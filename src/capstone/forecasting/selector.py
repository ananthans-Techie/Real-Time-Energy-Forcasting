"""Dynamic Model Selection Router and Performance Tracker for Workstream 2.

Implements:
  1. PerformanceTracker: Rolling 24-hour window tracking actual vs predicted values (MAE, RMSE, latency).
  2. DynamicModelSelector: Multi-objective utility routing policy:
       U(m) = w1 * Accuracy(m) - w2 * Latency(m) - w3 * ResourceCost(m)
     with emergency triggers (degraded accuracy, latency spikes, drift signals) and hysteresis dwell-time.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .. import constants as C
from ..events import build_model_event

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Historical tracking stats for a specific model."""

    predictions: deque[tuple[str, float]] = field(default_factory=lambda: deque(maxlen=200))  # (event_time, pred)
    actuals: deque[tuple[str, float]] = field(default_factory=lambda: deque(maxlen=200))      # (event_time, actual)
    errors: deque[float] = field(default_factory=lambda: deque(maxlen=200))                   # |pred - actual|
    sq_errors: deque[float] = field(default_factory=lambda: deque(maxlen=200))                # (pred - actual)^2
    latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=200))

    def record_prediction(self, event_time: str, prediction: float, latency_ms: float) -> None:
        self.predictions.append((event_time, prediction))
        self.latencies_ms.append(max(0.0, float(latency_ms)))

    def record_actual(self, event_time: str, actual: float) -> None:
        self.actuals.append((event_time, actual))
        # Match with prediction for this same event_time
        for t, pred in reversed(self.predictions):
            if t == event_time:
                err = abs(pred - actual)
                self.errors.append(err)
                self.sq_errors.append(err ** 2)
                break

    def rolling_mae(self, window: int = 24) -> float:
        if not self.errors:
            return 0.35  # default prior
        recent = list(self.errors)[-window:]
        return float(np.mean(recent))

    def rolling_rmse(self, window: int = 24) -> float:
        if not self.sq_errors:
            return 0.50  # default prior
        recent = list(self.sq_errors)[-window:]
        return float(np.sqrt(np.mean(recent)))

    def avg_latency_ms(self, window: int = 24) -> float:
        if not self.latencies_ms:
            return 1.0
        recent = list(self.latencies_ms)[-window:]
        return float(np.mean(recent))


class DynamicModelSelector:
    """Multi-Objective Dynamic Model Selection Policy and Router.

    Objective:
      U(m) = w_acc * Accuracy(m) - w_lat * Latency(m) - w_cost * ResourceCost(m)
    """

    DEFAULT_RESOURCE_COSTS = {
        "lstm": 0.70,        # Higher GPU/CPU compute requirements
        "sarima": 0.25,      # Medium statistical filter overhead
        "lightweight": 0.05, # Ultra-low CPU/memory footprint
    }

    def __init__(
        self,
        candidate_models: list[str] | None = None,
        default_model: str = "lstm",
        w_acc: float = 0.60,
        w_lat: float = 0.25,
        w_cost: float = 0.15,
        rolling_window_hours: int = 24,
        switch_margin: float = 0.05,
        min_dwell_hours: int = 6,
        error_threshold_mae: float = 0.65,
        latency_budget_ms: float = 50.0,
        resource_costs: Mapping[str, float] | None = None,
    ) -> None:
        self.candidate_models = candidate_models or ["lstm", "sarima", "lightweight"]
        self.active_model = default_model if default_model in self.candidate_models else self.candidate_models[0]

        # Multi-objective weights
        total_w = w_acc + w_lat + w_cost
        self.w_acc = w_acc / total_w
        self.w_lat = w_lat / total_w
        self.w_cost = w_cost / total_w

        self.rolling_window = rolling_window_hours
        self.switch_margin = switch_margin
        self.min_dwell_hours = min_dwell_hours
        self.error_threshold_mae = error_threshold_mae
        self.latency_budget_ms = latency_budget_ms
        self.resource_costs = dict(resource_costs or self.DEFAULT_RESOURCE_COSTS)

        # Performance trackers per model
        self.trackers: dict[str, ModelPerformance] = {
            m: ModelPerformance() for m in self.candidate_models
        }

        # Switch management state
        self.hours_since_switch: int = min_dwell_hours
        self.last_switch_time: str | None = None
        self.switch_history: list[dict[str, Any]] = []

    def compute_accuracy_score(self, model: str) -> float:
        """Accuracy score in (0, 1], where 1 is perfect (0 error)."""
        mae = self.trackers[model].rolling_mae(self.rolling_window)
        # Normalized hyperbolic score: 1.0 / (1.0 + MAE)
        return float(1.0 / (1.0 + max(0.0, mae)))

    def compute_latency_score(self, model: str) -> float:
        """Normalized latency penalty in [0, 1]."""
        lat = self.trackers[model].avg_latency_ms(self.rolling_window)
        return float(min(1.0, lat / max(1.0, self.latency_budget_ms)))

    def compute_resource_score(self, model: str) -> float:
        """Resource penalty in [0, 1]."""
        return float(self.resource_costs.get(model, 0.5))

    def compute_utility(self, model: str) -> float:
        """Compute scalar utility U(m)."""
        acc = self.compute_accuracy_score(model)
        lat = self.compute_latency_score(model)
        cost = self.compute_resource_score(model)
        return float(self.w_acc * acc - self.w_lat * lat - self.w_cost * cost)

    def evaluate_selection(
        self,
        event_time_iso: str,
        drift_signal: bool = False,
        observed_latency_ms: float | None = None,
    ) -> tuple[str, bool, str, float | None, dict[str, Any]]:
        """Evaluate whether to switch active model.

        Returns:
            (selected_model, switched_bool, trigger_metric, trigger_value, score_snapshot)
        """
        self.hours_since_switch += 1
        active = self.active_model
        tracker = self.trackers[active]
        rolling_mae = tracker.rolling_mae(self.rolling_window)
        curr_lat = observed_latency_ms if observed_latency_ms is not None else tracker.avg_latency_ms()

        # Compute utility scores across all candidates
        scores = {m: self.compute_utility(m) for m in self.candidate_models}
        snapshot = {
            "utilities": scores,
            "active_model": active,
            "rolling_mae": rolling_mae,
            "latency_ms": curr_lat,
            "hours_since_switch": self.hours_since_switch,
        }

        # -----------------------------------------------------------------
        # Emergency Check 1: External Concept Drift Signal
        # -----------------------------------------------------------------
        if drift_signal:
            alt_candidates = [m for m in self.candidate_models if m != active]
            best_alt = max(alt_candidates, key=lambda m: scores[m])
            self._apply_switch(best_alt, event_time_iso)
            return best_alt, True, "drift_signal", 1.0, snapshot

        # -----------------------------------------------------------------
        # Emergency Check 2: Latency Spike Violation
        # -----------------------------------------------------------------
        if curr_lat > self.latency_budget_ms:
            # Switch to faster/lightweight model
            fallback = "sarima" if "sarima" in self.candidate_models and active != "sarima" else "lightweight"
            if fallback in self.candidate_models and fallback != active:
                self._apply_switch(fallback, event_time_iso)
                return fallback, True, "latency_spike", float(curr_lat), snapshot

        # -----------------------------------------------------------------
        # Emergency Check 3: Degraded Accuracy
        # -----------------------------------------------------------------
        if rolling_mae > self.error_threshold_mae:
            # Pick best alternative model by utility
            best_alt = max([m for m in self.candidate_models if m != active], key=lambda m: scores[m])
            self._apply_switch(best_alt, event_time_iso)
            return best_alt, True, "mae_threshold", float(rolling_mae), snapshot

        # -----------------------------------------------------------------
        # Routine Scheduled Selection with Hysteresis Dwell-Time
        # -----------------------------------------------------------------
        best_model = max(self.candidate_models, key=lambda m: scores[m])
        if best_model != active:
            score_diff = scores[best_model] - scores[active]
            if self.hours_since_switch >= self.min_dwell_hours and score_diff > self.switch_margin:
                self._apply_switch(best_model, event_time_iso)
                return best_model, True, "utility_score", float(score_diff), snapshot

        return active, False, "none", None, snapshot

    def _apply_switch(self, new_model: str, event_time_iso: str) -> None:
        logger.info(f"DynamicModelSelector: Switching active model from '{self.active_model}' -> '{new_model}' at {event_time_iso}")
        self.active_model = new_model
        self.hours_since_switch = 0
        self.last_switch_time = event_time_iso

    def create_model_event(
        self,
        run_id: str,
        meter_id: str,
        event_time_iso: str,
        prev_model: str,
        new_model: str,
        trigger_metric: str,
        trigger_value: float | None = None,
        reason: str | None = None,
        metrics: dict[str, Any] | None = None,
        trigger_event_id: str | None = None,
        trigger_replay_time_ms: int | None = None,
        seq: int | None = None,
    ) -> dict[str, Any]:
        """Build structured energy.model_events compliant payload."""
        desc = reason or f"Model switched from '{prev_model}' to '{new_model}' due to {trigger_metric}."
        ev = build_model_event(
            run_id=run_id,
            meter_id=meter_id,
            event_time_iso=event_time_iso,
            event_type="model_switch",
            previous_model=prev_model,
            new_model=new_model,
            trigger_metric=trigger_metric,
            trigger_value=trigger_value,
            reason=desc,
            metrics=metrics,
            trigger_event_id=trigger_event_id,
            trigger_replay_time_ms=trigger_replay_time_ms,
            seq=seq,
        )
        self.switch_history.append(ev)
        return ev
