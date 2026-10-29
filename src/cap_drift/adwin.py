"""ADWIN (Adaptive Windowing) and Page-Hinkley Change Detection Algorithms.

Reference:
  Albert Bifet and Ricard Gavaldà (2007).
  "Learning from Time-Changing Data with Adaptive Windowing."
  SIAM International Conference on Data Mining (SDM).
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np


class ADWIN:
    """Adaptive Windowing algorithm for concept drift detection."""

    def __init__(
        self,
        delta: float = 0.002,
        max_window_size: int = 500,
        min_subwindow_size: int = 5,
        step: int = 2,
    ) -> None:
        """Initialize ADWIN.

        Args:
            delta: Confidence value in (0, 1). Smaller delta means fewer false alarms.
            max_window_size: Maximum memory bound for the sliding window.
            min_subwindow_size: Minimum elements in subwindows W0, W1 to test.
            step: Step size for evaluating candidate splits (optimizes search time).
        """
        self.delta = float(delta)
        self.max_window_size = int(max_window_size)
        self.min_subwindow_size = int(min_subwindow_size)
        self.step = max(1, int(step))

        self.window: deque[float] = deque(maxlen=self.max_window_size)
        self.total: float = 0.0
        self.width: int = 0

    def add_element(self, value: float) -> tuple[bool, float, float, float]:
        """Insert a new observation into ADWIN.

        Returns:
            (drift_detected, drift_score, mean_before, mean_after)
        """
        x = float(value)
        self.window.append(x)
        self.width = len(self.window)
        self.total += x

        drift_detected = False
        max_ratio = 0.0
        best_mu0 = float(np.mean(self.window))
        best_mu1 = best_mu0

        # Check subwindow splits W = W0 + W1
        n = self.width
        if n >= 2 * self.min_subwindow_size:
            arr = np.array(self.window, dtype=float)
            cumsum = np.cumsum(arr)
            total_sum = cumsum[-1]

            # Evaluate split points
            split_candidates = range(self.min_subwindow_size, n - self.min_subwindow_size + 1, self.step)
            for split in split_candidates:
                n0 = split
                n1 = n - split
                sum0 = cumsum[split - 1]
                sum1 = total_sum - sum0
                mu0 = sum0 / n0
                mu1 = sum1 / n1
                diff = abs(mu0 - mu1)

                m = (n0 * n1) / n  # harmonic mean / 2
                delta_prime = self.delta / n
                eps_cut = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 / max(1e-12, delta_prime)))

                ratio = diff / max(1e-8, eps_cut)
                if ratio > max_ratio:
                    max_ratio = ratio
                    best_mu0 = mu0
                    best_mu1 = mu1

                if diff >= eps_cut:
                    drift_detected = True
                    # Drop older observations from W0
                    drop_count = max(1, n0 // 2)
                    for _ in range(drop_count):
                        if self.window:
                            dropped = self.window.popleft()
                            self.total -= dropped
                    self.width = len(self.window)
                    break

        return drift_detected, float(max_ratio), float(best_mu0), float(best_mu1)

    def reset(self) -> None:
        """Clear the window."""
        self.window.clear()
        self.total = 0.0
        self.width = 0


class PageHinkley:
    """Page-Hinkley sequential analysis change detection test."""

    def __init__(
        self,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 1.0 - 0.0001,
        min_instances: int = 10,
    ) -> None:
        """Initialize Page-Hinkley test.

        Args:
            delta: Permissible error parameter.
            threshold: Cumulative deviation detection threshold.
            alpha: Forgetting factor for updating the running mean.
            min_instances: Minimum observations before allowing detections.
        """
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.min_instances = min_instances

        self.x_mean = 0.0
        self.sum = 0.0
        self.min_sum = 0.0
        self.n = 0
        self.history: list[float] = []

    def add_element(self, value: float) -> tuple[bool, float, float, float]:
        """Add observation and check for change.

        Returns:
            (drift_detected, drift_score, mean_before, mean_after)
        """
        x = float(value)
        self.n += 1
        self.history.append(x)
        if len(self.history) > 200:
            self.history.pop(0)

        # Update running mean with exponential forgetting
        self.x_mean = self.x_mean + (x - self.x_mean) / self.n
        self.sum = self.sum + (x - self.x_mean - self.delta)
        if self.sum < self.min_sum:
            self.min_sum = self.sum

        ph_value = self.sum - self.min_sum
        score = ph_value / max(1.0, self.threshold)

        drift_detected = bool(self.n >= self.min_instances and ph_value > self.threshold)
        mean_before = float(self.x_mean)
        mean_after = float(x)

        if drift_detected:
            self.reset()

        return drift_detected, float(score), mean_before, mean_after

    def reset(self) -> None:
        """Reset test statistics."""
        self.sum = 0.0
        self.min_sum = 0.0
        self.n = 0
