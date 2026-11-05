from __future__ import annotations

from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, PageBreak, SimpleDocTemplate, Spacer, ListFlowable


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = ROOT / "deliverables"
OUT.mkdir(exist_ok=True)


TITLE_STYLE = ParagraphStyle(
    "TitleStyle",
    parent=getSampleStyleSheet()["Title"],
    fontName="Helvetica-Bold",
    fontSize=20,
    leading=24,
    textColor=colors.HexColor("#1f2d3d"),
    spaceAfter=16,
)

SUBTITLE_STYLE = ParagraphStyle(
    "SubtitleStyle",
    parent=getSampleStyleSheet()["BodyText"],
    fontName="Helvetica",
    fontSize=11,
    leading=14,
    textColor=colors.HexColor("#4b5563"),
    spaceAfter=10,
)

HEADING1 = ParagraphStyle(
    "Heading1",
    parent=getSampleStyleSheet()["Heading1"],
    fontName="Helvetica-Bold",
    fontSize=16,
    leading=20,
    spaceBefore=8,
    spaceAfter=8,
    textColor=colors.HexColor("#1f2d3d"),
)

HEADING2 = ParagraphStyle(
    "Heading2",
    parent=getSampleStyleSheet()["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=13,
    leading=16,
    spaceBefore=10,
    spaceAfter=6,
    textColor=colors.HexColor("#1f2d3d"),
)

HEADING3 = ParagraphStyle(
    "Heading3",
    parent=getSampleStyleSheet()["Heading3"],
    fontName="Helvetica-Bold",
    fontSize=11,
    leading=14,
    spaceBefore=8,
    spaceAfter=4,
    textColor=colors.HexColor("#1f2d3d"),
)

BODY = ParagraphStyle(
    "Body",
    parent=getSampleStyleSheet()["BodyText"],
    fontName="Helvetica",
    fontSize=10,
    leading=14,
    spaceAfter=5,
    textColor=colors.HexColor("#1f2937"),
)

BULLET = ParagraphStyle(
    "Bullet",
    parent=BODY,
    leftIndent=18,
    bulletIndent=10,
    spaceAfter=4,
)


def sanitize_text(text: str) -> str:
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace("_", "_")
    return text


def md_to_story(markdown: str, page_size, is_presentation: bool = False):
    story = []
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            story.append(Spacer(1, 6))
            i += 1
            continue

        if stripped == "---":
            story.append(PageBreak())
            i += 1
            continue

        if stripped.startswith("# "):
            title = stripped[2:]
            story.append(Paragraph(sanitize_text(title), TITLE_STYLE))
            i += 1
            continue

        if stripped.startswith("## "):
            story.append(Paragraph(sanitize_text(stripped[3:]), HEADING1 if is_presentation else HEADING2))
            i += 1
            continue

        if stripped.startswith("### "):
            story.append(Paragraph(sanitize_text(stripped[4:]), HEADING3))
            i += 1
            continue

        if stripped.startswith("#### "):
            story.append(Paragraph(sanitize_text(stripped[5:]), ParagraphStyle("TinyBold", parent=BODY, fontName="Helvetica-Bold", fontSize=9, leading=12, spaceBefore=8, spaceAfter=4)))
            i += 1
            continue

        if stripped.startswith("- ") or stripped.startswith("* "):
            bullet = stripped[2:].strip()
            story.append(Paragraph(f"• {sanitize_text(bullet)}", BULLET))
            i += 1
            continue

        if re.match(r"^\d+\.\s", stripped):
            numbered = stripped.split(". ", 1)
            if len(numbered) == 2:
                story.append(Paragraph(f"{numbered[0]}. {sanitize_text(numbered[1])}", BODY))
            else:
                story.append(Paragraph(sanitize_text(stripped), BODY))
            i += 1
            continue

        if stripped.startswith("**") and stripped.endswith("**"):
            story.append(Paragraph(sanitize_text(stripped[2:-2]), SUBTITLE_STYLE))
            i += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            i += 1
            continue

        story.append(Paragraph(sanitize_text(stripped), BODY))
        i += 1

    return story


def export_pdf(source_md: Path, output_pdf: Path, page_size=A4, landscape_mode: bool = False, is_presentation: bool = False):
    if landscape_mode:
        page_size = landscape(page_size)
    document = SimpleDocTemplate(
        str(output_pdf),
        pagesize=page_size,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )
    story = md_to_story(source_md.read_text(encoding="utf-8"), page_size=page_size, is_presentation=is_presentation)
    document.build(story)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    report_md = DOCS / "CAPSTONE_REPORT_WORKSTREAM_3.md"
    presentation_md = DOCS / "PRESENTATION_SLIDES_WORKSTREAM_3.md"

    report_pdf = OUT / "Workstream_3_Report.pdf"
    presentation_pdf = OUT / "Workstream_3_Presentation.pdf"

    export_pdf(report_md, report_pdf, page_size=A4, landscape_mode=False, is_presentation=False)
    export_pdf(presentation_md, presentation_pdf, page_size=A4, landscape_mode=True, is_presentation=True)

    print(f"Report PDF: {report_pdf}")
    print(f"Presentation PDF: {presentation_pdf}")
