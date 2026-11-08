from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    ListFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = ROOT / "deliverables"
OUT.mkdir(exist_ok=True)


TITLE_STYLE = ParagraphStyle(
    "TitleStyle",
    parent=getSampleStyleSheet()["Title"],
    fontName="Helvetica-Bold",
    fontSize=18,
    leading=22,
    textColor=colors.HexColor("#1f2d3d"),
    spaceAfter=12,
)

SUBTITLE_STYLE = ParagraphStyle(
    "SubtitleStyle",
    parent=getSampleStyleSheet()["BodyText"],
    fontName="Helvetica",
    fontSize=10,
    leading=14,
    textColor=colors.HexColor("#4b5563"),
    spaceAfter=8,
)

HEADING1 = ParagraphStyle(
    "Heading1",
    parent=getSampleStyleSheet()["Heading1"],
    fontName="Helvetica-Bold",
    fontSize=15,
    leading=18,
    spaceBefore=8,
    spaceAfter=6,
    textColor=colors.HexColor("#1f2d3d"),
)

HEADING2 = ParagraphStyle(
    "Heading2",
    parent=getSampleStyleSheet()["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=12,
    leading=15,
    spaceBefore=8,
    spaceAfter=5,
    textColor=colors.HexColor("#1f2d3d"),
)

HEADING3 = ParagraphStyle(
    "Heading3",
    parent=getSampleStyleSheet()["Heading3"],
    fontName="Helvetica-Bold",
    fontSize=10,
    leading=13,
    spaceBefore=6,
    spaceAfter=3,
    textColor=colors.HexColor("#1f2d3d"),
)

BODY = ParagraphStyle(
    "Body",
    parent=getSampleStyleSheet()["BodyText"],
    fontName="Helvetica",
    fontSize=9,
    leading=13,
    spaceAfter=4,
    textColor=colors.HexColor("#1f2937"),
)

BULLET = ParagraphStyle(
    "Bullet",
    parent=BODY,
    leftIndent=14,
    bulletIndent=8,
    spaceAfter=3,
)

TABLE_CELL = ParagraphStyle(
    "TableCell",
    parent=BODY,
    fontSize=8,
    leading=10,
    spaceAfter=0,
)

TABLE_HEADER = ParagraphStyle(
    "TableHeader",
    parent=TABLE_CELL,
    fontName="Helvetica-Bold",
    textColor=colors.HexColor("#0f172a"),
)


def sanitize_text(text: str) -> str:
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    # Convert bold markdown **text** to <b>text</b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Convert inline code `text` to font
    text = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)
    return text


def md_to_story(markdown: str, page_size, is_presentation: bool = False):
    story = []
    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            story.append(Spacer(1, 4))
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
            story.append(Paragraph(sanitize_text(stripped[5:]), ParagraphStyle("TinyBold", parent=BODY, fontName="Helvetica-Bold", fontSize=9, leading=12, spaceBefore=6, spaceAfter=3)))
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

        # Render Markdown tables
        if stripped.startswith("|") and stripped.endswith("|"):
            table_rows = []
            is_first = True
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                tl = lines[i].strip()
                # Skip markdown separator rows like |---|---|
                if not all(c in "| -:\t" for c in tl):
                    cells = [c.strip() for c in tl.strip("|").split("|")]
                    style_to_use = TABLE_HEADER if is_first else TABLE_CELL
                    table_rows.append([Paragraph(sanitize_text(c), style_to_use) for c in cells])
                    is_first = False
                i += 1
            if table_rows:
                tbl = Table(table_rows)
                tbl.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f8fafc")),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                    ('TOPPADDING', (0, 0), (-1, -1), 3),
                    ('LEFTPADDING', (0, 0), (-1, -1), 4),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ]))
                story.append(tbl)
                story.append(Spacer(1, 6))
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
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    )
    story = md_to_story(source_md.read_text(encoding="utf-8"), page_size=page_size, is_presentation=is_presentation)
    document.build(story)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Workstream 3 Deliverables
    ws3_report_md = DOCS / "CAPSTONE_REPORT_WORKSTREAM_3.md"
    ws3_presentation_md = DOCS / "PRESENTATION_SLIDES_WORKSTREAM_3.md"
    ws3_report_pdf = OUT / "Workstream_3_Report.pdf"
    ws3_presentation_pdf = OUT / "Workstream_3_Presentation.pdf"

    export_pdf(ws3_report_md, ws3_report_pdf, page_size=A4, landscape_mode=False, is_presentation=False)
    export_pdf(ws3_presentation_md, ws3_presentation_pdf, page_size=A4, landscape_mode=True, is_presentation=True)

    # 2. Grand Capstone Integrated Deliverables
    grand_report_md = DOCS / "GRAND_CAPSTONE_FINAL_REPORT.md"
    grand_presentation_md = DOCS / "GRAND_CAPSTONE_FINAL_PRESENTATION.md"
    grand_report_pdf = OUT / "Grand_Capstone_Final_Report.pdf"
    grand_presentation_pdf = OUT / "Grand_Capstone_Final_Presentation.pdf"

    export_pdf(grand_report_md, grand_report_pdf, page_size=A4, landscape_mode=False, is_presentation=False)
    export_pdf(grand_presentation_md, grand_presentation_pdf, page_size=A4, landscape_mode=True, is_presentation=True)

    print("Successfully exported all PDF deliverables:")
    print(f"  - {ws3_report_pdf}")
    print(f"  - {ws3_presentation_pdf}")
    print(f"  - {grand_report_pdf}")
    print(f"  - {grand_presentation_pdf}")
