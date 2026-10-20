"""Build REPORT.pdf from REPORT.md (cover page, styled tables, cropped screenshots, captions)."""

import re
import tempfile
from pathlib import Path

from PIL import Image as PILImage
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "REPORT.md"
OUT = ROOT / "REPORT.pdf"

NAVY = colors.HexColor("#14325c")
BLUE = colors.HexColor("#2b6cb0")
LIGHT = colors.HexColor("#eaf1fb")
GREY = colors.HexColor("#5a6472")
GRID = colors.HexColor("#c5d0de")
STUDENT = "Akhilesh Gadagkar"
STUDENT_ID = "2025AE05712"

PAGE_W, PAGE_H = A4
MARGIN = 2 * cm
CONTENT_W = PAGE_W - 2 * MARGIN
TMP = Path(tempfile.mkdtemp(prefix="report_img_"))

ss = getSampleStyleSheet()
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontName="Helvetica", fontSize=10, leading=14.5, spaceAfter=6)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=16, bulletIndent=4, spaceAfter=3)
H1 = ParagraphStyle("H1", parent=BODY, fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=NAVY, spaceBefore=16, spaceAfter=2, keepWithNext=1)
H2 = ParagraphStyle("H2", parent=BODY, fontName="Helvetica-Bold", fontSize=12, leading=16, textColor=BLUE, spaceBefore=10, spaceAfter=4, keepWithNext=1)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=8.8, leading=11.5, spaceAfter=0)
CELL_H = ParagraphStyle("CellH", parent=CELL, fontName="Helvetica-Bold", textColor=colors.white)
CAPTION = ParagraphStyle("Caption", parent=BODY, fontSize=8.8, leading=11.5, textColor=GREY, alignment=TA_CENTER, spaceBefore=3, spaceAfter=12)
COVER_TITLE = ParagraphStyle("CoverTitle", parent=BODY, fontName="Helvetica-Bold", fontSize=26, leading=32, textColor=NAVY, spaceAfter=6)
COVER_SUB = ParagraphStyle("CoverSub", parent=BODY, fontSize=14, leading=19, textColor=BLUE, spaceAfter=18)


def inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font name="Courier" size="9">\1</font>', text)
    return text


# Fractions (x0, y0, x1, y1) of the original capture; the top 158 px (URL bar with session token) is always dropped.
TERMINAL = (0.23, 0.17, 0.915, 0.995)
CROPS = {
    "eda_summary": TERMINAL,
    "pytest_results": TERMINAL,
    "api_metrics2": TERMINAL,
    "api_health": (0, 0, 0.86, 0.40),
    "api_metrics": (0, 0, 0.55, 0.62),
    "api_swagger": (0, 0, 1, 0.885),
    "ci_pass": (0, 0, 1, 0.885),
}


def prepare_image(rel_path: str) -> Path:
    """Crop the browser chrome (URL bar with session token) off full-desktop captures and compress."""
    img = PILImage.open(ROOT / rel_path).convert("RGB")
    if img.width >= 3000 and img.height > 1000:
        x0, y0, x1, y1 = CROPS.get(Path(rel_path).stem, (0, 0, 1, 1))
        top = max(158, round(y0 * img.height))
        img = img.crop((round(x0 * img.width), top, round(x1 * img.width), round(y1 * img.height)))
    if img.width > 1800:
        img = img.resize((1800, round(img.height * 1800 / img.width)), PILImage.LANCZOS)
    out = TMP / (Path(rel_path).stem + ".jpg")
    img.save(out, "JPEG", quality=88, optimize=True)
    return out


def figure(caption: str, rel_path: str):
    path = prepare_image(rel_path)
    w, h = PILImage.open(path).size
    width = CONTENT_W * (0.72 if rel_path.startswith("reports/") else 1)
    img = Image(str(path), width=width, height=width * h / w)
    img.hAlign = "CENTER"
    box = Table([[img]], colWidths=[width])
    box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, GRID), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    m = re.match(r"(Figure \d+):\s*(.*)", caption)
    cap = f"<b>{m.group(1)}.</b> {inline(m.group(2))}" if m else inline(caption)
    return KeepTogether([box, Paragraph(cap, CAPTION)])


def make_table(rows: list[list[str]], width: float = CONTENT_W) -> Table:
    ncol = len(rows[0])
    weights = [max(8, min(45, max(len(r[i]) for r in rows))) for i in range(ncol)]
    widths = [width * w / sum(weights) for w in weights]
    data = [[Paragraph(inline(c), CELL_H if r == 0 else CELL) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def architecture() -> Drawing:
    w, h = CONTENT_W, 250
    d = Drawing(w, h)

    def box(x, y, bw, bh, title, sub, fill):
        d.add(Rect(x, y, bw, bh, rx=6, ry=6, fillColor=fill, strokeColor=NAVY, strokeWidth=0.8))
        d.add(String(x + bw / 2, y + bh / 2 + 3, title, textAnchor="middle", fontName="Helvetica-Bold", fontSize=8.5, fillColor=colors.white))
        d.add(String(x + bw / 2, y + bh / 2 - 9, sub, textAnchor="middle", fontName="Helvetica", fontSize=7, fillColor=colors.white))

    def arrow(x1, y1, x2, y2):
        d.add(Line(x1, y1, x2, y2, strokeColor=GREY, strokeWidth=1.2))
        if x1 != x2:
            s = 1 if x2 > x1 else -1
            d.add(Polygon([x2, y2, x2 - 6 * s, y2 + 3, x2 - 6 * s, y2 - 3], fillColor=GREY, strokeColor=GREY))
        else:
            s = 1 if y2 > y1 else -1
            d.add(Polygon([x2, y2, x2 - 3, y2 - 6 * s, x2 + 3, y2 - 6 * s], fillColor=GREY, strokeColor=GREY))

    bw, bh, gap = 100, 42, (w - 4 * 100) / 3
    xs = [i * (bw + gap) for i in range(4)]
    y_top, y_mid, y_low = 195, 110, 30
    top = [("UCI Dataset", "download_data.py", BLUE), ("EDA + Cleaning", "scripts/eda.py", BLUE), ("Train + Tune", "5-fold CV, 2 models", BLUE), ("MLflow Tracking", "params, metrics, plots", BLUE)]
    mid = [("model.joblib", "pipeline + metrics.json", NAVY), ("FastAPI Service", "/predict /health /metrics", NAVY), ("Docker Image", "python:3.11-slim", NAVY), ("Kubernetes", "2 replicas, probes", NAVY)]
    for x, (t, s, c) in zip(xs, top):
        box(x, y_top, bw, bh, t, s, c)
    for x, (t, s, c) in zip(xs, mid):
        box(x, y_mid, bw, bh, t, s, c)
    for i in range(3):
        arrow(xs[i] + bw, y_top + bh / 2, xs[i + 1], y_top + bh / 2)
        arrow(xs[i] + bw, y_mid + bh / 2, xs[i + 1], y_mid + bh / 2)
    mid_gap = (y_top + y_mid + bh) / 2
    d.add(Line(xs[2] + bw / 2, y_top, xs[2] + bw / 2, mid_gap, strokeColor=GREY, strokeWidth=1.2))
    d.add(Line(xs[2] + bw / 2, mid_gap, xs[0] + bw / 2, mid_gap, strokeColor=GREY, strokeWidth=1.2))
    arrow(xs[0] + bw / 2, mid_gap, xs[0] + bw / 2, y_mid + bh)
    box(xs[1], y_low, bw + 40, bh - 6, "Monitoring", "Prometheus metrics + api.log", colors.HexColor("#2f855a"))
    arrow(xs[1] + bw / 2, y_mid, xs[1] + bw / 2, y_low + bh - 6)
    d.add(Rect(xs[3] - 40, y_low, bw + 40, bh - 6, rx=6, ry=6, fillColor=colors.HexColor("#b7791f"), strokeColor=NAVY, strokeWidth=0.8))
    d.add(String(xs[3] + bw / 2 - 20, y_low + 18, "GitHub Actions CI/CD", textAnchor="middle", fontName="Helvetica-Bold", fontSize=8.5, fillColor=colors.white))
    d.add(String(xs[3] + bw / 2 - 20, y_low + 7, "lint, test, train, build", textAnchor="middle", fontName="Helvetica", fontSize=7, fillColor=colors.white))
    return d


def parse_blocks(lines: list[str]):
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            yield "table", rows
        else:
            yield "line", line
            i += 1


def cover(title: str, subtitle: str, details: list[list[str]], headings: list[str]):
    band = Table([[Paragraph("BITS Pilani - Machine Learning Operations", ParagraphStyle("b", parent=BODY, textColor=colors.white, fontName="Helvetica-Bold", fontSize=11))]], colWidths=[CONTENT_W])
    band.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), NAVY), ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10), ("LEFTPADDING", (0, 0), (-1, -1), 12)]))
    kv = [[Paragraph(f"<b>{inline(k)}</b>", ParagraphStyle("k", parent=CELL, fontSize=10.5, leading=14, textColor=NAVY)), Paragraph(inline(v), ParagraphStyle("v", parent=CELL, fontSize=10.5, leading=14))] for k, v in details[1:]]
    info = Table(kv, colWidths=[CONTENT_W * 0.30, CONTENT_W * 0.70])
    info.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, GRID),
        ("BOX", (0, 0), (-1, -1), 0.8, NAVY),
        ("LINEBEFORE", (0, 0), (0, -1), 4, BLUE),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
    ]))
    kpi_style = ParagraphStyle("kpi", parent=BODY, alignment=TA_CENTER, fontSize=9, leading=12, textColor=GREY, spaceAfter=0)
    kpi_val = ParagraphStyle("kv", parent=kpi_style, fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=NAVY)
    kpis = [("88.52%", "Test accuracy"), ("0.9654", "Test ROC-AUC"), ("20 / 20", "Unit tests passed"), ("2", "Models compared")]
    kpi = Table([[Paragraph(v, kpi_val) for v, _ in kpis], [Paragraph(t, kpi_style) for _, t in kpis]], colWidths=[CONTENT_W / 4] * 4)
    kpi.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, GRID), ("LINEAFTER", (0, 0), (-2, -1), 0.5, GRID), ("TOPPADDING", (0, 0), (-1, 0), 10), ("BOTTOMPADDING", (0, 1), (-1, 1), 10)]))
    toc = [Paragraph("<b>Contents</b>", ParagraphStyle("t", parent=BODY, textColor=NAVY, fontSize=11))] + [Paragraph(inline(h), ParagraphStyle("toc", parent=BULLET, leading=12, spaceAfter=1)) for h in headings]
    return [band, Spacer(1, 30), Paragraph(inline(title), COVER_TITLE), Paragraph(inline(subtitle), COVER_SUB),
            HRFlowable(width="100%", thickness=1.5, color=BLUE, spaceAfter=18), info, Spacer(1, 22), kpi, Spacer(1, 26), *toc, PageBreak()]


def decorate(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(MARGIN, PAGE_H - 1.2 * cm, "AIMLCZG523 - Machine Learning Operations - Assignment 01")
    canvas.setStrokeColor(GRID)
    canvas.line(MARGIN, PAGE_H - 1.35 * cm, PAGE_W - MARGIN, PAGE_H - 1.35 * cm)
    canvas.line(MARGIN, 1.4 * cm, PAGE_W - MARGIN, 1.4 * cm)
    canvas.drawString(MARGIN, 0.9 * cm, f"{STUDENT}  |  {STUDENT_ID}")
    canvas.drawRightString(PAGE_W - MARGIN, 0.9 * cm, f"Page {doc.page}")
    canvas.restoreState()


def main() -> None:
    lines = SRC.read_text().splitlines()
    blocks = list(parse_blocks(lines))
    title = subtitle = ""
    details: list[list[str]] = []
    headings: list[str] = []
    body = []
    seen_table = False
    for kind, val in blocks:
        if kind == "table" and not seen_table:
            details, seen_table = val, True
            continue
        if kind == "line":
            if val.startswith("# ") and not title:
                title = val[2:]
                continue
            if val.startswith("## ") and not subtitle:
                subtitle = val[3:]
                continue
            if val.startswith("# "):
                headings.append(val[2:])
        body.append((kind, val))

    story = cover(title, subtitle, details, headings)
    for kind, val in body:
        if kind == "table":
            story += [make_table(val), Spacer(1, 10)]
        elif val.startswith("# "):
            head = Table([[Paragraph(inline(val[2:]), H1)]], colWidths=[CONTENT_W])
            head.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 1, BLUE), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
            head.keepWithNext = True
            head.spaceAfter = 6
            story.append(head)
        elif val.startswith("## "):
            story.append(Paragraph(inline(val[3:]), H2))
        elif val == "[[ARCHITECTURE]]":
            story += [architecture(), Paragraph("<b>Architecture.</b> Data flows left to right; CI/CD automates lint, tests, training and container build.", CAPTION)]
        elif (m := re.match(r"!\[(.*)\]\((.*)\)", val)):
            story.append(figure(m.group(1), m.group(2)))
        elif val.startswith("- "):
            story.append(Paragraph(inline(val[2:]), BULLET, bulletText="\u2022"))
        elif (m := re.match(r"(\d+)\.\s+(.*)", val)):
            story.append(Paragraph(inline(m.group(2)), BULLET, bulletText=f"{m.group(1)}."))
        else:
            story.append(Paragraph(inline(val), BODY))

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=2 * cm, bottomMargin=2 * cm,
                            title=f"{title} - {STUDENT} ({STUDENT_ID})", author=f"{STUDENT} ({STUDENT_ID})")
    doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=decorate)
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
