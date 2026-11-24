"""make_pdf(path, markdown) -- the document tool for family doc_make.

Bionic audit (2026-10-02): its tool registry (bionic/llama-studio/src/lib/builtinTools.ts)
has `office_document` (DOCX/XLSX/PPTX via OfficeCLI, line 172) and no PDF tool, so
there is nothing to mirror for PDF. This is the minimal tool the task needs: the model
hands over Markdown, reportlab typesets it, the file lands inside the workspace jail.

Markdown subset: `#`/`##`/`###` headings, paragraphs, `-`/`*`/`1.` list items, pipe
tables (header row + `---` separator), **bold**. Anything else is plain text.
"""

from __future__ import annotations

import re
from io import BytesIO
from xml.sax.saxutils import escape

MAX_MD_CHARS = 20_000

_ASCII = {"–": "-", "—": "-", "‘": "'", "’": "'", "“": '"',
          "”": '"', "•": "-", " ": " ", "→": "->"}


def _clean(s: str) -> str:
    for k, v in _ASCII.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "replace").decode("latin-1")


def _inline(s: str) -> str:
    s = escape(_clean(s))
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return re.sub(r"`([^`]+)`", r"\1", s)


def _is_sep(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c.strip()) for c in cells)


def render(markdown: str) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    flow, bullets, rows = [], [], []

    def flush_bullets():
        if bullets:
            flow.append(ListFlowable([ListItem(Paragraph(_inline(b), ss["BodyText"])) for b in bullets],
                                     bulletType="bullet", start="-"))
            bullets.clear()

    def flush_table():
        if rows:
            data = [[Paragraph(_inline(c), ss["BodyText"]) for c in r] for r in rows]
            t = Table(data, repeatRows=1)
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                                   ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
            flow.append(t)
            flow.append(Spacer(1, 8))
            rows.clear()

    for raw in markdown.replace("\r\n", "\n").split("\n"):
        ln = raw.rstrip()
        if ln.lstrip().startswith("|"):
            flush_bullets()
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if not _is_sep(cells):
                rows.append(cells)
            continue
        flush_table()
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        b = re.match(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$", ln)
        if m:
            flush_bullets()
            style = ss["Heading%d" % min(len(m.group(1)), 3)]
            flow.append(Paragraph(_inline(m.group(2)), style))
        elif b:
            bullets.append(b.group(1))
        elif ln.strip():
            flush_bullets()
            flow.append(Paragraph(_inline(ln.strip()), ss["BodyText"]))
    flush_bullets()
    flush_table()
    if not flow:
        flow.append(Paragraph(" ", ss["BodyText"]))
    buf = BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, title="document").build(flow)
    return buf.getvalue()


def make_pdf(ws, args: dict) -> str:
    """Tool body: args {path, markdown}. Overwrites (she may fix and re-run)."""
    path, md = str(args["path"]), str(args["markdown"])
    if not path.lower().endswith(".pdf"):
        return f"error: path must end in .pdf (got {path!r})"
    if not md.strip():
        return "error: markdown is empty"
    if len(md) > MAX_MD_CHARS:
        return f"error: markdown is {len(md)} chars; the limit is {MAX_MD_CHARS}"
    f = ws.resolve(path)
    f.parent.mkdir(parents=True, exist_ok=True)
    data = render(md)
    f.write_bytes(data)
    from pypdf import PdfReader
    pages = len(PdfReader(BytesIO(data)).pages)
    return f"wrote {path} ({pages} page{'s' if pages != 1 else ''}, {len(data)} bytes)"
