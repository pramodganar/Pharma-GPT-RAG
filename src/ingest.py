"""Parse the WHO/PPRI glossary PDF into structured term entries.

pdfplumber renders the document's bold text (term headings and inline emphasised
terms) as every glyph repeated four times, e.g. "AAAABBBBCCCC" for "ABC". That
4x pattern is exploited both to recover the real text and to tell headings from
body lines. See DECISIONS.md for why pdfplumber replaced pypdf here.
"""

import json
import re

import pdfplumber

from . import config as cfg

# A run of one character repeated 4+ times (a single bold glyph).
_RUN = re.compile(r"(.)\1{3,}")
# A bold region: two or more consecutive glyph runs. Requiring two guards against
# collapsing an incidental run in body text (e.g. the zeros in "10000").
_BOLD_REGION = re.compile(r"(?:(.)\1{3,}){2,}")
_CID = re.compile(r"\(cid:\d+\)")
# The PDF's list-bullet glyphs decode inconsistently (»/•/(cid:2)); drop them all.
_BULLET = re.compile(r"[»•▪●]")
_SOURCE_OPEN = re.compile(r"^\s*\[Source", re.IGNORECASE)


def debold(s):
    """Collapse pdfplumber's 4x bold rendering back to plain text."""
    return _BOLD_REGION.sub(
        lambda m: _RUN.sub(lambda r: r.group(1) * (len(r.group(0)) // 4), m.group(0)),
        s,
    )


def _clean(line):
    line = _CID.sub(" ", line)
    line = _BULLET.sub(" ", line)
    return re.sub(r"[ \t]+", " ", line).strip()


def _is_footer(line):
    s = line.strip()
    if re.fullmatch(r"\d+", s):
        return True
    return "Collaborating Centre" in s and "Glossary 2016" in s


def _is_heading(raw):
    s = raw.strip()
    if not s:
        return False
    d = debold(s)
    if d == s or len(d) < 2:
        return False
    # Term headings start with a letter and never contain a colon; a bold bullet
    # list item inside an entry fails both.
    if not d[0].isalnum() or ":" in d:
        return False
    # A real heading is entirely bold, so every glyph is quadrupled and the raw
    # line is ~4x its debolded length. A line that merely starts with a bold
    # sub-label followed by normal text scores well under 3.
    return len(s) / len(d) >= 3.0


def _is_back_matter(line):
    return line.strip().lower().startswith("list of references")


def _finish_source(text):
    text = text.strip()
    text = re.sub(r"^\[Source:\s*", "", text, flags=re.IGNORECASE)
    return text.rstrip("]").strip()


def parse_pages(pages, start_page=1):
    """Parse a list of raw page texts (bold still 4x-rendered) into entries.

    pages[i] is the text of physical page start_page + i.
    """
    entries = []
    current = None
    source_buf = None  # open, unclosed [Source: ...] accumulator

    def flush():
        nonlocal current
        if current and current["definition_text"].strip():
            current["definition_text"] = current["definition_text"].strip()
            entries.append(current)
        current = None

    for i, text in enumerate(pages):
        pageno = start_page + i
        prev_heading = False
        for raw in (text or "").splitlines():
            if _is_footer(raw):
                continue
            if _is_back_matter(debold(raw)):
                flush()
                return entries

            heading = _is_heading(raw)
            line = _clean(debold(raw))
            if not line:
                continue

            if source_buf is not None and not heading:
                source_buf += " " + line
                if "]" in line:
                    current["sources"].append(_finish_source(source_buf))
                    source_buf = None
                prev_heading = False
                continue

            if heading:
                if prev_heading and current is not None:
                    current["term"] = (current["term"] + " " + line).strip()
                else:
                    flush()
                    current = {
                        "term": line,
                        "definition_text": "",
                        "sources": [],
                        "page_start": pageno,
                    }
                prev_heading = True
                continue

            prev_heading = False
            # A stray [Source: line before any heading has no entry to attach to.
            if _SOURCE_OPEN.match(line) and current is not None:
                if "]" in line:
                    current["sources"].append(_finish_source(line))
                else:
                    source_buf = line
            elif current is not None:
                current["definition_text"] += line + "\n"

    flush()
    return entries


def parse_pdf(pdf_path=None):
    pdf_path = str(pdf_path or cfg.RAW_PDF)
    with pdfplumber.open(pdf_path) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    content = pages[cfg.FRONT_MATTER_LAST_PAGE:]
    return parse_pages(content, start_page=cfg.FRONT_MATTER_LAST_PAGE + 1)


def main():
    entries = parse_pdf()
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg.ENTRIES_JSON, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(entries)} entries to {cfg.ENTRIES_JSON}")


if __name__ == "__main__":
    main()
