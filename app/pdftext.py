"""
Reading a PDF the way a person reads it.

A PDF has no paragraphs, no headings and no reading order. It has glyphs at
coordinates. Every "extract text" call is really a reconstruction, and the
naive one -- concatenate whatever the library hands back, page after page --
produces something that looks like text and behaves badly in a retrieval index
in five specific ways. This module exists to fix those five, because each of
them silently costs the assistant answers it had the document to give:

  1. Running headers and footers. A 180-page manual with "VEX Robotics -
     Confidential" at the top of every page contributes that phrase 180 times.
     BM25 scores a chunk by term frequency against the corpus average, so a
     phrase in every chunk of a document becomes the thing that document is
     most "about", and a search for it returns all 180 chunks in arbitrary
     order. The boilerplate does not merely waste space; it actively out-ranks
     the content.

  2. Line-wrap hyphenation. A PDF breaks "tolerance" across a line as "toler-"
     and "ance". Joined naively that is two tokens, neither of which is the
     word, and the chunk containing the best explanation of tolerances in the
     document does not match a search for "tolerance". This is invisible in the
     extracted text unless you go looking for it.

  3. Hard line breaks mid-sentence. Every line of a PDF ends in a newline
     whether or not the sentence did. app/kb.py splits documents into
     paragraphs on blank lines and the grounding checker splits answers into
     sentences -- both of which read a wrapped line as a finished thought. A
     number and the thing it measures end up in different chunks.

  4. Columns. Two-column pages extract as left-line, right-line, left-line,
     interleaved into gibberish. A datasheet in two columns is worse than no
     datasheet: it retrieves on the right terms and hands the model a sentence
     assembled from two unrelated ones.

  5. No headings. app/kb.py leans on markdown headings to keep chunks
     readable and to index a chunk under its section title. Straight PDF text
     has none, so every chunk from a PDF arrives headless -- the one document
     type where a chunk from page 94 most needs to say what section it is from.

And one thing that is not a bug but is worth as much as all five: page
numbers. A citation to "Machinery's Handbook" is unusable; a citation to
"Machinery's Handbook, p. 412" can be checked in thirty seconds. Page markers
are emitted into the text and app/kb.py carries them onto each chunk, so the
assistant can tell a reader where to look.

On scanned documents
--------------------
A scan has no text layer at all -- it is a picture of a page. Nothing in here
can read one, and the honest failure is to say so loudly with the exact command
that fixes it, rather than to index four characters of noise and report
success. OCR runs only if the user has already installed the pieces for it, is
never required, and never silently changes what a document says.

On dependencies
---------------
pypdf if present, pdfminer.six if not, and neither is imported until a PDF is
actually opened. Layout-preserving extraction (pypdf 4+) is used when
available because column detection needs to know where on the line a word sat;
without it, everything above still works except the column split.
"""
from __future__ import annotations

import os
import re

class LockedPDF(Exception):
    """The file needs a password we do not have.

    Its own type because it is the one failure the caller can do something
    about, and because it must not be answered by falling back to the other
    extractor: pdfminer will fail on an encrypted file too, and it fails with a
    stack trace rather than a sentence.
    """


# Emitted on its own line ahead of each page's text. app/kb.py recognises it,
# records the number on every chunk that follows, and strips it before the text
# is indexed, so it never reaches the model or the reader. Written in a form a
# human would understand anyway, because a marker that leaks is going to leak in
# front of somebody and "[[page 12]]" explains itself.
PAGE_MARK = "[[page %d]]"
PAGE_RE = re.compile(r"^\s*\[\[page (\d+)\]\]\s*$")

# Below this many word characters, a page has no text layer worth the name --
# it is a scan, a full-page figure, or a cover. Set above zero because a scanned
# page usually yields a few stray characters from a stamp or a page number, and
# treating those as "text present" is what turns a 300-page scan into a
# successful-looking ingest of nothing.
MIN_PAGE_CHARS = 15

# Ligatures and typographic glyphs. A PDF that renders "fillet" with the fi
# ligature extracts as "ﬁllet", which tokenises to a word that appears in no
# query anyone will ever type. The private-use codepoints are the bullet glyphs
# Word and Acrobat emit from Symbol and Wingdings.
_GLYPHS = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
    "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "′": "'", "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": " - ", "―": " - ", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ",
    "​": "", "‌": "", "‍": "", "﻿": "",
    "­": "",                      # soft hyphen: a wrap hint, not a hyphen
    "•": "- ", "‣": "- ", "●": "- ", "▪": "- ",
    "■": "- ", "·": "- ", "⁃": "- ",
    "": "- ", "": "- ", "": "- ", "": "- ",
    "…": "...",
}
_GLYPH_RE = re.compile("|".join(re.escape(k) for k in _GLYPHS))

# A line that is only a page number, or "Page 7 of 92", or "- 7 -".
_FOLIO_RE = re.compile(
    r"^\s*(?:[-–—|]\s*)?(?:page\s+)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?"
    r"\s*(?:[-–—|]\s*)?$", re.I)

# "3.2.1 Bearing selection" -- the most reliable heading signal a PDF gives.
_NUM_HEAD_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+(\S.{0,88})$")

# A bullet or an enumerated list item; never a heading and never joined onto
# the line above.
_BULLET_RE = re.compile(r"^\s*(?:[-*•+]|\(?[a-z0-9]{1,3}[.)])\s+\S")

# Two or more wide gaps on one line: a table row, which must survive verbatim.
# Reflowing one destroys the only thing that made it useful -- which number was
# under which column heading.
_TABLE_RE = re.compile(r"\S {3,}\S")


# ---------------------------------------------------------------------------
# Page ranges
# ---------------------------------------------------------------------------
def parse_pages(spec, total=None):
    """"1-40", "3", "1,4,9-12" -> a sorted list of 1-based page numbers.

    Returns None for an empty spec, meaning "all of it". Out-of-range numbers
    are dropped rather than raising: asking for 1-500 of a 90-page manual is a
    reasonable thing to type and obviously means "as much as there is".
    """
    if not spec:
        return None
    want = set()
    for part in str(spec).replace(" ", "").split(","):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            want.update(range(min(a, b), max(a, b) + 1))
            continue
        if part.isdigit():
            want.add(int(part))
            continue
        raise ValueError("page range %r should look like 1-40 or 2,5,9-12" % part)
    if total:
        want = {p for p in want if 1 <= p <= total}
    return sorted(p for p in want if p >= 1)


# ---------------------------------------------------------------------------
# Extraction back-ends
# ---------------------------------------------------------------------------
def _clean_glyphs(s):
    return _GLYPH_RE.sub(lambda m: _GLYPHS[m.group(0)], s or "")


def _pypdf_pages(path, password="", wanted=None):
    """[(page_number, raw_text)] via pypdf, plus (title, layout_used, n_total).

    Layout mode is tried per page rather than once for the file. It is the
    better extractor and it is also the one that throws on malformed content
    streams, and a single bad page in a 200-page manual must cost that page's
    layout, not the whole document's.
    """
    import warnings
    with warnings.catch_warnings():
        # pypdf emits a cryptography deprecation notice on import that has
        # nothing to do with the file being read, and a red block above a
        # successful ingest reads as a failure to anyone who has not seen it.
        warnings.simplefilter("ignore")
        from pypdf import PdfReader

        reader = PdfReader(path)
        if getattr(reader, "is_encrypted", False):
            # An empty password is the overwhelmingly common case: the file is
            # encrypted to forbid printing or copying, not to keep anyone out.
            ok = 0
            try:
                ok = reader.decrypt(password or "")
            except Exception:
                ok = 0
            if not ok:
                raise LockedPDF(
                    "this PDF is password-protected and %s"
                    % ("the password given did not open it"
                       if password else
                       "no password was given - add --password YOURPASSWORD"))

        title = ""
        try:
            meta = reader.metadata or {}
            title = (meta.get("/Title") or "").strip()
        except Exception:
            title = ""

        total = len(reader.pages)
        out, layout_used = [], False
        for i in range(total):
            n = i + 1
            if wanted and n not in wanted:
                continue
            try:
                page = reader.pages[i]
            except Exception:
                out.append((n, ""))
                continue
            txt = None
            try:
                txt = page.extract_text(extraction_mode="layout")
                layout_used = True
            except Exception:
                # pypdf < 4 has no extraction_mode; some pages fail in layout
                # mode alone. Either way plain mode is still worth having.
                txt = None
            if txt is None:
                try:
                    txt = page.extract_text() or ""
                except Exception:
                    txt = ""
            out.append((n, _clean_glyphs(txt)))
    return out, title, layout_used, total


def _pdfminer_pages(path, password="", wanted=None):
    """Same shape, via pdfminer.six. No layout coordinates, so no columns."""
    from pdfminer.high_level import extract_text
    raw = extract_text(path, password=password or "") or ""
    # pdfminer separates pages with a form feed, which is the only page
    # boundary information it gives up without using the low-level API.
    parts = raw.split("\f")
    if parts and not parts[-1].strip():
        parts.pop()
    out = []
    for i, t in enumerate(parts):
        n = i + 1
        if wanted and n not in wanted:
            continue
        out.append((n, _clean_glyphs(t)))
    return out, "", False, len(parts)


# ---------------------------------------------------------------------------
# OCR (optional, never required)
# ---------------------------------------------------------------------------
def ocr_available():
    """(bool, explanation). The explanation is the point of this function.

    A scanned PDF that ingests as nothing is the single most confusing failure
    this tool has, because every step reports success. Being able to say
    exactly which of three pieces is missing, and the command that installs it,
    is worth more than the OCR itself.
    """
    try:
        import pytesseract
    except Exception:
        return False, ("pytesseract is not installed  (pip install pytesseract)")
    try:
        import pypdfium2  # noqa: F401
    except Exception:
        try:
            import pdf2image  # noqa: F401
        except Exception:
            return False, ("no PDF renderer  (pip install pypdfium2)")
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return False, ("the Tesseract program itself is missing - install it from "
                       "https://github.com/UB-Mannheim/tesseract/wiki and reopen "
                       "the terminal")
    return True, "ready"


def _ocr_page(path, page_no, dpi=300):
    """Text for one page, rendered and OCR'd. '' if anything at all goes wrong."""
    try:
        import pytesseract
    except Exception:
        return ""
    img = None
    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(path)
        try:
            img = doc[page_no - 1].render(scale=dpi / 72.0).to_pil()
        finally:
            doc.close()
    except Exception:
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(path, dpi=dpi, first_page=page_no,
                                      last_page=page_no)
            img = pages[0] if pages else None
        except Exception:
            return ""
    if img is None:
        return ""
    try:
        return _clean_glyphs(pytesseract.image_to_string(img))
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
# A gutter has to be genuinely empty and genuinely wide. Three spaces occur
# inside ordinary layout-mode prose all the time; a real column gap is four or
# more, unbroken, down almost the whole page.
GUTTER_MIN_WIDTH = 4
GUTTER_MAX_INK = 0.06        # fraction of lines allowed to cross it
COLUMN_MIN_LINES = 12
COLUMN_MIN_SHARE = 0.18      # each column must hold this much of the text


def _split_columns(lines):
    """Un-interleave a multi-column page. Returns lines in reading order.

    Works on the whitespace picture rather than on coordinates, because that is
    what layout-mode extraction gives us and it is enough: a column gap is a
    vertical band that almost no line puts a character into. "Almost" matters --
    a page title spanning both columns crosses every gutter, and a rule that
    demanded a perfectly empty band would fail on every page that has a title.

    A line that does cross the gutter is treated as full width and kept in
    place, which is both the simplest rule and the one that matches how a
    person reads a heading that spans the page.
    """
    body = [ln for ln in lines if ln.strip()]
    if len(body) < COLUMN_MIN_LINES:
        return lines
    width = max(len(ln) for ln in body)
    if width < 40:
        return lines

    ink = [0] * (width + 1)
    for ln in body:
        for x, ch in enumerate(ln):
            if ch != " ":
                ink[x] += 1
    limit = max(0, int(len(body) * GUTTER_MAX_INK))

    # Candidate gutters: runs of near-empty columns, not touching the margins.
    gutters, run = [], None
    for x in range(width + 1):
        if ink[x] <= limit:
            run = (x, x) if run is None else (run[0], x)
        else:
            if run and run[1] - run[0] + 1 >= GUTTER_MIN_WIDTH:
                gutters.append(run)
            run = None
    if run and run[1] - run[0] + 1 >= GUTTER_MIN_WIDTH:
        gutters.append(run)
    gutters = [g for g in gutters if g[0] > 12 and g[1] < width - 12]
    if not gutters:
        return lines

    # More than two gutters means an indented list or a table, not columns.
    # Guessing at three-column reading order on that evidence does more damage
    # than leaving it alone.
    if len(gutters) > 2:
        return lines

    # A column ends where the gutter starts and the next one begins where the
    # gutter ends. Cutting both at the gutter's left edge would hand the right
    # column the whole empty band as permanent leading indent, and every
    # measurement of how full that column is would then be wrong by the width
    # of the gap -- which on a generous layout is most of the column.
    starts = [0] + [g[1] + 1 for g in gutters]
    ends = [g[0] for g in gutters] + [width + 1]
    cuts = list(zip(starts, ends))

    cols = [[] for _ in cuts]
    col_ink = [0] * len(cuts)
    fills = [[] for _ in cuts]
    for ln in lines:
        # "Crosses the gutter" means ink *inside* the empty band, not merely
        # text on both sides of it. Every body line of a two-column page has
        # text on both sides -- that is what two columns are -- so testing for
        # that instead is a rule that declares the whole page to be one
        # spanning heading and un-interleaves nothing.
        if any(ln[a:b + 1].strip() for a, b in gutters):
            cols[0].append(ln.rstrip())
            for i in range(1, len(cols)):
                cols[i].append("")
            continue
        for i, (a, b) in enumerate(cuts):
            s = ln[a:b].rstrip()
            cols[i].append(s if s.strip() else "")
            if s.strip():
                col_ink[i] += len(s.strip())
                fills[i].append(len(s.rstrip()) - (len(s) - len(s.lstrip())))

    # Measured against the same population it is summed from: the ink that
    # actually landed in a column, not the spanning lines that landed in
    # neither. Counting those in the denominator made the test unpassable.
    total = sum(col_ink)
    if not total or min(col_ink) < total * COLUMN_MIN_SHARE:
        # One side is a margin note, a line-number rail or a figure caption,
        # not a column. Splitting on it would scatter the real text.
        return lines

    # Prose wrapped into a column reaches near its right edge on most lines;
    # a table's cells are short and leave the same gaps on every row. Both
    # produce a clean vertical band, and only one of them should be read
    # column-first, so the shape of the lines is what tells them apart.
    for (a, b), f in zip(cuts, fills):
        if not f:
            return lines
        f = sorted(f)
        if f[len(f) // 2] < (b - a) * 0.55:
            return lines

    out = []
    for c in cols:
        # Strip the leading indent the cut leaves behind, uniformly, so the
        # table detector downstream is not fooled by it.
        trimmed = [s.lstrip() if s.strip() else "" for s in c]
        while trimmed and not trimmed[0].strip():
            trimmed.pop(0)
        while trimmed and not trimmed[-1].strip():
            trimmed.pop()
        if trimmed:
            out.extend(trimmed)
            out.append("")
    return out or lines


# ---------------------------------------------------------------------------
# Running headers and footers
# ---------------------------------------------------------------------------
HEADFOOT_ZONE = 3            # lines examined at each end of a page
HEADFOOT_MIN_PAGES = 4       # below this the statistic means nothing
HEADFOOT_SHARE = 0.5


def _hf_key(line):
    """A header's fingerprint: what stays the same from page to page.

    Digits are the part that changes -- "Page 4 of 92", "Section 3-1" -- so they
    are collapsed. Without this, a footer that carries the page number is
    unique on every page and survives, which is the most common footer there is.
    """
    s = re.sub(r"\d+", "#", (line or "").lower())
    return " ".join(s.split())


def _strip_headers_footers(pages, sample=()):
    """Drop the lines that repeat at the top or bottom of most pages.

    Top and bottom are counted separately. The same string can legitimately be
    a running header on every page and a real sentence in the body, and only the
    positional evidence tells them apart.

    `sample` is extra pages to count but not to return. It exists for --pages:
    asking for three pages of a two-hundred-page manual leaves too few pages to
    tell a running header from a sentence, so the evidence is borrowed from
    pages the caller did not ask to keep. Without it, the flag most likely to
    be used on the documents with the worst boilerplate is the one flag that
    turns boilerplate removal off.
    """
    counted = list(pages) + list(sample)
    if len(counted) < HEADFOOT_MIN_PAGES:
        # Still worth removing bare page numbers, which need no statistics.
        return [(n, [ln for i, ln in enumerate(lines)
                     if not (_FOLIO_RE.match(ln)
                             and (i < HEADFOOT_ZONE
                                  or i >= len(lines) - HEADFOOT_ZONE))])
                for n, lines in pages]

    top, bottom = {}, {}
    for _, lines in counted:
        body = [ln for ln in lines if ln.strip()]
        for ln in body[:HEADFOOT_ZONE]:
            k = _hf_key(ln)
            if k:
                top[k] = top.get(k, 0) + 1
        for ln in body[-HEADFOOT_ZONE:]:
            k = _hf_key(ln)
            if k:
                bottom[k] = bottom.get(k, 0) + 1

    need = max(3, int(len(counted) * HEADFOOT_SHARE))
    drop_top = {k for k, v in top.items() if v >= need and len(k) <= 120}
    drop_bot = {k for k, v in bottom.items() if v >= need and len(k) <= 120}

    out = []
    for n, lines in pages:
        keep, seen_body = [], 0
        # Index within the non-blank lines, so a page that starts with three
        # blank lines still has its header examined.
        idx = [i for i, ln in enumerate(lines) if ln.strip()]
        head_at = set(idx[:HEADFOOT_ZONE])
        foot_at = set(idx[-HEADFOOT_ZONE:]) if idx else set()
        for i, ln in enumerate(lines):
            k = _hf_key(ln)
            if i in head_at and (k in drop_top or _FOLIO_RE.match(ln)):
                continue
            if i in foot_at and (k in drop_bot or _FOLIO_RE.match(ln)):
                continue
            keep.append(ln)
            if ln.strip():
                seen_body += 1
        out.append((n, keep))
    return out


# ---------------------------------------------------------------------------
# Reflow: lines -> paragraphs, headings and tables
# ---------------------------------------------------------------------------
def _is_table_row(line):
    if not _TABLE_RE.search(line or ""):
        return False
    cells = [c for c in re.split(r"\s{3,}", line.strip()) if c]
    return len(cells) >= 3 or (len(cells) == 2 and any(
        re.search(r"\d", c) for c in cells) and len(line) > 24)


def _tidy_table_row(line):
    """Keep the columns, lose the padding.

    A row rendered as forty spaces of padding is unreadable once whitespace is
    collapsed anywhere downstream, and a material table where the value has
    drifted away from its label is worse than not having the table. The pipe is
    doing one job: holding "6061-T6" and "276" together as one fact.
    """
    cells = [c.strip() for c in re.split(r"\s{3,}", line.strip()) if c.strip()]
    return " | ".join(cells)


def _heading_level(line, prev_blank, next_line):
    """0 if this line is body text, else a markdown heading level.

    Deliberately conservative. A false heading costs more than a missed one:
    app/kb.py starts a new chunk at every heading change, so a wrapped sentence
    mistaken for a heading cuts a paragraph in half and puts half a sentence in
    the index under a title that is the other half.
    """
    s = (line or "").strip()
    if not s or len(s) > 92 or _is_table_row(line) or _BULLET_RE.match(s):
        return 0
    if s.endswith((".", ",", ";", ":", "-")) and not s.endswith("..."):
        # A colon introduces a list rather than titling a section, and either
        # way a chunk boundary in front of the list it introduces is wrong.
        return 0
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 3:
        return 0

    m = _NUM_HEAD_RE.match(s)
    if m and not re.match(r"^\d+(\.\d+)?\s*(in|mm|cm|ft|mpa|psi|lbs?|kg)\b",
                          s, re.I):
        # "3.2 Bearing selection" is a heading; "0.5 in clearance" is not.
        depth = m.group(1).count(".") + 2
        return min(depth, 6)

    if not prev_blank:
        # Everything below relies on the line standing alone. Inside a
        # paragraph, a short capitalised line is just a short line.
        return 0

    upper = [c for c in letters if c.isupper()]
    if len(upper) == len(letters) and len(s) <= 64:
        return 2

    words = [w for w in re.split(r"\s+", s) if w]
    if len(words) >= 2:
        capped = sum(1 for w in words if w[:1].isupper() or w[:1].isdigit())
        nxt = (next_line or "").strip()
        if capped >= max(2, int(len(words) * 0.6)) and (not nxt or nxt[:1].isupper()
                                                        or _BULLET_RE.match(nxt)):
            return 3
    return 0


def _dehyphenate(a, b):
    """Join a line ending in a hyphen to the next one. Returns the joined text.

    The hyphen is dropped when it was a line-break artefact and kept when it was
    part of the token. "toler-/ance" is one word; "10-/32" is a thread call-out
    and losing the hyphen turns it into the number 1032.
    """
    left = a[:-1]
    tail = re.search(r"([\w.#/]+)$", left)
    token = tail.group(1) if tail else ""
    head = re.match(r"[\w.#/]+", b or "")
    nxt = head.group(0) if head else ""
    if not token or not nxt:
        return a + " " + b
    if any(c.isdigit() for c in token) or any(c.isdigit() for c in nxt) \
            or nxt[:1].isupper():
        # 10-32, 6061-T6, ANSI-B: a real hyphen inside an identifier.
        return left + "-" + b
    if nxt[:1].islower():
        return left + b
    return a + " " + b


def _reflow(lines):
    """Lines -> markdown-ish blocks: headings, paragraphs, lists, tables."""
    out, para = [], []

    def flush():
        if para:
            out.append(" ".join(para).strip())
            para.clear()

    n = len(lines)
    for i, raw in enumerate(lines):
        line = (raw or "").rstrip()
        nxt = lines[i + 1] if i + 1 < n else ""
        prev_blank = (i == 0) or not (lines[i - 1] or "").strip()

        if not line.strip():
            flush()
            continue

        if _is_table_row(line):
            flush()
            out.append(_tidy_table_row(line))
            continue

        lvl = _heading_level(line, prev_blank, nxt)
        if lvl:
            flush()
            out.append("\n" + "#" * lvl + " " + line.strip() + "\n")
            continue

        stripped = line.strip()
        if _BULLET_RE.match(stripped):
            flush()
            para.append(re.sub(r"^\s*[•*+]\s+", "- ", stripped))
            continue

        if para and para[-1].endswith("-"):
            para[-1] = _dehyphenate(para[-1], stripped)
            continue

        para.append(stripped)

        # A line that ends a sentence and is followed by an obviously new one
        # closes the paragraph. Without this every page becomes one block and
        # the chunker's only cut points are its own length limit.
        if stripped.endswith((".", "!", "?", '."', ".'")) and nxt.strip() \
                and (nxt.strip()[:1].isupper() or _BULLET_RE.match(nxt.strip())) \
                and len(" ".join(para)) > 400:
            flush()

    flush()
    text = "\n\n".join(b for b in out if b.strip())
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Titles
# ---------------------------------------------------------------------------
_BAD_TITLE = re.compile(
    r"^(untitled|document\d*|microsoft word|print|slide \d+|.*\.(docx?|pptx?|"
    r"indd|pdf|tex))$", re.I)


def _clean_title(meta_title, first_text, path):
    """The best name available, in order of how much it can be trusted."""
    t = (meta_title or "").strip()
    # "Microsoft Word - Design Handbook v3.docx" is a real title wearing a
    # producer's clothes.
    m = re.match(r"^Microsoft Word\s*-\s*(.+?)(?:\.docx?)?$", t, re.I)
    if m:
        t = m.group(1).strip()
    if t and len(t) >= 4 and not _BAD_TITLE.match(t):
        return t[:200]
    for line in (first_text or "").splitlines():
        if PAGE_RE.match(line):
            # The first line of the first block is the page marker this module
            # put there itself. Naming a document "[[page 1]]" is a bad enough
            # outcome to be worth one explicit check.
            continue
        s = line.strip().lstrip("#").strip()
        if 8 <= len(s) <= 120 and re.search(r"[A-Za-z]{3}", s) \
                and not _FOLIO_RE.match(s):
            return s[:200]
    return os.path.splitext(os.path.basename(path))[0][:200]


# ---------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------
def read(path, pages=None, ocr="auto", password="", dpi=300, ocr_max=40):
    """Read a PDF into indexable text.

    `pages` is a range spec ("1-40", "2,5,9-12") or None for everything.
    `ocr` is "auto" (OCR only pages with no text layer, and only if the tools
    are already installed), "off", or "force".

    Returns a dict:
        title       best available document title
        text        the extracted text, with [[page N]] markers
        pages       page numbers actually read
        n_pages     pages in the file
        n_empty     pages that had no text layer
        n_ocr       pages recovered by OCR
        columns     True if a multi-column page was un-interleaved
        engine      which extractor ran
        warnings    plain-English problems worth telling a human about
        chars       length of `text`

    Never raises for a readable-but-difficult PDF. It raises only when the file
    cannot be opened at all, because at that point there is nothing to report.
    """
    warns = []
    wanted = parse_pages(pages)
    wanted_set = set(wanted) if wanted else None

    engine = "pypdf"
    try:
        raw_pages, meta_title, layout, n_total = _pypdf_pages(
            path, password, wanted_set)
    except ImportError:
        try:
            raw_pages, meta_title, layout, n_total = _pdfminer_pages(
                path, password, wanted_set)
            engine = "pdfminer"
        except ImportError:
            raise SystemExit(
                "Reading PDFs needs pypdf:\n"
                "    pip install pypdf\n"
                "(START_APP.bat installs it for you.)")
    except LockedPDF as e:
        raise SystemExit(str(e))
    except Exception as e:
        # A file pypdf cannot parse is quite often one pdfminer can.
        try:
            raw_pages, meta_title, layout, n_total = _pdfminer_pages(
                path, password, wanted_set)
            engine = "pdfminer"
            warns.append("pypdf could not read this file (%s); fell back to "
                         "pdfminer" % e)
        except Exception:
            raise

    if wanted_set and not raw_pages:
        warns.append("no pages matched %r; the file has %d" % (pages, n_total))

    if not layout and engine == "pypdf":
        warns.append("this pypdf is too old for layout-aware extraction, so "
                     "columns cannot be detected - pip install -U pypdf")

    # --- per page: columns, then lines ------------------------------------
    split_any = False
    laid = []
    for n, txt in raw_pages:
        lines = (txt or "").replace("\r\n", "\n").replace("\f", "\n").split("\n")
        if layout:
            new = _split_columns(lines)
            if new is not lines and new != lines:
                split_any = True
                lines = new
        laid.append((n, [ln.rstrip() for ln in lines]))

    # --- OCR the pages that have no text layer ----------------------------
    def _ink(lines):
        return len(re.findall(r"\w", "\n".join(lines)))

    empty = [n for n, lines in laid if _ink(lines) < MIN_PAGE_CHARS]
    n_ocr = 0
    if ocr == "force":
        targets = [n for n, _ in laid]
    elif ocr == "off":
        targets = []
    else:
        targets = list(empty)

    if targets:
        ready, why = ocr_available()
        if not ready:
            if empty:
                warns.append(
                    "%d page%s have no text layer (they are scans or images). "
                    "OCR is not set up: %s"
                    % (len(empty), "" if len(empty) == 1 else "s", why))
        else:
            if len(targets) > ocr_max:
                warns.append("only the first %d image pages were OCR'd; pass "
                             "--pages to do the rest in batches" % ocr_max)
                targets = targets[:ocr_max]
            done = {}
            for n in targets:
                t = _ocr_page(path, n, dpi=dpi)
                if len(re.findall(r"\w", t)) >= MIN_PAGE_CHARS:
                    done[n] = _clean_glyphs(t).split("\n")
                    n_ocr += 1
            if done:
                laid = [(n, done.get(n, lines)) for n, lines in laid]
    elif empty and ocr == "off":
        warns.append("%d page%s have no text layer and OCR was switched off"
                     % (len(empty), "" if len(empty) == 1 else "s"))

    # --- boilerplate, then reflow -----------------------------------------
    # Borrow evidence from pages outside the requested range when the range is
    # too short to supply its own. Bounded to a handful of extra extractions:
    # the point of --pages on a 900-page file is not to read 900 pages.
    sample = []
    if wanted_set and len(laid) < HEADFOOT_MIN_PAGES and n_total > len(laid):
        extra = [n for n in range(1, n_total + 1) if n not in wanted_set]
        step = max(1, len(extra) // 6)
        extra = extra[::step][:6]
        if extra:
            try:
                if engine == "pdfminer":
                    got, _, _, _ = _pdfminer_pages(path, password, set(extra))
                else:
                    got, _, _, _ = _pypdf_pages(path, password, set(extra))
                sample = [(n, _clean_glyphs(t or "").split("\n"))
                          for n, t in got]
            except Exception:
                sample = []
    laid = _strip_headers_footers(laid, sample)

    blocks = []
    for n, lines in laid:
        body = _reflow(lines)
        if not body.strip():
            continue
        # The marker goes on its own line with blank lines around it so the
        # paragraph splitter in app/kb.py sees it as a block of its own and
        # never welds it onto a sentence.
        blocks.append("%s\n\n%s" % (PAGE_MARK % n, body))

    text = "\n\n".join(blocks).strip()
    title = _clean_title(meta_title, blocks[0] if blocks else "", path)

    if n_total > 300 and not wanted_set:
        warns.append("this is a %d-page document; if only part of it is "
                     "relevant, --pages 1-60 keeps the index sharper" % n_total)
    # Thinness is only evidence of a problem when there was more to get. A
    # one-page table is legitimately 300 characters long, and warning about it
    # teaches the reader to ignore the warnings that matter.
    if len(re.findall(r"\w", text)) < 200 and (empty or len(raw_pages) > 1):
        warns.append("almost no text came out - if this is a scan, see the OCR "
                     "note above")

    return {
        "title": title,
        "text": text,
        "pages": [n for n, _ in raw_pages],
        "n_pages": n_total,
        "n_empty": len(empty),
        "n_ocr": n_ocr,
        "columns": split_any,
        "engine": engine,
        "layout": layout,
        "warnings": warns,
        "chars": len(text),
    }


def pdf_to_text(path, **kw):
    """Just the text. Kept for callers that only ever wanted a string."""
    return read(path, **kw)["text"]


def outline(text, limit=40):
    """The headings found, for a human deciding whether the extraction worked.

    Looking at 40 000 characters of extracted text tells you nothing. Looking at
    its heading list tells you in two seconds whether the document came through
    as a document or as soup.
    """
    out = []
    for line in (text or "").splitlines():
        m = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if m:
            out.append((len(m.group(1)), m.group(2).strip()))
            if len(out) >= limit:
                break
    return out
