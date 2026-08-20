"""
The local knowledge base — documents you feed it, searchable offline.

Why this exists
---------------
Web search is a guess. Tavily has to infer, from a handful of words, which of a
billion pages answers a question, and when it guesses wrong the assistant
answers from whatever it was handed. That is how a practice-exercise page
became the source for "what to consider when designing a shooter". No amount of
prompt tuning fixes a retrieval miss, because by the time the model sees the
context the right page is already absent.

A curated corpus removes the guess for the material that matters. Roughly
thirty pages — the FRCDesign book, WPILib's docs, Onshape4FRC, a few vendor
datasheets — answer most of what a team actually asks. Ingest them once and
those answers stop depending on a search engine having a good day. It also
means the assistant still knows things with no network and no Tavily key, which
is the state most machines are in the night before a competition.

Why BM25 and not embeddings
---------------------------
Embeddings would catch synonyms BM25 misses. They would also mean a model
download on first run, a few hundred megabytes of torch on a school laptop, and
a new way for START_APP.bat to fail in front of a student who cannot debug it.
BM25 is a hundred lines, needs nothing but the standard library, is instant on a
corpus this size, and is deterministic — the same question retrieves the same
passages every time, which is what makes the eval story possible at all.

The synonym gap is covered from the other side: chat.analyze() already rewrites
the question into expert search terms before retrieval runs, so "how thick
should my plate be" arrives here as the vocabulary the documents actually use.

Storage
-------
One JSON file, data/kb.json. Plain text, diffable, hand-editable, and copyable
between machines by dragging one file. The index is rebuilt in memory on load
(about 20 ms for a few thousand chunks) rather than serialised, so the file
never goes stale against the code that reads it.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from urllib.parse import urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KB_PATH = os.environ.get("STRESSVIZ_KB") or os.path.join(ROOT, "data", "kb.json")

# Tuned for a small technical corpus. k1 low because engineering pages repeat a
# part number many times without becoming more about it; b at 0.75 is the usual
# default and keeps a long reference page from being unfairly penalised.
K1 = 1.2
B = 0.75

# Words too common in this domain to discriminate. Kept short on purpose — an
# aggressive stop list throws away "load", "stress" and "fit", which are the
# words a mechanical question is actually made of.
_STOP = set("""a an the is are was were be been being am of for to in on at by with
from about into over under after before above below and or but if then than that this
these those it its as so such not no nor can could should would will shall may might
must do does did done doing have has had having i you he she we they me my your our
their there here what which who whom whose when where why how all any some more most
other another each every both few many much very own same too also just only
""".split())

# Engineering identifiers must survive tokenisation intact: 10-32, 1/4-20, #25,
# 6061-T6, M5, GT2. Splitting those on punctuation turns a precise query into a
# vague one, which is exactly backwards.
_TOKEN_RE = re.compile(r"#?[a-z0-9][a-z0-9#/\.\-]*")

# Words where -ing is part of the noun, not a verb ending. "bearing" naively
# stems to "bear" while "bearings" stems to "bearing", so the singular and the
# plural of the single most common component in the corpus would never match
# each other. These are returned untouched; their plurals reach the same form
# by the ordinary -s rule.
_NOUN_ING = frozenset("""bearing spring ring string housing casing coupling bushing
tubing webbing drawing setting opening coating plating fitting""".split())

# Only these endings take -es. Everything else takes a plain -s, and treating
# them alike is what turns "plates" into "plat" and "tolerances" into
# "toleranc" while leaving the singulars whole.
_SIBILANT = ("s", "x", "z", "ch", "sh")

# Pairs the suffix rules cannot reconcile, mapped to a shared arbitrary stem.
# The stem does not need to be a word -- it only needs to be the same one for
# every surface form of the same idea.
_IRREGULAR = {
    "machine": "machin", "machines": "machin",
    "load": "load", "loads": "load", "loaded": "load", "loading": "load",
    "weld": "weld", "welds": "weld", "welded": "weld", "welding": "weld",
    "mount": "mount", "mounts": "mount", "mounted": "mount", "mounting": "mount",
    "analysis": "analys", "analyses": "analys", "analyse": "analys",
    "analyze": "analys", "analyzing": "analys",
    "radius": "radius", "radii": "radius",
    "axis": "axis", "axes": "axis",
}


def _stem(w: str) -> str:
    """Crude, deliberately conservative suffix stripping.

    A real stemmer would map "bearing" to "bear", which in this corpus is a
    different subject entirely. This only collapses plurals and the two verb
    endings that cause the most obvious misses ("pocketing" vs "pockets"), and
    it refuses to touch anything containing a digit, so 6061-T6 is never
    mangled.

    The -ss guard is not a nicety. Without it "stress" stems to "stres" while
    "stresses" stems to "stress", so the two never match each other -- in a tool
    whose entire subject is stress, that one missing branch would quietly break
    the most important query in the corpus.
    """
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if len(w) < 4 or any(c.isdigit() for c in w):
        return w
    if w.endswith("ss") or w in _NOUN_ING:
        return w

    def undouble(base):
        # "pocketting" style doubling: pocketting -> pocket. -ll and -ss are
        # left alone; "drilling" really is drill, not dril.
        if len(base) > 3 and base[-1] == base[-2] and base[-1] not in "sl":
            return base[:-1]
        return base

    if w.endswith("ies") and len(w) >= 5:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) >= 5 and w[:-2].endswith(_SIBILANT):
        return w[:-2]
    if w.endswith("s") and len(w) >= 4:
        return w[:-1]
    if w.endswith("ing") and len(w) >= 7:
        return undouble(w[:-3])
    if w.endswith("ed") and len(w) >= 6:
        return undouble(w[:-2])
    return w


def tokens(text: str, stem: bool = True):
    """Text → searchable terms. The one place tokenisation is defined."""
    out = []
    for m in _TOKEN_RE.finditer((text or "").lower()):
        w = m.group(0).strip(".-/")
        if not w or w in _STOP or len(w) < 2:
            continue
        out.append(_stem(w) if stem else w)
    return out


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK = 900
OVERLAP = 150

_HEAD_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")

# app/pdftext.py writes one of these ahead of each page it extracts. They are
# consumed here and never indexed: a page number is metadata about where a
# passage came from, not part of what it says, and leaving it in the text would
# put a stray integer in front of the grounding checker on every PDF chunk.
_PAGE_RE = re.compile(r"^\s*\[\[page (\d+)\]\]\s*$")


def chunk_text(text: str, win: int = CHUNK, overlap: int = OVERLAP):
    """Split a document into overlapping windows, carrying the heading down.

    Three things matter here. Overlap, so a definition split across a boundary
    is still whole in one of the two pieces — without it the single most useful
    sentence on a page is the one most likely to be cut in half. Headings,
    because a chunk from the middle of a long page is often unreadable without
    the section title, and the title is usually the best evidence of what the
    chunk is about; the heading is prepended to the indexed text as well as
    stored, so matching a section name ranks its whole section. And the page
    number, for PDFs, which is what turns "it says so in the handbook" into a
    claim somebody can go and check.

    Returns [(heading, text, page)]. `page` is 0 for anything that did not come
    from a PDF.
    """
    text = (text or "").replace("\r\n", "\n").strip()
    if not text:
        return []
    # Break into blocks, remembering the most recent heading above each and the
    # page it was on.
    blocks, head, page = [], "", 0
    for para in re.split(r"\n\s*\n+", text):
        para = para.strip()
        if not para:
            continue
        pm = _PAGE_RE.match(para)
        if pm:
            page = int(pm.group(1))
            continue
        m = _HEAD_RE.match(para)
        if m and len(para) < 200:
            head = m.group(2).strip()
            continue
        blocks.append((head, para, page))

    out, cur, cur_head, cur_page = [], "", "", 0
    for h, para, pg in blocks:
        if cur and (h != cur_head or len(cur) + len(para) + 2 > win):
            out.append((cur_head, cur, cur_page))
            # Carry the tail of the previous chunk forward.
            tail = cur[-overlap:] if overlap and h == cur_head else ""
            cur = (tail + "\n" + para).strip() if tail else para
            cur_head = h
            # The new chunk is cited by where it starts, not by where the
            # overlap it inherited came from.
            cur_page = pg
            continue
        cur = (cur + "\n" + para).strip() if cur else para
        cur_head = cur_head or h
        cur_page = cur_page or pg
        while len(cur) > win * 1.6:
            cut = cur.rfind(" ", 0, win)
            cut = cut if cut > win // 2 else win
            out.append((cur_head, cur[:cut].strip(), cur_page))
            cur = cur[max(0, cut - overlap):].strip()
    if cur:
        out.append((cur_head, cur, cur_page))
    return [(h, t, p) for h, t, p in out if len(t.strip()) > 40]


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------
def empty():
    return {"version": 1, "docs": [], "chunks": []}


def load_raw(path=None):
    """Read the JSON file. A corrupt or missing KB is an empty KB, never a crash.

    The assistant working without its knowledge base is a degraded assistant.
    The app failing to start because a JSON file got truncated is a broken app.
    """
    path = path or KB_PATH
    if not os.path.exists(path):
        return empty()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or "chunks" not in data:
            return empty()
        data.setdefault("docs", [])
        data.setdefault("version", 1)
        return data
    except Exception:
        return empty()


def save_raw(data, path=None):
    """Write via a temp file so an interrupted ingest cannot destroy the KB."""
    path = path or KB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


class Index:
    """An in-memory BM25 index over the chunk list."""

    def __init__(self, data):
        self.docs = data.get("docs") or []
        self.chunks = data.get("chunks") or []
        self.tf = []        # per chunk: {term: count}
        self.length = []    # per chunk: token count
        self.df = {}        # term: number of chunks containing it
        self.avgdl = 1.0
        self._build()

    def _build(self):
        total = 0
        for c in self.chunks:
            # The heading is indexed with the body: a chunk under "Bearing
            # selection" is about bearing selection even where the paragraph
            # itself says only "use a flanged one".
            blob = (c.get("head") or "") + " " + (c.get("text") or "")
            tf = {}
            n = 0
            for t in tokens(blob):
                tf[t] = tf.get(t, 0) + 1
                n += 1
            # The document title matters more than any single sentence in it,
            # so its terms are weighted rather than merely present.
            doc = self.docs[c["doc"]] if 0 <= c.get("doc", -1) < len(self.docs) else {}
            for t in tokens(doc.get("title") or ""):
                tf[t] = tf.get(t, 0) + 2
                n += 2
            self.tf.append(tf)
            self.length.append(max(n, 1))
            total += n
            for t in tf:
                self.df[t] = self.df.get(t, 0) + 1
        self.avgdl = (total / len(self.chunks)) if self.chunks else 1.0

    def _idf(self, term):
        n = len(self.chunks)
        df = self.df.get(term, 0)
        if not df:
            return 0.0
        # Robertson/Sparck-Jones with the +1 that keeps a term appearing in
        # every chunk at zero rather than negative.
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query, k=6, per_doc=2, boost_terms=None):
        """Top chunks for a query.

        `per_doc` caps how many chunks one document may contribute. Without it a
        single long, well-matched page fills every slot and the answer is built
        from one source that may simply be wrong — the cap is what keeps
        corroboration possible.
        """
        if not self.chunks:
            return []
        q = tokens(query)
        extra = tokens(" ".join(boost_terms or []))
        # Terms the planner pulled out of the question are what the user
        # actually named; they get counted twice.
        weights = {}
        for t in q:
            weights[t] = weights.get(t, 0.0) + 1.0
        for t in extra:
            weights[t] = weights.get(t, 0.0) + 1.0
        if not weights:
            return []
        idf = {t: self._idf(t) for t in weights}
        phrase = " ".join(tokens(query, stem=False)[:8])

        scored = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            hits = 0
            dl = self.length[i]
            norm = K1 * (1 - B + B * dl / self.avgdl)
            for t, w in weights.items():
                f = tf.get(t)
                if not f:
                    continue
                hits += 1
                s += w * idf[t] * (f * (K1 + 1.0)) / (f + norm)
            if s <= 0:
                continue
            # Coverage matters as much as raw frequency: a chunk touching four
            # of the question's five terms once each is a better answer than one
            # hammering a single term twenty times.
            s *= 0.55 + 0.45 * (hits / float(len(weights)))
            if phrase and len(phrase) > 12:
                blob = ((self.chunks[i].get("head") or "") + " "
                        + self.chunks[i].get("text", "")).lower()
                if phrase in blob:
                    s *= 1.35
            scored.append((s, i))

        scored.sort(key=lambda x: -x[0])
        out, used = [], {}
        for s, i in scored:
            c = self.chunks[i]
            d = c.get("doc", -1)
            if used.get(d, 0) >= per_doc:
                continue
            used[d] = used.get(d, 0) + 1
            doc = self.docs[d] if 0 <= d < len(self.docs) else {}
            out.append({
                "title": doc.get("title") or "knowledge base",
                "url": doc.get("url") or "",
                "kind": doc.get("kind") or "reference",
                "source": doc.get("source") or "",
                "head": c.get("head") or "",
                "text": c.get("text") or "",
                # 0 for anything that is not a PDF. Carried all the way to the
                # source line the reader sees, because "p. 412" is the
                # difference between a citation and a gesture.
                "page": int(c.get("page") or 0),
                "page_word": doc.get("page_word") or "p.",
                "score": round(s, 4),
            })
            if len(out) >= k:
                break
        return out


# A reload-on-change cache. Ingesting from the CLI while the server is running
# should take effect on the next question, not the next restart — otherwise
# every correction to the corpus costs a restart and people stop making them.
_lock = threading.Lock()
_cache = {"mtime": None, "index": None, "path": None}


def index(path=None):
    path = path or KB_PATH
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None
    with _lock:
        if (_cache["index"] is not None and _cache["mtime"] == mtime
                and _cache["path"] == path):
            return _cache["index"]
        idx = Index(load_raw(path))
        _cache.update(mtime=mtime, index=idx, path=path)
        return idx


def search(query, k=6, boost_terms=None, path=None, per_doc=2):
    """Public entry point used by the assistant.

    `per_doc` is exposed rather than fixed because the right value depends on
    the question: two chunks of the best page is the right shape for "how does
    this work", and one chunk each of six pages is the right shape for "what
    kinds are there". app/chat.py decides which.
    """
    try:
        return index(path).search(query, k=k, boost_terms=boost_terms,
                                  per_doc=per_doc)
    except Exception:
        return []


def stats(path=None):
    idx = index(path)
    by_source = {}
    for d in idx.docs:
        s = d.get("source") or "other"
        by_source[s] = by_source.get(s, 0) + 1
    return {
        "docs": len(idx.docs),
        "chunks": len(idx.chunks),
        "by_source": by_source,
        "path": path or KB_PATH,
        "exists": os.path.exists(path or KB_PATH),
    }


def add(documents, path=None, replace=True):
    """Insert or update documents. Returns (added, replaced, chunks).

    `documents` is a list of {title, url, text, kind, source}. Re-ingesting the
    same URL replaces it rather than duplicating it, so a refresh crawl is safe
    to run repeatedly — the alternative is a corpus that silently accumulates
    three versions of the same page and votes with all of them.
    """
    path = path or KB_PATH
    data = load_raw(path)
    docs, chunks = data["docs"], data["chunks"]

    added = replaced = made = 0
    for d in documents:
        text = (d.get("text") or "").strip()
        # The floor exists to catch the two documents that arrive looking like
        # successes and hold nothing: a scanned PDF, and a page whose body is
        # built in the browser. A note somebody sat down and typed is neither,
        # and "our pocket floors are 0.100 in" is one sentence long and among
        # the most useful things in the corpus -- so that caller can waive it.
        if len(text) < (40 if d.get("typed") else 200):
            continue
        url = (d.get("url") or "").strip()
        rec = {
            "title": (d.get("title") or url or "untitled").strip()[:300],
            "url": url,
            "kind": d.get("kind") or "reference",
            "source": d.get("source") or "manual",
            "added": time.strftime("%Y-%m-%d"),
            "chars": len(text),
        }
        if d.get("pages"):
            rec["pages"] = int(d["pages"])
        # What to call the number in a citation. Absent means "p.", which is
        # what every PDF already in the corpus is, so nothing has to re-ingest.
        # A deck sets "slide" and a workbook sets "tab": "(p. 7)" against a
        # document that has no pages sends a reader looking for one.
        if d.get("page_word"):
            rec["page_word"] = str(d["page_word"])[:12]
        old = None
        if url and replace:
            for i, ex in enumerate(docs):
                if ex.get("url") == url:
                    old = i
                    break
        if old is None:
            docs.append(rec)
            di = len(docs) - 1
            added += 1
        else:
            docs[old] = rec
            di = old
            chunks[:] = [c for c in chunks if c.get("doc") != di]
            replaced += 1
        for head, body, page in chunk_text(text):
            c = {"doc": di, "head": head, "text": body}
            if page:
                # Only written when there is one, so a knowledge base built
                # before PDFs had page numbers stays byte-identical after a
                # re-ingest of everything else.
                c["page"] = page
            chunks.append(c)
            made += 1

    data["docs"], data["chunks"] = docs, chunks
    save_raw(data, path)
    _cache["index"] = None
    return added, replaced, made


def remove(url_or_title, path=None):
    """Drop a document and its chunks. Returns how many documents went."""
    path = path or KB_PATH
    data = load_raw(path)
    keep_idx, gone = [], 0
    needle = (url_or_title or "").lower().strip()
    remap = {}
    for i, d in enumerate(data["docs"]):
        hit = needle and (needle in (d.get("url") or "").lower()
                          or needle in (d.get("title") or "").lower())
        if hit:
            gone += 1
            continue
        remap[i] = len(keep_idx)
        keep_idx.append(d)
    data["docs"] = keep_idx
    data["chunks"] = [dict(c, doc=remap[c["doc"]]) for c in data["chunks"]
                      if c.get("doc") in remap]
    save_raw(data, path)
    _cache["index"] = None
    return gone


# ---------------------------------------------------------------------------
# Siblings
# ---------------------------------------------------------------------------
def siblings(urls, exclude=(), limit=6, path=None):
    """Other documents living in the same section of a site as `urls`.

    BM25 cannot answer "what mechanisms are common in FRC" completely, and no
    amount of extra ranking slots fixes it: the query does not contain the word
    "shooter", so the shooter page scores nothing and never enters the ranking
    at all. Every category the question does not happen to name is invisible to
    a keyword index by construction.

    What is not invisible is the shape of the corpus. A site that documents five
    mechanisms puts them side by side -- /mechanism-examples/shooter/,
    /mechanism-examples/intake/, /mechanism-examples/elevator/ -- so once ONE of
    them has been matched, the rest are its neighbours in the URL tree. This
    walks from a hit to its siblings, which is how the missing category gets
    found without knowing its name in advance.

    Returns the first chunk of each sibling document, shaped like a search hit
    with score 0.0 -- the caller decides what a neighbour is worth, since it was
    reached by structure rather than by matching anything.
    """
    idx = index(path)
    if not idx.docs:
        return []
    seen = {u for u in exclude if u}
    parents = []
    for u in urls:
        if not u:
            continue
        parts = urlsplit(u)
        segs = [s for s in (parts.path or "/").split("/") if s]
        # Drop the leaf to get the section. A trailing-slash URL and a
        # /page.html URL have to land on the same parent or a site that uses
        # one style gets no siblings at all.
        if segs:
            segs = segs[:-1]
        if not segs:
            # A top-level page has no section; the whole host is not a section.
            continue
        parent = "%s://%s/%s/" % (parts.scheme, parts.netloc, "/".join(segs))
        if parent not in parents:
            parents.append(parent)
    if not parents:
        return []

    # First chunk per document: the top of a page is where it says what it is,
    # which is the sentence a list answer needs. Later chunks are detail about a
    # thing the reader has not been told exists yet.
    first = {}
    for c in idx.chunks:
        d = c.get("doc", -1)
        if d not in first:
            first[d] = c

    out = []
    for d, doc in enumerate(idx.docs):
        u = doc.get("url") or ""
        if not u or u in seen:
            continue
        if not any(u.startswith(p) and u.rstrip("/") != p.rstrip("/")
                   for p in parents):
            continue
        c = first.get(d)
        if not c:
            continue
        out.append({
            "title": doc.get("title") or "knowledge base",
            "url": u,
            "kind": doc.get("kind") or "reference",
            "source": doc.get("source") or "",
            "head": c.get("head") or "",
            "text": c.get("text") or "",
            "page": int(c.get("page") or 0),
            "page_word": doc.get("page_word") or "p.",
            "score": 0.0,
            "sibling": True,
        })
        if len(out) >= limit:
            break
    return out
