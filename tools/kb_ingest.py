"""
Feed the assistant's knowledge base.

    python tools/kb_ingest.py url  https://example.com/page [more urls...]
    python tools/kb_ingest.py url  https://docs.google.com/presentation/d/<id>/edit
    python tools/kb_ingest.py crawl https://example.com/guide/ --max 40 --depth 2
    python tools/kb_ingest.py frcdesign               # that whole site, tagged right
    python tools/kb_ingest.py hub  https://www.firstinspires.org/resources/library/frc/technical-resources
    python tools/kb_ingest.py hub  <that url> --dry-run     # what would come in
    python tools/kb_ingest.py file docs/*.pdf notes/*.md
    python tools/kb_ingest.py file manual.pdf --pages 1-40 --ocr auto
    python tools/kb_ingest.py pdfcheck manual.pdf     # dry run: what would go in
    python tools/kb_ingest.py text "Team rule: ..." --title "Shop rules"
    python tools/kb_ingest.py seed
    python tools/kb_ingest.py list | stats | search "query" | remove <url or title>

Everything lands in data/kb.json, which the running server picks up on the next
question -- no restart. Re-ingesting a URL replaces it instead of adding a
second copy, so refreshing the corpus is just running the same command again.

On Google links
---------------
`url` recognises Google Docs, Slides, Sheets and Drive files and exports them
before reading, because the page at a /edit address is an empty shell that
draws the document in the browser -- fetched as a web page a deck yields about
forty characters of menu labels. app/gdocs.py picks the export per type: HTML
for a Doc so its headings survive, .pptx for a deck so slide titles, tables and
speaker notes stay distinguishable, .xlsx for a Sheet so every tab comes in.
Slides and Sheets are cited by slide and by tab rather than by page.

The document has to be link-shared -- Share > General access > "Anyone with the
link" -- because nothing here signs in. A document that is not gets a message
naming that setting, rather than an indexed copy of Google's sign-in page.

On --kind
---------
The kind is not decoration. app/chat.py ranks a source partly by what type of
document it is, and demotes narrow ones when the question is general. Ingesting
a team handbook as "reference" tells the assistant it may state that handbook's
numbers as engineering fact; ingest it as "convention" and it will attribute
them to you instead. When in doubt, under-claim: an attributed true statement
costs a reader nothing, and an unattributed local convention presented as
physics costs them a part.

    reference   general engineering practice; the default
    docs        official documentation for a tool or product
    data        material properties, standards tables
    vendor      a manufacturer's page for a specific part
    forum       a thread; one team's experience, not consensus
    blog        one author's opinion
    exercise    a design challenge or assignment -- its stated limits are
                rules of that exercise only
    rules       a competition manual; binds one season
    convention  your own team's handbook or shop notes -- true where you
                work, not everywhere
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import kb  # noqa: E402
from app import webtext  # noqa: E402
from app import pdftext  # noqa: E402
from app import gdocs  # noqa: E402

UA = webtext.UA

# Kept in step with KIND_INFO in app/chat.py -- a kind this tool accepts but
# chat.py does not know about would silently fall back to "reference", which is
# the most permissive kind there is and therefore the worst place to land by
# accident.
KINDS = ("reference", "docs", "data", "vendor", "forum", "blog", "exercise",
         "rules", "convention")


# ---------------------------------------------------------------------------
# HTML -> readable text
# ---------------------------------------------------------------------------
# Lives in app/webtext.py because app/chat.py reads pages too, and two copies
# of an extractor means a page that indexes well and reads badly with no way to
# tell which copy is wrong. Re-exported under the old names so the rest of this
# file (and anything importing it) is unchanged.
_html_to_text_bs4 = webtext._bs4
_html_to_text_regex = webtext._regex
html_to_text = webtext.html_to_text
pdf_to_text = webtext.pdf_to_text
read_pdf = pdftext.read
_DROP_TAGS = webtext.DROP_TAGS
_DROP_HINT = webtext.DROP_HINT


def _report_pdf(path, info, indent="    "):
    """Say what actually came out of a PDF, in the terms a person cares about.

    Printed on every PDF ingest, not just failing ones. "added 1 document" is
    equally true of a clean 90-page manual and of a scan that yielded three
    characters of noise, and the moment to notice the difference is now --
    before the corpus quietly starts answering questions out of it.
    """
    bits = ["%d page%s" % (len(info["pages"]),
                           "" if len(info["pages"]) == 1 else "s")]
    if info["n_pages"] and len(info["pages"]) != info["n_pages"]:
        bits[0] += " of %d" % info["n_pages"]
    bits.append("%d chars" % info["chars"])
    if info["columns"]:
        bits.append("columns un-interleaved")
    if info["n_ocr"]:
        bits.append("%d page%s recovered by OCR"
                    % (info["n_ocr"], "" if info["n_ocr"] == 1 else "s"))
    heads = pdftext.outline(info["text"], limit=3)
    print("%s%s" % (indent, ", ".join(bits)))
    if heads:
        print("%ssections: %s%s"
              % (indent, "; ".join(h[1][:40] for h in heads),
                 " ..." if len(pdftext.outline(info["text"], limit=4)) > 3 else ""))
    for w in info["warnings"]:
        print("%s! %s" % (indent, w))


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
_HTML_HEAD_RE = re.compile(rb"(?i)<(!doctype|html|head|body|meta|\?xml|link|"
                           rb"script|div|title)\b")


def fetch_doc(url, timeout=25):
    """Download one URL and return {title, text, info, is_pdf}.

    `info` is the full extraction report when the URL turned out to be a PDF,
    and None otherwise, so a caller can print how many pages were read and
    whether any of them were scans. fetch() below keeps the (title, text) shape
    every older caller wants.
    """
    # Before anything else, because a Google URL is not a page. Fetching
    # /presentation/d/<id>/edit gets an empty shell and a megabyte of script --
    # the slides are drawn in the browser. Every Google document has to be
    # exported to something readable first; app/gdocs.py knows which export per
    # type and hands back this same shape.
    if gdocs.is_google_url(url):
        return gdocs.fetch(url, timeout=max(timeout, gdocs.TIMEOUT))
    import requests
    r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
    r.raise_for_status()
    ctype = (r.headers.get("content-type") or "").lower()
    # The bare type, not the whole header. Asking whether "xml" appears
    # anywhere in the content-type says yes to a .docx, whose type is
    # application/vnd.openxmlformats-officedocument.wordprocessingml.document
    # -- which is how a zip archive gets read as a web page.
    mime = ctype.split(";")[0].strip()
    head = r.content[:512].lstrip()
    # Header first, the bytes themselves second, and the file extension never.
    # Servers hand back application/octet-stream for PDFs and for HTML alike,
    # and a URL ending in .pdf is quite often a redirect to a sign-in page, so
    # no one of the three can be trusted on its own.
    if "pdf" in mime or head.startswith(b"%PDF"):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
            fh.write(r.content)
            tmp = fh.name
        try:
            info = read_pdf(tmp)
            # Falling back to the last path segment only when the PDF has no
            # usable title of its own: a source list full of "AN-1421.pdf" is a
            # list nobody can read.
            return {"title": info["title"] or url.rsplit("/", 1)[-1],
                    "text": info["text"], "info": info, "is_pdf": True}
        finally:
            os.unlink(tmp)
    known_text = (mime.startswith("text/")
                  or mime in ("application/xhtml+xml", "application/xml",
                              "application/html", "application/json",
                              "application/rss+xml", "application/atom+xml"))
    if not known_text and not _HTML_HEAD_RE.match(head):
        # A .docx, a zip, a firmware image. html_to_text would strip tags out of
        # binary and hand back a page of mojibake that is long enough to clear
        # the thin-text floor and means nothing. This is the only place it can
        # be caught: the link that led here looked like every other link on the
        # page, and by the time it is text it is indistinguishable from text.
        raise ValueError("not a readable document (%s)"
                         % (mime or "unknown type"))
    r.encoding = r.encoding or "utf-8"
    title, text = html_to_text(r.text)
    return {"title": title, "text": text, "info": None, "is_pdf": False}


def fetch(url, timeout=25):
    """(title, text) for one URL, PDF or HTML."""
    d = fetch_doc(url, timeout=timeout)
    return d["title"], d["text"]


_ASSET_RE = re.compile(r"\.(png|jpe?g|gif|svg|zip|step|stp|stl|mp4|webm|css|js|"
                       r"ico|woff2?|ttf|xml|json|rss|atom)$", re.I)
_PDF_RE = re.compile(r"\.pdf(\?|$)", re.I)
# Documents that are real resources and that this tool has no reader for. Kept
# separate from the asset list because these deserve to be reported by name --
# somebody may want to download them by hand.
_UNREADABLE_RE = re.compile(r"\.(docx?|pptx?|xlsx?|rtf|odt|ods|odp|"
                            r"7z|rar|exe|msi|iso|dmg)$", re.I)


def _main_region(html):
    """(the part of the page worth reading links from, is it really isolated).

    A resource-library page's navigation and footer hold dozens of links -- to
    the donate page, to social media, to every other section of the site. On
    the page this was written for they outnumber the resources it exists to
    list by about four to one. Dropping the furniture before reading the links
    is the difference between ingesting a library and ingesting a site map.

    The second value is the honest part. True means a landmark element said
    "the page's content is here", so a link found inside it is one the editors
    put there on purpose and can be trusted on that basis alone. False means
    the best that could be done was to strip the obvious furniture out of the
    whole document, which leaves enough navigation behind that the caller still
    has to be suspicious of what it finds. Confusing the two is how a rule that
    should protect a crawl from the site menu ends up deleting real resources.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return html, False
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(list(_DROP_TAGS)):
            tag.decompose()
        for tag in soup.find_all(attrs={"class": True}):
            if _DROP_HINT.search(" ".join(tag.get("class") or [])):
                tag.decompose()
        for tag in soup.find_all(attrs={"id": True}):
            if _DROP_HINT.search(tag.get("id") or ""):
                tag.decompose()
        node = (soup.find("main") or soup.find(attrs={"role": "main"})
                or soup.find(id="content") or soup.find(id="main-content"))
        if node is not None:
            return str(node), True
        return str(soup.body or soup), False
    except Exception:
        return html, False


def link_items(html, base):
    """[(url, link text)] in page order: absolute, de-fragmented, no assets.

    The link text is carried along because it is often a better name for the
    document than anything inside the document. A PDF's own title is whatever
    sat in the Word file's properties -- frequently a filename, frequently the
    name of the document it was copied from, frequently empty -- while the link
    text is what an editor wrote to tell a human what they are about to open.
    """
    out, seen = [], set()
    for m in re.finditer(r"(?is)<a\b([^>]*)>(.*?)</a>", html):
        attrs, inner = m.group(1), m.group(2)
        hm = re.search(r'(?is)href\s*=\s*["\']([^"\']+)', attrs)
        if not hm:
            continue
        u = urllib.parse.urljoin(base, hm.group(1).strip())
        u, _ = urllib.parse.urldefrag(u)
        if not u.startswith(("http://", "https://")):
            continue
        if _ASSET_RE.search(urllib.parse.urlsplit(u).path):
            continue
        if u in seen:
            continue
        seen.add(u)
        text = " ".join(re.sub(r"(?s)<[^>]+>", " ", inner).split())
        if not text:
            # An image link -- the alt or aria-label is what a reader was meant
            # to see in its place.
            am = re.search(r'(?is)(?:aria-label|title)\s*=\s*["\']([^"\']+)',
                           attrs)
            text = " ".join(am.group(1).split()) if am else ""
        out.append((u, text))
    return out


def links(html, base):
    """In-page links, absolute, de-fragmented, HTML only."""
    return [u for u, _ in link_items(html, base)]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def _source_of(url):
    host = urllib.parse.urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def cmd_url(args):
    docs = []
    for u in args.urls:
        try:
            d = fetch_doc(u)
        except gdocs.NotShared as e:
            # Its own branch because the fix is two clicks and nothing else
            # here has a fix at all. Printed on its own lines rather than
            # squeezed into "SKIP <url> (...)", which truncates the one part
            # that matters.
            print("  SKIP %s\n        %s" % (u, e))
            continue
        except Exception as e:
            print("  SKIP %s (%s)" % (u, e))
            continue
        title, text = d.get("title"), d.get("text") or ""
        if len(text.strip()) < 200:
            print("  SKIP %s (only %d chars of text)" % (u, len(text.strip())))
            continue
        # gdocs normalises an /edit?usp=sharing link to the canonical /edit
        # form. kb.add() replaces on exact URL match, so without that the same
        # deck reached three ways is three documents that all vote.
        canon = d.get("url") or u
        rec = {"title": args.title or title or u, "url": canon, "text": text,
               "kind": args.kind, "source": args.source or _source_of(canon)}
        if d.get("pages"):
            rec["pages"] = d["pages"]
        if d.get("page_word"):
            # So slide 7 of a deck is cited as "(slide 7)". A page number
            # pointing into a document with no pages is worse than no citation:
            # the reader goes looking for it.
            rec["page_word"] = d["page_word"]
        docs.append(rec)
        what = ""
        if d.get("gkind"):
            what = " [%s%s]" % (gdocs.LABEL.get(d["gkind"], d["gkind"]),
                                (", %d %ss" % (d["pages"], d["page_word"]))
                                if d.get("pages") and d.get("page_word")
                                else "")
        print("  read %-52.52s %6d chars%s" % (u, len(text), what))
        if d.get("note"):
            print("        note: %s" % d["note"])
        if d.get("is_pdf") and d.get("info"):
            _report_pdf(u, d["info"], indent="        ")
        time.sleep(args.delay)
    _commit(docs, args)


def _prefix_of(start):
    """The directory the start URL lives in.

    --prefix means "stay in this section of the site", and a section is a
    directory, not a file. Using the start URL verbatim works by accident when
    it ends in a slash and fails silently when it does not: no link can begin
    with ".../index.html", so the crawl visits exactly one page, prints nothing,
    and looks like a site with no links rather than a bug.
    """
    parts = urllib.parse.urlsplit(start)
    path = parts.path or "/"
    if not path.endswith("/"):
        last = path.rsplit("/", 1)[-1]
        if "." in last:
            # ".../index.html" -- the section is the folder holding it.
            path = path[:-len(last)]
        else:
            # ".../best-practices" -- a section someone typed without the
            # trailing slash. Treating it as a file would widen the crawl to
            # the entire site, which is the opposite of what --prefix is for.
            path = path + "/"
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, "", ""))


# A crawl sweeps up whatever the navigation happens to link to, so it wants a
# higher bar than a deliberate single-page ingest: index pages, tag listings and
# "coming soon" stubs are all real pages with almost no prose, and each one that
# gets indexed is a page that can win a ranking slot from a page with answers.
CRAWL_MIN_CHARS = 400


def _kind_rules(specs):
    """Parse --kind-for REGEX=KIND into [(compiled, kind), ...].

    One --kind across a whole crawl is right for a site that is one type of
    thing and wrong for a course. FRCDesign's learning course is the case that
    forced this: it teaches motors, gears, belts and ball trajectory on some
    pages and sets graded design challenges on others, under one URL prefix.
    Stamped "exercise", the teaching pages inherit a -0.9 general-question bias
    in app/chat.py and are buried on exactly the broad questions they answer
    best. Stamped "reference", a challenge's acceptance criteria get quoted as
    though they were engineering fact. Neither is a defensible default, so the
    kind is decided per URL and the rules are visible in the command that runs.

    Rules are tried in order and the first match wins, which is what lets the
    narrow rule be written first and a broad fallback after it.
    """
    out = []
    for spec in specs or ():
        if "=" not in spec:
            raise SystemExit(
                "--kind-for wants REGEX=KIND, e.g. --kind-for 'exercise=exercise'"
                "\n  got: %s" % spec)
        pat, kind = spec.rsplit("=", 1)
        kind = kind.strip()
        if kind not in KINDS:
            raise SystemExit("--kind-for: %r is not a kind. Pick one of: %s"
                             % (kind, ", ".join(KINDS)))
        try:
            out.append((re.compile(pat, re.I), kind))
        except re.error as e:
            raise SystemExit("--kind-for: %r is not a valid pattern (%s)"
                             % (pat, e))
    return out


def _crawl_kind(url, rules, default):
    for rx, kind in rules:
        if rx.search(url):
            return kind
    return default


def cmd_crawl(args):
    import requests
    start = args.url
    host = urllib.parse.urlparse(start).netloc
    prefix = _prefix_of(start) if args.prefix else None
    rules = _kind_rules(getattr(args, "kind_for", None))
    seen, queue, docs, thin = set(), [(start, 0)], [], 0
    kinds_used = {}
    while queue and len(docs) < args.max:
        u, d = queue.pop(0)
        if u in seen:
            continue
        seen.add(u)
        try:
            r = requests.get(u, headers={"User-Agent": UA}, timeout=25)
            r.raise_for_status()
            html = r.text
        except Exception as e:
            print("  SKIP %s (%s)" % (u[:70], e))
            continue
        title, text = html_to_text(html)
        if len((text or "").strip()) >= CRAWL_MIN_CHARS:
            kind = _crawl_kind(u, rules, args.kind)
            kinds_used[kind] = kinds_used.get(kind, 0) + 1
            docs.append({"title": title or u, "url": u, "text": text,
                         "kind": kind,
                         "source": args.source or _source_of(u)})
            # The kind is printed because it is the one thing about a crawled
            # page that changes how it is ranked and cannot be seen by reading
            # the page. A rule that matches nothing, or matches everything, is
            # otherwise invisible until an answer goes wrong weeks later.
            print("  [%d/%d] %-10s %-52s %6d chars"
                  % (len(docs), args.max, kind, (title or u)[:52], len(text)))
        else:
            thin += 1
        # Links are followed out of every page fetched, including the thin ones
        # -- a table-of-contents page is exactly the page worth not indexing and
        # worth following.
        if d < args.depth:
            for link in links(html, u):
                if urllib.parse.urlparse(link).netloc != host:
                    continue
                if prefix and not link.startswith(prefix):
                    continue
                if link not in seen:
                    queue.append((link, d + 1))
        # Politeness is not optional when the crawler is pointed at a
        # volunteer-run community site.
        time.sleep(args.delay)

    print("  visited %d page(s); %d had under %d characters and were left out"
          % (len(seen), thin, CRAWL_MIN_CHARS))
    if len(kinds_used) > 1:
        print("  tagged: %s"
              % ", ".join("%d %s" % (n, k)
                          for k, n in sorted(kinds_used.items(),
                                             key=lambda kv: -kv[1])))
    if not docs and thin:
        print("  (if that seems wrong, the site may build its pages in the\n"
              "   browser -- try 'url' on one page to see what text comes back)")
    _commit(docs, args)


# ---------------------------------------------------------------------------
# Link hubs (a page whose value is its list, not its text)
# ---------------------------------------------------------------------------
# Destinations a resource list will always contain and that can never become an
# indexed document. Naming them is not the same as letting them fail: a video
# URL fetches perfectly well, returns a page of player chrome and channel
# blurb, clears the thin-text floor, and lands in the corpus as a real document
# with no content in it -- and the reader who later wonders why the assistant
# cited a YouTube page has no way to find out. Skipping by name also means the
# run prints "3 video" instead of three separate confusing successes.
HUB_SKIP = (
    ("youtube.com", "video"),
    ("youtu.be", "video"),
    ("vimeo.com", "video"),
    ("facebook.com", "social media"),
    ("twitter.com", "social media"),
    ("x.com", "social media"),
    ("instagram.com", "social media"),
    ("linkedin.com", "social media"),
    ("tiktok.com", "social media"),
    ("learn.onshape.com", "page body is built in the browser"),
    ("help.autodesk.com", "refuses non-browser requests"),
    ("docs.google.com", "needs a sign-in"),
    ("drive.google.com", "needs a sign-in"),
    ("accounts.google.com", "sign-in"),
    ("my.firstinspires.org", "sign-in"),
    ("googletagmanager.com", "tracking"),
    ("doubleclick.net", "tracking"),
)

# A hub points at several different kinds of source at once, and the kind is
# what app/chat.py uses to decide whether a document may be quoted as general
# fact. Guessing it per host beats stamping one --kind across the whole run: a
# tool's own documentation and a forum thread are not interchangeable evidence,
# and the alternative to guessing is not "no guess", it is "reference", which
# is the most permissive kind there is. --kind still overrides everything.
KIND_BY_HOST = (
    ("docs.wpilib.org", "docs"),
    ("wpilib.org", "docs"),
    ("cad.onshape.com", "docs"),
    ("onshape4frc.com", "docs"),
    ("github.com", "docs"),
    ("chiefdelphi.com", "forum"),
)

# A linked PDF is a document somebody chose to publish, so a one-page pit
# checklist is short on purpose and worth keeping. A linked HTML page that
# short is a stub, a redirect notice or a body that never rendered.
HUB_MIN_PDF = 200
HUB_MIN_HTML = 400


def _kind_for(url, override):
    if override:
        return override
    host = urllib.parse.urlparse(url).netloc.lower()
    for h, k in KIND_BY_HOST:
        if host == h or host.endswith("." + h):
            return k
    return "reference"


def _best_title(found, anchor, url):
    """What to call this document in a source list."""
    found = " ".join((found or "").split())
    anchor = " ".join((anchor or "").split())
    if (len(anchor) < 8 or len(anchor) > 120
            or not re.search(r"[A-Za-z]{3}", anchor)):
        # "here", "download", "PDF" -- a label, not a name for anything.
        anchor = ""
    unusable = (not found
                or " " not in found
                or found.lower().endswith((".pdf", ".doc", ".docx")))
    if anchor and unusable:
        return anchor[:200]
    return found or anchor or url.rsplit("/", 1)[-1]


def _hub_reject(url, hub_host, section, pdf_only, onsite_only, isolated):
    """Why this link is not worth fetching, or "" if it is."""
    host = urllib.parse.urlparse(url).netloc.lower()
    for bad, why in HUB_SKIP:
        if host == bad or host.endswith("." + bad):
            return why
    if _UNREADABLE_RE.search(urllib.parse.urlsplit(url).path):
        # Named rather than fetched-and-refused so the run says "1 office file"
        # instead of spending a request to print a content-type nobody asked
        # about. A .docx worksheet is a real resource; it is just not one this
        # tool can read, and saying which is the useful half of the message.
        return "office file this tool cannot read"
    is_pdf = _PDF_RE.search(url) is not None
    if pdf_only and not is_pdf:
        return "not a PDF (--pdf-only)"
    if host == hub_host and not isolated:
        # Only when the page's main content could not be isolated, so this list
        # still has the site menu in it. Same-site links are then mostly the
        # site's own furniture, and the two exceptions are the documents
        # themselves -- which live under an uploads path with no relationship
        # to where the hub page sits -- and neighbouring pages of the same
        # library section.
        #
        # When the main content *was* isolated this test is switched off, and
        # deliberately: inside it, a same-site link is a resource the editors
        # chose to name, and the section rule would throw away the real ones
        # that happen to live elsewhere on the site. On the page this was built
        # for that is the KitBot, filed under /resource-library/ while the hub
        # sits under /resources/library/.
        if is_pdf or url.startswith(section):
            return ""
        return "site navigation"
    if onsite_only:
        return "another site"
    return ""


def cmd_hub(args):
    """Ingest the documents a link-hub page points at, rather than the page.

    A resource library -- FIRST's FRC technical-resources page is the one this
    was written for -- is worth almost nothing as a document and a great deal
    as a list. Its own text is link labels and one-line blurbs, so it shares
    vocabulary with every question a team will ever ask and answers none of
    them: indexing it spends a ranking slot on every search to say "these
    documents exist". What is worth having is the two dozen PDFs behind it and
    the handful of sites it names.

    `crawl` cannot do this, for two reasons that are right there and wrong
    here. It stays on one host, and a hub's whole value is that it points off
    one. And it hands everything it fetches to the HTML extractor, so the
    Pneumatics Manual would arrive as tag-stripped PDF bytes.
    """
    import requests
    start = args.url
    try:
        r = requests.get(start, headers={"User-Agent": UA}, timeout=25)
        r.raise_for_status()
        r.encoding = r.encoding or "utf-8"
        html = r.text
    except Exception as e:
        raise SystemExit(
            "could not read the hub page: %s\n"
            "  If that was a 403 or a 503, the site is refusing requests that\n"
            "  do not come from a browser. Open the page, save it as HTML, and:\n"
            "      python tools/kb_ingest.py hubfile \"saved page.html\" --base %s"
            % (e, start))
    _hub_from_html(html, start, args)


def cmd_hubfile(args):
    """Same as `hub`, but reading a page you saved from your browser.

    The escape hatch for a site that blocks automated requests. The links in a
    saved page are usually relative, so --base says what they were relative to;
    without it there is nothing to resolve them against and every link is
    dropped as non-absolute.
    """
    if not os.path.exists(args.path):
        raise SystemExit("no such file: %s" % args.path)
    with open(args.path, "r", encoding="utf-8", errors="replace") as fh:
        html = fh.read()
    base = args.base
    if not base:
        m = re.search(r'(?is)<base\b[^>]*href\s*=\s*["\']([^"\']+)', html)
        if m:
            base = m.group(1).strip()
    if not base:
        raise SystemExit(
            "give --base <the url this page came from>, so the links in it can\n"
            "be turned into addresses that mean something outside your Downloads\n"
            "folder.")
    _hub_from_html(html, base, args)


def _hub_from_html(html, start, args):
    if getattr(args, "whole_page", False):
        region, isolated = html, False
    else:
        region, isolated = _main_region(html)
    items = link_items(region, start)
    if isolated and len(items) < 5:
        # The landmark was there and held almost nothing -- some sites put the
        # real list in a sibling element. A noisier list the filters below can
        # trim beats a hub that reports, wrongly, that it has nothing on it.
        items, isolated = link_items(html, start), False

    hub_host = urllib.parse.urlparse(start).netloc.lower()
    parts = urllib.parse.urlsplit(start)
    section = urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, parts.path.rsplit("/", 1)[0] + "/", "", ""))

    wanted, skipped, seen = [], [], {start.rstrip("/")}
    for url, anchor in items:
        key = url.rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        why = _hub_reject(url, hub_host, section, args.pdf_only,
                          args.onsite_only, isolated)
        (skipped if why else wanted).append((url, anchor, why))

    print("hub: %s" % start)
    print("  links read from %s"
          % ("this page's main content" if isolated else
             "the whole page, minus its navigation and footer"))
    print("  %d link(s) worth trying, %d left out" % (len(wanted), len(skipped)))
    if skipped:
        by = {}
        for _, _, why in skipped:
            by[why] = by.get(why, 0) + 1
        print("  left out: %s"
              % ", ".join("%d %s" % (v, k)
                          for k, v in sorted(by.items(), key=lambda x: -x[1])))

    if args.dry_run:
        print("\nwould fetch (--max %d):" % args.max)
        for url, anchor, _ in wanted[:args.max]:
            print("  %-10s %-44.44s %s"
                  % (_kind_for(url, args.kind), anchor or "-", url[:70]))
        if len(wanted) > args.max:
            print("  ... and %d more, above --max" % (len(wanted) - args.max))
        print("\nwould skip:")
        for url, anchor, why in skipped:
            print("  %-34.34s %-36.36s %s" % (why, anchor or "-", url[:56]))
        print("\nnothing was added. Drop --dry-run to ingest the list above.")
        return

    docs, thin = [], 0
    for url, anchor, _ in wanted[:args.max]:
        try:
            got = fetch_doc(url)
        except Exception as e:
            print("  SKIP %-58.58s %s" % (url, e))
            continue
        text = (got["text"] or "").strip()
        floor = HUB_MIN_PDF if got["is_pdf"] else HUB_MIN_HTML
        if len(text) < floor:
            print("  thin %-58.58s %d chars" % (url, len(text)))
            thin += 1
            continue
        kind = _kind_for(url, args.kind)
        rec = {"title": _best_title(got["title"], anchor, url), "url": url,
               "text": text, "kind": kind,
               "source": args.source or _source_of(url)}
        if got["info"]:
            rec["pages"] = got["info"]["n_pages"]
        docs.append(rec)
        print("  [%d/%d] %-9s %-44.44s %6d chars"
              % (len(docs), min(len(wanted), args.max), kind, rec["title"],
                 len(text)))
        if got["info"]:
            _report_pdf(url, got["info"], indent="          ")
        time.sleep(args.delay)

    if thin:
        print("  %d link(s) had too little text to be worth indexing" % thin)
    _commit(docs, args)
    print("\n  The hub page itself was not added: its text is link labels and\n"
          "  blurbs, which match every question and answer none. For a site in\n"
          "  that list worth reading in full rather than one page deep:\n"
          "      python tools/kb_ingest.py crawl <that site> --prefix --depth 2")


# Extensions that hold no indexable text and must never reach the `else` branch
# in cmd_file.
#
# That branch opens whatever it is handed as UTF-8 with errors="replace", which
# cannot fail. A PNG comes back as roughly 1600 replacement characters, clears
# the 200-character floor the scan detector uses to decide a document holds
# something, and lands in kb.json as a document whose every token is noise. It
# then competes for a retrieval slot on every question asked afterwards.
#
# Dropping a photo of a machined part onto ADD_TO_LIBRARY.bat is an obvious
# thing to try -- photos are exactly what the calibration side wants -- so this
# is the likely way it happens, and the message points at the tool that does
# want them.
_BINARY_EXT = {
    ".png": "photo", ".jpg": "photo", ".jpeg": "photo", ".gif": "photo",
    ".bmp": "photo", ".tif": "photo", ".tiff": "photo", ".webp": "photo",
    ".heic": "photo",
    ".step": "CAD", ".stp": "CAD", ".stl": "CAD", ".iges": "CAD",
    ".igs": "CAD", ".sldprt": "CAD", ".sldasm": "CAD", ".x_t": "CAD",
    ".dxf": "CAD", ".dwg": "CAD", ".f3d": "CAD", ".3mf": "CAD",
    ".zip": "", ".exe": "", ".dll": "", ".xlsx": "", ".docx": "",
    ".pptx": "", ".mp4": "", ".mov": "", ".mp3": "", ".7z": "",
}


def cmd_file(args):
    paths = []
    for pat in args.paths:
        hits = glob.glob(pat, recursive=True)
        paths.extend(hits or ([pat] if os.path.exists(pat) else []))
    docs = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext in _BINARY_EXT:
            what = _BINARY_EXT[ext]
            print("  SKIP %s" % p)
            if what:
                print("       A %s file holds no text to index." % what)
                print("       To use a pocketed part as a reference for the"
                      " pocketing engine,")
                print("       put it in reference_parts\\ and run"
                      " CALIBRATE_POCKETS.bat instead.")
            else:
                print("       Not a text document -- nothing to index.")
            continue
        info = None
        try:
            if ext == ".pdf":
                info = read_pdf(p, pages=getattr(args, "pages", None),
                                ocr=getattr(args, "ocr", "auto"),
                                password=getattr(args, "password", "") or "")
                text = info["text"]
                # The PDF's own title beats the filename. A file called
                # "doc_2019_final_v2.pdf" is a filename somebody typed once;
                # the title inside it is what the document calls itself, and it
                # is what a reader has to recognise in a source list.
                title = info["title"] or os.path.basename(p)
            elif ext in (".html", ".htm"):
                with open(p, "r", encoding="utf-8", errors="replace") as fh:
                    title, text = html_to_text(fh.read())
                title = title or os.path.basename(p)
            else:
                with open(p, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
                title = os.path.basename(p)
                m = re.search(r"^\s*#\s+(.+)$", text, re.M)
                if m:
                    title = m.group(1).strip()
        except Exception as e:
            print("  SKIP %s (%s)" % (p, e))
            continue
        rec = {"title": args.title or title,
               "url": "file://" + os.path.abspath(p),
               "text": text, "kind": args.kind,
               "source": args.source or "local files"}
        if info:
            rec["pages"] = info["n_pages"]
        docs.append(rec)
        print("  read %-70s %6d chars" % (p[:70], len(text)))
        if info:
            _report_pdf(p, info, indent="       ")
    _commit(docs, args)


def cmd_pdfcheck(args):
    """Show what a PDF would contribute, without putting it in the corpus.

    Ingesting a difficult PDF and then judging it by asking the assistant
    questions is a slow and ambiguous way to find out that its columns came out
    interleaved -- the answer is merely worse, with nothing pointing at why.
    This puts the extraction itself in front of you: how many pages had text,
    what sections were found, and the first stretch of what the index will
    actually hold.
    """
    p = args.path
    if not os.path.exists(p):
        raise SystemExit("no such file: %s" % p)
    info = read_pdf(p, pages=args.pages, ocr=args.ocr,
                    password=args.password or "")
    print("file    : %s" % p)
    print("title   : %s" % info["title"])
    print("engine  : %s%s" % (info["engine"],
                              "" if info["layout"] else "  (no layout mode)"))
    print("pages   : %d read of %d in the file" % (len(info["pages"]),
                                                   info["n_pages"]))
    print("text    : %d characters" % info["chars"])
    print("empty   : %d page(s) with no text layer" % info["n_empty"])
    if info["n_ocr"]:
        print("ocr     : %d page(s) recovered" % info["n_ocr"])
    print("columns : %s" % ("multi-column pages were un-interleaved"
                            if info["columns"] else "single column"))
    ready, why = pdftext.ocr_available()
    print("ocr tool: %s" % ("ready" if ready else why))

    heads = pdftext.outline(info["text"])
    print("\nsections found (%d):" % len(heads))
    for lvl, h in heads[:25]:
        print("  %s%s" % ("  " * (lvl - 1), h[:88]))
    if not heads:
        print("  (none - the assistant will still index it, but chunks from the")
        print("   middle of the document will not carry a section name)")

    for w in info["warnings"]:
        print("\n!  %s" % w)

    body = info["text"]
    print("\n--- first 1200 characters as the assistant will see them "
          "-------------")
    print(body[:1200] if body.strip() else "(nothing)")
    print("--- end ---------------------------------------------------"
          "-------------")
    n_chunks = len(kb.chunk_text(body))
    print("\nthis would add %d chunk(s) to the knowledge base." % n_chunks)
    print("looks right?  python tools/kb_ingest.py file \"%s\"%s" % (
        p, (" --pages %s" % args.pages) if args.pages else ""))


def cmd_text(args):
    body = args.body
    if body == "-":
        body = sys.stdin.read()
    # `typed` waives the 200-character floor. That floor is a scan detector,
    # and a shop rule someone typed on purpose is not a scan -- it is short
    # because rules are short.
    _commit([{"title": args.title or "note", "url": args.url or "",
              "text": body, "kind": args.kind, "typed": True,
              "source": args.source or "manual"}], args)


# ---------------------------------------------------------------------------
# One site, done properly
# ---------------------------------------------------------------------------
# FRCDesign is three separate bodies of writing living under one domain, and
# the only way to get all of it in without wrecking the ranking is to crawl
# each with its own settings. Kept here as a named recipe rather than in
# kb_seed.txt because `seed` fetches one URL per line and cannot follow links,
# and rather than as four lines in a README because a four-command instruction
# is a four-command opportunity to leave out the --kind-for and quietly bury
# half the corpus.
#
# `course` is the interesting one. Its pages are a mix: sections 1b, 2a and 2c
# teach motors, gears, ball trajectory and intake geometry -- general
# engineering, the best answers in the corpus for a broad question -- while the
# numbered exercises and project-overview pages set challenges whose stated
# limits are true only inside the challenge. The --kind-for rule splits them so
# each is ranked as what it is.
FRCDESIGN = (
    ("handbook", "https://www.frcdesign.org/design-handbook/",
     dict(depth=3, max=60, kind="reference", kind_for=None),
     "the design handbook: structure, materials, fasteners, 3D printing"),
    ("course", "https://www.frcdesign.org/learning-course/",
     dict(depth=4, max=200, kind="reference",
          kind_for=["/exercise=exercise", "project-overview=exercise"]),
     "the learning course, every stage and section"),
    ("mechanisms", "https://www.frcdesign.org/mechanism-examples/",
     dict(depth=3, max=40, kind="reference", kind_for=None),
     "worked mechanisms: drivebases, intakes, shooters, elevators, pivots"),
    ("practices", "https://www.frcdesign.org/best-practices/",
     dict(depth=3, max=30, kind="reference", kind_for=None),
     "CAD practice: document, sketch, feature tree and assembly setup"),
)


def cmd_frcdesign(args):
    """Crawl the parts of frcdesign.org worth having, each tagged correctly."""
    import argparse as _ap
    want = args.only or [name for name, _, _, _ in FRCDESIGN]
    unknown = [w for w in want if w not in {n for n, _, _, _ in FRCDESIGN}]
    if unknown:
        raise SystemExit("unknown section(s): %s\n  choose from: %s"
                         % (", ".join(unknown),
                            ", ".join(n for n, _, _, _ in FRCDESIGN)))
    for name, url, opts, blurb in FRCDESIGN:
        if name not in want:
            continue
        print("\n=== %s -- %s" % (name, blurb))
        print("    %s" % url)
        sub = _ap.Namespace(
            url=url, prefix=True, delay=args.delay, source=args.source,
            kb=args.kb, **opts)
        cmd_crawl(sub)


SEED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "kb_seed.txt")


def cmd_seed(args):
    """Ingest the starter list in tools/kb_seed.txt.

    Shipped as an editable text file rather than a hardcoded list because the
    right corpus is team-specific, and because a URL that is right today is a
    404 in two seasons. Lines are `url [| kind]`; blanks and # comments ignored.
    """
    if not os.path.exists(SEED_FILE):
        raise SystemExit("no seed list at %s" % SEED_FILE)
    todo = []
    with open(SEED_FILE, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            url = parts[0]
            kind = parts[1] if len(parts) > 1 and parts[1] in KINDS else "reference"
            todo.append((url, kind))
    print("seeding %d pages from %s" % (len(todo), os.path.basename(SEED_FILE)))
    docs = []
    for url, kind in todo:
        try:
            title, text = fetch(url)
        except Exception as e:
            print("  SKIP %s (%s)" % (url[:70], e))
            continue
        if len((text or "").strip()) < 200:
            print("  SKIP %s (thin)" % url[:70])
            continue
        docs.append({"title": title or url, "url": url, "text": text,
                     "kind": kind, "source": _source_of(url)})
        print("  read %-60s %-10s %6d chars" % ((title or url)[:60], kind, len(text)))
        time.sleep(args.delay)
    _commit(docs, args)


def cmd_list(args):
    idx = kb.index(args.kb)
    if not idx.docs:
        print("knowledge base is empty (%s)" % (args.kb or kb.KB_PATH))
        return
    counts = {}
    for c in idx.chunks:
        counts[c["doc"]] = counts.get(c["doc"], 0) + 1
    print("%-52s %-9s %-18s %5s" % ("TITLE", "KIND", "SOURCE", "CHUNKS"))
    for i, d in enumerate(idx.docs):
        print("%-52s %-9s %-18s %5d"
              % ((d.get("title") or "")[:52], (d.get("kind") or "")[:9],
                 (d.get("source") or "")[:18], counts.get(i, 0)))


def cmd_stats(args):
    s = kb.stats(args.kb)
    print("path   : %s%s" % (s["path"], "" if s["exists"] else "  (not created yet)"))
    print("docs   : %d" % s["docs"])
    print("chunks : %d" % s["chunks"])
    for k, v in sorted(s["by_source"].items(), key=lambda x: -x[1]):
        print("         %-30s %d" % (k, v))


def cmd_search(args):
    hits = kb.search(args.query, k=args.k, path=args.kb)
    if not hits:
        print("nothing matched.")
        return
    for h in hits:
        where = (" > " + h["head"]) if h["head"] else ""
        if h.get("page"):
            where += "  (%s %d)" % (h.get("page_word") or "p.", h["page"])
        print("\n%.3f  %s%s\n        %s"
              % (h["score"], h["title"], where, h["url"]))
        body = " ".join(h["text"].split())
        print("        " + body[:300] + ("..." if len(body) > 300 else ""))


def cmd_remove(args):
    n = kb.remove(args.needle, path=args.kb)
    print("removed %d document(s) matching %r" % (n, args.needle))


def _commit(docs, args):
    if not docs:
        print("nothing to add.")
        return

    # kb.add() drops anything under 200 characters. Say so out loud: the usual
    # cause is a PDF whose text layer is an image, or a page that rendered its
    # body in JavaScript, and both look exactly like success if the only thing
    # printed is "added 0".
    thin = [d for d in docs
            if len((d.get("text") or "").strip())
            < (40 if d.get("typed") else 200)]
    for d in thin:
        print("  skipped %-58.58s  only %d chars of text"
              % (d.get("title") or d.get("url") or "untitled",
                 len((d.get("text") or "").strip())))
    if thin and len(thin) == len(docs):
        print("\nnothing had enough text to index.")
        # The advice has to match what was actually handed in. Telling someone
        # who typed two words that their scanner is at fault is worse than
        # saying nothing.
        if all(d.get("typed") for d in docs):
            print("  A typed note needs about 40 characters -- roughly one\n"
                  "  full sentence. Say what the rule is and why, and it will\n"
                  "  index and be quotable back to you.")
            return
        ready, why = pdftext.ocr_available()
        print("  If these were PDFs they are almost certainly scans -- pictures\n"
              "  of pages, with no text in them to read. Check with:\n"
              "      python tools/kb_ingest.py pdfcheck <file.pdf>")
        if ready:
            print("  OCR is installed here, so try:  ... file <file.pdf> "
                  "--ocr force")
        else:
            print("  To read scans you would need OCR, which is not set up: %s"
                  % why)
        print("  If they were web pages the body is likely built by JavaScript,\n"
              "  so save the page from your browser and ingest the saved .html.")
        return

    added, replaced, chunks = kb.add(docs, path=args.kb)
    print("\nadded %d, replaced %d, %d chunks -> %s"
          % (added, replaced, chunks, args.kb or kb.KB_PATH))
    s = kb.stats(args.kb)
    print("knowledge base now holds %d documents / %d chunks"
          % (s["docs"], s["chunks"]))


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="kb_ingest",
        description="Add documents to the StressViz assistant's knowledge base.")
    ap.add_argument("--kb", default=None, help="path to kb.json (default data/kb.json)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, net=False, kind_default="reference", titled=True):
        p.add_argument("--kind", default=kind_default, choices=KINDS,
                       help="how far this document's claims reach (see module "
                            "docstring)"
                            + ("" if kind_default else "; default: per site"))
        p.add_argument("--source", default=None, help="label shown in listings")
        # Not offered where one command brings back many different documents:
        # a --title that renamed all of them to the same string would make the
        # source list unreadable in exactly the case it matters most.
        if titled:
            p.add_argument("--title", default=None,
                           help="override the detected title")
        if net:
            p.add_argument("--delay", type=float, default=1.0,
                           help="seconds between requests")

    p = sub.add_parser("url", help="ingest web pages, or Google Docs/Slides/"
                                   "Sheets share links")
    p.add_argument("urls", nargs="+")
    common(p, net=True)
    p.set_defaults(func=cmd_url)

    p = sub.add_parser("crawl", help="follow links from a starting page")
    p.add_argument("url")
    p.add_argument("--max", type=int, default=30, help="page limit")
    p.add_argument("--depth", type=int, default=1, help="link depth")
    p.add_argument("--prefix", action="store_true",
                   help="only follow links under the starting URL")
    p.add_argument("--kind-for", action="append", metavar="REGEX=KIND",
                   help="tag pages whose URL matches REGEX with KIND instead "
                        "of --kind; repeatable, first match wins")
    common(p, net=True, titled=False)
    p.set_defaults(func=cmd_crawl)

    p = sub.add_parser(
        "frcdesign",
        help="crawl frcdesign.org's handbook, learning course, mechanism "
             "examples and CAD practices, each tagged correctly")
    p.add_argument("--only", action="append", metavar="SECTION",
                   choices=[n for n, _, _, _ in FRCDESIGN],
                   help="just one section (%s); repeatable"
                        % ", ".join(n for n, _, _, _ in FRCDESIGN))
    p.add_argument("--source", default=None, help="label shown in listings")
    p.add_argument("--delay", type=float, default=1.0,
                   help="seconds between requests")
    p.set_defaults(func=cmd_frcdesign)

    def hubopts(p):
        p.add_argument("--max", type=int, default=50,
                       help="how many linked documents to fetch")
        p.add_argument("--pdf-only", action="store_true",
                       help="take only the PDFs, not the sites it links to")
        p.add_argument("--onsite-only", action="store_true",
                       help="skip links that lead off the hub's own site")
        p.add_argument("--whole-page", action="store_true",
                       help="read links from the whole page, not just its main "
                            "content (use if the list comes back too short)")
        p.add_argument("--dry-run", action="store_true",
                       help="list what would be fetched and skipped, add nothing")
        common(p, net=True, kind_default=None, titled=False)

    p = sub.add_parser(
        "hub", help="ingest the documents a resource-list page links to")
    p.add_argument("url")
    hubopts(p)
    p.set_defaults(func=cmd_hub)

    p = sub.add_parser(
        "hubfile",
        help="same as hub, for a resource-list page saved from your browser")
    p.add_argument("path")
    p.add_argument("--base", default=None,
                   help="the URL the saved page came from, so its links resolve")
    hubopts(p)
    p.set_defaults(func=cmd_hubfile)

    def pdfopts(p):
        p.add_argument("--pages", default=None, metavar="RANGE",
                       help="only these pages, e.g. 1-40 or 2,5,9-12")
        p.add_argument("--ocr", default="auto", choices=("auto", "off", "force"),
                       help="read scanned pages as images (auto: only pages "
                            "with no text layer, and only if OCR is installed)")
        p.add_argument("--password", default=None,
                       help="for a PDF that asks for one to open")

    p = sub.add_parser("file", help="ingest local pdf/md/txt/html files")
    p.add_argument("paths", nargs="+")
    pdfopts(p)
    common(p)
    p.set_defaults(func=cmd_file)

    p = sub.add_parser("pdfcheck",
                       help="show what a PDF would contribute, without adding it")
    p.add_argument("path")
    pdfopts(p)
    p.set_defaults(func=cmd_pdfcheck)

    p = sub.add_parser("text", help="ingest text given on the command line or stdin")
    p.add_argument("body", help="the text, or - to read stdin")
    p.add_argument("--url", default=None)
    common(p)
    p.set_defaults(func=cmd_text)

    p = sub.add_parser("seed", help="ingest tools/kb_seed.txt")
    common(p, net=True, titled=False)
    p.set_defaults(func=cmd_seed)

    p = sub.add_parser("list", help="show what is in the knowledge base")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("stats", help="counts")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("search", help="query the knowledge base directly")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("remove", help="delete documents by url or title substring")
    p.add_argument("needle")
    p.set_defaults(func=cmd_remove)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
