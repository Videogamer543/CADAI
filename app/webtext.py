"""
Turning a web page into the text that is actually on it.

Two callers need exactly the same thing and used to have separate answers.
tools/kb_ingest.py needs it to put a page into the knowledge base, and
app/chat.py needs it to read the top search results in full instead of trusting
the search engine's snippet. A snippet is roughly the first seven hundred
characters of a page, and the first seven hundred characters of an engineering
article are the introduction -- the part that says what the article is about
rather than the part that answers the question. Every specific number the
assistant is being asked for is further down.

So the extraction lives here, in one place, and both sides import it. The
alternative -- two copies that drift -- means a page that indexes cleanly and
reads badly, or the reverse, with no way to tell which copy is at fault.

What makes this more than a tag stripper:

  * Navigation, headers, footers and sidebars are removed. They are the same on
    every page of a site, so leaving them in means every page of that site
    shares a large block of identical text: retrieval then ranks pages by how
    much boilerplate they have in common with the question.
  * Headings are kept, as markdown. A chunk taken from the middle of a long
    page is close to unreadable without its section title, and the title is the
    single best clue to what the chunk is about.
  * BeautifulSoup is used when it is installed and a regex fallback when it is
    not, because the fallback is the difference between "one optional package
    failed to install" and "the feature does not work".
"""
from __future__ import annotations

import re

UA = ("Mozilla/5.0 (compatible; StressViz/1.0; "
      "+https://github.com/stressviz) Python-requests")

# Elements that never contain the article.
DROP_TAGS = ("script", "style", "nav", "header", "footer", "aside", "form",
             "noscript", "svg", "button", "iframe")

# Wrappers whose contents are navigation furniture on nearly every docs site.
DROP_HINT = re.compile(
    r"(^|[\s_-])(nav|menu|sidebar|breadcrumb|toc|table-of-contents|footer|"
    r"header|banner|cookie|subscribe|newsletter|share|social|related|"
    r"pagination|skip-link|search)([\s_-]|$)", re.I)

_TAG_RE = re.compile(r"<[^>]+>")

_TEXT_ELEMENTS = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre",
                  "td", "th", "dd", "dt", "figcaption"]


def _bs4(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(list(DROP_TAGS)):
        tag.decompose()
    for tag in soup.find_all(attrs={"class": True}):
        if DROP_HINT.search(" ".join(tag.get("class") or [])):
            tag.decompose()
    for tag in soup.find_all(attrs={"id": True}):
        if DROP_HINT.search(tag.get("id") or ""):
            tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        # The <h1> beats the <title>: a <title> usually carries the site name
        # as well ("Materials | FRCDesign"), and the site name is noise in a
        # source list where every entry already shows its host.
        title = h1.get_text(strip=True) or title

    out = []
    body = soup.body or soup
    for el in body.find_all(_TEXT_ELEMENTS):
        txt = " ".join(el.get_text(" ", strip=True).split())
        if not txt:
            continue
        name = el.name
        if name.startswith("h") and len(name) == 2 and name[1].isdigit():
            out.append("\n" + "#" * int(name[1]) + " " + txt + "\n")
        elif name == "li":
            out.append("- " + txt)
        else:
            out.append(txt)
    return title, "\n\n".join(out)


def _regex(html):
    """Fallback for when BeautifulSoup is not installed. Deliberately crude."""
    html = re.sub(r"(?is)<(%s)\b.*?</\1>" % "|".join(DROP_TAGS), " ", html)
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
    title = " ".join(_TAG_RE.sub("", m.group(1)).split()) if m else ""
    html = re.sub(r"(?is)<h([1-6])[^>]*>(.*?)</h\1>",
                  lambda m: "\n\n" + "#" * int(m.group(1)) + " "
                  + _TAG_RE.sub("", m.group(2)) + "\n\n", html)
    html = re.sub(r"(?is)</(p|div|li|tr|br)\s*>", "\n\n", html)
    html = re.sub(r"(?is)<li[^>]*>", "\n- ", html)
    text = _TAG_RE.sub(" ", html)
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&")
                .replace("&lt;", "<").replace("&gt;", ">")
                .replace("&#39;", "'").replace("&quot;", '"'))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return title, text.strip()


def html_to_text(html):
    """(title, text). Never raises -- a page that will not parse yields ''."""
    try:
        return _bs4(html)
    except ImportError:
        return _regex(html)
    except Exception:
        try:
            return _regex(html)
        except Exception:
            return "", ""


def pdf_to_text(path, **kw):
    """PDF text. The real work is in app/pdftext.py.

    It lives there rather than here because a PDF needs an entirely different
    kind of repair from an HTML page -- headers that repeat on every page,
    words hyphenated across a line break, columns interleaved into gibberish --
    and none of that has an analogue in a DOM. Re-exported under this name so
    the callers that only ever wanted a string are unchanged.
    """
    try:
        from . import pdftext
    except ImportError:      # pragma: no cover - run as a loose script
        import pdftext       # type: ignore
    return pdftext.pdf_to_text(path, **kw)


def fetch_text(url, timeout=12, max_bytes=2_000_000):
    """Download a page and return (title, text). ('', '') on any failure.

    Failure is not exceptional here and must not be loud. This runs inside the
    request that answers a user's question, alongside results that arrived
    fine; one page behind Cloudflare, one timeout, one PDF that is really a
    scan should cost that one source's detail and nothing else.

    The size cap is not about disk. It is about the case where a URL turns out
    to be a firmware image or a video: streaming it into memory to look for
    prose would stall the answer for as long as the connection holds up.
    """
    try:
        import requests
        with requests.get(url, headers={"User-Agent": UA}, timeout=timeout,
                          stream=True) as r:
            r.raise_for_status()
            # The bare type, not the whole header. Asking whether "xml" appears
            # anywhere in the content-type says yes to a .docx, whose type is
            # application/vnd.openxmlformats-officedocument.wordprocessingml.
            # document -- and a zip archive run through a tag stripper comes
            # back as several thousand characters of mojibake, which is long
            # enough to look like a successfully read source.
            mime = (r.headers.get("content-type") or "").lower()
            mime = mime.split(";")[0].strip()
            if mime and not (mime.startswith("text/")
                             or mime in ("application/xhtml+xml",
                                         "application/xml",
                                         "application/html",
                                         "application/json",
                                         "application/rss+xml",
                                         "application/atom+xml")):
                return "", ""
            buf, total = [], 0
            for part in r.iter_content(65536):
                if not part:
                    break
                total += len(part)
                buf.append(part)
                if total >= max_bytes:
                    break
            html = b"".join(buf).decode(r.encoding or "utf-8", "replace")
        return html_to_text(html)
    except Exception:
        return "", ""
