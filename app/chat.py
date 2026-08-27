"""
Server-side AI assistant + web search + FRC live data.

Keys live in environment variables here — they NEVER reach the browser (the
biggest publish blocker from the HTML version). Ports the browser assistant:
engineering/CAD-first identity, credible open-web search, The Blue Alliance
live data for FRC, and inline-citation formatting.
"""
from __future__ import annotations
import os
import base64
import datetime
import hashlib
import hmac
import time
import requests

# The local knowledge base. Imported defensively because chat.py is run both as
# part of the app package and, occasionally, straight from the tools directory
# during testing -- and because a missing or broken kb.py must degrade the
# assistant to web-only rather than stopping the server from starting at all.
try:
    from . import kb as _kb
except Exception:          # pragma: no cover - import-path fallback
    try:
        import kb as _kb   # type: ignore
    except Exception:
        _kb = None

try:
    from . import webtext as _webtext
except Exception:          # pragma: no cover - import-path fallback
    try:
        import webtext as _webtext   # type: ignore
    except Exception:
        _webtext = None

try:
    from . import grounding as _grounding
except Exception:          # pragma: no cover - import-path fallback
    try:
        import grounding as _grounding   # type: ignore
    except Exception:
        _grounding = None

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
TAVILY_ENDPOINT = "https://api.tavily.com/search"
TBA_BASE = "https://www.thebluealliance.com/api/v3"

SYSTEM_PROMPT = (
    "You are an expert mechanical engineering, hardware, and CAD assistant. "
    "Primary domain: statics and strength of materials, CAD (Onshape/Fusion/"
    "SolidWorks), design-for-manufacturing, hardware selection (motors, "
    "bearings, fasteners, belts, gears), tolerances/fits, and materials. You "
    "also have deep FRC robotics knowledge. Lead with specifics - real numbers, "
    "parts, materials, formulas. State facts directly (no 'According to...'). "
    "\n\nCITATION RULES - follow exactly:\n"
    "- If a SOURCES block is present, cite each factual claim inline with a "
    "bracketed number like [1] placed right after the claim. The number must "
    "match the numbering in the SOURCES block. The app turns [n] into a link.\n"
    "- Only cite a number that actually appears in the SOURCES block.\n"
    "- If NO SOURCES block is present, write the answer with NO bracketed "
    "numbers at all, and open with one short line saying the answer is from "
    "general engineering knowledge rather than retrieved sources.\n"
    "- Never invent a source, a URL, or a citation number.\n\n"
    "When the sources include a page that answers the question directly, lead "
    "with what that page actually says rather than generic advice. Prefer "
    "datasheets, MatWeb, ASM, .edu and official vendor docs for engineering "
    "fundamentals, and FRCDesign.org / Chief Delphi / WPILib for FRC-specific "
    "design practice.\n\n"
    "SOURCE SCOPE - follow exactly. Every source is tagged with a KIND, and the "
    "kind decides how far that page's claims reach:\n"
    "- 'exercise brief' is a practice problem, design challenge or assignment. "
    "Every dimension, range, limit and requirement it states is a rule of THAT "
    "exercise, not of engineering. Never restate one as a general design "
    "requirement. If it is worth using at all, name where it came from in the "
    "same sentence - e.g. 'FRCDesign's ball-shooter exercise sets its target at "
    "X for that challenge' - and say what the underlying reason is.\n"
    "- 'forum thread' is one team's experience, not consensus. Attribute it "
    "('one team reports...') unless an independent source agrees.\n"
    "- 'season rules' binds one game year only. Name the year.\n"
    "- 'blog' is one author's opinion; attribute it.\n"
    "- 'official docs', 'vendor spec' and 'material data' may be stated "
    "flatly.\n"
    "- Any number, limit or range that appears in only ONE source, where that "
    "source is an exercise brief, forum thread, season rules page or blog, must "
    "be attributed inside the sentence that uses it. A bracketed [n] alone is "
    "not enough - the reader cannot see the tag, only you can.\n"
    "- When the user asks how to design or choose something in general, lead "
    "with the governing principle and the quantities that always matter. Reach "
    "for a specific figure only when a broad-scope source supports it, or when "
    "you attribute it.\n\n"
    "ANSWERING THE ACTUAL QUESTION - follow exactly:\n"
    "- If a QUESTION ANALYSIS block is present, it lists what the user is "
    "really asking, the entities they named, and the constraints they stated. "
    "Honour every constraint literally (a stated material, size, budget, "
    "motor, year or process is not optional background).\n"
    "- Open with a direct answer to the exact question asked - the specific "
    "number, part, procedure or verdict - in the first sentence or two. Put "
    "supporting detail after that, never before.\n"
    "- Do not pad the answer with adjacent topics the user did not ask about. "
    "A question about one mechanism is not an invitation to survey the robot. "
    "This does not apply when the QUESTION ANALYSIS block says Shape: LIST - "
    "there the user asked what exists in a category, and naming every member "
    "the sources cover IS answering the question.\n"
    "- If the sources do not actually cover what was asked, say so plainly in "
    "one line and then answer from engineering fundamentals."
)

# Prompt for the cheap analysis pass. Kept separate from SYSTEM_PROMPT so the
# planner is not told to cite sources it has not seen.
PLANNER_PROMPT = (
    "You analyse a user's engineering/CAD/FRC question and plan how to search "
    "for it. Reply with ONE JSON object and nothing else - no prose, no code "
    "fence. Schema:\n"
    '{"restated": "<the question rewritten as one precise sentence>",\n'
    ' "intent": "<fact_lookup|design_guidance|comparison|how_to|troubleshoot'
    '|calculation|opinion>",\n'
    ' "entities": ["<specific things named: parts, materials, motors, tools, '
    'mechanisms, software, teams>"],\n'
    ' "constraints": ["<hard requirements stated or clearly implied: '
    'material, size, weight, budget, year, process, rule>"],\n'
    ' "scope": "<general|specific>",\n'
    ' "topics": ["<zero or more of the ALLOWED TOPICS, most relevant first>"],\n'
    ' "queries": ["<2-4 web search queries>"],\n'
    ' "sites": ["<zero or more of the ALLOWED SITES worth restricting to>"]}\n\n'
    'Rules for scope: "specific" only when the user names a particular '
    "document, design challenge, competition season/year, team or product and "
    'wants what THAT one says. Otherwise "general" - they want engineering '
    "practice that holds regardless of which year or exercise it came from.\n"
    "Rules for queries: write what an expert would type into a search box, not "
    "the user's conversational phrasing. Make them DIFFERENT from each other - "
    "one should target the specific named thing, one should target the "
    "underlying engineering principle or a known document/page title. Keep each "
    "under 12 words. When the scope is general, do NOT write a query that would "
    "land on a practice problem, design challenge, assignment or one season's "
    "rules - those pages state requirements that are true only inside "
    "themselves, and they are the single most common way an answer ends up "
    "quoting a rule that does not exist. Never invent a topic or site outside "
    "the allowed lists."
)

# ---------------------------------------------------------------------------
# Topic routing.
#
# The old retrieval ran at most one un-steered Tavily query, and only when the
# question happened to contain a word from a short keyword list -- "intake
# design frc golden rules" matched nothing, so no search ran at all and the
# model answered from memory while still emitting [n] markers that pointed at
# nothing. Now every real question is routed to a topic first, and the topic
# decides which domains get searched. FRCDesign.org has a page literally titled
# "Intake Golden Rules"; the point of routing is that questions like that land
# on it instead of on generic advice.
# ---------------------------------------------------------------------------
ROUTES = [
    ("FRC mechanism design", [
        r"\b(intake|shooter|flywheel|hopper|indexer|feeder|climber|elevator|"
        r"telescop\w*|arm|wrist|pivot|slapdown|linkage|four ?bar|drivetrain|"
        r"swerve|west ?coast|tank drive|gearbox|bumper|chassis|bellypan|"
        r"end ?effector|dead ?axle|live ?axle|game ?piece|golden rules?|"
        r"frc|first robotics|robot)\b"],
     ["frcdesign.org", "chiefdelphi.com", "onshape4frc.com", "docs.wpilib.org",
      "wcproducts.com", "thethriftybot.com", "andymark.com", "revrobotics.com",
      "vexrobotics.com", "firstinspires.org"]),
    ("CAD workflow", [
        r"\b(onshape|fusion ?360|solidworks|inventor|sketch|mate|assembly|"
        r"configuration|feature ?script|variable ?studio|part ?studio|drawing|"
        r"gd&t|datum|tolerance ?stack)\b"],
     ["onshape.com", "forum.onshape.com", "onshape4frc.com",
      "help.autodesk.com", "solidworks.com"]),
    ("motors and power transmission", [
        r"\b(neo|kraken|falcon|cim|775|vortex|minion|brushless|motor|stall|"
        r"free ?speed|gear ?ratio|reduction|encoder|spark ?max|talon|"
        r"current ?limit|torque|rpm|belt|pulley|htd|gt2|chain|#25|#35|sprocket)\b"],
     ["motors.vex.com", "docs.wpilib.org", "docs.revrobotics.com",
      "ctr-electronics.com", "vexrobotics.com", "wcproducts.com",
      "sdp-si.com", "gates.com"]),
    ("fasteners and bearings", [
        r"\b(bearing|flanged|thrust|bushing|shoulder ?bolt|rivet|helicoil|"
        r"heli-?coil|thread|tap|10-32|1/4-20|8-32|m3|m5|loctite|nyloc|"
        r"press ?fit|slip ?fit|retaining ?ring|snap ?ring|shaft ?collar)\b"],
     ["mcmaster.com", "skf.com", "boltdepot.com", "engineeringtoolbox.com",
      "vexrobotics.com", "wcproducts.com"]),
    ("machining and manufacturing", [
        r"\b(cnc|router|mill|lathe|waterjet|laser ?cut|end ?mill|feed|speed|"
        r"chip ?load|toolpath|fixture|deburr|tap ?drill|pocket\w*|lighten\w*|"
        r"tolerance|fit|sendcutsend|3d ?print)\b"],
     ["sendcutsend.com", "practicalmachinist.com", "protolabs.com",
      "haascnc.com", "mcmaster.com", "engineeringtoolbox.com"]),
    ("materials and strength", [
        r"\b(6061|7075|2024|4130|aluminum|aluminium|steel|titanium|polycarb\w*|"
        r"delrin|acetal|abs|pla|carbon ?fib\w*|yield|ultimate|tensile|modulus|"
        r"fatigue|anodiz\w*|heat ?treat|temper|stress|strain|buckl\w*|"
        r"safety ?factor|von ?mises)\b"],
     ["matweb.com", "engineeringtoolbox.com", "asm.org", "efunda.com",
      "mcmaster.com", "azom.com"]),
]

# Short, search-friendly anchor per topic. Appending the verbose label
# ("materials and strength") to a query is noise; these are the words that
# actually improve retrieval.
ANCHOR = {
    "FRC mechanism design": "FRC robot design",
    "CAD workflow": "CAD",
    "motors and power transmission": "",
    "fasteners and bearings": "engineering fit",
    "machining and manufacturing": "machining",
    "materials and strength": "material properties",
}

_STOP = set("""a an the is are was were be been being of for to in on at by with from
about into over after before under above and or but if then than that this these those
what which who whom whose when where why how do does did doing can could should would
will shall may might must i you he she it we they me my your our their its as so such
some any all more most other please tell explain give show me best good better""".split())


def route(question: str):
    """Pick the topic(s) this question belongs to and the domains to search."""
    import re
    hits, domains = [], []
    for label, pats, doms in ROUTES:
        if any(re.search(p, question, re.I) for p in pats):
            hits.append(label)
            for d in doms:
                if d not in domains:
                    domains.append(d)
    return hits, domains[:14]


def narrow(question: str, topics):
    """Build a tighter query from the question's own salient terms.

    Search engines do badly with conversational phrasing; stripping the filler
    words and re-anchoring on the topic turns 'intake design frc golden rules'
    into 'FRC intake design golden rules mechanism design', which is what
    actually retrieves the FRCDesign.org page of that name.
    """
    import re
    words = [w for w in re.findall(r"[A-Za-z0-9#/\-\.]+", question.lower())
             if w not in _STOP and len(w) > 1]
    seen, keep = set(), []
    for w in words:
        if w not in seen:
            seen.add(w); keep.append(w)
    core = " ".join(keep[:12])
    if topics:
        return f"{core} {ANCHOR.get(topics[0], '')}".strip()
    return core or question


# Every domain any route is allowed to use. The planner picks from this set;
# anything it invents is discarded, so a hallucinated site can never become a
# Tavily include_domains entry.
KNOWN_DOMAINS = sorted({d for _, _, doms in ROUTES for d in doms})
KNOWN_TOPICS = [label for label, _, _ in ROUTES]
INTENTS = {"fact_lookup", "design_guidance", "comparison", "how_to",
           "troubleshoot", "calculation", "opinion"}


def _json_block(text: str):
    """Pull the first JSON object out of a model reply, fence or no fence."""
    import json
    import re
    if not text:
        return None
    text = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(),
                  flags=re.I | re.M).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


def _clean_list(val, limit, maxlen=80):
    out = []
    if isinstance(val, str):
        val = [val]
    for x in (val or []):
        if not isinstance(x, str):
            continue
        s = " ".join(x.split())[:maxlen].strip()
        if s and s.lower() not in {o.lower() for o in out}:
            out.append(s)
        if len(out) >= limit:
            break
    return out


def analyze(question: str, model=None):
    """Read the question before searching for it.

    The regex router can only see words it was told about in advance, so it
    treats 'what end mill for 1/4" 6061 on a Shapeoko' and 'why did my end mill
    snap in 6061' as the same query. This pass asks a small fast model what is
    actually being asked - the intent, the specific things named, the
    constraints that must be respected - and lets that drive the searches.

    Returns a dict that is always safe to use: every field is validated against
    the known topic/site lists, and any failure degrades to the regex router.
    """
    fallback_topics, fallback_domains = route(question)
    out = {
        "restated": question.strip(),
        "intent": "",
        "entities": [],
        "constraints": [],
        "topics": fallback_topics,
        "domains": fallback_domains,
        "queries": [],
        "scope": "",
        "source": "regex",
    }
    if not os.environ.get("GROQ_API_KEY"):
        return out
    prompt = (
        PLANNER_PROMPT
        + "\n\nALLOWED TOPICS: " + ", ".join(KNOWN_TOPICS)
        + "\nALLOWED SITES: " + ", ".join(KNOWN_DOMAINS)
    )
    try:
        resp = chat(
            [{"role": "user", "content": "QUESTION: " + question.strip()[:1200]}],
            model=model or os.environ.get("GROQ_PLANNER_MODEL",
                                          "llama-3.1-8b-instant"),
            temperature=0.0, max_tokens=400, system=prompt,
        )
        raw = resp["choices"][0]["message"]["content"]
    except Exception:
        return out
    data = _json_block(raw)
    if not isinstance(data, dict):
        return out

    restated = data.get("restated")
    if isinstance(restated, str) and 5 < len(restated.strip()) < 400:
        out["restated"] = restated.strip()
    intent = str(data.get("intent") or "").strip().lower().replace("-", "_")
    if intent in INTENTS:
        out["intent"] = intent
    out["entities"] = _clean_list(data.get("entities"), 8)
    out["constraints"] = _clean_list(data.get("constraints"), 6, maxlen=120)
    scope = str(data.get("scope") or "").strip().lower()
    if scope in {"general", "specific"}:
        out["scope"] = scope

    topics = [t for t in _clean_list(data.get("topics"), 3, maxlen=60)
              if t in KNOWN_TOPICS]
    # The regex router is precise when it fires (it matches literal part
    # numbers and mechanism names), so union rather than replace.
    for t in fallback_topics:
        if t not in topics:
            topics.append(t)
    out["topics"] = topics[:3]

    sites = [s.lower().lstrip("*.").strip("/")
             for s in _clean_list(data.get("sites"), 10, maxlen=60)]
    domains = [s for s in sites if s in KNOWN_DOMAINS]
    for label, _, doms in ROUTES:
        if label in out["topics"]:
            for d in doms:
                if d not in domains:
                    domains.append(d)
    out["domains"] = domains[:14]

    out["queries"] = [q for q in _clean_list(data.get("queries"), 4, maxlen=120)
                      if len(q.split()) >= 2]
    if out["queries"] or out["entities"] or out["intent"]:
        out["source"] = "llm"
    return out


def _groq(messages, model, temperature, max_tokens, system, stream):
    """One place that talks to Groq, so streaming and not-streaming cannot
    drift apart in model, prompt or error handling."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")
    model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
    full = [{"role": "system", "content": system or SYSTEM_PROMPT}] + messages
    r = requests.post(
        GROQ_ENDPOINT,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "messages": full, "temperature": temperature,
              "max_tokens": max_tokens, "stream": bool(stream)},
        timeout=60,
        stream=bool(stream),
    )
    if r.status_code >= 400:
        # Surface Groq's own message (bad key, retired model, rate limit)
        # instead of a bare "502 Bad Gateway". Reading .text here consumes the
        # streamed body, which is exactly right: an error response is a single
        # JSON object, not a stream, and there is nothing else to read from it.
        try:
            msg = r.json().get("error", {}).get("message", r.text[:300])
        except Exception:
            msg = r.text[:300]
        raise RuntimeError(f"Groq {r.status_code}: {msg}")
    return r


def chat(messages, model=None, temperature=0.3, max_tokens=1024, system=None):
    return _groq(messages, model, temperature, max_tokens, system, False).json()


def chat_stream(messages, model=None, temperature=0.3, max_tokens=1024,
                system=None):
    """The same call as chat(), yielding the answer as it is written.

    The wire format is server-sent events: each line is `data: {json}` and the
    last is `data: [DONE]`. A frame that will not parse is skipped rather than
    raised on -- a single malformed keep-alive must not destroy an answer that
    is nine tenths written and already on the reader's screen.
    """
    import json
    r = _groq(messages, model, temperature, max_tokens, system, True)
    for raw in r.iter_lines(decode_unicode=True):
        if not raw:
            continue
        if raw.startswith("data:"):
            raw = raw[5:].strip()
        if raw == "[DONE]":
            break
        if not raw:
            continue
        try:
            obj = json.loads(raw)
            piece = obj["choices"][0]["delta"].get("content")
        except Exception:
            continue
        if piece:
            yield piece


class _MarkerStrip:
    """Removes [n] citation markers that point past the end of the source list.

    The non-streaming path did this in one regex over the finished answer. It
    cannot here, because "[12]" arrives as "[1" and then "2]", and a marker
    split across two frames would both escape the filter and flash a half-drawn
    "[1" on screen. So anything that could still become a marker is held back --
    at most three characters -- and released the moment it cannot. Fed the whole
    answer in any chunking, this produces exactly what the single regex did.
    """

    def __init__(self, n_src):
        import re
        self._re = re
        self._marker = re.compile(r"\[(\d{1,2})\]")
        self._partial = re.compile(r"\[\d{0,2}$")
        self.n = n_src
        self.buf = ""

    def _clean(self, s):
        return self._marker.sub(
            lambda m: m.group(0) if 1 <= int(m.group(1)) <= self.n else "", s)

    def feed(self, chunk):
        self.buf += chunk
        i = self.buf.rfind("[")
        if i != -1 and self._partial.match(self.buf[i:]):
            out, self.buf = self.buf[:i], self.buf[i:]
        else:
            out, self.buf = self.buf, ""
        return self._clean(out)

    def flush(self):
        out, self.buf = self.buf, ""
        return self._clean(out)


def search(query, domains=None, depth="advanced", max_results=6, raw=False):
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY not set")
    # No include_answer. Tavily's "answer" is its own LLM summary of the
    # results, which costs a second or so per search -- and nothing in this
    # file ever reads it. The summarising is done later, once, by the answering
    # model, over the merged pool of every search. Asking for it was paying for
    # four paragraphs per question that were thrown away unread.
    body = {"api_key": key, "query": query, "search_depth": depth,
            "max_results": max_results,
            "include_raw_content": bool(raw)}
    if domains:
        body["include_domains"] = domains
    r = requests.post(TAVILY_ENDPOINT, json=body, timeout=30)
    r.raise_for_status()
    return r.json()


def _tba_get(path):
    key = os.environ.get("TBA_API_KEY")
    if not key:
        raise RuntimeError("TBA_API_KEY not set")
    r = requests.get(TBA_BASE + path, headers={"X-TBA-Auth-Key": key}, timeout=20)
    r.raise_for_status()
    return r.json()


def tba_team(num, year=None):
    year = year or datetime.datetime.now().year
    info = _tba_get(f"/team/frc{num}")
    out = {"info": info, "sources": [
        {"title": f"The Blue Alliance — Team {num}",
         "url": f"https://www.thebluealliance.com/team/{num}"}]}
    if info.get("website"):
        out["sources"].append({"title": f"Team {num} website", "url": info["website"]})
    try:
        socials = _tba_get(f"/team/frc{num}/social_media")
        out["socials"] = socials
    except Exception:
        out["socials"] = []
    return out


_CHITCHAT = r"^\s*(hi|hey|hello|yo|thanks|thank you|ty|ok|okay|cool|nice|sup|" \
            r"good (morning|afternoon|evening)|how are you)\b[\s!.?]*$"


MAX_SEARCHES = 4


def _plan_queries(question, topics, domains, analysis=None):
    """Complementary searches derived from what the question actually asks.

    When the analysis pass produced queries, the first one (the planner is told
    to put the most specific first) runs against the routed allowlist and the
    rest run open-web, so we get both the authoritative page and whatever the
    allowlist does not cover. Without an analysis we fall back to the old
    narrowed / raw pair.
    """
    nq = narrow(question, topics)
    plan, seen = [], set()

    def add(q, doms, raw, n, tag):
        q = (q or "").strip()
        key = (q.lower(), bool(doms))
        if not q or key in seen or len(plan) >= MAX_SEARCHES:
            return
        seen.add(key)
        plan.append({"q": q, "domains": doms, "raw": raw, "n": n, "tag": tag})

    aq = list((analysis or {}).get("queries") or [])
    if aq:
        if domains:
            add(aq[0], domains, True, 6, "site-steered")
        for q in (aq[1:] if domains else aq):
            add(q, None, False, 4, "analysed")
        # Keep one un-analysed shot so a bad plan cannot blind the search.
        add(question, None, False, 4, "verbatim")
    else:
        if domains:
            add(nq, domains, True, 6, "site-steered")
        add(question, None, False, 5, "open web")
        if not domains:
            add(nq, None, False, 4, "narrowed")
    return plan[:MAX_SEARCHES]


def _host(url):
    import re
    m = re.match(r"https?://(?:www\.)?([^/]+)", url or "")
    return m.group(1).lower() if m else ""


# ---------------------------------------------------------------------------
# What kind of document is this, and how far do its claims reach?
#
# This is the fix for the failure that motivated it: asked what to consider
# when designing a shooter, the assistant answered that the base must sit no
# higher than 3.5 ft and that it must shoot from 3-15 ft. Both numbers are
# real, both were correctly cited -- and neither is a fact about shooters. They
# are the acceptance criteria of one FRCDesign practice exercise, true only
# inside that exercise. Retrieval had no way to tell the difference, so the
# answer laundered an assignment brief into an engineering rule.
#
# A page's TYPE predicts that reach far better than its domain does. The same
# host serves reference material and practice problems; a forum serves one
# team's experiment next to hard-won consensus. So every result is typed, the
# type biases the ranking, and the type is handed to the answering model so it
# knows which claims it is allowed to state flatly.
# ---------------------------------------------------------------------------
# (regex on title+url, regex on body, kind). First match wins, so the narrow
# kinds are listed before the broad ones.
_KIND_PATTERNS = [
    (r"design[- ]challenge|challenge\s*#?\d|practice[- ]problem|"
     r"\bexercise\b|\bassignment\b|\bhomework\b|\bworkshop\b|\bquiz\b|"
     r"\btutorial\b|walk-?through|\blesson\b|getting[- ]started",
     r"your (?:design|robot|task|goal) must|the goal of this challenge|"
     r"requirements? for this|acceptance criteria|when you are done|"
     r"in this challenge|for this exercise",
     "exercise"),
    (r"game[- ]manual|\bmanual\b|\brule[s]?\b|\bR\d{2,3}\b|competition manual",
     r"", "rules"),
]

# Domains whose reach is settled by the host, not the page.
_KIND_HOSTS = [
    ({"chiefdelphi.com", "reddit.com", "forum.onshape.com",
      "practicalmachinist.com"}, "forum"),
    ({"medium.com", "substack.com", "blogspot.com", "wordpress.com"}, "blog"),
    ({"docs.wpilib.org", "firstinspires.org", "help.autodesk.com",
      "docs.revrobotics.com", "ctr-electronics.com", "onshape.com",
      "solidworks.com", "motors.vex.com"}, "docs"),
    ({"matweb.com", "asm.org", "efunda.com", "azom.com",
      "engineeringtoolbox.com"}, "data"),
    ({"mcmaster.com", "wcproducts.com", "andymark.com", "revrobotics.com",
      "vexrobotics.com", "thethriftybot.com", "sdp-si.com", "gates.com",
      "skf.com", "boltdepot.com", "sendcutsend.com", "protolabs.com",
      "haascnc.com"}, "vendor"),
]

# How each kind is shown to the model and to the reader, and how much its score
# moves -- once for a general question, once for a question about a particular
# document. The two columns are the whole point: a practice brief is the worst
# possible source for "how do I design a shooter" and the best possible source
# for "what does challenge 3 require", so its bias flips rather than merely
# switching off. Both are nudges, not bans: an exercise page can still be the
# only thing on the open web about a niche mechanism, and burying it outright
# would trade one failure for another.
#                (label, caveat for the model, bias_general, bias_specific)
KIND_INFO = {
    "exercise": ("exercise brief",
                 "a practice problem - its stated limits and dimensions are "
                 "rules of that exercise only, never general requirements",
                 -0.9, 0.5),
    "rules":    ("season rules",
                 "binds one competition year; name the year if you use it",
                 -0.4, 0.5),
    # A team's own handbook or shop notes. Neither of the two kinds it would
    # otherwise land in fits: it is not general engineering practice, and it is
    # not a competition rule either. "Pocket floors are 0.100 in" can be
    # perfectly true of one shop's end mill and completely wrong as advice, so
    # it has to be attributed rather than either promoted or buried -- a team's
    # own convention is usually the single most useful source for a question
    # about what THEY should do, and the most misleading one for a question
    # about what is true in general.
    "convention": ("team convention",
                   "one team's own practice, not a general requirement - say "
                   "whose convention it is",
                   -0.35, 0.6),
    "forum":    ("forum thread",
                 "one team's experience, not consensus - attribute it",
                 -0.3, 0.1),
    "blog":     ("blog",
                 "one author's opinion - attribute it", -0.5, 0.0),
    "docs":     ("official docs", "authoritative; may be stated flatly",
                 0.4, 0.2),
    "data":     ("material data", "authoritative; may be stated flatly",
                 0.35, 0.2),
    "vendor":   ("vendor spec",
                 "authoritative for that product; check it generalises",
                 0.05, 0.2),
    "reference": ("reference", "general design reference", 0.2, 0.0),
}


def classify_source(title, url, text=""):
    """Type a retrieved page. Body evidence can override a weak title match."""
    import re
    head = f"{title or ''} {url or ''}".lower()
    body = (text or "")[:4000].lower()
    for t_pat, b_pat, kind in _KIND_PATTERNS:
        if re.search(t_pat, head):
            return kind
        # A page can be an assignment without saying so in the title -- the
        # give-away is second person plus a requirements list in the body.
        if b_pat and len(re.findall(b_pat, body)) >= 2:
            return kind
    host = _host(url)
    for hosts, kind in _KIND_HOSTS:
        if any(host == h or host.endswith("." + h) for h in hosts):
            return kind
    return "reference"


_SPECIFIC_RE = (
    r"\b(20\d\d|19\d\d)\b|\bchallenge\b|\bgame manual\b|\bthis (?:year|season)\b"
    r"|\bteam\s*#?\d{1,5}\b|\brule\s*[A-Z]?\d|\bmanual\b|\bin the (?:rules|manual)\b"
)


def wants_specific(question, analysis=None):
    """Is the user asking what one particular document says?

    If they are, an exercise brief or a season rules page is exactly the right
    answer and must not be demoted. The planner's own read is trusted first
    because it can see phrasing a regex cannot; the regex is the floor, so a
    question naming a year or a rule number is treated as specific even when
    the planner is unavailable.
    """
    import re
    if (analysis or {}).get("scope") == "specific":
        return True
    if (analysis or {}).get("scope") == "general":
        # Still honour an explicit year/rule reference in the raw text.
        return bool(re.search(r"\b(20\d\d)\b|\brule\s*[A-Z]?\d|\bteam\s*#?\d",
                              question, re.I))
    return bool(re.search(_SPECIFIC_RE, question, re.I))


# A question that asks for a *set* rather than an answer. "What mechanisms are
# common in FRC" wants five names; "how do I design a shooter" wants one thing
# explained. Nothing else in this file could tell those apart, and the answer to
# the first was being graded as though it were the second -- an answer naming
# four of five categories reads as complete, cites real sources, passes the
# grounding check, and is wrong in the only way that matters.
#
# Deliberately a regex and not a second model call, for the reason given in
# app/grounding.py: an LLM judging the shape of a question fails in correlated
# ways with the LLM answering it, and the failure is not reproducible. A regex
# is dumb, instant, and can be fixed by whoever reads the wrong answer.
#
# The plural is load-bearing. "what is a common mechanism" is a definition
# question and stays out; "what are the common mechanisms" is a list question.
_ENUM_RE = (
    r"\bwhat\s+(?:are|were)\b.*\b(?:the\s+)?(?:different|various|common|main|"
    r"typical|standard|usual|popular)?\s*\w+s\b"
    r"|\b(?:list|name|enumerate)\b\s+(?:the\s+|some\s+|all\s+|a few\s+)?\w+"
    r"|\bwhat\s+(?:kinds?|types?|sorts?|categories|options|varieties)\s+of\b"
    r"|\bwhich\s+\w+s\s+(?:are|do|can|should)\b"
    r"|\b(?:most\s+)?common(?:ly used)?\s+\w+s\b"
    r"|\ball\s+(?:the\s+)?(?:kinds?|types?|sorts?|options)\b"
    r"|\bexamples?\s+of\b"
    r"|\boverview\s+of\b|\bsurvey\s+of\b"
    r"|\bwhat\s+\w+s\s+(?:are|do)\s+(?:there|teams|people|most)\b"
)

# Written out because the intent of a list question is often carried entirely by
# a plural noun the regex above would need context to see: "common FRC
# mechanisms" has no verb at all.
_ENUM_NOUNS = (
    r"\b(?:mechanisms|subsystems|materials|alloys|fasteners|gearboxes|"
    r"drivetrains|bearings|motors|sensors|tools|methods|approaches|techniques|"
    r"strategies|options|alternatives|types|kinds|categories|examples)\b"
)


def wants_enumeration(question, analysis=None):
    """Is the user asking for a list of things, rather than about one thing?

    This changes two things downstream: the local corpus is searched wider and
    flatter (more documents, fewer chunks each), because a category the corpus
    only mentions on one page is exactly the category that goes missing; and the
    system prompt's instruction not to wander into adjacent topics is replaced,
    because for this question the adjacent topics *are* the question.
    """
    import re
    q = (question or "").strip()
    if not q:
        return False
    an = analysis or {}
    # A question naming one document or one season is not a survey, whatever
    # its grammar looks like -- "what are the rules for the 2024 game" is a
    # lookup in one manual.
    if an.get("scope") == "specific":
        return False
    if re.search(_ENUM_RE, q, re.I):
        return True
    # Plural subject with no interrogative narrowing it: "common FRC
    # mechanisms", "FRC drivetrain types".
    return bool(re.search(_ENUM_NOUNS, q, re.I)
                and not re.search(r"\bhow\s+(?:do|does|can|should|would)\b|"
                                  r"\bwhy\b|\bshould\s+i\b", q, re.I))


def _key_terms(analysis):
    """Distinctive words the answer must be about.

    Two tiers. `strong` is what the user actually named - the entities and
    constraints the planner pulled out - and it decides whether a result is
    on-topic at all. `weak` is the rest of the restated question, which is too
    generic to filter on but is exactly what tells one paragraph of a long page
    apart from another.
    """
    import re
    an = analysis or {}

    def words(phrases):
        out = set()
        for phrase in phrases:
            for w in re.findall(r"[a-z0-9#/\-\.]{3,}", str(phrase).lower()):
                if w not in _STOP:
                    out.add(w)
        return out

    strong = words(an.get("entities", []) + an.get("constraints", []))
    weak = words([an.get("restated", "")]) - strong
    return {"strong": strong, "weak": weak}


def _relevance(res, terms):
    """Fraction of the question's key terms this result mentions."""
    strong = (terms or {}).get("strong") or set()
    if not strong:
        return 1.0
    blob = ((res.get("title") or "") + " " + (res.get("text") or "")).lower()
    return sum(1 for t in strong if t in blob) / float(len(strong))


def _chunks(text, win):
    """Split a page into readable windows, preferring paragraph boundaries."""
    import re
    parts = [p.strip() for p in re.split(r"\n\s*\n|(?<=[.!?])\n", text or "")
             if p.strip()]
    out, buf = [], ""
    for p in parts:
        while len(p) > win:
            if buf:
                out.append(buf); buf = ""
            out.append(p[:win]); p = p[win:].lstrip()
        if len(buf) + len(p) + 1 <= win:
            buf = (buf + "\n" + p).strip() if buf else p
        else:
            out.append(buf); buf = p
    if buf:
        out.append(buf)
    return out


def _passages(text, terms, budget, win=460):
    """The parts of a page that are about the question, not just its opening.

    Truncating at the top is what put a challenge page's acceptance-criteria
    list into the context in the first place: on that page the criteria ARE the
    opening. Scoring windows against the question instead means a long page
    contributes the paragraphs that answer it, and a page that only mentions the
    subject in passing contributes little enough to be visibly thin.
    """
    text = (text or "").strip()
    if not text:
        return ""
    strong = (terms or {}).get("strong") or set()
    weak = (terms or {}).get("weak") or set()
    if not strong and not weak:
        return text[:budget]
    ch = _chunks(text, win)
    if len(ch) <= 1:
        return text[:budget]
    scored = []
    for i, c in enumerate(ch):
        low = c.lower()
        s = 2.0 * sum(1 for t in strong if t in low) + sum(1 for t in weak if t in low)
        # Mild preference for earlier text when scores tie: a page's own summary
        # is usually near the front, and ordering the picks by position keeps
        # the excerpt readable rather than shuffled.
        scored.append((s - i * 1e-3, i, c))
    scored.sort(key=lambda x: -x[0])
    picked, used = [], 0
    for s, i, c in scored:
        if s <= 0 and picked:
            break
        if used + len(c) > budget:
            continue
        picked.append((i, c))
        used += len(c) + 2
        if used >= budget:
            break
    if not picked:
        return text[:budget]
    picked.sort()
    joined = "\n...\n".join(c for _, c in picked)
    return joined[:budget]


# How many knowledge-base chunks may enter the pool, and how hard a chunk has
# to match before it is allowed in at all.
#
# Both floors exist for the same reason. BM25 always returns its best guess,
# and the best guess against a corpus that simply does not cover the question is
# still some page about something else. Injecting it costs more than leaving it
# out: it occupies a ranking slot, it arrives wearing the authority of "the
# team's own handbook", and the model will dutifully cite it. An empty local
# result is a correct answer to "does our corpus know about this".
#
# The relative floor catches the case the absolute one cannot -- a question
# whose vocabulary happens to be common in the corpus scores everything highly,
# and only the ratio between the best hit and the rest says whether anything
# actually stood out.
KB_MAX_HITS = 4
KB_MIN_SCORE = 1.0
KB_MIN_RATIO = 0.25

# A list question needs the opposite shape of retrieval from a depth question.
# "How do I design a shooter" is best answered by four chunks of the best
# shooter page; "what mechanisms are common in FRC" is best answered by one
# chunk each from six different pages, because a category the corpus covers on
# exactly one page is the category that goes missing -- which is how an answer
# came back naming intakes, drivebases, elevators, pivots and extenders from a
# corpus whose own mechanism index lists shooters third.
#
# So the local budget widens and flattens: more slots, and at most one chunk per
# document rather than two. The relative floor also has to relax, because the
# fifth category's page will legitimately score well below the first's -- the
# question names none of them, so nothing in the query favours any one.
KB_MAX_HITS_LIST = 7
KB_PER_DOC_LIST = 1
KB_MIN_RATIO_LIST = 0.12

# A curated corpus earns a nudge over an arbitrary search result, because
# somebody chose it on purpose -- but only a nudge. It is smaller than the
# preferred-domain bonus (1.0) on purpose: the team handbook should beat a
# random blog on a tie, and should not beat a manufacturer's own spec sheet
# that matches the question far better.
KB_BONUS = 0.6


def kb_query(question, analysis=None):
    """What to actually search the corpus for.

    Not the raw question. `analyze()` has already rewritten it into the
    vocabulary the documents use -- "how do I make my plate lighter" becomes
    pocketing, rib, 6061-T6 -- and that rewrite is worth more to a keyword
    index than to a web search engine, because BM25 cannot bridge a synonym on
    its own the way a search engine can.
    """
    an = analysis or {}
    parts = [an.get("restated") or question]
    parts += [str(x) for x in (an.get("entities") or [])]
    parts += [str(x) for x in (an.get("constraints") or [])]
    for q in (an.get("queries") or [])[:2]:
        parts.append(str(q))
    return " ".join(p for p in parts if p)[:600]


def kb_hits(question, analysis=None, terms=None, specific=False,
            listing=False):
    """Knowledge-base chunks, shaped exactly like web results.

    Shaped identically on purpose: the ranking, the entity filter, the
    broad-source rescue and the citation numbering downstream all then treat a
    local chunk and a web page the same way, and there is only one ranking
    policy in this file to keep correct instead of two that drift apart.

    `listing` widens and flattens the search for a question that wants a set of
    things rather than one thing -- see KB_MAX_HITS_LIST.
    """
    if _kb is None:
        return []
    an = analysis or {}
    boost = [str(x) for x in (an.get("entities") or [])
             + (an.get("constraints") or [])]
    cap = KB_MAX_HITS_LIST if listing else KB_MAX_HITS
    floor_ratio = KB_MIN_RATIO_LIST if listing else KB_MIN_RATIO
    try:
        kw = {"k": cap * 2, "boost_terms": boost or None}
        if listing:
            kw["per_doc"] = KB_PER_DOC_LIST
        raw = _kb.search(kb_query(question, an), **kw)
    except TypeError:
        # An older kb.search without per_doc. Better a slightly narrower list
        # than no local sources at all.
        try:
            raw = _kb.search(kb_query(question, an), k=cap * 2,
                             boost_terms=boost or None)
        except Exception:
            return []
    except Exception:
        return []
    if not raw:
        return []

    best = max((h.get("score") or 0.0) for h in raw) or 1.0
    out = []
    for h in raw:
        s = float(h.get("score") or 0.0)
        if s < KB_MIN_SCORE or s < best * floor_ratio:
            continue
        kind = h.get("kind") or "reference"
        if kind not in KIND_INFO:
            kind = "reference"
        title = h.get("title") or "knowledge base"
        head = h.get("head") or ""
        url = h.get("url") or ""
        page = int(h.get("page") or 0)
        # A PDF chunk carries the page it came from. It goes in the title
        # rather than being kept for the UI, because the title is what the
        # model is shown and what gets copied out of the answer -- and a
        # citation to a 400-page manual with no page number is a citation
        # nobody is ever going to check.
        shown = (f"{title} - {head}" if head and head not in title else title)
        if page:
            # "p." unless the document said otherwise: a deck says "slide", a
            # workbook says "tab". Anything ingested before page_word existed
            # has none, so every PDF citation already in the corpus is
            # unchanged.
            shown = f"{shown} ({h.get('page_word') or 'p.'} {page})"
        rec = {
            "title": shown,
            "page": page,
            "url": url,
            "host": _host(url) or (h.get("source") or "knowledge base"),
            "text": (h.get("text") or "").strip(),
            "preferred": True,
            "kind": kind,
            "kind_label": KIND_INFO[kind][0],
            "local": True,
            "kb_source": h.get("source") or "knowledge base",
            # Normalised against the best local hit so this is comparable with
            # Tavily's 0..1 relevance rather than with raw BM25, which has no
            # ceiling and would otherwise swamp every web result.
            "match": round(s / best, 3),
            "bm25": round(s, 3),
        }
        rel = _relevance(rec, terms)
        rec["relevance"] = round(rel, 3)
        bias = KIND_INFO[kind][3] if specific else KIND_INFO[kind][2]
        rec["kind_bias"] = bias
        rec["score"] = rec["match"] + KB_BONUS + 1.5 * rel + bias
        out.append(rec)
        if len(out) >= cap:
            break

    if listing and out:
        out += _kb_siblings(out, terms, specific, cap)
    return out


# How many neighbours a list question may pull in. Six because the categories a
# site documents side by side are usually five or six -- enough to complete the
# set, few enough that they cannot crowd out the pages that actually matched.
KB_SIBLINGS = 6

# What a page reached by structure rather than by matching is worth. Below every
# real hit (which start at KB_BONUS = 0.6 plus their match) and above nothing,
# because a neighbour has earned a place in the context window and has not
# earned the top of it.
KB_SIBLING_SCORE = 0.25


def _kb_siblings(hits, terms, specific, cap):
    """Complete a list answer with the neighbours of what matched.

    The gap this closes is not a ranking gap, so widening the budget above does
    not close it. Asked "what mechanisms are common in FRC", a keyword index
    scores the shooter page at zero -- the question never says "shooter" -- and
    a page scoring zero is not near the bottom of the ranking, it is not in the
    ranking. Slots do not help; the page was never a candidate.

    Its neighbours are, though. Once the intake page has matched, the shooter
    page is the directory next door, and that is a fact about the corpus rather
    than about the wording of the question. So for a list question only, the
    documents sharing a section with what matched are pulled in at a low score:
    present in the context, clearly not the answer, and exactly the thing whose
    absence made a five-category answer name four.
    """
    try:
        sibs = _kb.siblings([h["url"] for h in hits[:3]],
                            exclude=[h["url"] for h in hits],
                            limit=KB_SIBLINGS)
    except Exception:
        return []
    extra = []
    for h in sibs:
        kind = h.get("kind") or "reference"
        if kind not in KIND_INFO:
            kind = "reference"
        title = h.get("title") or "knowledge base"
        head = h.get("head") or ""
        url = h.get("url") or ""
        shown = (f"{title} - {head}" if head and head not in title else title)
        rec = {
            "title": shown,
            "page": int(h.get("page") or 0),
            "url": url,
            "host": _host(url) or (h.get("source") or "knowledge base"),
            "text": (h.get("text") or "").strip(),
            "preferred": True,
            "kind": kind,
            "kind_label": KIND_INFO[kind][0],
            "local": True,
            "sibling": True,
            "kb_source": h.get("source") or "knowledge base",
            "match": 0.0,
            "bm25": 0.0,
        }
        rel = _relevance(rec, terms)
        rec["relevance"] = round(rel, 3)
        bias = KIND_INFO[kind][3] if specific else KIND_INFO[kind][2]
        rec["kind_bias"] = bias
        # No KB_BONUS and no relevance multiplier: this page did not match, and
        # scoring it as though it had would let a neighbour outrank a hit.
        rec["score"] = KB_SIBLING_SCORE + bias
        extra.append(rec)
    return extra


# How many of the best web results get fetched in full, and how long that is
# allowed to take.
#
# Two, not eight. This runs inside the request the user is waiting on, and the
# value drops off a cliff after the first couple: the top results are the ones
# the answer will actually rest on, and fetching the eighth costs the same
# second as fetching the first while changing nothing. Both pages are fetched
# at once, so the cost is one page's latency rather than two.
DEEPEN_N = 2
DEEPEN_TIMEOUT = 8

# A full page has to be substantially bigger than the snippet to be worth
# swapping in. Some pages really are 700 characters long, and some "fetches"
# come back as a cookie wall of exactly the same size -- neither is an
# improvement over what the search engine already gave us.
DEEPEN_MIN_GAIN = 1.6


def _deepen(pool, terms, specific):
    """Replace the top results' snippets with the real page, in place.

    This is the fix for a specific and very common failure: the user asks for a
    number, the page has the number in a table two thirds of the way down, and
    the assistant answers from the introduction because the introduction is all
    it was given. Re-running the passage picker over the full text puts the
    paragraph that actually answers the question in front of the model.

    The page is re-typed as well as re-excerpted. A design challenge often
    reads like a general article until its requirements list, which lives well
    past the snippet -- so deepening is also the point at which a page can be
    correctly recognised as narrow and demoted for it.
    """
    if _webtext is None:
        return
    targets = [r for r in pool[:4]
               if r.get("snippet_only") and not r.get("local")
               and (r.get("url") or "").startswith("http")][:DEEPEN_N]
    if not targets:
        return
    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(targets)) as ex:
            fetched = list(ex.map(
                lambda r: _webtext.fetch_text(r["url"], timeout=DEEPEN_TIMEOUT),
                targets))
    except Exception:
        return

    for rec, (title, text) in zip(targets, fetched):
        text = (text or "").strip()
        if len(text) < len(rec.get("text") or "") * DEEPEN_MIN_GAIN:
            continue
        kind = classify_source(title or rec["title"], rec["url"], text)
        if kind not in KIND_INFO:
            kind = "reference"
        rec["text"] = _passages(text, terms, 1800)
        rec["kind"] = kind
        rec["kind_label"] = KIND_INFO[kind][0]
        rec["snippet_only"] = False
        rec["full_page"] = True
        rec["page_chars"] = len(text)
        rel = _relevance(rec, terms)
        rec["relevance"] = round(rel, 3)
        bias = KIND_INFO[kind][3] if specific else KIND_INFO[kind][2]
        rec["kind_bias"] = bias
        rec["score"] = (rec.get("match", 0.0)
                        + (1.0 if rec.get("preferred") else 0.0)
                        + 1.5 * rel + bias)


def gather(question, topics, domains, analysis=None, timing=None):
    """Run the plan, merge, de-duplicate, type and rank.

    Four things decide a result's place: how well Tavily matched it, whether it
    sits on a domain this topic actually lives on, how much of what the user
    named it mentions, and -- new -- what KIND of document it is relative to
    what was asked. A practice-problem page is the right answer to "what does
    challenge 3 require" and the wrong answer to "how do I design a shooter",
    and until that distinction existed the two were ranked identically.

    `timing`, if given a dict, is filled in with how long each stage took in
    milliseconds. It is an out-parameter rather than part of the return value
    so that nothing which already calls this has to change.
    """
    pool, seen = [], set()
    used_queries = []
    terms = _key_terms(analysis)
    specific = wants_specific(question, analysis)
    listing = (not specific) and wants_enumeration(question, analysis)
    # A list answer is only as complete as the number of distinct sources that
    # reach the model. Eight is the right ceiling for a question with one
    # answer; for "name the kinds of X" it is the ceiling that drops the last
    # kind on the floor after everything upstream did its job.
    keep = 11 if listing else 8

    # The corpus is consulted first and unconditionally. First because it is
    # local and free, and unconditionally because the one machine that most
    # needs a working assistant -- a school laptop with no Tavily key, on
    # competition-venue wifi, the night before a match -- is exactly the machine
    # where every web step below is going to fail.
    _t0 = time.time()
    local = kb_hits(question, analysis, terms, specific, listing=listing)
    if timing is not None:
        timing["kb"] = int((time.time() - _t0) * 1000)
    for rec in local:
        # Two chunks of one document are two different excerpts, so the URL
        # alone cannot be the identity here the way it can for a web page.
        seen.add((rec["url"], rec["text"][:60]))
        # Also block the web loop from returning the same page: if it is in the
        # corpus we already hold the paragraph that matches, rather than
        # whatever Tavily's snippet happened to start with.
        if rec["url"]:
            seen.add(rec["url"])
        pool.append(rec)

    if not os.environ.get("TAVILY_API_KEY"):
        pool.sort(key=lambda r: -r["score"])
        return pool[:keep], used_queries

    plan = _plan_queries(question, topics, domains, analysis)

    # All of the plan's searches at once, not one after another.
    #
    # This is the single largest thing the user waits on. Every query in the
    # plan runs at Tavily's "advanced" depth, which is a few seconds each, and
    # run serially four of them stacked up into most of the response time --
    # although no query depends on any other one's result. They are independent
    # by construction: _plan_queries builds the whole list up front from the
    # question and the analysis, and never looks at a result.
    #
    # The MERGE below still walks the plan IN ORDER. That matters and is not
    # incidental: `seen` gives a shared URL to whichever query comes first in
    # the plan, and the plan is ordered most-authoritative-first, so merging in
    # completion order instead would quietly change which query's snippet and
    # raw-content budget a page arrives with. Fetch concurrently, merge in
    # order, and the output is byte-for-byte what it was before.
    results = [None] * len(plan)

    def _run(step):
        return search(step["q"], domains=step["domains"],
                      max_results=step["n"], raw=step["raw"])

    _t0 = time.time()
    if plan:
        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(plan)) as ex:
                futures = [ex.submit(_run, s) for s in plan]
                for i, fut in enumerate(futures):
                    try:
                        results[i] = fut.result()
                    except Exception:
                        results[i] = None
        except Exception:
            # A machine that cannot start threads at all still gets an answer,
            # just at the old speed. Better slow than sourceless.
            for i, step in enumerate(plan):
                try:
                    results[i] = _run(step)
                except Exception:
                    results[i] = None
    if timing is not None:
        timing["search"] = int((time.time() - _t0) * 1000)
        timing["searches"] = sum(1 for r in results if r is not None)

    for step, sr in zip(plan, results):
        if sr is None:
            continue
        used_queries.append({"query": step["q"], "scope": step["tag"],
                             "domains": step["domains"] or "open web"})
        for res in (sr.get("results") or []):
            url = res.get("url") or ""
            if not url or url in seen:
                continue
            seen.add(url)
            host = _host(url)
            preferred = any(host == d or host.endswith("." + d) for d in domains)
            body = (res.get("raw_content") or res.get("content") or "")
            title = res.get("title") or host
            kind = classify_source(title, url, body)
            budget = 1800 if step["raw"] else 700
            rec = {
                "title": title,
                "url": url,
                "host": host,
                # Typed off the FULL body, excerpted down to the part that
                # answers the question -- in that order, so a page is never
                # mistyped because the passage picker cropped away the
                # sentence that gave it away.
                "text": _passages(body, terms, budget),
                "preferred": preferred,
                "kind": kind,
                "kind_label": KIND_INFO[kind][0],
                # Tavily's `content` is roughly the first 700 characters of the
                # page -- the introduction, not the answer. Flagged here so the
                # deepening pass below knows which results are worth fetching
                # in full and which already arrived complete.
                "snippet_only": not step["raw"],
                "match": float(res.get("score") or 0.0),
            }
            rel = _relevance(rec, terms)
            rec["relevance"] = round(rel, 3)
            # A narrow-scope page is penalised on a general question and
            # promoted on a specific one -- same page, opposite usefulness.
            bias = KIND_INFO[kind][3] if specific else KIND_INFO[kind][2]
            rec["kind_bias"] = bias
            rec["score"] = (rec["match"]
                            + (1.0 if preferred else 0.0)
                            + 1.5 * rel + bias)
            pool.append(rec)

    pool.sort(key=lambda r: -r["score"])
    _t0 = time.time()
    _deepen(pool, terms, specific)
    if timing is not None:
        timing["deepen"] = int((time.time() - _t0) * 1000)
    pool.sort(key=lambda r: -r["score"])

    # Drop results that mention none of the named entities/constraints, but
    # only while enough on-topic material survives - a thin pool is better
    # kept whole than filtered down to nothing.
    if (terms or {}).get("strong"):
        strong = [r for r in pool if r["relevance"] > 0.0]
        if len(strong) >= 3:
            pool = strong
    # Do not let one narrow-scope page dominate a general answer. If the top of
    # a general-question pool is all exercise briefs and forum posts, the model
    # has nothing to state flatly and will either hedge everything or -- worse --
    # promote an assignment's numbers. Pull the best broad source up to the top
    # three whenever one exists further down.
    top = pool[:keep]
    if not specific and top:
        broad = {"docs", "data", "reference", "vendor"}
        if not any(r["kind"] in broad for r in top[:3]):
            for i, r in enumerate(pool):
                if r["kind"] in broad:
                    pool = [r] + [x for j, x in enumerate(pool) if j != i]
                    break
    return pool[:keep], used_queries


def ask(question, model=None):
    """Answer the question and return the whole thing at once.

    A thin drain of ask_stream, deliberately: there is one copy of the pipeline
    and the streaming and non-streaming answers are the same answer by
    construction rather than by two implementations agreeing. Kept because
    /api/chat is still served for anything that calls it directly.
    """
    out = None
    for ev in ask_stream(question, model=model):
        if ev.get("type") == "done":
            out = ev["result"]
    return out


def ask_stream(question, model=None):
    """Route the question to a topic, search the domains that topic actually
    lives on, then answer strictly from what came back -- yielding as it goes.

    Events, in order:
      {"type": "stage",  "stage": "analyse"|"search"|"answer"}
      {"type": "plan",   "plan": {...}, "sources": [...]}   once, before a word
                         of the answer exists, so the reader can see WHAT was
                         found while the model is still deciding what to say
                         about it
      {"type": "delta",  "text": "..."}                     many
      {"type": "done",   "result": {answer, sources, plan, grounding}}

    The `done` result is the complete payload the non-streaming endpoint has
    always returned. A client can therefore ignore every event but the last and
    be exactly where it was before streaming existed.
    """
    import re
    sources, context_blocks, excerpts = [], [], []
    # Where the time actually goes, measured rather than guessed, and reported
    # back with the answer so a slow question can be diagnosed on the machine
    # it was slow on. Every number is milliseconds.
    timing = {}
    _t_start = time.time()
    chitchat = bool(re.match(_CHITCHAT, question, re.I)) or len(question.strip()) <= 3
    if chitchat:
        an = {"restated": question.strip(), "intent": "", "entities": [],
              "constraints": [], "topics": [], "domains": [], "queries": [],
              "scope": "", "source": "skipped"}
    else:
        yield {"type": "stage", "stage": "analyse"}
        _t0 = time.time()
        an = analyze(question)
        timing["analyse"] = int((time.time() - _t0) * 1000)
    topics, domains = an["topics"], an["domains"]
    specific = (not chitchat) and wants_specific(question, an)
    listing = (not chitchat) and (not specific) and wants_enumeration(question, an)
    plan_meta = {"topics": topics, "domains": domains, "queries": [],
                 "search": "off", "analysis": {
                     "restated": an["restated"], "intent": an["intent"],
                     "entities": an["entities"], "constraints": an["constraints"],
                     "scope": "specific" if specific else "general",
                     "shape": "list" if listing else "",
                     "by": an["source"]}}

    # FRC team lookups still go straight to The Blue Alliance
    tm = re.search(r"\b(?:team|frc)\s*#?(\d{1,5})\b", question, re.I)
    if tm:
        try:
            t = tba_team(tm.group(1))
            info = str(t["info"])[:800]
            for srec in t.get("sources", []):
                sources.append(srec)
                # `excerpts` shadows `sources` one-for-one so the grounding
                # check can ask "what did source 3 actually say". It cannot use
                # context_blocks for that: those carry a title, a URL and a KIND
                # line as well, and a URL full of digits would let a made-up
                # measurement match a thread ID.
                excerpts.append(info)
            context_blocks.append("[TBA] " + info)
        except Exception:
            pass

    # Search every real question. The old keyword gate silently skipped most of
    # them, which is the direct cause of un-sourced answers.
    if not chitchat:
        yield {"type": "stage", "stage": "search"}
        have_web = bool(os.environ.get("TAVILY_API_KEY"))
        hits, used = gather(question, topics, domains, an, timing=timing)
        n_local = sum(1 for h in hits if h.get("local"))
        n_full = sum(1 for h in hits if h.get("full_page"))
        plan_meta["queries"] = used
        plan_meta["kb"] = {"hits": n_local}
        plan_meta["full_pages"] = n_full
        if hits:
            plan_meta["search"] = "on" if have_web else "knowledge base only"
        else:
            plan_meta["search"] = "no results" if have_web else "no TAVILY_API_KEY"
        # De-duplicate against anything already listed (the TBA block) BEFORE
        # numbering, never after. Dropping a source once the citation markers
        # have been assigned does not renumber them -- it just makes every [n]
        # above the gap point at the wrong document, or get stripped as
        # out-of-range further down. Cheaper to not create the collision.
        known = {s.get("url") for s in sources if s.get("url")}
        hits = [h for h in hits if not h.get("url") or h["url"] not in known]

        base = len(sources)
        for i, h in enumerate(hits):
            # The KIND line travels with the excerpt, not in a separate
            # preamble. Stated once at the top of the block the model
            # forgets it by source 6; stated on the source itself it is
            # right next to the sentence it governs.
            label, caveat = KIND_INFO[h["kind"]][:2]
            if h.get("local"):
                # Named explicitly so the model does not describe a document
                # somebody deliberately put in the corpus as something it
                # "found online", and so a reader can tell which claims came
                # from their own handbook and are therefore theirs to fix.
                label = f"{label} (knowledge base: {h['kb_source']})"
            context_blocks.append(
                f"[{base + i + 1}] {h['title']} - {h['host']}\n"
                f"KIND: {label} - {caveat}\n"
                f"{h['url']}\n{h['text']}")
            sources.append({"title": h["title"], "url": h["url"],
                            "host": h["host"], "preferred": h["preferred"],
                            "kind": h["kind"], "kind_label": label,
                            "local": bool(h.get("local")),
                            "page": int(h.get("page") or 0),
                            "full_page": bool(h.get("full_page"))})
            excerpts.append(h["text"])

    msgs = []
    # The analysis goes in ahead of the sources so the model reads what was
    # asked before it reads what came back, not the other way round.
    if not chitchat:
        lines = ["QUESTION ANALYSIS:", f"The user is asking: {an['restated']}"]
        if an["intent"]:
            lines.append(f"Intent: {an['intent'].replace('_', ' ')}")
        if an["entities"]:
            lines.append("Specifically about: " + "; ".join(an["entities"]))
        if an["constraints"]:
            lines.append("Constraints that must be respected: "
                         + "; ".join(an["constraints"]))
        # Spelled out rather than implied. Told only "answer the question", the
        # model happily answers a general one with a specific document's
        # acceptance criteria, because those criteria look like exactly the
        # concrete numbers it was asked to lead with.
        lines.append(
            "Scope: SPECIFIC - the user wants what one named document, "
            "challenge, season or product actually says, so quote it directly."
            if specific else
            "Scope: GENERAL - the user wants engineering practice that holds "
            "regardless of which exercise, season or team a page came from. Do "
            "NOT present a limit, dimension or range taken from a design "
            "challenge, assignment or one season's rules as a requirement of "
            "the mechanism itself. Give the governing principle and the "
            "quantities that always matter; attribute anything narrower.")
        if listing:
            # This overrides the standing "do not pad the answer with adjacent
            # topics" rule in SYSTEM_PROMPT, and says so, because a model given
            # two instructions that contradict each other follows whichever it
            # read last and neither reliably. The rule is right for "how do I
            # design a shooter" and exactly wrong here: the adjacent topics ARE
            # the question.
            lines.append(
                "Shape: LIST - the user asked what things exist in a category, "
                "not how one of them works. Completeness is the answer here, so "
                "the standing instruction not to cover adjacent topics does NOT "
                "apply to this question. Before writing, read every source "
                "below and collect every member of the category any of them "
                "names - including one named only in a heading, a nav list or a "
                "single passing line. Name them all, each with a one-line note "
                "on what it is for, and only then expand on any. If a source "
                "lists a member you have not mentioned, you have not finished. "
                "If your engineering knowledge covers a member none of the "
                "sources name, include it too and mark it as not from the "
                "sources rather than leaving it out.")
        lines.append("Answer this exact question first, then support it.")
        msgs.append({"role": "system", "content": "\n".join(lines)})
    if context_blocks:
        hint = ""
        if topics:
            hint = (f"\n\nThe question was routed to: {', '.join(topics)}. "
                    f"Sources were drawn from: {', '.join(domains[:8])}.")
        msgs.append({"role": "system",
                     "content": "SOURCES:\n" + "\n\n".join(context_blocks) + hint})
    msgs.append({"role": "user", "content": question})

    # Everything the reader needs to judge the answer BEFORE the answer exists.
    # The sources are already decided at this point -- the model is about to
    # write from them, not to change them -- so there is no reason to sit on
    # them for the several seconds it takes to write. This is the event that
    # turns a blank wait into a visible one.
    n_src = len(sources)
    yield {"type": "plan", "plan": plan_meta, "sources": sources}

    yield {"type": "stage", "stage": "answer"}
    _t0 = time.time()
    # Strip citation markers that point at nothing, as they arrive. Without this
    # the model can still emit [1] on an un-sourced answer and the UI renders a
    # bare stub -- and on the streaming path it would render it, briefly, even
    # if it were cleaned up afterwards.
    strip = _MarkerStrip(n_src)
    parts = []
    try:
        for piece in chat_stream(msgs, model=model):
            clean = strip.feed(piece)
            if clean:
                parts.append(clean)
                yield {"type": "delta", "text": clean}
        tail = strip.flush()
        if tail:
            parts.append(tail)
            yield {"type": "delta", "text": tail}
    except Exception as e:
        # Half an answer plus an explanation beats a bare error page. If
        # nothing arrived at all, say what went wrong in the answer itself --
        # that is where the reader is looking, and it is what the
        # non-streaming path has always done with a Groq error.
        if parts:
            note = "\n\n_The answer was cut short: %s_" % e
            parts.append(note)
            yield {"type": "delta", "text": note}
        else:
            raise
    timing["answer"] = int((time.time() - _t0) * 1000)

    answer = "".join(parts)
    if not answer.strip():
        answer = "The model returned an empty response - try rephrasing the question."
        yield {"type": "delta", "text": answer}

    # What the model actually claimed, before any of the notes below are stuck
    # on the end. Those notes are interface furniture, not assertions, and
    # checking them would let a filename or a version number in a hint line be
    # reported to the reader as an unsupported measurement.
    claimed = answer

    if n_src == 0 and plan_meta.get("search") == "no TAVILY_API_KEY":
        answer += ("\n\n_No sources: add a free Tavily key with SET_API_KEY.bat "
                   "for web search, or build a local knowledge base with "
                   "`python tools/kb_ingest.py seed` - the knowledge base works "
                   "with no key and no internet._")
    elif plan_meta.get("search") == "knowledge base only" and n_src:
        answer += ("\n\n_Answered from the local knowledge base only (no Tavily "
                   "key). Sources are the documents you ingested._")

    # Check the numbers against the excerpts they cite. This runs last, on the
    # final text, because everything above can still change it: a stripped [n]
    # marker turns a cited claim into an uncited one, and checking before the
    # strip would report a citation the reader never sees.
    #
    # It runs after the model, not instead of it, and it never edits the
    # answer. A verifier that silently deleted a flagged sentence would leave
    # a gap in the reasoning with nothing to say why -- worse than a visible
    # number with a visible warning next to it. The reader is told and decides.
    #
    # Chitchat is exempt. It never searched, so "no sources to check against"
    # is a description of the greeting rather than a finding about it, and a
    # verification badge on "hello" teaches the reader that the badge means
    # nothing.
    ground = None
    if _grounding is not None and not chitchat:
        try:
            _t0 = time.time()
            ground = _grounding.check(claimed, sources, excerpts)
            timing["check"] = int((time.time() - _t0) * 1000)
        except Exception:
            # A failure here must cost the check and nothing else. The answer
            # is already written and is still worth returning unverified.
            ground = None

    if ground and ground["verdict"] == "partial":
        # Named individually, because "some numbers are unsupported" sends the
        # reader back through the whole answer, and the point of the check is
        # to tell them exactly where to look.
        WORDS = {
            "not-in-sources": "not found in any source",
            "wrong-source": "appears in a different source than the one cited",
            "uncited": "not cited, and not found in the sources",
        }
        lines = []
        for f in ground["flags"][:4]:
            lines.append("- %s - %s" % (f["claim"], WORDS.get(
                f["reason"], f["reason"])))
        extra = len(ground["flags"]) - len(lines)
        if extra > 0:
            lines.append("- ...and %d more" % extra)
        # Written into the answer rather than left in the JSON for the UI to
        # draw, because the answer is the part that gets copied into a Slack
        # message or a build-log entry, and a warning that does not survive
        # being copied is a warning that stops existing the moment it matters.
        answer += ("\n\n**Check these numbers** - they could not be matched to "
                   "the sources cited:\n" + "\n".join(lines)
                   + "\n\n_A text match, not a judgement: a number can be "
                   "flagged because the source gives it in different units, or "
                   "because the answer worked it out rather than read it. It "
                   "means look, not wrong._")

    timing["total"] = int((time.time() - _t_start) * 1000)
    plan_meta["timing_ms"] = timing
    # The complete payload, identical to what this function returned before it
    # streamed. The trailing notes above -- the no-key hint, the grounding
    # warning -- are deliberately NOT sent as deltas: they are decided after the
    # last word arrives, and a client that redraws from this final answer gets
    # them in one paint instead of watching them appear.
    yield {"type": "done",
           "result": {"answer": answer, "sources": sources, "plan": plan_meta,
                      "grounding": ground}}
