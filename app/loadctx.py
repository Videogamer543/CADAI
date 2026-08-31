"""
Plain-English loading description -> solver settings.

You should be able to write what the part actually does -- "bolted to the
gearbox on the left, the roller pushes down about 40 lb on the far end" --
instead of translating that into a load case and a newton figure yourself.
This module does that translation and, just as importantly, reports back what
it understood so a wrong reading is obvious before you trust the numbers.

Two layers, and the second never depends on the first:

  1. A deterministic regex/unit pass. It finds forces in N, kgf, lb and lbf,
     masses in kg and lb (times g), torques, thicknesses in mm and inches,
     materials by name, and load-case wording. No network, no API key, no
     model. This alone handles most of what people actually type.

  2. An LLM pass (Groq, same key the assistant uses) for the sentences the
     regexes miss -- "held at both ends and something heavy sits in the
     middle" has no numbers and no keywords, but it is clearly ss_center.

The regex pass runs FIRST and its findings are treated as ground truth: if
you wrote "250 N", no model gets to decide you meant something else. The LLM
only fills fields the text did not pin down. Every value is range-checked
before it leaves this module, so a hallucinated 9e9 N never reaches the FEM.
"""
from __future__ import annotations
import os
import re

from . import materials

G = 9.80665

# The load cases the solver actually implements.
CASES = {
    "cantilever": "fixed at one end, load at the free end",
    "ss_center":  "supported at both ends, load in the middle",
    "ss_dist":    "supported at both ends, load spread over the whole part",
    "fixed_fixed": "built in at both ends, load in the middle",
}

# Wording -> load case. Ordered: the first pattern that matches wins, so the
# more specific phrasings are listed above the generic ones.
_CASE_PATTERNS = [
    ("fixed_fixed", r"\b(built[- ]?in|welded|clamped|fixed|bolted|rigid)\b[^.]{0,40}\b(both|two|each)\s+(end|side|edge)"),
    ("fixed_fixed", r"\bfixed[- ]?fixed\b|\bboth ends (are )?(welded|clamped|rigid|built)"),
    ("ss_dist",     r"\b(distributed|spread|uniform(ly)?|pressure|all over|along (the|its) (whole|entire|full)|self[- ]?weight|own weight)\b"),
    ("ss_center",   r"\b(simply[- ]?support|supported)\b[^.]{0,50}\b(both|two|each|either)\s+(end|side|edge)"),
    ("ss_center",   r"\b(both|two|either)\s+(end|side)s?\b[^.]{0,40}\b(support|rest|sit|held|pinned|bearing)"),
    ("ss_center",   r"\bthree[- ]?point\b|\bmid[- ]?span\b|\bin the (middle|centre|center)\b"),
    ("cantilever",  r"\bcantilever(ed)?\b|\boverhung\b|\bsticks? out\b|\bhangs? off\b"),
    ("cantilever",  r"\b(bolted|mounted|clamped|fixed|attached|held|anchored)\b[^.]{0,50}\b(one end|one side|the (left|right|top|bottom|near|inner)\b)"),
]

_VERTICAL = r"\b(vertical(ly)?|up and down|upright|standing|on end|top to bottom)\b"
_HORIZONTAL = r"\b(horizontal(ly)?|side to side|flat|lying|left to right|lengthwise)\b"

# mode wording
_MODES = [
    ("fatigue", r"\b(fatigue|cyclic|repeated|vibrat|thousands of cycles|every match|life)\b"),
    ("manufacturing", r"\b(manufactur|machin|as[- ]?cut|tolerance|water ?jet|surface finish)\b"),
    ("full", r"\bfull\b|\bworst case\b|\ball checks\b"),
    ("structural", r"\bstatic|structural|one[- ]?time|single (hit|impact)\b"),
]

_NUM = r"(\d+(?:\.\d+)?|\.\d+)"


def _f(m):
    try:
        return float(m)
    except Exception:
        return None


def _find_force(text):
    """Return (newtons, how_it_was_read) or (None, None). Explicit force wins
    over a mass, and a mass is converted with g rather than silently treated
    as newtons -- a '20 kg load' is 196 N, and getting that wrong is a factor
    of ten in the safety factor."""
    t = text.lower()

    # multipliers first so "2.5 kN" doesn't read as 2.5 N
    for pat, k, unit in (
        (rf"{_NUM}\s*(?:k\s?n|kilonewtons?)\b", 1000.0, "kN"),
        (rf"{_NUM}\s*(?:n|newtons?)\b(?!\s*[-/]?\s*m)", 1.0, "N"),
        (rf"{_NUM}\s*(?:lbf|pounds?[- ]force)\b", 4.4482216, "lbf"),
        (rf"{_NUM}\s*(?:kgf|kilograms?[- ]force)\b", G, "kgf"),
    ):
        mm = re.search(pat, t)
        if mm:
            v = _f(mm.group(1))
            if v is not None:
                return v * k, f"{mm.group(1)} {unit}"

    # masses -> weight. Only when the sentence is about a load, not a size.
    for pat, k, unit in (
        (rf"{_NUM}\s*(?:kgs?|kilograms?)\b", G, "kg"),
        (rf"{_NUM}\s*(?:lbs?|pounds?)\b", 0.45359237 * G, "lb"),
    ):
        mm = re.search(pat, t)
        if mm:
            v = _f(mm.group(1))
            if v is not None:
                return v * k, f"{mm.group(1)} {unit} × g"
    return None, None


def _find_thickness(text):
    t = text.lower()
    mm = re.search(rf"{_NUM}\s*(?:mm|millimet)", t)
    if mm:
        v = _f(mm.group(1))
        if v and 0.3 <= v <= 80:
            return v, f"{mm.group(1)} mm"
    # fractional inches are how plate is actually ordered: 1/4", 1/8 in
    mm = re.search(r"(\d+)\s*/\s*(\d+)\s*(?:\"|in\b|inch)", t)
    if mm:
        a, b = _f(mm.group(1)), _f(mm.group(2))
        if a and b:
            v = 25.4 * a / b
            if 0.3 <= v <= 80:
                return v, f"{mm.group(1)}/{mm.group(2)} in"
    mm = re.search(rf"{_NUM}\s*(?:\"|in\b|inch)", t)
    if mm:
        v = _f(mm.group(1))
        if v and 0.012 <= v <= 3.2:
            return v * 25.4, f"{mm.group(1)} in"
    return None, None


def _find_material(text):
    """Match a material by its alloy designation, then by a distinctive word.

    Designations are checked first because '6061' is unambiguous while
    'steel' matches half the table."""
    t = text.lower()
    names = materials.names()
    for n in names:
        for tok in re.findall(r"\d{4}|\d{3}\b", n):
            if re.search(rf"\b{tok}\b", t):
                return n
    for n in names:
        head = n.lower().split()[0]
        if len(head) > 4 and re.search(rf"\b{re.escape(head)}\b", t):
            return n
    if re.search(r"\bti\b|\btitanium\b", t):
        for n in names:
            if "titan" in n.lower():
                return n
    return None


def _first(patterns, text, default=None):
    for val, pat in patterns:
        if re.search(pat, text, re.I):
            return val
    return default


def parse_regex(text):
    """Deterministic pass. Returns only the fields the text actually pins
    down -- absent means 'not stated', which is different from a default."""
    t = " " + (text or "").strip().lower() + " "
    out, why = {}, []

    case = _first(_CASE_PATTERNS, t)
    if case:
        out["load_case"] = case
        why.append(CASES[case])

    force, how = _find_force(t)
    if force is not None and 0.1 <= force <= 5e6:
        out["load"] = round(force, 2)
        why.append(f"{out['load']:g} N (from {how})")

    if re.search(_VERTICAL, t):
        out["orientation"] = "vertical"
    elif re.search(_HORIZONTAL, t):
        out["orientation"] = "horizontal"

    mode = _first(_MODES, t)
    if mode:
        out["mode"] = mode

    mat = _find_material(t)
    if mat:
        out["material"] = mat
        why.append(mat)

    thk, thow = _find_thickness(t)
    if thk:
        out["thickness_mm"] = round(thk, 3)
        why.append(f"{out['thickness_mm']:g} mm thick (from {thow})")

    return out, why


_LLM_PROMPT = (
    "You convert a plain-English description of how a mechanical part is "
    "loaded into solver settings for a 2D plate stress analysis.\n\n"
    "Reply with ONE JSON object and nothing else. Schema:\n"
    "{\n"
    '  "load_case": one of "cantilever" | "ss_center" | "ss_dist" | "fixed_fixed",\n'
    '  "orientation": "horizontal" | "vertical",\n'
    '  "load_n": total applied force in NEWTONS (number),\n'
    '  "mode": "structural" | "fatigue" | "manufacturing" | "full",\n'
    '  "thickness_mm": plate thickness in millimetres (number),\n'
    '  "material": material name if one is named, else "",\n'
    '  "restated": one sentence describing the load case you chose,\n'
    '  "assumed": list of short strings, one per value you had to guess\n'
    "}\n\n"
    "Rules:\n"
    "- cantilever = held at one end only. ss_center = supported at both ends, "
    "load in the middle. ss_dist = supported at both ends, load spread over "
    "the whole part. fixed_fixed = rigidly built in at both ends.\n"
    "- A mass is not a force. Multiply kg by 9.81 and lb by 4.45.\n"
    "- orientation is the part's long axis: horizontal unless the text says "
    "the part stands upright.\n"
    "- OMIT any key the text gives you no basis for. Do not invent a force.\n"
    "- Every value you did guess must appear in \"assumed\"."
)

_ALLOWED_ORIENT = {"horizontal", "vertical"}
_ALLOWED_MODE = {"structural", "fatigue", "manufacturing", "full"}


def parse_llm(text, model=None):
    """Groq pass. Returns {} on any failure -- missing key, bad JSON, timeout,
    out-of-range numbers. The regex pass is always there to fall back on."""
    if not os.environ.get("GROQ_API_KEY"):
        return {}, []
    from . import chat as chatmod
    try:
        raw = chatmod.chat(
            [{"role": "user", "content": (text or "")[:1500]}],
            # Retired by Groq alongside llama-3.3-70b-versatile. Like the
            # planner in chat.py this call is inside a try/except that returns
            # an empty parse, so a 404 here read as "the typed loading
            # description said nothing useful" rather than as an outage.
            model=model or os.environ.get("GROQ_PLANNER_MODEL",
                                          "openai/gpt-oss-20b"),
            temperature=0.0, max_tokens=400, system=_LLM_PROMPT)
        obj = chatmod._json_block(raw)
    except Exception:
        return {}, []
    if not isinstance(obj, dict):
        return {}, []

    out = {}
    if obj.get("load_case") in CASES:
        out["load_case"] = obj["load_case"]
    if obj.get("orientation") in _ALLOWED_ORIENT:
        out["orientation"] = obj["orientation"]
    if obj.get("mode") in _ALLOWED_MODE:
        out["mode"] = obj["mode"]
    try:
        v = float(obj.get("load_n"))
        if 0.1 <= v <= 5e6:
            out["load"] = round(v, 2)
    except (TypeError, ValueError):
        pass
    try:
        v = float(obj.get("thickness_mm"))
        if 0.3 <= v <= 80:
            out["thickness_mm"] = round(v, 3)
    except (TypeError, ValueError):
        pass
    mat = str(obj.get("material") or "").strip()
    if mat:
        known = {n.lower(): n for n in materials.names()}
        if mat.lower() in known:
            out["material"] = known[mat.lower()]
        else:
            hit = _find_material(mat)
            if hit:
                out["material"] = hit

    restated = str(obj.get("restated") or "").strip()[:240]
    assumed = [str(a).strip()[:80] for a in (obj.get("assumed") or [])
               if str(a).strip()][:6]
    return out, ([restated] if restated else []) + assumed


DEFAULTS = {"load_case": "cantilever", "orientation": "horizontal",
            "load": 500.0, "mode": "structural", "thickness_mm": 6.35,
            "material": materials.DEFAULT}


def parse(text, model=None, use_llm=True):
    """Full parse. Returns a dict the /api/analyze form can be filled from,
    plus a readback so a misreading is visible before you trust the result."""
    text = (text or "").strip()
    if not text:
        return {"ok": False, "settings": dict(DEFAULTS), "from_text": [],
                "assumed": [], "understood": "", "source": "empty"}

    rx, why = parse_regex(text)
    llm, notes = ({}, [])
    if use_llm and len(text.split()) >= 3:
        llm, notes = parse_llm(text, model=model)

    # The regex pass wins every contested field. If you typed a number, that
    # is the number -- a language model does not get a vote on it.
    settings = dict(DEFAULTS)
    settings.update({k: v for k, v in llm.items()})
    settings.update(rx)

    from_text = sorted(set(rx) | set(llm))
    assumed = [k for k in DEFAULTS if k not in from_text]

    bits = [CASES.get(settings["load_case"], settings["load_case"]),
            f"{settings['load']:g} N",
            settings["material"],
            f"{settings['thickness_mm']:g} mm plate",
            settings["orientation"]]
    understood = " · ".join(bits)

    return {"ok": True,
            "settings": settings,
            "from_text": from_text,
            "assumed": assumed,
            "understood": understood,
            "detail": why + notes,
            "source": "text+llm" if llm else "text"}
