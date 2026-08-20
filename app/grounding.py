"""
Checking the answer against the sources it cites.

The problem this solves is narrow and specific. Retrieval can put the right
page in front of the model and the model can still write a number that is not
on it -- transposed digits, a value carried over from a different material, a
tolerance recalled from training rather than read from the excerpt. Everything
upstream of here improves the odds; nothing upstream can catch it after the
fact. This can, for the one class of claim where being wrong does physical
damage: the numbers.

Why numbers and not prose
-------------------------
A number is the only part of an engineering answer that can be checked
mechanically and unambiguously. "6061-T6 is a good choice for structural plate"
is a judgement; "the yield strength is 276 MPa" is either in the cited source
or it is not. Numbers are also the claims most worth checking, on both counts
that matter: they are what a reader acts on, and they are what a language model
is most likely to get subtly wrong while sounding completely certain. A
plausible sentence with one wrong digit is far more dangerous than an obviously
vague paragraph, because nothing about it invites a second look.

Why this is not another model call
----------------------------------
Asking a language model whether a language model's answer is supported is
appealing and mostly theatre. It doubles latency and cost, it fails in
correlated ways with the thing it is checking -- the same model that invented
"0.090 in" will happily confirm it -- and its verdict cannot be reproduced, so
a flag that turns out to be wrong cannot be traced to a rule anyone can fix.
String and interval matching over the excerpt is dumb, instant, free, and
auditable. When it says a number is not in the sources, that is a fact about
the text, not an opinion about it.

What it deliberately does NOT do
--------------------------------
It does not convert units, so a source in ksi and an answer in MPa reads as
unsupported. It does not follow arithmetic: an answer that correctly computes
a section modulus from two cited dimensions will flag the result, because the
result genuinely is not in any source. It does not judge whether a supported
number is being used correctly, or whether the surrounding sentence is true.

That asymmetry is on purpose and runs the safe way. A flag means "a human
should look at this", never "this is wrong", and the absence of a flag means
only "this number appears in a source it cites" -- never "this answer is
correct". Every message this module produces is worded to keep those apart,
because a verifier that overstates what it verified is worse than no verifier
at all: it converts a reader's healthy suspicion into misplaced trust.
"""
from __future__ import annotations

import re

# A quantity: optional sign, digits, optional decimal or fraction, optional
# unit. Also matches engineering identifiers (10-32, 1/4-20, 6061-T6, #25)
# because those are exactly the tokens that must survive intact -- a fastener
# called out by the wrong thread pitch is the same class of error as a wrong
# stress value, and is caught by the same literal comparison.
_NUM = re.compile(r"""
    (?<![\w.])
    (?:\#\d+                              # #25 chain
      |\d+(?:\.\d+)?/\d+(?:\.\d+)?        # 1/4, 5/16
      |\d+(?:\.\d+)?                      # 0.090, 276, 2024
      |\.\d+)                             # .090
    (?:-[A-Za-z0-9]+)*                    # -32, -T6, -20
    (?![\w.])
""", re.X)

# Units worth treating as part of the quantity, so "0.090 in" and "0.090in"
# compare equal and a bare "0.090" is still recognised as a measurement.
#
# Compound units come first. Python's alternation takes the first branch that
# matches, not the longest, and "in" is a prefix of "in-lb" that ends on a word
# boundary -- so with the short form first, a torque of "40 in-lb" gets reported
# to the reader as "40 in". The value checked is the same either way, but a flag
# that misquotes the claim it is flagging is a flag nobody trusts.
_UNIT = re.compile(
    r"\s*(?:in-?lbs?\b|ft-?lbs?\b|inch(?:es)?\b|in\b|\"|mm\b|cm\b|m\b|ft\b"
    r"|feet\b|thou\b|mil\b|mpa\b|gpa\b|ksi\b|psi\b|pa\b|kn\b|n\b|lbf\b|lbs?\b"
    r"|kg\b|g\b|oz\b|nm\b|deg\b|degrees?\b|%|rpm\b|hz\b|amps?\b"
    r"|a\b|v\b|w\b|hp\b)", re.I)

# Sentence boundary: a full stop, question or exclamation mark followed by
# whitespace and a capital. The lookbehind keeps "0.090" and "e.g." from
# splitting a sentence in half and orphaning the number from its citation.
#
# A blank line ends a sentence too, whatever it ends with. Bullet lists and
# headings routinely have no terminating punctuation, and without this a "[3]"
# on the last bullet would be read as covering every number in the paragraph
# after it -- a citation the reader can see does not apply, silently counted as
# support.
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])|\n\s*\n")

_CITE = re.compile(r"\[(\d{1,2})\]")

# Numbers that are not claims. Ordinals and small bare counts ("two or three
# ribs", "step 3") are prose, and checking them produces flags nobody can act
# on, which is the fastest way to teach a reader to ignore the flags that
# matter.
_MIN_BARE_INT = 10


def _norm(tok):
    """Compare-ready form of a quantity: lower case, no spaces or commas."""
    return re.sub(r"[\s,]", "", (tok or "").lower())


def _claims(sentence):
    """Quantities in one sentence, with their units attached where present."""
    out = []
    for m in _NUM.finditer(sentence):
        raw = m.group(0)
        # Skip the citation markers themselves.
        if sentence[max(0, m.start() - 1):m.start()] == "[":
            continue
        tail = _UNIT.match(sentence, m.end())
        unit = tail.group(0).strip() if tail else ""
        bare = raw.replace("#", "")
        is_plain_int = bool(re.fullmatch(r"\d+", bare))
        if is_plain_int and not unit and int(bare) < _MIN_BARE_INT:
            continue
        out.append({"raw": raw, "unit": unit,
                    "text": (raw + (" " + unit if unit else "")).strip()})
    return out


def _numeric_value(tok):
    """Float value of a quantity token, or None if it is an identifier."""
    t = tok.replace("#", "").strip()
    if re.fullmatch(r"\d+(?:\.\d+)?/\d+(?:\.\d+)?", t):
        a, b = t.split("/")
        try:
            return float(a) / float(b)
        except (ValueError, ZeroDivisionError):
            return None
    if re.fullmatch(r"\d*\.?\d+", t):
        try:
            return float(t)
        except ValueError:
            return None
    return None       # 10-32, 6061-T6: identifiers, matched literally only


def _in_text(claim, blob, values):
    """Is this quantity present in the source text?

    Three passes, loosest last. The literal comparison catches the ordinary
    case. The zero-padding variants exist because ".090" and "0.090" are the
    same dimension written by two different people. The tolerance pass exists
    because an answer that rounds 0.0905 to 0.090, or writes 276 where the
    table says 275.8, is reporting the source faithfully, and flagging it would
    train the reader to dismiss the flags.
    """
    raw = _norm(claim["raw"])
    if raw and raw in blob:
        return True
    if raw.startswith("0."):
        if raw[1:] in blob:
            return True
    elif raw.startswith("."):
        if "0" + raw in blob:
            return True

    val = _numeric_value(claim["raw"])
    if val is None:
        return False
    tol = max(abs(val) * 0.01, 1e-9)
    return any(abs(val - v) <= tol for v in values)


def _values_in(blob):
    """Every numeric value appearing in a source, for the tolerance pass."""
    out = []
    for m in _NUM.finditer(blob):
        v = _numeric_value(m.group(0))
        if v is not None:
            out.append(v)
    return out


def check(answer, sources, blocks):
    """Compare an answer's quantities against the excerpts it cites.

    `sources` is the list ask() returns; `blocks` is the matching list of
    excerpt texts, same order, so blocks[i] is what source i+1 actually said.

    Returns {verdict, checked, supported, flags, note}. `verdict` is:
        "ok"          every quantity was found in a source it cites
        "partial"     at least one was not
        "unsourced"   the answer cites nothing at all
        "skipped"     nothing numeric to check
    """
    blocks = list(blocks or [])
    norm_blocks = [_norm(b) for b in blocks]
    vals_blocks = [_values_in(b or "") for b in blocks]
    all_norm = _norm(" ".join(blocks))
    all_vals = [v for vs in vals_blocks for v in vs]

    checked = supported = 0
    flags = []
    for sentence in _SENT.split(answer or ""):
        s = sentence.strip()
        if not s:
            continue
        cites = [int(n) for n in _CITE.findall(s)]
        cites = [n for n in cites if 1 <= n <= len(blocks)]
        for claim in _claims(s):
            checked += 1
            if cites:
                ok = any(_in_text(claim, norm_blocks[n - 1], vals_blocks[n - 1])
                         for n in cites)
                if ok:
                    supported += 1
                    continue
                # Present in the sources, just not in the one cited. That is a
                # citation error rather than an invention, and saying which it
                # is decides what the reader does about it.
                elsewhere = _in_text(claim, all_norm, all_vals)
                flags.append({
                    "claim": claim["text"],
                    "sentence": s[:240],
                    "cites": cites,
                    "reason": "wrong-source" if elsewhere else "not-in-sources",
                })
            else:
                if _in_text(claim, all_norm, all_vals):
                    # Correct number, missing marker. Worth noting and not
                    # worth alarming anyone about.
                    supported += 1
                    continue
                flags.append({
                    "claim": claim["text"],
                    "sentence": s[:240],
                    "cites": [],
                    "reason": "uncited",
                })

    if not checked:
        verdict = "skipped"
    elif not blocks:
        verdict = "unsourced"
    elif not flags:
        verdict = "ok"
    else:
        verdict = "partial"

    return {"verdict": verdict, "checked": checked, "supported": supported,
            "flags": flags[:8], "note": _note(verdict, checked, supported, flags)}


def _note(verdict, checked, supported, flags):
    """One line a human can read, worded to claim only what was actually done."""
    if verdict == "skipped":
        return "No numeric claims to check."
    if verdict == "unsourced":
        return ("%d numeric claim%s, no sources to check them against."
                % (checked, "" if checked == 1 else "s"))
    if verdict == "ok":
        return ("%d of %d numeric claim%s appear in the sources cited. This "
                "checks that the numbers were copied, not that they are right "
                "for your part." % (supported, checked,
                                    "" if checked == 1 else "s"))
    n = len(flags)
    return ("%d of %d numeric claims appear in the sources cited; %d could not "
            "be found and %s worth checking before you cut anything."
            % (supported, checked, n, "is" if n == 1 else "are"))
