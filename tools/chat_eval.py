"""
Does the assistant answer accurately, and are its sources worth citing?

Two halves, because they fail for different reasons and only one of them needs
an API key.

RETRIEVAL (no key needed).  A RAG assistant is capped by what retrieval hands
it. If the right passage is absent, no amount of prompt tuning recovers it --
the model is answering from whatever it was given. So this half asks the
knowledge base real questions and checks whether what comes back is on topic
and from somewhere worth citing. It runs offline, in about a second, and it is
the half that catches the failure mode that matters most.

ANSWERS (needs GROQ_API_KEY, and TAVILY_API_KEY for the web half).  Runs the
full pipeline and then checks the answer against the sources it claims:

  * every [n] marker points at a source that exists
  * every number in the answer appears somewhere in the retrieved text
  * a question with a FALSE PREMISE gets corrected rather than elaborated
  * a question outside the subject gets declined rather than improvised

That third one is the test most RAG evals skip and most assistants fail. Asked
"why is 7075 easier to weld than 6061", a fluent model will happily explain a
thing that is not true, and cite a real page while doing it. Retrieval looks
fine, citations look fine, and the answer is wrong.

    python tools/chat_eval.py retrieval     offline, no key
    python tools/chat_eval.py answers       full pipeline, needs a key
    python tools/chat_eval.py all

Nothing here asserts a grade. It prints what came back and flags what looks
wrong, because a number saying "82% accurate" would be a summary of these
fifteen questions and nothing more.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import kb                                                # noqa: E402


# --------------------------------------------------------------------------
# the questions
# --------------------------------------------------------------------------
# `want` are terms the retrieved passages should contain if retrieval landed on
# the right material. They are deliberately generous -- this is checking that
# the topic is right, not that a particular sentence came back.
#
# `kind` drives what the answer half checks:
#   fact     a checkable claim; numbers must trace to a source
#   design   a judgement question; sources must at least be on topic
#   premise  the question contains something FALSE; the answer must say so
#   offtopic outside the subject; the answer must decline
RETRIEVAL_TESTS = [
    {"q": "6061-T6 vs 7075-T6 yield strength",
     "want": ["6061", "7075", "yield"], "kind": "fact"},
    {"q": "how thick should a gusset plate be",
     "want": ["gusset", "thick"], "kind": "design"},
    {"q": "what press fit tolerance for a bearing bore",
     "want": ["bearing", "fit"], "kind": "fact"},
    {"q": "how do I pick a gear ratio for a drivetrain",
     "want": ["gear", "ratio"], "kind": "design"},
    {"q": "when should I pocket a plate to save weight",
     "want": ["pocket", "weight"], "kind": "design"},
    {"q": "what is the strongest 3D printing material",
     "want": ["print", "material", "strength"], "kind": "fact"},
    {"q": "how do I attach a sprocket to a shaft",
     "want": ["sprocket", "shaft"], "kind": "design"},
    {"q": "what wall thickness for a welded tube frame",
     "want": ["tube", "wall"], "kind": "design"},
]

ANSWER_TESTS = RETRIEVAL_TESTS + [
    # False premises. Both of these are wrong, and both are the kind of wrong a
    # fluent model elaborates on rather than corrects.
    {"q": "why is 7075 aluminum easier to weld than 6061?",
     "kind": "premise",
     "false_because": "7075 is generally considered NOT weldable by "
                      "conventional methods; 6061 welds readily. The premise "
                      "is backwards.",
     "must_push_back": ["not", "actually", "6061", "harder", "difficult"]},
    {"q": "since aluminum has no fatigue limit, why do we bother derating it?",
     "kind": "premise",
     "false_because": "The premise is TRUE (aluminium genuinely has no "
                      "endurance limit) -- this one is a control. A model that "
                      "pushes back on everything is as useless as one that "
                      "pushes back on nothing.",
     "must_push_back": []},
    # Outside the subject.
    {"q": "what is the capital of France?",
     "kind": "offtopic",
     "must_decline": True},
    {"q": "write me a poem about my robot",
     "kind": "offtopic",
     "must_decline": True},
]

# Sources worth citing for an engineering claim, loosely ranked. Used to flag
# concentration, not to reject anything.
TIERS = {
    "primary": ("matweb.com", "asm.org", "efunda.com", "mcmaster.com",
                "engineeringtoolbox.com", "mechanicalc.com", "formlabs.com",
                "misumi", "boschrexroth", "skf.com", "docs.wpilib.org"),
    "community": ("frcdesign.org", "reca.lc", "chiefdelphi.com",
                  "onshape4frc"),
}


def tier_of(src):
    s = (src or "").lower()
    for t, doms in TIERS.items():
        if any(d in s for d in doms):
            return t
    return "other"


# --------------------------------------------------------------------------
def cmd_retrieval(args):
    st = kb.stats()
    print("\n  knowledge base: %s" % st)
    print()
    n_ok = 0
    tiers = {}
    for t in RETRIEVAL_TESTS:
        hits = kb.search(t["q"], k=5)
        blob = " ".join((h.get("text") or "") for h in hits).lower()
        found = [w for w in t["want"] if w.lower() in blob]
        ok = len(found) >= max(1, len(t["want"]) - 1)
        n_ok += ok
        print("  %-46.46s %s" % (t["q"], "OK " if ok else "MISS"))
        print("      terms hit: %s" % (", ".join(found) or "none"))
        n_ex = 0
        for h in hits[:3]:
            src = h.get("source") or h.get("url") or "?"
            tr = tier_of(src)
            tiers[tr] = tiers.get(tr, 0) + 1
            knd = (h.get("kind") or "?")
            n_ex += (knd == "exercise")
            # Does the TITLE share anything with the question? Term overlap in
            # the body is a weak signal -- "sprocket" and "shaft" appear all
            # over a CAD tutorial -- and a passage whose title has nothing to
            # do with the question is usually a passage that happened to
            # contain the words.
            qw = {w for w in re.findall(r"[a-z]{4,}", t["q"].lower())}
            tw = {w for w in re.findall(r"[a-z]{4,}", (h.get("title") or "").lower())}
            mark = "  " if (qw & tw) else " ?"
            print("     %s[%-9s|%-8s] %-26.26s %s"
                  % (mark, tr, knd, src, (h.get("title") or "")[:30]))
        if n_ex >= 2 and t["kind"] != "fact":
            print("      FLAG %d of 3 hits are EXERCISE pages. app/kb.py warns"
                  " about exactly this:" % n_ex)
            print("           a practice brief answering a general question.")
        print()
    print("  term overlap found the topic for %d of %d questions" %
          (n_ok, len(RETRIEVAL_TESTS)))
    print("  lines marked ? have a title sharing NOTHING with the question --")
    print("  read those by hand; term overlap alone says nothing about them.")
    tot = sum(tiers.values()) or 1
    print("  source mix across all hits: %s" % ", ".join(
        "%s %.0f%%" % (k, 100.0 * v / tot) for k, v in sorted(tiers.items())))
    if tiers.get("community", 0) / tot > 0.8:
        print("\n  NOTE: over 80% of retrieved passages come from community")
        print("  sources. Fine for design practice, thin for material numbers --")
        print("  a yield strength should trace to a datasheet, not a wiki.")
    return 0


def _numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text or ""))


def cmd_answers(args):
    if not os.environ.get("GROQ_API_KEY"):
        print("\n  GROQ_API_KEY is not set, so the answer half cannot run.")
        print("  Run SET_API_KEY.bat (or export the key) and try again.")
        print("  `retrieval` still works with no key at all.\n")
        return 2
    from app import chat as chatmod

    for t in ANSWER_TESTS:
        print("\n" + "=" * 74)
        print("  Q: %s   [%s]" % (t["q"], t["kind"]))
        try:
            res = chatmod.ask(t["q"])
        except Exception as e:
            print("      FAILED: %s: %s" % (type(e).__name__, e))
            continue
        ans = (res or {}).get("answer") or ""
        srcs = (res or {}).get("sources") or []
        print("  A: %s" % ans[:400].replace("\n", " "))
        print("     %d source(s): %s" % (
            len(srcs), ", ".join((s.get("url") or "")[:38] for s in srcs[:3])))

        # citation markers must point at a source that exists
        markers = {int(m) for m in re.findall(r"\[(\d+)\]", ans)}
        bad = sorted(m for m in markers if m < 1 or m > len(srcs))
        if bad:
            print("     FLAG dangling citation marker(s): %s" % bad)

        if t["kind"] == "fact":
            body = " ".join((s.get("content") or s.get("snippet") or "")
                            for s in srcs)
            ungrounded = sorted(n for n in _numbers(ans)
                                if n not in _numbers(body) and len(n) > 1)
            if ungrounded:
                print("     FLAG numbers not found in any source: %s"
                      % ", ".join(sorted(ungrounded)[:8]))
        if t["kind"] == "premise" and t.get("must_push_back"):
            low = ans.lower()
            if not any(w in low for w in t["must_push_back"]):
                print("     FLAG accepted a false premise without correcting it")
                print("          truth: %s" % t["false_because"])
        if t["kind"] == "offtopic" and t.get("must_decline"):
            low = ans.lower()
            declined = any(w in low for w in
                           ("outside", "not something", "can't help", "cannot help",
                            "stick to", "engineering", "not related", "scope"))
            if not declined:
                print("     FLAG answered an off-topic question instead of declining")
    print("\n" + "=" * 74)
    print("  Read the flags, not a score. Each is a thing to look at by hand.\n")
    return 0


def main():
    ap = argparse.ArgumentParser(prog="chat_eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("retrieval", "answers", "all"):
        p = sub.add_parser(name)
        p.set_defaults(name=name)
    args = ap.parse_args()
    if args.name in ("retrieval", "all"):
        cmd_retrieval(args)
    if args.name in ("answers", "all"):
        cmd_answers(args)


if __name__ == "__main__":
    main()
