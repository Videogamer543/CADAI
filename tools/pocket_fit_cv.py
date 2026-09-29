"""
Fit the pocketing constants on the trusted reference parts, and check whether
the result actually generalises.

Why this is not just `pocket_ref.py fit`
----------------------------------------
Two reasons, both learned from this repository's own history.

1. **The last fit overfitted.** Constants fitted on 9 parts scored 0.1298 on
   those 9 and 0.260 on all 27. A single in-sample number cannot tell you that
   happened; only held-out parts can. So this does k-fold: fit on four fifths,
   score the fifth nobody fitted on, five times round. The number that matters
   is the held-out one.

2. **Two of the 27 parts are mis-measured.** One reads 127 mm "stock"; two
   report 94-95% removal with a 1.4 mm rib. They are level-detection failures,
   not parts, and `fit` weighted them equally -- dragging every constant toward
   geometry that does not exist. The same plausibility test tools/pocket_learn
   uses is applied here, and what it drops is printed rather than assumed.

Everything is measured ONCE and reused. `pocket_ref.py fit` re-measures the
folder on every invocation, which is ten minutes; doing that per fold would
turn an hour into two for no new information.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import pocket_ref as PR                                # noqa: E402
from app import pocketing                                         # noqa: E402

DENSITY = "normal"
PASSES = 2
GRID = 5
FOLDS = 5
SEED = 12345


def trusted(p):
    """Same plausibility test as tools/pocket_learn.py, same reasons."""
    masks = p.get("masks")
    if not masks:
        return False, "no masks"
    part, pockets, bores, ppm = masks
    part = np.asarray(part, bool)
    removal = float(np.asarray(pockets, bool).sum()) / max(float(part.sum()), 1.0)
    thick = float((p.get("extras") or {}).get("thickness_mm") or 0.0)
    rib = float((p.get("real") or {}).get("rib_mm") or 0.0)
    if thick > 25.0:
        return False, "stock %.0f mm is not plate" % thick
    if removal > 0.88:
        return False, "removal %.0f%% is a mis-read" % (100 * removal)
    if 0 < rib < 2.5:
        return False, "rib %.1f mm is not machinable" % rib
    return True, ""


def score_on(subset, vec, cache=None):
    cal = PR._expand(vec)
    key = (tuple(round(float(v), 5) for v in vec),
           tuple(sorted(p["name"] for p in subset)))
    if cache is not None and key in cache:
        return cache[key]
    g = PR.aggregate(PR.evaluate(subset, cal=cal, density=DENSITY))
    if cache is not None:
        cache[key] = g
    return g


def descend(train, active, start, log=None):
    """The same coordinate descent pocket_ref.fit uses, on a given subset."""
    cache = {}
    vec = list(start)
    best = score_on(train, vec, cache)
    for sweep in range(PASSES):
        span_f = 1.0 if sweep == 0 else 0.35 ** sweep
        for i in active:
            name, lo, hi = PR.FIT_PARAMS[i]
            cur, width = vec[i], (hi - lo) * span_f
            grid = np.unique(np.clip(
                np.linspace(cur - width / 2.0, cur + width / 2.0, GRID), lo, hi))
            lb, lg = cur, best
            for val in grid:
                t = list(vec)
                t[i] = float(val)
                g = score_on(train, t, cache)
                if g < lg - 1e-9:
                    lb, lg = float(val), g
            if lb != cur and lg <= best - max(best * PR.MIN_GAIN, 1e-4):
                vec[i], best = lb, lg
                if log:
                    log("      %-14s -> %8.3f   train gap %.4f" % (name, lb, best))
    return vec, best


def main():
    t_start = time.time()
    print("measuring reference parts (once)...", flush=True)
    parts = PR.find_parts()
    prepped = PR.prepare(parts, verbose=False)

    usable, dropped = [], []
    for p in prepped:
        if p.get("error") or not p.get("masks"):
            dropped.append((p.get("name"), p.get("error") or "unmeasurable"))
            continue
        ok, why = trusted(p)
        (usable if ok else dropped).append(p if ok else (p["name"], why))

    print("\n  %d usable, %d dropped" % (len(usable), len(dropped)))
    for name, why in dropped:
        print("    dropped %-28.28s %s" % (name, why))
    if len(usable) < 6:
        print("need at least 6 trusted parts"); return 1

    # Which constants this set is entitled to an opinion about.
    ts = [float((p.get("extras") or {}).get("thickness_mm") or 0) for p in usable]
    ts = [t for t in ts if t > 0]
    spread = (max(ts) / min(ts)) if len(ts) >= 2 else 1.0
    groups = []
    for t in sorted(ts):
        for g in groups:
            if abs(t - g[0]) <= PR.THICK_SAME_TOL * g[0]:
                g.append(t); break
        else:
            groups.append([t])
    groups.sort(key=len, reverse=True)
    n_off = sum(len(g) for g in groups[1:])
    fit_thick = spread >= PR.THICK_SPREAD_MIN and n_off >= PR.THICK_MIN_OFF_STOCK
    active = [i for i, (n, _l, _h) in enumerate(PR.FIT_PARAMS)
              if fit_thick or n != "thick_exp"]
    print("  stock spread %.1fx over %d parts, %d off the commonest stock"
          % (spread, len(ts), n_off))
    print("  thick_exp is %s" % ("FITTABLE" if fit_thick else "held at its default"))

    start = PR._start_vec()
    base_all = score_on(usable, start)
    print("\n  current constants score %.4f across all %d trusted parts"
          % (base_all, len(usable)))

    # ---- k-fold: the number that actually means something ----------------
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(usable))
    folds = [order[i::FOLDS] for i in range(FOLDS)]
    print("\n  %d-fold cross-validation" % FOLDS, flush=True)
    held_base, held_fit = [], []
    for k, test_idx in enumerate(folds):
        test = [usable[i] for i in test_idx]
        train = [usable[i] for i in range(len(usable)) if i not in set(test_idx)]
        print("    fold %d: train %d, test %d" % (k + 1, len(train), len(test)),
              flush=True)
        vec, _tr = descend(train, active, start,
                           log=lambda s: print(s, flush=True))
        b = score_on(test, start)
        f = score_on(test, vec)
        held_base.append(b); held_fit.append(f)
        print("      held-out: before %.4f  after %.4f  %s"
              % (b, f, "better" if f < b else "WORSE"), flush=True)

    hb, hf = float(np.mean(held_base)), float(np.mean(held_fit))
    print("\n  HELD-OUT MEAN   before %.4f   after %.4f   (%+.1f%%)"
          % (hb, hf, 100.0 * (hf - hb) / max(hb, 1e-9)))
    generalises = hf < hb - 1e-4

    # ---- final fit on everything trusted ---------------------------------
    print("\n  final fit on all %d trusted parts" % len(usable), flush=True)
    vec, best = descend(usable, active, start,
                        log=lambda s: print(s, flush=True))
    new, old = PR._expand(vec), pocketing.calibration()
    print("\n  %-14s %10s    %-10s" % ("CONSTANT", "FROM", "TO"))
    for k in sorted(new):
        if abs(new[k] - old[k]) > 1e-9:
            print("  %-14s %10.3f -> %-10.3f" % (k, old[k], new[k]))
    print("\n  in-sample gap %.4f -> %.4f" % (base_all, best))

    out = {
        "_comment": ("Written by tools/pocket_fit_cv.py. Fitted only on parts "
                     "that passed the plausibility test, and validated on "
                     "held-out folds -- see held_out below, which is the "
                     "number that says whether these generalise."),
        "fitted_on": sorted(p["name"] for p in usable),
        "n_parts": len(usable),
        "dropped": [{"name": n, "why": w} for n, w in dropped],
        "span_mm": [round(min(p["real"]["span_mm"] for p in usable), 1),
                    round(max(p["real"]["span_mm"] for p in usable), 1)],
        "gap_before": round(base_all, 4),
        "gap_after": round(best, 4),
        "held_out": {"folds": FOLDS,
                     "before": round(hb, 4), "after": round(hf, 4),
                     "generalises": bool(generalises)},
        "density": DENSITY,
        "constants": {k: round(v, 4) for k, v in new.items()},
    }
    dest = os.path.join(ROOT, "data", "pocket_cal_candidate.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n  wrote %s" % os.path.relpath(dest, ROOT))
    print("  %s" % ("HELD-OUT IMPROVED - safe to adopt" if generalises else
                    "HELD-OUT DID NOT IMPROVE - do NOT adopt these"))
    print("  total %.1f min" % ((time.time() - t_start) / 60.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
