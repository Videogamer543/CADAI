"""
What the reference parts actually DO — a training manual, not a scoreboard.

`tools/pocket_ref.py` answers one question: how far is our plan from the real
one, as a single number. That is the right question when you are tuning six
global constants, and the wrong one when you want to know *why* a machinist put
a rib where they put it.

This reads the same folder and asks the other question. For every part it
recovers the finished geometry -- the blank, the pockets, the bores -- and then
measures the decisions:

  * **Boss**, the ring of solid kept around each bore before any pocket starts.
    A bolt needs material to bear against and a bearing needs a seat. Real
    parts leave one; the engine reserves a keep-out whose size was guessed.
    Reported as boss_radius / bore_radius, which is scale-free, so a 5 mm bolt
    hole and a 40 mm bearing bore can be compared on the same axis.

  * **Spokes**, the number of rib-width arms of solid material radiating from
    each boss. This is the quantity the whole exercise is about. A hole that
    exists for a reason is a hole that carries load, and a machinist ties it
    into the structure with ribs. Zero spokes means an island; three or four
    means a hub. The engine currently draws a lattice and then punches
    clearance around bores, which produces whatever spoke count falls out.

  * **Rim**, the solid border kept around the silhouette, as a multiple of the
    rib width.

  * **Web**, the narrowest solid ligament anywhere on the part, which is the
    machinability floor the part was actually built to.

  * Plus the plain descriptors -- span, short axis, aspect ratio, stock and
    floor thickness, bore count and bore-area share, removal fraction.

Everything here is MEASURED off the finished part. Nothing is fitted and
nothing is assumed, because the point of the file is to be the evidence that a
later fit is argued from.

    python tools/pocket_learn.py report      the manual, part by part
    python tools/pocket_learn.py rules       the numbers a rule could use
    python tools/pocket_learn.py json OUT    the whole table, machine-readable

A caution that belongs at the top rather than in a footnote: 27 parts is a
small sample from a small number of designers. A quantity with a tight spread
across all of them is worth turning into a rule; one that scatters is a
description of these parts and nothing more. `rules` reports the spread next to
every number for exactly that reason, and refuses to suggest a constant whose
scatter is wider than the effect it would have.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import pocket_ref as PR                                # noqa: E402


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def _dt(mask):
    """Distance from every pixel to the nearest zero, in pixels."""
    import cv2
    return cv2.distanceTransform(np.ascontiguousarray(mask.astype(np.uint8)),
                                 cv2.DIST_L2, 3)


def _bore_records(bores, px_per_mm):
    """One record per bore: centre, radius, area — in millimetres."""
    import cv2
    out = []
    m = np.ascontiguousarray((bores > 0).astype(np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    for i in range(1, n):
        area_px = float(stats[i, cv2.CC_STAT_AREA])
        if area_px < 4:
            continue
        # Equivalent radius from AREA, not from the bounding box: a bore that
        # the rasteriser clipped at the part edge has a misleading box and an
        # honest area.
        r_px = math.sqrt(area_px / math.pi)
        out.append({"cx": float(cent[i][0]), "cy": float(cent[i][1]),
                    "r_mm": r_px / px_per_mm, "r_px": r_px,
                    "label": i})
    out.sort(key=lambda b: -b["r_mm"])
    return out, lab


def _boss_and_spokes(solid, bores, lab, recs, px_per_mm, rib_px):
    """For each bore: how much solid is kept around it, and how it ties in.

    `boss` is measured by walking radially outward from the bore rim and
    finding the first radius at which the ring is no longer mostly solid. That
    is deliberately not "distance to the nearest pocket pixel": a pocket that
    approaches from one side only should not be allowed to claim the boss is
    zero, because the bolt is still fully supported on the other three.

    `spokes` counts the connected runs of solid around a ring drawn just
    outside the boss. Each run at least a rib wide is an arm tying this hole
    into the rest of the part -- which is the thing a machinist decides and the
    engine currently leaves to chance.
    """
    H, W = solid.shape
    yy, xx = np.mgrid[0:H, 0:W]
    out = []
    for b in recs:
        r0 = max(b["r_px"], 1.0)
        d = np.hypot(xx - b["cx"], yy - b["cy"])

        # --- boss: first ring that stops being mostly solid ----------------
        boss_px = 0.0
        step = max(1.0, rib_px * 0.25)
        r = r0 + step
        while r < r0 + 14.0 * max(rib_px, 1.0) and r < max(H, W):
            ring = (d >= r - step * 0.5) & (d < r + step * 0.5)
            if not ring.any():
                break
            frac = float(solid[ring].mean())
            if frac < 0.80:
                break
            boss_px = r - r0
            r += step
        # --- spokes: solid runs on a ring just outside the boss ------------
        r_probe = r0 + boss_px + max(rib_px, 1.0)
        spokes, arc_frac = 0, 0.0
        if r_probe < 0.5 * max(H, W):
            n_theta = 720
            th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
            px = np.clip((b["cx"] + r_probe * np.cos(th)).astype(int), 0, W - 1)
            py = np.clip((b["cy"] + r_probe * np.sin(th)).astype(int), 0, H - 1)
            ring = solid[py, px].astype(bool)
            arc_frac = float(ring.mean())
            if ring.all():
                spokes = -1                      # fully embedded, not an island
            else:
                # circular run-length: count rising edges
                prev = np.roll(ring, 1)
                starts = np.flatnonzero(ring & ~prev)
                min_arc = max(2, int(n_theta * (rib_px / (2 * np.pi * r_probe))))
                runs = []
                for s in starts:
                    ln = 0
                    i = s
                    while ring[i % n_theta] and ln < n_theta:
                        ln += 1
                        i += 1
                    runs.append(ln)
                spokes = int(sum(1 for ln in runs if ln >= min_arc))
        out.append({"r_mm": b["r_mm"],
                    "boss_mm": boss_px / px_per_mm,
                    "boss_ratio": (boss_px / r0) if r0 > 0 else 0.0,
                    "spokes": spokes,
                    "arc_solid": arc_frac})
    return out


def features(rec, verbose=False):
    """Every measurement this file makes, for one reference part."""
    loaded = PR.load_part(rec, verbose=verbose)
    real = loaded.get("real")
    masks = loaded.get("masks")
    if not real or not masks:
        return None
    # load_part hands back a plain tuple, not a dict -- unpacked here rather
    # than indexed by name so a change to its shape fails loudly at this line
    # instead of silently measuring the wrong array.
    part, pockets, bores, ppm = masks
    part = np.asarray(part, bool)
    pockets = np.asarray(pockets, bool)
    bores = (np.asarray(bores, bool) if bores is not None
             else np.zeros_like(part))
    ppm = float(ppm)
    extras = loaded.get("extras") or {}

    solid = part & ~pockets & ~bores
    if not solid.any():
        return None

    ys, xs = np.nonzero(part)
    span_px = max(np.ptp(xs), np.ptp(ys)) or 1
    short_px = min(np.ptp(xs), np.ptp(ys)) or 1
    span_mm, short_mm = span_px / ppm, short_px / ppm

    # rib width: twice the distance-to-edge along the solid's medial ridge, at a
    # low percentile -- the same measure pocket_ref uses, so the two files agree
    rib_mm = PR._rib_mm(solid, ppm)
    rib_px = max(1.0, rib_mm * ppm)

    # rim: solid retained just inside the silhouette
    import cv2
    # Rim is the distance from the silhouette inward to the first POCKET, not
    # to the first non-solid pixel. The earlier version measured the latter and
    # reported 0.2-1.1 mm on every part regardless of size -- which is exactly
    # what it should report, because a boundary pixel is by construction one
    # pixel from the outside. It was measuring the rasteriser, not the part.
    er = cv2.erode(part.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    edge = part & ~er.astype(bool)
    rim_px = 0.0
    if edge.any() and pockets.any():
        d_to_pocket = _dt(~pockets)          # 0 inside a pocket, grows outward
        rim_px = float(np.percentile(d_to_pocket[edge], 50))

    recs, lab = _bore_records(bores, ppm)
    holes = _boss_and_spokes(solid, bores, lab, recs, ppm, rib_px)

    part_area = float(part.sum())
    out = {
        "name": rec.get("name") or os.path.basename(rec["path"]),
        "span_mm": round(span_mm, 1),
        "short_mm": round(short_mm, 1),
        "aspect": round(span_mm / max(short_mm, 1e-6), 2),
        "stock_mm": extras.get("thickness_mm") or real.get("thickness_mm"),
        "floor_mm": extras.get("floor_mm") or real.get("floor_mm"),
        "rib_mm": round(rib_mm, 2),
        "rim_mm": round(rim_px / ppm, 2),
        "rim_over_rib": round((rim_px / rib_px), 2) if rib_px else None,
        "removal": round(float(pockets.sum()) / part_area, 4),
        "n_bores": len(recs),
        "bore_area_frac": round(float(bores.sum()) / part_area, 4),
        "holes": holes,
    }
    if holes:
        br = [h["boss_ratio"] for h in holes]
        sp = [h["spokes"] for h in holes if h["spokes"] >= 0]
        out["boss_ratio_med"] = round(float(np.median(br)), 3)
        out["boss_mm_med"] = round(float(np.median([h["boss_mm"] for h in holes])), 2)
        out["spokes_med"] = float(np.median(sp)) if sp else None
        out["embedded_frac"] = round(
            float(np.mean([h["spokes"] < 0 for h in holes])), 3)
    # The narrowest-web metric that used to live here has been removed rather
    # than fixed. A low percentile of the distance transform over a mask that
    # was rasterised to a fixed pixel budget returns a roughly constant number
    # of PIXELS, so in millimetres it just tracked 1/px_per_mm -- it rose
    # monotonically with span across all 27 parts, which is the signature of a
    # measurement of the raster rather than of the geometry. rib_mm already
    # carries the real ligament information and is measured on the medial ridge
    # where that is meaningful.

    # Plausibility. These parts are the evidence a fit is argued from, so a
    # mis-measured one is worse than a missing one: `fit` would weight it
    # equally and quietly drag every constant toward a part that does not
    # exist. Flagged, not dropped, so the report still shows what happened.
    flags = []
    if out["stock_mm"] and out["stock_mm"] > 25.0:
        flags.append("stock %.0f mm is not plate - level detection probably "
                     "failed" % out["stock_mm"])
    if out["removal"] > 0.88:
        flags.append("removal %.0f%% - a pocketed plate that keeps under an "
                     "eighth of its blank is almost certainly a mis-read"
                     % (100 * out["removal"]))
    if rib_mm < 2.5:
        flags.append("rib %.1f mm is below anything machinable here" % rib_mm)
    out["flags"] = flags
    out["trusted"] = not flags
    return out


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def _collect(args):
    parts = PR.find_parts()
    rows = []
    for rec in parts:
        try:
            f = features(rec, verbose=args.verbose)
            if f:
                rows.append(f)
            else:
                print("  skipped %s (no usable geometry)" % rec.get("name"))
        except Exception as e:
            print("  skipped %s %s: %s" % (rec.get("name"), type(e).__name__,
                                           str(e)[:70]))
    return rows


def _spread(vals):
    """median, and the spread as a fraction of it. Small = worth a rule."""
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return None, None, 0
    med = float(np.median(v))
    # median absolute deviation, scaled to be comparable with a std dev
    mad = 1.4826 * float(np.median(np.abs(v - med)))
    return med, (mad / med if med else None), int(v.size)


def cmd_report(args):
    rows = _collect(args)
    if not rows:
        print("no usable parts")
        return
    print()
    print("  %-26s %6s %5s %6s %5s %5s %4s %5s %6s  %s" % (
        "PART", "SPAN", "AR", "STOCK", "RIB", "RIM", "BOR", "BOSS", "REMOV", ""))
    for r in sorted(rows, key=lambda r: -r["span_mm"]):
        print("  %-26.26s %6.0f %5.1f %6s %5.1f %5.1f %4d %5s %5.0f%%  %s" % (
            r["name"], r["span_mm"], r["aspect"],
            ("%.2f" % r["stock_mm"]) if r["stock_mm"] else "  -  ",
            r["rib_mm"], r["rim_mm"], r["n_bores"],
            ("%.2f" % r["boss_ratio_med"]) if r.get("boss_ratio_med") is not None else "  -  ",
            100 * r["removal"], "" if r["trusted"] else "<-- EXCLUDED"))
    bad = [r for r in rows if not r["trusted"]]
    if bad:
        print()
        print("  EXCLUDED FROM ANY FIT (%d):" % len(bad))
        for r in bad:
            for f in r["flags"]:
                print("    %-26.26s %s" % (r["name"], f))
    print()
    print("  SPAN/RIB/RIM in mm. BOSS = boss radius / bore radius.")
    print("  %d part(s)." % len(rows))


def cmd_rules(args):
    rows = [r for r in _collect(args) if r["trusted"]]
    if not rows:
        print("no usable parts")
        return
    print()
    print("  WHAT THE PARTS AGREE ON")
    print("  (spread is a robust relative scatter; under ~0.25 is tight enough")
    print("   to build a rule on, above ~0.5 is a description of these parts.)")
    print()

    def line(label, vals, unit="", note=""):
        med, spread, n = _spread(vals)
        if med is None:
            print("  %-30s  no data" % label)
            return
        verdict = ("tight"  if spread is not None and spread < 0.25 else
                   "usable" if spread is not None and spread < 0.50 else
                   "SCATTERED - do not fit")
        print("  %-30s %8.3f %-4s  spread %5s  n=%-3d %s" % (
            label, med, unit,
            ("%.2f" % spread) if spread is not None else "   - ", n, verdict))
        if note:
            print("      %s" % note)

    line("boss radius / bore radius", [r.get("boss_ratio_med") for r in rows], "x",
         "how much solid a real part keeps around a hole before pocketing starts")
    line("boss (mm)", [r.get("boss_mm_med") for r in rows], "mm")
    line("rim / rib width", [r.get("rim_over_rib") for r in rows], "x",
         "perimeter band kept solid, in rib widths")
    line("rib width (mm)", [r["rib_mm"] for r in rows], "mm")
    line("removal fraction", [r["removal"] for r in rows], "",
         "share of the blank taken away")

    sp = [r.get("spokes_med") for r in rows if r.get("spokes_med") is not None]
    if sp:
        print()
        print("  HOW HOLES TIE INTO THE STRUCTURE")
        vals, counts = np.unique(np.asarray(sp), return_counts=True)
        for v, c in zip(vals, counts):
            print("      %-4.1f spokes per hole   %d part(s)" % (v, c))
        emb = [r.get("embedded_frac", 0.0) for r in rows]
        print("      holes fully inside solid (no pocket adjacent): %.0f%% on average"
              % (100 * float(np.mean(emb))))

    print()
    print("  BY STOCK THICKNESS")
    for t in sorted({round(r["stock_mm"], 2) for r in rows if r.get("stock_mm")}):
        sub = [r for r in rows if r.get("stock_mm") and abs(r["stock_mm"] - t) < 0.01]
        med_rm, _, n = _spread([r["removal"] for r in sub])
        med_bs, _, _ = _spread([r.get("boss_ratio_med") for r in sub])
        print("      %.2f mm  n=%-3d removal %.0f%%   boss %s" % (
            t, n, 100 * (med_rm or 0),
            ("%.2fx" % med_bs) if med_bs else "-"))


def cmd_json(args):
    rows = _collect(args)
    out = args.out or os.path.join(ROOT, "data", "pocket_features.json")
    with open(out, "w") as fh:
        json.dump({"n_parts": len(rows), "parts": rows}, fh, indent=1)
    print("wrote %s (%d parts)" % (out, len(rows)))


def main():
    ap = argparse.ArgumentParser(prog="pocket_learn")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("report", cmd_report), ("rules", cmd_rules),
                     ("json", cmd_json)):
        p = sub.add_parser(name)
        p.add_argument("-v", "--verbose", action="store_true")
        if name == "json":
            p.add_argument("out", nargs="?")
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
