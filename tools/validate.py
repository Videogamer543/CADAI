"""Does CADAI agree with the textbook? Four closed-form cases, measured.

Written in response to a PE's review, which asked for exactly this and was
right to. Every number a solver prints is believable; the only way to know
whether it is correct is to put it next to a problem whose answer was known
before the solver existed.

The four cases are the standard ones, and each is chosen because it breaks a
different way:

  Kirsch      a hole in a plate under tension. Tests whether the mesh resolves
              a stress concentration. A coarse mesh does not fail here -- it
              quietly reports a lower peak, which is the dangerous direction.
  Cantilever  sigma = Mc/I. Tests the bending field itself, at several
              stations, so a right answer at one point cannot hide a wrong
              distribution.
  Curved beam Winkler-Bach. Tests whether curvature is handled: the neutral
              axis shifts toward the centre and the inner fibre runs ~15%
              over what a straight-beam formula predicts. A solver that got
              this wrong would be conservative on the outside and unsafe on
              the inside.
  Lame        a thick cylinder under internal pressure. Tests a load with no
              reaction and a stress gradient through a wall -- and it is the
              only one of the four where the part is held by nothing at all.

Each runs at four mesh densities, so the report shows not just the error but
which way it is moving. An answer that is 6% off and converging is a different
statement from one that is 6% off and flat.

    python tools/validate.py              everything, with plots
    python tools/validate.py --no-plots   numbers only, no matplotlib
    python tools/validate.py --case kirsch

Writes validation_report.md and validation_*.png next to the repo root.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.mesh import build_mesh                              # noqa: E402
from app.fem import solve_plane_stress                       # noqa: E402

E, NU = 69e9, 0.33
THK_MM = 6.35
THK = THK_MM / 1000.0
PPM = 4.0                       # pixels per mm in the solver's image frame
M_PER_PX = 1.0 / (PPM * 1000.0)

# Mesh densities to sweep. Not the app's presets: these are raw triangle
# targets, because a convergence study wants the element size as a continuous
# knob rather than four named rungs.
LEVELS = (4000, 9000, 18000, 34000)


def ring(cx, cy, r, n=256):
    return [(cx + r * math.cos(t), cy + r * math.sin(t))
            for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]


def mean_edge_mm(verts, tris):
    """Typical element edge, in millimetres -- the x axis of a convergence plot."""
    p = verts[tris]
    a = 0.5 * np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) -
                     (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    return math.sqrt(float(a.mean()) * 4.0 / math.sqrt(3.0)) / PPM


def vm_near(res, x_px, y_px, rad_px):
    """Highest von Mises within a disc.

    The right reading for an extreme-fibre or peak-concentration check, where
    the question is "how hot does it get here". It is the WRONG reading inside
    a steep gradient -- see vm_at_radius.
    """
    p = np.asarray(res["nodes"])
    vm = np.asarray(res["von_mises"])
    d = np.hypot(p[:, 0] - x_px, p[:, 1] - y_px)
    sel = d <= rad_px
    return float(vm[sel].max()) if sel.any() else float("nan")


def vm_at_radius(res, cx_px, cy_px, r_mm, band_mm=0.35):
    """Mean von Mises on a thin ring -- a reading AT a radius, not near one.

    Written because the first Lame run came out 14% high and dead flat across
    four mesh densities, which is the signature of a measurement error rather
    than a solver one: a converging solver that is wrong gets less wrong.

    The cause was vm_near. Hoop stress through a thick wall falls by half from
    bore to outside, and a disc of radius 3 mm centred at r = 26 mm reaches
    the bore itself at r = 25, where the true value is 93 MPa against the
    86 MPa being compared against. Taking the maximum over that disc does not
    report the stress at 26 mm; it reports the stress at the hottest point the
    disc happened to touch, which was always the bore.

    Averaging a thin annulus asks the question that was meant.
    """
    p = np.asarray(res["nodes"])
    vm = np.asarray(res["von_mises"])
    d = np.hypot(p[:, 0] - cx_px, p[:, 1] - cy_px) / PPM
    sel = np.abs(d - r_mm) <= band_mm
    return float(vm[sel].mean()) if sel.any() else float("nan")


def solve(ext, holes, tgt, **kw):
    verts, tris = build_mesh(ext, holes, target_tris=tgt)
    res = solve_plane_stress(verts, tris, E=E, nu=NU, thickness=THK,
                             m_per_px=M_PER_PX, **kw)
    res["_h_mm"] = mean_edge_mm(verts, tris)
    res["_n"] = tris.shape[0]
    return res


# ===========================================================================
# 1. Kirsch - circular hole in a plate under uniaxial tension
# ===========================================================================
def case_kirsch():
    """Kt at the hole edge. Infinite-plate answer is exactly 3.

    The plate here is finite, so the honest comparison is Peterson's
    finite-width fit, which is a curve through Howland's exact solution:

        Kt_net = 3.000 - 3.140(d/w) + 3.667(d/w)^2 - 1.527(d/w)^3

    It is quoted on NET section stress, and the test of whether it has been
    transcribed correctly is its own endpoints: it gives exactly 3 as d/w goes
    to zero and exactly 2 at d/w = 1, which are the two values theory requires.
    Both forms are reported, because the "3" everyone remembers is the gross
    one and the one that is actually right here is the net one.
    """
    w_mm, L_mm, d_mm = 120.0, 360.0, 16.0
    w, L, a = w_mm * PPM, L_mm * PPM, (d_mm / 2) * PPM
    ext = [(0, 0), (L, 0), (L, w), (0, w)]
    holes = [ring(L / 2, w / 2, a)]

    dw = d_mm / w_mm
    kt_net = 3.000 - 3.140 * dw + 3.667 * dw ** 2 - 1.527 * dw ** 3
    P = 20000.0                                   # N, total pull
    s_gross = P / (w_mm * THK_MM * 1e-6)          # Pa
    s_net = P / ((w_mm - d_mm) * THK_MM * 1e-6)
    want = kt_net * s_net

    rows = []
    for tgt in LEVELS:
        r = solve(ext, holes, tgt, load_case="axial", orientation="horizontal",
                  load=P)
        # Read the rim, not the whole part: the grips at the ends carry their
        # own concentration and it is not the one being measured.
        got = vm_near(r, L / 2, w / 2, a * 1.35)
        rows.append({"h": r["_h_mm"], "n": r["_n"], "got": got,
                     "kt_gross": got / s_gross, "kt_net": got / s_net})
    return {
        "name": "Kirsch - hole in a plate, uniaxial tension",
        "want": want, "want_label": "Kt_net = %.3f (Peterson, d/w = %.3f)"
                                    % (kt_net, dw),
        "extra": "infinite-plate Kt = 3.000 on gross stress; "
                 "gross stress %.2f MPa, net %.2f MPa"
                 % (s_gross / 1e6, s_net / 1e6),
        "rows": rows, "unit": "MPa", "key": "got",
        "report": [("Kt on net section", "kt_net", kt_net),
                   ("Kt on gross section", "kt_gross", 3.0)],
    }


# ===========================================================================
# 2. Cantilever - sigma = Mc/I
# ===========================================================================
def case_cantilever():
    """Bending stress at four stations, not one.

    One station proves nothing: a solver with the wrong span, the wrong
    section or the wrong load can still be right somewhere by accident. Mc/I
    varies linearly along a tip-loaded cantilever, so checking the whole line
    checks the distribution.

    The root is deliberately excluded. A perfectly clamped corner is a
    singularity in elasticity -- the stress there is unbounded and refining the
    mesh makes it grow without limit, which is a property of the model, not of
    the part. See the reentrant-corner study below.
    """
    L_mm, b_mm = 300.0, 50.0
    L, b = L_mm * PPM, b_mm * PPM
    ext = [(0, 0), (L, 0), (L, b), (0, b)]
    P = 800.0
    I = THK_MM * b_mm ** 3 / 12.0 * 1e-12         # m^4
    c = (b_mm / 2) * 1e-3

    stations = (0.3, 0.45, 0.6, 0.75)
    rows = []
    for tgt in LEVELS:
        r = solve(ext, holes=[], tgt=tgt, load_case="cantilever",
                  orientation="horizontal", load=P)
        errs, got0 = [], None
        for fr in stations:
            x_m = (L_mm - fr * L_mm) * 1e-3       # distance from the free end
            want = P * x_m * c / I
            got = vm_near(r, fr * L, b, 0.03 * L)
            errs.append(100.0 * (got - want) / want)
            if fr == 0.45:
                got0, want0 = got, want
        rows.append({"h": r["_h_mm"], "n": r["_n"], "got": got0,
                     "want": want0, "errs": errs})
    return {
        "name": "Cantilever - sigma = Mc/I",
        "want": rows[-1]["want"],
        "want_label": "Mc/I at 45%% of span = %.2f MPa" % (rows[-1]["want"]/1e6),
        "extra": "stations at %s of span; the clamped root is excluded as a "
                 "singularity" % ", ".join("%.0f%%" % (100*s) for s in stations),
        "rows": rows, "unit": "MPa", "key": "got",
        "stations": stations,
    }


# ===========================================================================
# 3. Curved beam - Winkler-Bach
# ===========================================================================
def case_curved():
    """A curved bar in pure bending. The inner fibre runs hot.

    R_n = h / ln(ro/ri) is the neutral-axis radius, which sits INSIDE the
    centroid. That offset e = R_c - R_n is what a straight-beam formula misses,
    and it is small -- 0.67 mm on a 20 mm deep section -- while its effect is
    not: it lifts the inner fibre about 15% above Mc/I and drops the outer one
    about 12% below. A solver that quietly used Mc/I would be unsafe on the
    inside of every hook and link it ever saw, by exactly that margin, and
    nothing on the screen would say so.

    The couple is applied at one end and the other is held. The section read is
    90 degrees from both, which is several section depths away -- far enough
    that how the couple was distributed over the end face stops mattering
    (Saint-Venant), which is what makes this a test of the curvature and not of
    the end condition.
    """
    ri_mm, ro_mm = 40.0, 60.0
    h_mm = ro_mm - ri_mm
    Rn = h_mm / math.log(ro_mm / ri_mm)
    Rc = 0.5 * (ri_mm + ro_mm)
    e = Rc - Rn
    A = h_mm * THK_MM
    M = 50.0                                      # N*m
    M_nmm = M * 1000.0
    want_i = M_nmm * (Rn - ri_mm) / (A * e * ri_mm) * 1e6      # Pa
    want_o = abs(M_nmm * (Rn - ro_mm) / (A * e * ro_mm)) * 1e6
    straight = M_nmm * (h_mm / 2) / (THK_MM * h_mm ** 3 / 12.0) * 1e6

    # Half annulus, with a straight leg on each end.
    #
    # The legs are not decoration. Without them the end faces sit directly
    # against the curved edges, and the disc that selects an end face to load
    # also catches a slice of the inner and outer curves -- so the "couple"
    # arrives partly as traction on a curved surface, which is not a couple at
    # all. The first run of this benchmark did exactly that and read 67% high
    # and FALLING with refinement, which is the signature of a rig that is
    # wrong rather than a solver that is imprecise: a converging solver walks
    # towards an answer, it does not walk away from one.
    #
    # A leg of 1.5 section depths puts the loaded face that far from anything
    # curved, so the disc catches the face and nothing else, and it is also
    # the Saint-Venant distance that makes the crown indifferent to how the
    # couple was distributed.
    leg = 1.5 * h_mm * PPM
    cx, cy = ro_mm * PPM + 10, leg + 10.0
    ang = np.linspace(0, math.pi, 200)
    outer = [(cx + ro_mm * PPM * math.cos(t), cy + ro_mm * PPM * math.sin(t))
             for t in ang]
    inner = [(cx + ri_mm * PPM * math.cos(t), cy + ri_mm * PPM * math.sin(t))
             for t in ang[::-1]]
    ext = ([(cx + ro_mm * PPM, cy - leg)] + outer
           + [(cx - ro_mm * PPM, cy - leg), (cx - ri_mm * PPM, cy - leg)]
           + inner + [(cx + ri_mm * PPM, cy - leg)])

    end_r = (cx + Rc * PPM, cy - leg)
    end_l = (cx - Rc * PPM, cy - leg)
    cap = (h_mm * PPM) * 0.60

    rows = []
    for tgt in LEVELS:
        picks = [{"role": "fixed", "x": end_l[0], "y": end_l[1], "r": cap},
                 {"role": "load", "x": end_r[0], "y": end_r[1], "r": cap,
                  "torque": M}]
        r = solve(ext, holes=[], tgt=tgt, load_case="cantilever",
                  orientation="horizontal", load=0.0, picks=picks)
        # Crown of the arch, 90 degrees from both ends.
        gi = vm_near(r, cx, cy + ri_mm * PPM, 0.10 * h_mm * PPM)
        go = vm_near(r, cx, cy + ro_mm * PPM, 0.10 * h_mm * PPM)
        rows.append({"h": r["_h_mm"], "n": r["_n"], "got": gi, "outer": go})
    return {
        "name": "Curved beam - Winkler-Bach",
        "want": want_i,
        "want_label": "inner fibre %.2f MPa (outer %.2f); straight-beam Mc/I "
                      "would say %.2f" % (want_i/1e6, want_o/1e6, straight/1e6),
        "extra": "ri %.0f, ro %.0f mm; neutral axis at %.3f mm, centroid at "
                 "%.1f, offset e = %.3f mm" % (ri_mm, ro_mm, Rn, Rc, e),
        "rows": rows, "unit": "MPa", "key": "got",
        "second": ("outer fibre", "outer", want_o),
    }


# ===========================================================================
# 4. Lame - thick cylinder under internal pressure
# ===========================================================================
def case_lame():
    """A ring with pressure inside and nothing holding it.

    sigma_theta(r) = p ri^2/(ro^2-ri^2) (1 + ro^2/r^2)
    sigma_r(r)     = p ri^2/(ro^2-ri^2) (1 - ro^2/r^2)

    Two things are being tested at once. The first is the gradient: hoop stress
    falls by nearly half across the wall, and a solver that reported a single
    "ring stress" would be wrong everywhere except one radius.

    The second is subtler. This load is self-equilibrated -- the pressure
    pushes outward all the way round and the forces cancel exactly -- so there
    is no reaction anywhere and the stiffness matrix is singular by three rigid
    body modes. The part is held by three PINS, each constraining one
    displacement component at one node, placed where that component is not the
    direction the ring wants to grow. If the pins were taking any load, the
    numbers below would not match the closed form; that they do is the proof
    that they are not.
    """
    ri_mm, ro_mm = 25.0, 50.0
    p = 40e6                                      # Pa, internal
    k = p * ri_mm ** 2 / (ro_mm ** 2 - ri_mm ** 2)

    def hoop(r):
        return k * (1 + ro_mm ** 2 / r ** 2)

    def rad(r):
        return k * (1 - ro_mm ** 2 / r ** 2)

    def vm(r):
        a, b = hoop(r), rad(r)
        return math.sqrt(a * a - a * b + b * b)

    cx = cy = ro_mm * PPM + 10
    ext = ring(cx, cy, ro_mm * PPM, 220)
    holes = [ring(cx, cy, ri_mm * PPM, 180)]

    # Three pins: y at the two ends of a horizontal diameter kills drift in y
    # and the spin; x at the top kills drift in x. None of the three resists
    # the radial growth that pressure causes, which is the whole point.
    pins = [{"role": "pin", "x": cx + ro_mm * PPM, "y": cy, "r": 1, "dofs": "y"},
            {"role": "pin", "x": cx - ro_mm * PPM, "y": cy, "r": 1, "dofs": "y"},
            {"role": "pin", "x": cx, "y": cy + ro_mm * PPM, "r": 1, "dofs": "x"}]
    load = [{"role": "load", "x": cx, "y": cy, "r": ri_mm * PPM * 1.08,
             "pressure": p}]

    radii = (ri_mm * 1.04, 0.5 * (ri_mm + ro_mm), ro_mm * 0.96)
    rows = []
    for tgt in LEVELS:
        r = solve(ext, holes, tgt, load_case="cantilever",
                  orientation="horizontal", load=0.0, picks=pins + load)
        # Read on a thin ring at each radius, not in a disc around a point:
        # the hoop stress halves across this wall and a disc samples the
        # gradient instead of the station. See vm_at_radius.
        prof = [vm_at_radius(r, cx, cy, rr) for rr in radii]
        rows.append({"h": r["_h_mm"], "n": r["_n"], "got": prof[0],
                     "prof": prof})
    return {
        "name": "Lame - thick cylinder, internal pressure",
        "want": vm(radii[0]),
        "want_label": "von Mises at r = %.1f mm is %.2f MPa"
                      % (radii[0], vm(radii[0]) / 1e6),
        "extra": "ri %.0f, ro %.0f mm, p = %.0f MPa; hoop runs %.1f MPa at the "
                 "bore to %.1f at the outside"
                 % (ri_mm, ro_mm, p / 1e6, hoop(ri_mm) / 1e6, hoop(ro_mm) / 1e6),
        "rows": rows, "unit": "MPa", "key": "got",
        "profile": [(rr, vm(rr)) for rr in radii],
    }


# ===========================================================================
# 5. The reentrant corner - a singularity, demonstrated rather than hidden
# ===========================================================================
def case_corner():
    """An L with a sharp inside corner. The peak should NOT converge.

    This one has no right answer, and that is the point. Linear elasticity at a
    sharp reentrant corner gives stress that goes as r^(lambda-1) with lambda <
    1, which is unbounded at the corner itself. Every refinement finds a bigger
    number. A tool that reported that number as "the stress" would get
    arbitrarily more alarming the harder it worked, and would be wrong every
    time.

    Two honest readings are printed beside it: the stress at a fixed small
    distance from the corner, which does converge, and what the same corner
    reads with a fillet in it. Reporting either is defensible. Reporting the
    node value is not.
    """
    a = 120.0 * PPM
    t = 36.0 * PPM
    ext = [(0, 0), (a, 0), (a, t), (t, t), (t, a), (0, a)]
    P = 600.0
    corner = (t, t)
    rows = []
    for tgt in LEVELS:
        r = solve(ext, holes=[], tgt=tgt, load_case="cantilever",
                  orientation="horizontal", load=P)
        at_node = vm_near(r, corner[0], corner[1], 0.012 * a)
        # Along the bisector INTO the material. The first version of this
        # stepped away from the corner in +x and +y, which on an L is the
        # empty quadrant -- it sampled no nodes at all and read NaN, and the
        # table printed "nan%" next to a confident sentence about convergence.
        at_dist = vm_near(r, corner[0] - 0.07 * a, corner[1] - 0.07 * a,
                          0.025 * a)
        rows.append({"h": r["_h_mm"], "n": r["_n"], "got": at_node,
                     "at_dist": at_dist, "peak_rep": r["peak_vm"],
                     "peak_raw": r["peak_vm_raw"]})
    return {"name": "Reentrant corner - singular by construction",
            "rows": rows, "singular": True}


# ===========================================================================
CASES = {"kirsch": case_kirsch, "cantilever": case_cantilever,
         "curved": case_curved, "lame": case_lame, "corner": case_corner}


def plot(case, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = case["rows"]
    h = [r["h"] for r in rows]
    g = [r["got"] / 1e6 for r in rows]
    fig, ax = plt.subplots(figsize=(6.0, 3.8), dpi=140)
    ax.plot(h, g, "o-", color="#d97706", label="CADAI", zorder=3)
    if not case.get("singular"):
        ax.axhline(case["want"] / 1e6, color="#2563eb", ls="--",
                   label="closed form")
    ax.set_xlabel("mean element edge (mm)")
    ax.set_ylabel("stress (MPa)")
    ax.set_title(case["name"], fontsize=9)
    ax.invert_xaxis()
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(prog="validate")
    ap.add_argument("--case", default="all")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    names = list(CASES) if args.case == "all" else [args.case]
    out = ["# CADAI verification against closed-form solutions", "",
           "Material E = %.0f GPa, nu = %.2f, plate thickness %.2f mm. Each "
           "case is run at four mesh densities so the trend is visible, not "
           "just the error." % (E / 1e9, NU, THK_MM), ""]
    print()
    worst = []
    for n in names:
        case = CASES[n]()
        print("=" * 74)
        print(case["name"])
        out.append("## " + case["name"])
        out.append("")
        if case.get("extra"):
            print("  " + case["extra"])
            out.append(case["extra"] + "")
            out.append("")
        if case.get("singular"):
            print("  %-9s %8s %12s %12s %12s" %
                  ("edge mm", "elems", "at corner", "7% away", "CADAI peak"))
            out.append("| edge (mm) | elements | at the corner | 7% away | "
                       "CADAI's reported peak |")
            out.append("|---|---|---|---|---|")
            for r in case["rows"]:
                print("  %-9.3f %8d %12.1f %12.1f %12.1f"
                      % (r["h"], r["n"], r["got"] / 1e6, r["at_dist"] / 1e6,
                         r["peak_rep"] / 1e6))
                out.append("| %.3f | %d | %.1f | %.1f | %.1f |"
                           % (r["h"], r["n"], r["got"] / 1e6,
                              r["at_dist"] / 1e6, r["peak_rep"] / 1e6))
            first, last = case["rows"][0], case["rows"][-1]
            grow = 100.0 * (last["got"] - first["got"]) / first["got"]
            steady = 100.0 * (last["at_dist"] - first["at_dist"]) / first["at_dist"]
            print("\n  corner node rose %+.0f%% over the refinement - it does "
                  "not converge, and should not" % grow)
            print("  the reading 7%% away moved %+.1f%% - that one is a number"
                  % steady)
            out += ["", "The corner node rose **%+.0f%%** across the "
                        "refinement and will keep rising: a sharp reentrant "
                        "corner is a singularity in linear elasticity, so the "
                        "node value is a property of the mesh, not the part. "
                        "The reading a fixed distance away moved %+.1f%% and "
                        "is a number you can quote." % (grow, steady), ""]
            continue

        print("  closed form: " + case["want_label"])
        out.append("**Closed form:** " + case["want_label"])
        out.append("")
        print()
        print("  %-9s %8s %12s %12s %9s" %
              ("edge mm", "elems", "CADAI", "closed form", "error"))
        out.append("| edge (mm) | elements | CADAI | closed form | error |")
        out.append("|---|---|---|---|---|")
        for r in case["rows"]:
            want = r.get("want", case["want"])
            err = 100.0 * (r["got"] - want) / want
            print("  %-9.3f %8d %9.2f %s %9.2f %s %+8.1f%%"
                  % (r["h"], r["n"], r["got"] / 1e6, case["unit"],
                     want / 1e6, case["unit"], err))
            out.append("| %.3f | %d | %.2f %s | %.2f %s | %+.1f%% |"
                       % (r["h"], r["n"], r["got"] / 1e6, case["unit"],
                          want / 1e6, case["unit"], err))
        fin = case["rows"][-1]
        want = fin.get("want", case["want"])
        err = 100.0 * (fin["got"] - want) / want
        worst.append((case["name"], err))
        out.append("")

        if case.get("report"):
            print()
            for label, key, target in case["report"]:
                v = fin[key]
                print("  %-22s %7.3f   (theory %.3f, %+.1f%%)"
                      % (label, v, target, 100.0 * (v - target) / target))
                out.append("- %s: **%.3f** against %.3f (%+.1f%%)"
                           % (label, v, target, 100.0 * (v - target) / target))
            out.append("")
        if case.get("second"):
            label, key, target = case["second"]
            v = fin[key]
            print("  %-22s %7.2f MPa (theory %.2f, %+.1f%%)"
                  % (label, v / 1e6, target / 1e6,
                     100.0 * (v - target) / target))
            out.append("- %s: **%.2f MPa** against %.2f (%+.1f%%)"
                       % (label, v / 1e6, target / 1e6,
                          100.0 * (v - target) / target))
            out.append("")
        if case.get("profile"):
            print()
            print("  through the wall, finest mesh:")
            out.append("Through the wall, finest mesh:")
            out.append("")
            out.append("| radius (mm) | CADAI | Lame |")
            out.append("|---|---|---|")
            for (rr, wv), gv in zip(case["profile"], fin["prof"]):
                print("    r = %5.1f mm   CADAI %7.2f   Lame %7.2f  (%+.1f%%)"
                      % (rr, gv / 1e6, wv / 1e6, 100.0 * (gv - wv) / wv))
                out.append("| %.1f | %.2f MPa | %.2f MPa |"
                           % (rr, gv / 1e6, wv / 1e6))
            out.append("")
        if case.get("stations"):
            print()
            print("  along the span, finest mesh: %s"
                  % ", ".join("%+.1f%%" % e for e in fin["errs"]))
            out.append("Error at %s of span: %s"
                       % (", ".join("%.0f%%" % (100 * s)
                                    for s in case["stations"]),
                          ", ".join("%+.1f%%" % e for e in fin["errs"])))
            out.append("")

        if not args.no_plots:
            png = os.path.join(ROOT, "validation_%s.png" % n)
            try:
                plot(case, png)
                out.append("![convergence](%s)" % os.path.basename(png))
                out.append("")
            except Exception as ex:
                print("  (no plot: %s)" % ex)
        print()

    if worst:
        print("=" * 74)
        print("  SUMMARY, finest mesh")
        out.append("## Summary, finest mesh")
        out.append("")
        for name, err in worst:
            print("  %-46.46s %+7.1f%%" % (name, err))
            out.append("- %s: **%+.1f%%**" % (name, err))
        out.append("")

    dest = os.path.join(ROOT, "validation_report.md")
    with open(dest, "w") as fh:
        fh.write("\n".join(out))
    print("\n  wrote %s" % os.path.relpath(dest, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
