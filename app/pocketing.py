"""
Adaptive pocketing engine.

The old version carved one fixed pattern — a +/-45 degree diamond lattice locked
to the *image* axes — into every part, which is why a pentagonal intake plate
came out looking exactly like a square bellypan. This version does what a person
at a whiteboard does: it looks at the plate first, decides what kind of part it
is, and then picks a rib archetype that suits it, oriented to the part's own
axes rather than the pixel grid.

Underneath, there is only ONE pattern generator: a node web. Nodes are placed in
and around a pocket region, the node set is triangulated, and the ribs are the
triangulation's edges. Every rib therefore ends on a node, every node is a
junction of three or more ribs, and every bay is a triangle -- which is also the
only planar bay that cannot shear. Look at real pocketed 6061 plates and this is
what they are: a gusset is three nodes, a bellypan rail is a row of nodes down
two chords, an arm side plate is a node at every bearing plus a field of them
between. The archetypes below differ only in where the nodes go.

  truss   Long slender beams and arms. Nodes zig-zag between the two long
          edges at one bay spacing; triangulating a zig-zag *is* a Warren
          truss, with the two continuous chords falling out of it.
  waffle  Large flat panels (bellypans, electronics trays). A hexagonally
          packed lattice on the panel's own principal axes, so every rib in
          the field is roughly the same length.
  radial  Plates organised around bores (shooter side plates, gearbox plates).
          Every bore centre is forced into the node set, so ribs radiate from
          the bearings and, where two bores can see each other, one edge runs
          straight between them. Load enters and leaves a plate like this
          through its bearings; the triangulation puts the ribs on those paths
          without a second, separate spoke pattern crossing the first.
  xbrace  Small gussets and brackets. A single X across the region: below about
          a dozen rib-widths there isn't room for a lattice, and one brace beats
          a chopped-up grid.

Selection happens twice. Once for the whole part, from slenderness, area and
bore layout; then again per pocket region, because a long finger hanging off an
otherwise square plate still wants a truss.

Bay size scales with the PART, not just with the rib. A big plate given small
bays reads as a spider web rather than a machined part -- the reference 6061
pieces all have a handful of large pockets no matter how big they get, because
a pocket costs a tool path and a rib costs weight. So the cell floor is tied to
the part's own span, and the coverage loop then trims it to hit the removal
target.

The FEM field reaches this module three ways: it sizes each bore's boss collar
by that bore's own rim stress, it doubles the lattice where the solver named a
weak point, and it opens the bays out where the solver named a strong one.

Calibration. None of the above is learned -- it is geometry with constants, and
the constants were picked by eye off parts in the 200-400 mm range. `CAL_DEFAULTS`
gathers the dozen that a real part can disagree with, and `calibration()` lets
data/pocket_cal.json override them. tools/pocket_ref.py measures reference parts
you have actually machined and fits that file. Delete the file and the numbers
below are exactly what runs again.
"""
from __future__ import annotations
import os
import json
import numpy as np
import cv2
from scipy import ndimage as ndi
from PIL import Image


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _disk(r):
    r = int(max(1, r))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _rim_stress(hole_mask, field, pad):
    """Peak stress in the ring of material just outside every bore, at once.

    The obvious implementation — dilate each hole, read the field in the ring —
    costs one dilation per hole, and a bellypan has two hundred of them. One
    distance transform with `return_indices` gives, for every pixel of stock,
    which bore is nearest; the ring of each bore is then just a labelled
    selection and `ndi.maximum` reads all of them in a single pass.

    Neighbouring bores are excluded from the ring by construction: a pixel
    inside another hole is not stock, so it never enters the selection. On a
    tight bolt circle that matters — otherwise "rim stress" is partly a value
    sampled from empty space.
    """
    lab, n = ndi.label(hole_mask)
    if n == 0:
        return lab, 0, {}
    try:
        dist, (iy, ix) = ndi.distance_transform_edt(~hole_mask,
                                                    return_indices=True)
        ring = (~hole_mask) & (dist <= float(pad))
        if not ring.any():
            return lab, n, {}
        owner = lab[iy, ix][ring]
        vals = ndi.maximum(np.asarray(field)[ring], owner,
                           index=np.arange(1, n + 1))
        vals = np.nan_to_num(np.atleast_1d(np.asarray(vals, float)))
    except Exception:
        return lab, n, {}
    return lab, n, {i + 1: float(np.clip(vals[i], 0.0, 1.0))
                    for i in range(min(n, vals.size))}


def _hole_bosses(hole_mask, body, rib_px, field=None):
    """A solid ring around each hole, sized by the hole AND by its rim stress.

    A bolt needs a boss to clamp against and a bearing needs a seat, but on a
    real 6061 plate that boss is a thin collar, not a no-go island: the pocket
    pattern runs right up to it. The collar is sized off the hole's own radius
    (about a third of a radius, floored just under one rib width and capped at
    two and a half) so a #10 clearance hole gets a modest ring and a 2 in
    bearing bore gets a slightly larger seat, without either being tied to a
    fixed mm value. The cap matters: at half a radius a 20 mm bearing bore grew
    a 10 mm collar, and a plate with two bores and a bolt circle lost half its
    face to collars before a single rib was drawn.

    The stress term is the first place the FEM field reaches the pocket plan
    hole by hole. Two bores of identical diameter do not deserve identical
    collars if one sits on the load path and the other is a lightening hole in
    dead stock: the loaded one gets up to ~1.6x the ring, the idle one shrinks
    to about 0.75x. That is a real design move -- it is how a machinist decides
    which bores get a beefed-up boss -- and it makes "the stress points
    influence the pocketing" true per feature rather than only in aggregate.

    Holes are bucketed by the collar width they need, so a bellypan with two
    hundred fastener holes still costs a handful of dilations.
    """
    out = np.zeros_like(body)
    if not hole_mask.any():
        return out
    pad = int(max(2, round(rib_px)))
    if field is None:
        (lab, n), rim = ndi.label(hole_mask), {}
    else:
        lab, n, rim = _rim_stress(hole_mask, field, pad)
    if n == 0:
        return out
    boxes = ndi.find_objects(lab)
    buckets = {}
    for i in range(1, n + 1):
        slc = boxes[i - 1]
        if slc is None:
            continue
        a = int((lab[slc] == i).sum())
        if a < 4:
            continue
        r = float(np.sqrt(a / np.pi))
        # 0.75x in dead stock, 1.6x on a fully loaded rim.
        f = 0.75 + 0.85 * float(rim.get(i, 0.35))
        w = int(round(float(np.clip(0.35 * r * f,
                                    0.8 * rib_px, 2.5 * rib_px))))
        buckets.setdefault(w, []).append(i)
    for w, ids in buckets.items():
        out |= ndi.binary_dilation(np.isin(lab, ids), _disk(w))
    return out & body


def _solid_band(field, body, t0, cap):
    """The only material allowed to stay completely untouched — area-capped.

    The instinct is to leave everything above some stress threshold solid, but
    that is not how a pocketed plate is actually made: high stress earns *more
    ribs*, not more metal, and every reference part is lightened end to end.
    Worse, the clamped-edge stress singularity saturates a wide band of the
    display field at 1.0, so a fixed threshold hands back a large solid island
    for numerical reasons rather than physical ones.

    So the threshold is raised in small steps until the island it selects fits
    inside `cap` of the plate. If it never fits, nothing is exempt and the whole
    part gets ribbed.
    """
    tot = int(body.sum()) or 1
    t = float(t0)
    while t < 0.999:
        m = body & (field >= t)
        if int(m.sum()) <= cap * tot:
            # Drop confetti: a scatter of tiny hot specks is mesh noise, and
            # rendering each as a "do not machine" island is unreadable.
            lab, n = ndi.label(m)
            if n:
                small = max(40, int(0.004 * tot))
                sz = ndi.sum(np.ones_like(lab), lab, index=range(1, n + 1))
                for i in np.nonzero(np.asarray(sz) < small)[0] + 1:
                    m[lab == i] = False
            return m
        t += 0.02
    return np.zeros_like(body)


def _frame(mask):
    """Principal frame of a boolean mask.

    Returns (center_xy, u_hat, v_hat, Lu, Lv) where u is the long axis and
    Lu/Lv are the full extents along each axis, in pixels. Everything
    downstream is written in this frame, which is what makes a pattern follow
    the part instead of the screen.
    """
    ys, xs = np.nonzero(mask)
    if xs.size < 3:
        return np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0]), 1.0, 1.0
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    c = pts.mean(axis=0)
    d = pts - c
    cov = (d.T @ d) / len(pts)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    u = V[:, order[0]]
    v = V[:, order[1]]
    Lu = float(np.ptp(d @ u))
    Lv = float(np.ptp(d @ v))
    return c, u, v, max(Lu, 1.0), max(Lv, 1.0)


def _uv_grid(shape, center, u, v):
    """u/v coordinate arrays over a sub-window, measured from `center`."""
    h, w = shape
    Y, X = np.mgrid[0:h, 0:w]
    dx = X - center[0]
    dy = Y - center[1]
    return dx * u[0] + dy * u[1], dx * v[0] + dy * v[1]


# --------------------------------------------------------------------------
# rib archetypes -- each returns a boolean "rib material" mask
# --------------------------------------------------------------------------
def _ribs_xbrace(U, V, rib_px, Lu, Lv):
    """The two diagonals of the region's oriented bounding box."""
    n = np.hypot(Lu, Lv) or 1.0
    half = rib_px * 0.62      # slightly beefy: this is the only structure here
    d1 = np.abs(V * Lu - U * Lv) / n
    d2 = np.abs(V * Lu + U * Lv) / n
    return (d1 < half) | (d2 < half)


def _bores(hole_mask, min_r_px):
    """Hole blobs big enough to be structural bores (bearing/gearbox), not
    fastener clearance. Returns [(cx, cy, r_px), ...] largest first."""
    lab, n = ndi.label(hole_mask)
    out = []
    for i in range(1, n + 1):
        m = lab == i
        a = int(m.sum())
        r = float(np.sqrt(a / np.pi))
        if r >= min_r_px:
            ys, xs = np.nonzero(m)
            out.append((float(xs.mean()), float(ys.mean()), r))
    out.sort(key=lambda b: -b[2])
    return out


def _segment_inside(part_mask, hole_mask, p0, p1, samples=32, frac=0.97):
    """True if the straight run p0->p1 stays on the plate (bores count as on
    the plate, since a rib is allowed to terminate at a boss)."""
    t = np.linspace(0.0, 1.0, samples)[:, None]
    pts = np.round(np.array(p0)[None, :] * (1 - t) + np.array(p1)[None, :] * t).astype(int)
    H, W = part_mask.shape
    xs = np.clip(pts[:, 0], 0, W - 1)
    ys = np.clip(pts[:, 1], 0, H - 1)
    ok = part_mask[ys, xs] | hole_mask[ys, xs]
    return bool(ok.mean() > frac)


# --------------------------------------------------------------------------
# the node web -- one triangulator behind every pattern
# --------------------------------------------------------------------------
# Why a triangulation replaced the hand-drawn spoke fan: look at what the
# reference 6061 parts actually are. The gusset is one triangular pocket. The
# bellypan rail is a row of triangles. The arm side plate is a mixed field of
# triangles and trapezoids around its bores. Every one of them is the SAME
# object -- a set of nodes with ribs on the edges between them -- and the only
# thing that differs between the three is where the nodes go.
#
# The old radial routine fanned spokes outward from each bore at a fixed
# angular step and laid a waffle over the gaps. Two independent patterns
# crossing at arbitrary angles is precisely the spider web the plan came out
# looking like: ribs terminating in the middle of a pocket wall, bays with five
# and six sides, and no two pockets the same shape. Triangulating a node set
# cannot produce that. Every rib ends on a node, every node is a junction of
# three or more ribs, and every bay is a triangle -- which is also the only
# planar bay that cannot shear.


def _dedupe(pts, gap):
    """Drop nodes closer together than `gap`; two nodes a rib-width apart make
    a sliver bay that the cutter-radius opening deletes anyway."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    if pts.shape[0] < 2:
        return pts
    keep, g2 = [], float(gap) * float(gap)
    for p in pts:
        if keep:
            k = np.asarray(keep)
            d = (k[:, 0] - p[0]) ** 2 + (k[:, 1] - p[1]) ** 2
            if float(d.min()) < g2:
                continue
        keep.append(p)
    return np.asarray(keep, float)


def _contour_nodes(reg, cell):
    """Nodes around the region outline: every real corner, plus fill-ins.

    Corners are taken first and unconditionally. A rib that stops short of the
    corner of a pocket region leaves a floppy unsupported tab, and the corners
    are exactly where the reference parts put their rib junctions -- the
    triangular gusset is three corner nodes and nothing else.

    The fill-ins are spaced by arc length rather than by vertex index, because
    a rasterised contour has far more vertices per unit length on a diagonal
    than on an axis-aligned run, and index spacing would bunch every node on
    the diagonals.
    """
    try:
        cs, _ = cv2.findContours(np.ascontiguousarray(reg.astype(np.uint8)),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    except Exception:
        return np.zeros((0, 2))
    out = []
    for c_ in cs:
        p = c_.reshape(-1, 2).astype(float)
        if p.shape[0] < 8:
            continue
        corners = cv2.approxPolyDP(c_, max(2.0, cell * 0.22), True)
        out.append(corners.reshape(-1, 2).astype(float))
        seg = np.hypot(*np.diff(np.vstack([p, p[:1]]), axis=0).T)
        d = np.concatenate([[0.0], np.cumsum(seg)])
        L = float(d[-1])
        if L >= cell * 1.5:
            k = max(3, int(round(L / cell)))
            s = np.linspace(0.0, L, k, endpoint=False)
            out.append(np.stack([np.interp(s, d, np.r_[p[:, 0], p[0, 0]]),
                                 np.interp(s, d, np.r_[p[:, 1], p[0, 1]])], 1))
    if not out:
        return np.zeros((0, 2))
    return np.vstack(out)


def _lattice_nodes(reg, c, u, v, Lu, Lv, cell, pattern, rib_px):
    """Interior nodes. The pattern lives here and nowhere else.

    truss   A zig-zag: nodes alternating near the two long edges at one bay
            spacing. Triangulating a zig-zag *is* a Warren truss -- the
            alternating diagonals and the two continuous chords fall out of it,
            which is why the bellypan rail in the reference photos looks the
            way it does.
    else    A triangular (hexagonal-packed) lattice on the region's own axes.
            Rows offset by half a cell and spaced by cell*sqrt(3)/2 give
            near-equilateral triangles, which is the shape that keeps every rib
            in the field roughly the same length -- the arm side plate again.
    """
    if pattern == "truss":
        bay = float(np.clip(Lv * 0.95, rib_px * 5.0, max(Lu * 0.5, rib_px * 5.0)))
        n = max(2, int(round(Lu / bay)))
        us = np.linspace(-Lu * 0.5, Lu * 0.5, n + 1)
        vs = np.where(np.arange(n + 1) % 2 == 0, -Lv * 0.40, Lv * 0.40)
    else:
        step = max(float(cell), rib_px * 3.0)
        rows = np.arange(-Lv * 0.5, Lv * 0.5 + 1e-6, step * np.sqrt(3.0) / 2.0)
        us, vs = [], []
        for k, vv in enumerate(rows):
            off = 0.0 if k % 2 == 0 else step * 0.5
            uu = np.arange(-Lu * 0.5 + off, Lu * 0.5 + 1e-6, step)
            us.append(uu)
            vs.append(np.full(uu.shape, vv))
        if not us:
            return np.zeros((0, 2))
        us, vs = np.concatenate(us), np.concatenate(vs)
    xy = np.stack([c[0] + us * u[0] + vs * v[0],
                   c[1] + us * u[1] + vs * v[1]], 1)
    # Keep only nodes standing in real material. A node in the rib wall or off
    # the region drags edges through stock that is not ours to cut.
    h, w = reg.shape
    xi = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    return xy[reg[yi, xi]]


def _web_edges(nodes, max_len):
    """Delaunay edges shorter than `max_len`, as index pairs.

    The length cap is what stops the triangulation from bridging a concavity:
    Delaunay always fills the convex hull, so a C-shaped region gets a few very
    long edges straight across the mouth of the C. Those are the only edges
    that are not local, so a cap at a little over one cell removes them without
    touching anything real.
    """
    if nodes.shape[0] < 3:
        return []
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(nodes)
    except Exception:
        return []
    e = set()
    for s in tri.simplices:
        for a, b in ((s[0], s[1]), (s[1], s[2]), (s[2], s[0])):
            e.add((a, b) if a < b else (b, a))
    m2 = float(max_len) ** 2
    out = []
    for a, b in e:
        d = nodes[a] - nodes[b]
        if float(d[0] * d[0] + d[1] * d[1]) <= m2:
            out.append((a, b))
    return out


def _trim(reg, p0, p1, n=28):
    """Pull both endpoints in to the first sample standing in material.

    A bore centre is a node but is never *in* the pocket region -- it sits in
    its own boss collar, which was subtracted before any of this ran. Tested
    end to end, every rib radiating from a bearing therefore starts with a boss
    radius of non-material and fails the inside test, which is exactly the way
    a plate loses the ribs it most needs. Trimming the ends first asks the
    right question: does the rib hold up over the part of its run that is
    actually ours to cut?
    """
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    t = np.linspace(0.0, 1.0, int(n))
    xs, ys = p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t
    h, w = reg.shape
    ok = reg[np.clip(np.round(ys).astype(int), 0, h - 1),
             np.clip(np.round(xs).astype(int), 0, w - 1)]
    if not ok.any():
        return None
    i0 = int(np.argmax(ok))
    i1 = int(len(ok) - 1 - np.argmax(ok[::-1]))
    if i1 <= i0:
        return None
    return np.array([xs[i0], ys[i0]]), np.array([xs[i1], ys[i1]])


def _draw_web(shape, nodes, edges, reg, holes_sub, rib_px, thick_f=1.0,
              wall_dt=None):
    """Paint ribs on the edges that stay in material.

    `wall_dt` is the region's distance-to-boundary map, and it exists to throw
    away one specific family of edges. Contour nodes sit on the region outline,
    so the triangulation always connects each to its neighbours -- and that
    chain of edges paints a continuous rib around the whole region, one rib
    width inside the perimeter wall that is already there. It is invisible in a
    node diagram and enormous on a big plate: on a 660x550 region it was
    eighteen per cent of the area, a redundant second wall that no reference
    part has. Real pocketed 6061 opens its bays straight out to the perimeter.
    So an edge whose run hugs the boundary is dropped; the diagonals from the
    same nodes, which do real work, are kept.
    """
    canvas = np.zeros(shape, np.uint8)
    thick = max(2, int(round(rib_px * thick_f)))
    hug = None
    if wall_dt is not None and float(wall_dt.max()) > 2.5 * rib_px:
        hug = rib_px * 1.15
    h, w = shape
    for a, b in edges:
        seg = _trim(reg, nodes[a], nodes[b])
        if seg is None:
            continue
        p0, p1 = seg
        if hug is not None:
            # Sample the middle of the run, not the endpoints: the endpoints of
            # a contour-to-contour edge are ON the boundary whatever the edge
            # does, so they cannot tell a wall-hugger from a chord.
            t = np.linspace(0.25, 0.75, 7)
            mx = np.clip(np.round(p0[0] + (p1[0] - p0[0]) * t).astype(int), 0, w - 1)
            my = np.clip(np.round(p0[1] + (p1[1] - p0[1]) * t).astype(int), 0, h - 1)
            if float(wall_dt[my, mx].max()) < hug:
                continue
        # 0.88 rather than 0.97: an edge that clips a bore or nicks the rib
        # wall for a few pixels is still a rib a machinist would draw, and the
        # canvas is masked to the region afterwards anyway.
        if not _segment_inside(reg, holes_sub, p0, p1, samples=24, frac=0.88):
            continue
        cv2.line(canvas, (int(round(p0[0])), int(round(p0[1]))),
                 (int(round(p1[0])), int(round(p1[1]))), 1, thick, cv2.LINE_AA)
    return canvas.astype(bool)


def _subdivide(nodes, edges):
    """Nodes plus every edge midpoint — the dense tier's node set.

    Retriangulating this splits each triangle into four similar ones, so every
    base rib survives as two collinear halves of itself. That is what keeps the
    dense lattice a strict SUPERSET of the base one: ribs run straight through
    the hot/cold boundary instead of stopping dead on it, which is both how a
    real part is drawn and the difference between a stiffener and a crack
    starter.
    """
    if not edges:
        return nodes
    mids = np.asarray([(nodes[a] + nodes[b]) * 0.5 for a, b in edges], float)
    return np.vstack([nodes, mids]) if mids.size else nodes


def _ribs_web(reg, holes_sub, c, u, v, Lu, Lv, cell, rib_px, pattern,
              bore_pts=()):
    """The rib mask for one region, base tier and dense tier.

    `bore_pts` are bore centres in this region's window. They are forced into
    the node set so that ribs radiate from the bores and, where two bores can
    see each other, one edge runs straight between them -- the bore-to-bore
    load path the old radial routine drew by hand, now falling out of the
    triangulation for free and joined to the rest of the pattern at nodes
    instead of crossing it.
    """
    gap = max(rib_px * 2.2, cell * 0.34)
    parts = [_lattice_nodes(reg, c, u, v, Lu, Lv, cell, pattern, rib_px),
             _contour_nodes(reg, cell * 1.15)]
    if len(bore_pts):
        parts.insert(0, np.asarray(bore_pts, float).reshape(-1, 2))
    nodes = np.vstack([p for p in parts if p.size]) if any(p.size for p in parts) \
        else np.zeros((0, 2))
    nodes = _dedupe(nodes, gap)
    if nodes.shape[0] < 3:
        return None, None
    max_len = (2.2 * max(cell, rib_px * 3.0) if pattern != "truss"
               else 1.8 * max(Lv, cell))
    edges = _web_edges(nodes, max_len)
    if not edges:
        return None, None
    wall_dt = cv2.distanceTransform(np.ascontiguousarray(reg.astype(np.uint8)),
                                    cv2.DIST_L2, 3)
    base = _draw_web(reg.shape, nodes, edges, reg, holes_sub, rib_px,
                     wall_dt=wall_dt)
    dn = _dedupe(_subdivide(nodes, edges), gap * 0.45)
    de = _web_edges(dn, max_len * 0.62)
    dense = _draw_web(reg.shape, dn, de, reg, holes_sub, rib_px, thick_f=1.1,
                      wall_dt=wall_dt) if de else base
    return base, (dense | base)


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------
def classify_part(part_mask, hole_mask, px_per_mm, cal=None):
    """Pick the archetype for the whole plate. Returns (name, descriptors)."""
    _c = calibration() if cal is None else cal
    c, u, v, Lu, Lv = _frame(part_mask)
    mm = 1.0 / max(px_per_mm, 1e-6)
    L_mm, W_mm = Lu * mm, Lv * mm
    area_mm2 = float(part_mask.sum()) * mm * mm
    slender = L_mm / max(W_mm, 1e-6)
    # solidity: how much of the oriented bbox the part actually fills. A low
    # value means a branchy/organic outline (arms, forks), which reads as truss.
    solidity = area_mm2 / max(L_mm * W_mm, 1e-6)
    bores = _bores(hole_mask, min_r_px=max(2.0, 5.0 * px_per_mm))

    if len(bores) >= 2 and slender < 2.6 and area_mm2 > _c["radial_area_mm2"]:
        name = "radial"
    elif slender >= 3.0 or (slender >= 2.2 and solidity < 0.62):
        name = "truss"
    elif area_mm2 < _c["xbrace_area_mm2"]:
        name = "xbrace"
    elif area_mm2 >= _c["waffle_area_mm2"] and slender < 1.9:
        name = "waffle"
    elif slender >= 1.9:
        name = "truss"
    else:
        name = "waffle"

    return name, {
        "length_mm": round(L_mm, 1),
        "width_mm": round(W_mm, 1),
        "area_mm2": round(area_mm2, 1),
        "slenderness": round(slender, 2),
        "solidity": round(solidity, 3),
        "n_bores": len(bores),
    }, bores


def _region_pattern(part_pattern, area, Lu, Lv, rib_px, has_radial):
    """Refine the archetype for one pocket region."""
    slend = Lu / max(Lv, 1.0)
    if area < (7.0 * rib_px) ** 2 or Lv < 5.0 * rib_px:
        # Too small to subdivide, or the web is already narrower than a few rib
        # widths — chopping it into bays would leave nothing but slivers, so
        # this stays one continuous pocket.
        return "none"
    if slend >= 2.6:
        return "truss"                     # a slender pocket always wants a truss
    if has_radial:
        return "radial"
    if area < (13.0 * rib_px) ** 2:
        return "xbrace"
    return part_pattern if part_pattern in ("waffle", "truss") else "waffle"


# --------------------------------------------------------------------------
# calibration — the numbers in this file that reference parts get a vote on
# --------------------------------------------------------------------------
# Everything the engine decides comes from geometry, so there is no model here
# to train. There are, however, about a dozen constants that were chosen by
# looking at photographs of 6061 parts and picking a number that looked right,
# and "looked right" was judged mostly on parts around 200-400 mm. Those are
# the numbers below, pulled out of the code and given names so that
# tools/pocket_ref.py can measure real parts, fit them, and write the result to
# data/pocket_cal.json.
#
# Deleting that file restores exactly these values, which is the point of
# keeping the defaults here rather than in the file: an uncalibrated install and
# a reverted one are the same install.
CAL_DEFAULTS = {
    # Bay size. `cell` is the lattice spacing, and it is the largest of three
    # floors: a multiple of the rib width, a fraction of the pocket region's
    # short axis, and a fraction of the WHOLE PART's span. The third one is the
    # large-part term -- without it bay COUNT stays fixed and bay LENGTH grows,
    # which is what turns a 600 mm bellypan into a spider web.
    "cell_rib_f": 14.0,       # cell >= this many rib widths
    "cell_short_f": 0.3125,   # cell >= this fraction of the region's short axis
    "cell_span_f": 0.26,      # cell >= this fraction of the part's span
    # How hard stock thickness pulls on how much of the plate comes out. See
    # THICK_REF_MM below and the long note at the target band in generate().
    #
    # 0.5 is a measurement, rounded. Ten of team 2813's parts, seven on 1/4" and
    # three on 1/8", regress to removal ~ t^0.45 (standard error 0.23) once span
    # is controlled for. 0.45 and 0.50 are indistinguishable at that error bar,
    # and 0.50 has the tidier reading -- removal goes as the square root of the
    # stock -- so the default is the round number and this comment is the honest
    # version of where it came from.
    #
    # It was 1.0 for one afternoon, on the theory that rib buckling makes bay
    # size go linearly with thickness. The parts disagree; see generate(). Note
    # what 1.0 would have cost: at 1/8" it asks for HALF the removal of 1/4"
    # stock, where the parts want 0.71 of it. That is not a small overshoot on a
    # part that is already thin.
    "thick_exp": 0.5,
    # Removal band the coverage loop drives towards, for density="normal".
    # The aggressive/conservative presets shift by the same amount, so they keep
    # meaning "more than my default" / "less than my default".
    "target_lo": 0.40,
    "target_hi": 0.60,
    "cell_f": 1.0,            # global multiplier on the density preset's cell_f
    # Smallest pocket worth cutting, as a fraction of part area.
    "min_area_frac": 0.0040,
    # classify_part's archetype cut lines, in mm^2. These are absolute areas,
    # which makes them the constants most likely to be wrong on a part far from
    # the size they were picked at.
    "radial_area_mm2": 3000.0,
    "xbrace_area_mm2": 6000.0,
    "waffle_area_mm2": 30000.0,
}

# The stock every other constant in CAL_DEFAULTS was chosen on: 1/4" 6061.
#
# Deliberately NOT calibratable. The thickness multiplier is (t/THICK_REF_MM)
# ** thick_exp and it multiplies the target band, so moving the reference and
# moving target_lo/target_hi do the same thing to the output -- two knobs for
# one degree of freedom, which a coordinate search will happily wander along
# forever without the objective changing. It is an anchor, not a parameter: it
# records which stock the rest of the file was tuned against, and that is a
# historical fact rather than something to fit.
THICK_REF_MM = 6.35

# Bounds on the multiplier, also not calibratable, also on purpose.
#
# These are extrapolation limits, not physics. There is evidence at two stock
# thicknesses and only two: 1/8" and 1/4". Everything outside that is the curve
# being trusted where nothing has been measured, so the clamps mark how far it
# gets trusted -- roughly 1/16" at the bottom and 1/2" at the top, beyond which
# a thinner or thicker plate is simply treated as the thinnest or thickest one
# anybody here has actually cut.
#
# At thick_exp = 0.5 they almost never bite: 1/8" lands at 0.71 and 1/2" at
# 1.41, so only the top clamp catches anything in the range a robot gets built
# from. They earn their place if a fit ever pushes the exponent back up.
THICK_LO, THICK_HI = 0.45, 1.30


def thickness_factor(thick_mm, cal=None):
    """Removal-target multiplier for stock thickness. 1.0 at 1/4" and unknown.

    Thin plate is lighter-duty plate: it keeps more of its area. This scales
    the fraction of the pocket region the coverage loop is aiming to remove,
    and the bay size then falls out of that plus the cutter -- see the note at
    the target band in generate() for why it is applied there and nowhere else.

    Returning exactly 1.0 for `None` is the contract that makes this safe to
    add to an engine whose constants were all fitted without it: a photo
    carries no thickness, and neither did any of the tuning that produced the
    other constants, so "not stated" has to mean "the stock I was tuned on".
    """
    if thick_mm is None:
        return 1.0
    try:
        t = float(thick_mm)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(t) or t <= 0.0:
        return 1.0
    c = calibration() if cal is None else cal
    return float(np.clip((t / THICK_REF_MM) ** float(c["thick_exp"]),
                         THICK_LO, THICK_HI))


CAL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "pocket_cal.json")

_CAL_CACHE = {"mtime": None, "values": None}


def calibration(override=None):
    """The active constants: defaults, then data/pocket_cal.json, then override.

    `override` exists for the fitter, which has to evaluate a candidate set of
    constants without writing them to disk first -- writing every trial value to
    the file the running app reads would mean a fit in one window silently
    changing results in another.

    Unknown keys in the file are ignored rather than raising. A calibration file
    is written by a tool and edited by hand afterwards, and a typo in it should
    cost the one constant it was meant to set, not the ability to pocket.
    """
    values = dict(CAL_DEFAULTS)
    try:
        st = os.stat(CAL_PATH)
        if _CAL_CACHE["mtime"] != st.st_mtime or _CAL_CACHE["values"] is None:
            with open(CAL_PATH, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            got = raw.get("constants", raw) if isinstance(raw, dict) else {}
            clean = {}
            for k, v in (got or {}).items():
                if k in CAL_DEFAULTS:
                    try:
                        clean[k] = float(v)
                    except (TypeError, ValueError):
                        pass
            _CAL_CACHE["mtime"] = st.st_mtime
            _CAL_CACHE["values"] = clean
        values.update(_CAL_CACHE["values"] or {})
    except FileNotFoundError:
        _CAL_CACHE["mtime"], _CAL_CACHE["values"] = None, None
    except Exception:
        pass
    if override:
        for k, v in override.items():
            if k in CAL_DEFAULTS:
                try:
                    values[k] = float(v)
                except (TypeError, ValueError):
                    pass
    return values


def is_calibrated():
    """True when a calibration file is present and changed at least one value."""
    cal = calibration()
    return any(abs(cal[k] - CAL_DEFAULTS[k]) > 1e-12 for k in CAL_DEFAULTS)


def calibration_meta():
    """What the UI needs to say where the active constants came from.

    Always returns a dict, never None, and always with `active` set, so the
    caller never has to distinguish "no calibration" from "couldn't tell".

    This exists because a boolean was not enough. `is_calibrated()` answers
    "are these constants still the shipped ones", which is the question the
    engine cares about, but the question a user asks looking at a plan is
    "fitted to WHAT" -- constants fitted to nine 250-400 mm gussets are a
    different object from the same constants fitted to two bellypans, and a
    footer that says only "calibrated" invites reading the first as though it
    were the second. So the part count and the span range it was fitted over
    travel with the flag.

    `span_mm` is None when the file predates that field or was hand-edited.
    That is not an error worth surfacing; the caller just says less.
    """
    out = {"active": False, "n_parts": 0, "span_mm": None, "changed": []}
    if not is_calibrated():
        return out
    out["active"] = True
    cal = calibration()
    out["changed"] = sorted(k for k in CAL_DEFAULTS
                            if abs(cal[k] - CAL_DEFAULTS[k]) > 1e-12)
    try:
        with open(CAL_PATH, "r", encoding="utf-8") as fh:
            raw = json.load(fh) or {}
        out["n_parts"] = int(raw.get("n_parts") or 0)
        span = raw.get("span_mm")
        if isinstance(span, (list, tuple)) and len(span) == 2:
            out["span_mm"] = [round(float(span[0])), round(float(span[1]))]
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------
def thresholds(density="normal", cal=None):
    """The two cut lines on the stress scale, as the legend should show them.

    At or above `keep`, material stays. At or below `pocket`, it is a
    candidate to remove. Between the two is the transition band the rib
    pattern negotiates. Exposed so the colour bar can mark the exact numbers
    the engine used rather than a hand-copied guess that drifts.

    `keep`/`pocket` are the colour-scale cut lines the legend draws. They no
    longer decide whether material survives — that would leave a third of the
    plate solid, which no real pocketed 6061 part is. They now set rib DENSITY:
    at or above `keep` the lattice doubles up. `solid`/`solid_cap` bound the one
    genuinely untouchable island, and `target` is the removal band the coverage
    loop drives towards.

    Calibration moves the band and scales the cell factor. It shifts all three
    presets by the SAME amount rather than rewriting the normal one, because
    "aggressive" means "cut more than I usually would" -- a promise about the
    distance between the presets, not about the absolute number. Calibrating
    normal to 0.57 and leaving aggressive at 0.62 would quietly make the two
    settings the same setting.
    """
    t = {
        "aggressive": {"keep": 0.72, "pocket": 0.68, "cell_f": 1.30,
                       "solid": 0.88, "solid_cap": 0.06, "target": [0.52, 0.72]},
        "conservative": {"keep": 0.42, "pocket": 0.38, "cell_f": 0.80,
                         "solid": 0.72, "solid_cap": 0.18, "target": [0.28, 0.48]},
    }.get(density, {"keep": 0.55, "pocket": 0.45, "cell_f": 1.0,
                    "solid": 0.80, "solid_cap": 0.12, "target": [0.40, 0.60]})
    c = calibration() if cal is None else cal
    d_lo = c["target_lo"] - CAL_DEFAULTS["target_lo"]
    d_hi = c["target_hi"] - CAL_DEFAULTS["target_hi"]
    lo = float(np.clip(t["target"][0] + d_lo, 0.05, 0.90))
    hi = float(np.clip(t["target"][1] + d_hi, lo + 0.04, 0.94))
    t = dict(t)
    t["target"] = [lo, hi]
    t["cell_f"] = t["cell_f"] * c["cell_f"]
    return t


def _stress_seeds(shape, stress_pts, rib_px, kind="hot", span_px=None,
                  body=None, area_cap=0.30):
    """Disks around the FEM's weak/strong callouts.

    This is the second, more direct way the stress analysis reaches the pocket
    plan. The `field >= keep_t` test alone is a threshold on a smoothed raster:
    a genuine concentration at a fillet is a few pixels across and can sit just
    under the cut line after rasterisation, and then the ribs never notice it.
    Stamping a disk at every point the solver identified as a distinct weak
    point guarantees the dense tier lands on all of them, and sizes the disk by
    severity so a 90%-of-peak fillet claims more reinforced material than a
    35% one.

    Two bounds, both learned the hard way. The disk is capped against the
    PART's span, not just the rib width -- and the whole set is shrunk together
    until it claims no more than `area_cap` of the body. The solver now names
    every weak point it finds instead of the worst three, and a radius that
    read as a local reinforcement for three callouts stamps two thirds of the
    plate when there are fourteen. At that point "reinforce here" means nothing,
    because it is everywhere, and the pocket plan collapses back towards the
    solid part it was supposed to lighten. Shrinking the whole set by a common
    factor keeps their RELATIVE sizes -- the 90% fillet still outranks the 35%
    one -- while bringing the total back to something a rib pattern can act on.
    """
    pts = [p for p in (stress_pts or []) if p.get("kind") == kind]
    if not pts:
        return np.zeros(shape, bool)
    h, w = shape
    span = float(span_px or max(h, w))
    pct = np.array([float(p.get("pct") or 0.0) for p in pts])
    r = (rib_px * (1.8 + 3.2 * pct)) if kind == "hot" \
        else np.full(pct.shape, rib_px * 3.0)
    r = np.clip(r, rib_px * 1.5, max(rib_px * 2.0, span * 0.055))

    # Normalised distance: min over points of (distance / that point's radius).
    # Thresholding it at s is exactly "every disk scaled by s", so the shrink
    # loop below costs one comparison per pass instead of re-stamping every
    # disk.
    ys, xs = np.mgrid[0:h, 0:w]
    m = np.full(shape, np.inf, float)
    for i, p in enumerate(pts):
        dx = xs - float(p.get("x", 0.0))
        dy = ys - float(p.get("y", 0.0))
        m = np.minimum(m, np.sqrt(dx * dx + dy * dy) / max(1e-6, r[i]))

    denom = float(body.sum()) if (body is not None and body.any()) else float(h * w)
    s, out = 1.0, None
    for _ in range(6):
        out = m <= s
        if body is not None:
            out &= body
        if float(out.sum()) / denom <= area_cap:
            break
        s *= 0.78
    return out


# Above this fraction of the POCKETABLE area reading "hot", the stress field is
# treated as saturated and the rib doubling is switched off. The long argument
# for why lives at the guard itself, inside generate().
#
# 0.60 is picked so a part with a genuinely large working area -- a gusset in
# heavy shear, a plate loaded across most of its face -- still earns its doubled
# ribs, while a plate sitting uniformly at the allowable does not.
#
# Deliberately NOT a member of CAL_DEFAULTS. Everything in that dict is fair
# game for tools/pocket_ref.py to fit, and a sanity check the search is free to
# tune away is not a sanity check.
HOT_SATURATION_FRAC = 0.60

# How much of a rib width the reserved perimeter ring shrinks to on a TRUSS
# part, where the lattice supplies its own chords along both long edges. See
# the long note at the relief itself, inside generate().
#
# 0.60 is measured, not chosen. Swept against the nine reference parts with
# tools/pocket_ref.py report, scoring the two truss parts and the aggregate gap
# (0 = we match the real parts):
#
#     TRUSS_RIM_F   P4041 (real 55%)   P3003 (real 26%)   gap
#        1.00            41%                25%          0.155   <- old behaviour
#        0.60            49%                27%          0.134
#        0.34            54%                31%          0.154
#
# 0.34 removes more material on P4041 and scores WORSE overall, which is the
# useful part of the result: it buys removal by leaving a wall thinner than the
# real parts carry, and the measured rib drops to 0.0097 against a real 0.0195.
# The objective notices. 0.60 improves both truss parts without overshooting
# either, and takes the aggregate gap down 13% -- the largest single move since
# the original calibration went 0.215 -> 0.155.
#
# Not in CAL_DEFAULTS for the same reason HOT_SATURATION_FRAC is not: this is a
# structural decision about what the rim is FOR, and letting the fitter slide it
# to zero would trade a closed perimeter for a slightly better removal score.
TRUSS_RIM_F = 0.60



def generate(field, part_mask, hole_mask, px_per_mm=1.0,
             rib_mm=3.0, keep_t=None, pocket_t=None, density="normal",
             stress_pts=None, cal=None, thick_mm=None):
    H, W = part_mask.shape
    rib_px = max(3, int(round(rib_mm * px_per_mm)))
    _c = calibration() if cal is None else dict(calibration(cal))
    _t = thresholds(density, _c)
    cell_f = _t["cell_f"]
    thick_m = thickness_factor(thick_mm, _c)
    keep_t = _t["keep"] if keep_t is None else keep_t
    pocket_t = _t["pocket"] if pocket_t is None else pocket_t

    body = part_mask & ~hole_mask
    part_px = int(part_mask.sum())

    # --------------------------------------------------------------------
    # What is off limits. Everything else is fair game -- that is the whole
    # change: on a real pocketed plate the *entire* face is lightened, and the
    # only things that survive are structural, not "stress was high here".
    # --------------------------------------------------------------------
    # (a) Outer rim. Every reference part keeps one continuous perimeter wall;
    #     it is what carries the bending chord and what the pockets stop at.
    #     Eroded from the FILLED outline, not from the drilled plate: eroding a
    #     mask with the holes already punched out of it grows a full rib-width
    #     ring around every bore as well -- both a second, uncapped boss and, on
    #     a plate with a bolt circle, a larger area than the perimeter wall it
    #     was meant to be. On t_bar that one substitution read the rim as 60% of
    #     the plate instead of 33%.
    rim = body & ~ndi.binary_erosion(part_mask | hole_mask, _disk(rib_px))
    # (b) A thin collar at each hole, sized off that hole (see _hole_bosses).
    #     Beyond the collar, the material around a hole IS pocketable.
    boss = _hole_bosses(hole_mask, body, rib_px, field)
    # (c) A genuinely untouchable island, area-capped so it cannot swallow the
    #     part when the clamp singularity saturates the field.
    solid = _solid_band(field, body, _t["solid"], _t["solid_cap"])

    keep = solid | boss
    # Rib wall: pockets stand off the solid island by one rib width, the same
    # way they stand off the rim, so nothing meets at a knife edge.
    domain = body & ~rim & ~boss & ~ndi.binary_dilation(solid, _disk(rib_px))

    part_pattern, desc, bores = classify_part(part_mask, hole_mask, px_per_mm, _c)

    # ---- slender-truss rim relief ----------------------------------------
    # `rim` above reserves a full rib-width ring around the whole perimeter,
    # and then the lattice draws its own members inside that ring. On a Warren
    # truss that is the same wall built twice: the truss ends in continuous
    # chords along both long edges -- the plan text on screen says exactly that
    # -- so the ring and the chords are redundant with each other. On a deep
    # plate nobody notices. On a shallow one it is most of the part.
    #
    # Measured on a 367.7 x 48.9 mm beam: at rib 6 mm the ring alone takes
    # 27.1% of the plate and the hole collars another 8.8%, leaving a 36.7 mm
    # strip that `_region_pattern` then refuses to subdivide, because it wants
    # 5 x rib = 30 mm and the bosses chop the strip below that. Removal falls
    # from 32% at rib 4 mm to 5% at rib 6 mm on geometry that the real 6061
    # part pockets happily at 6 mm -- because on the real part the chords ARE
    # the edges, with no separate ring behind them.
    #
    # A thin ring is still kept rather than none. It guarantees a closed
    # perimeter even where a chord lands badly against a tapered edge, and at a
    # third of a rib it costs almost nothing. Truss parts only: waffle and
    # radial plates draw their members across the interior, not along the
    # boundary, so for those the ring is the only perimeter wall there is and
    # removing it would open the pockets straight through the outside edge.
    if part_pattern == "truss":
        _thin = max(2, int(round(rib_px * TRUSS_RIM_F)))
        rim = body & ~ndi.binary_erosion(part_mask | hole_mask, _disk(_thin))
        domain = body & ~rim & ~boss & ~ndi.binary_dilation(solid, _disk(rib_px))

    # The part's own size. Used to floor the bay size and to cap the stress
    # seeds below. Taken from the oriented frame rather than the bitmap, so a
    # part drawn on the diagonal is not treated as bigger than the same part
    # drawn square.
    _c0, _u0, _v0, span_u, span_v = _frame(part_mask)
    span_px = float(max(span_u, span_v))

    # Where the lattice doubles up: the high-stress raster band, closed so the
    # dense zone is a region rather than a stipple, UNION a disk at every weak
    # point the solver named. Two mechanisms because they fail in opposite
    # directions -- the band catches broad working areas but misses pinpoint
    # concentrations, the seeds catch the concentrations but say nothing about
    # area.
    hot = ndi.binary_closing(body & (field >= keep_t), _disk(max(2, rib_px // 2)))
    hot |= _stress_seeds((H, W), stress_pts, rib_px, "hot",
                         span_px=span_px, body=body, area_cap=0.22)
    n_hot_pts = sum(1 for p in (stress_pts or []) if p.get("kind") == "hot")
    n_cool_pts = sum(1 for p in (stress_pts or []) if p.get("kind") == "safe")

    # ---- saturation guard ------------------------------------------------
    # `hot` doubles the rib density underneath it (`np.where(hot_sub, dense,
    # base)`, ~150 lines down). That is the right answer to a real stress
    # concentration covering PART of the plate. It is the wrong answer to a
    # field that is hot almost everywhere -- and a field that is hot almost
    # everywhere is very rarely a part in trouble. It is a load, a thickness or
    # a material that was never entered, so every pixel sits near the allowable
    # at once.
    #
    # The failure this prevents is quiet and expensive, and it is worth writing
    # down because the symptom points somewhere else entirely. On a 49 mm deep
    # truss solving at SF 1.0x, doubling a 6 mm rib spends 24 mm of the depth on
    # chords before a single diagonal is drawn. The bays that survive are then
    # rounded by `fillet` (which itself scales with rib width) and deleted
    # outright by `min_area`, because a bay does not shrink gracefully -- it
    # either clears the sliver threshold or vanishes. Measured on one real part:
    # 32% removed at rib 4 mm, 5% at rib 6 mm. That reads as "the pocketer can't
    # handle 6 mm ribs", which is false. It handles 6 mm ribs on that exact
    # geometry perfectly well once the inputs are right.
    #
    # Measured over `domain`, not the whole part. The rim, the bosses and the
    # solid island are excluded from pocketing anyway, and they are precisely
    # where a healthy field is legitimately hot -- counting them would trip the
    # guard on parts that are entirely fine.
    _dom_px = int(domain.sum())
    hot_frac = (float((hot & domain).sum()) / _dom_px) if _dom_px else 0.0
    hot_saturated = bool(hot_frac > HOT_SATURATION_FRAC)
    # Only the doubling is suppressed, and `hot` itself is left alone: `cool`
    # still subtracts the true hot mask below, because a "safe" seed landing
    # inside a high-stress band is wrong whether or not the field is saturated.
    hot_dense = np.zeros_like(hot) if hot_saturated else hot

    # And where it thins out: material the solver called a strong point is
    # along for the ride, so its cells open up by a quarter. The pocket gets
    # bigger exactly where the analysis says nothing is happening -- which is
    # the whole argument for pocketing in the first place.
    cool = _stress_seeds((H, W), stress_pts, rib_px, "safe",
                         span_px=span_px, body=body, area_cap=0.40) & ~hot

    cr = max(2, int(round(2.0 * px_per_mm)))
    # Generous inside radii. The cutter radius alone rounds the pocket's own
    # convex corners, but the corners a person notices on a real 6061 plate are
    # the CONCAVE ones -- the fillets where two ribs meet. Those are added by
    # closing the rib material, and they are most of what separates a photo of
    # a machined plate from a lattice subtracted from a silhouette.
    fillet = int(round(float(np.clip(rib_px * 1.05, cr,
                                     max(cr, 6.0 * px_per_mm)))))
    # Slivers no sane machinist would cut. Scaled with the part for the same
    # reason the cell is: a 400 px offcut is a nuisance chip on a bellypan and
    # a legitimate pocket on a gusset.
    min_area = max(70, int(part_px * _c["min_area_frac"]))

    # Bay size per region, per trial scale, area-weighted at the end. Recorded
    # because `cell` is the number the thickness term is really about, and it is
    # otherwise invisible: `cell_scale` is only the search's multiplier on it,
    # and pocket count conflates it with how the sliver drop happened to land.
    # When someone asks "did the 1/8" setting actually do anything", this is the
    # field that answers it.
    cell_log = {}

    def _build(scale):
        """Cut the pattern at a given cell scale and clean it up.

        Wrapped as a function because coverage is not something you can solve
        for in closed form: the cutter-radius opening and the sliver drop both
        eat material in ways that depend on the region shapes. Far cheaper to
        cut it, measure it, and adjust the cell size than to model it.
        """
        core = domain.copy()
        lab_, n_ = ndi.label(core)
        boxes_ = ndi.find_objects(lab_)
        used_ = {}
        for i in range(1, n_ + 1):
            slc = boxes_[i - 1]
            if slc is None:
                continue
            reg = (lab_[slc] == i)
            area = int(reg.sum())
            c, u, v, Lu, Lv = _frame(reg)

            # Bores this region is organised around: those whose centre falls
            # inside the region's window, plus a little reach, since the bore
            # itself sits in its boss collar and so is never in `domain`.
            y0, x0 = slc[0].start, slc[1].start
            reach = max(Lu, Lv) * 0.08 + 2.0 * rib_px
            bore_pts = [(bx - x0, by - y0) for bx, by, _r in bores
                        if (x0 - reach) <= bx <= (slc[1].stop + reach)
                        and (y0 - reach) <= by <= (slc[0].stop + reach)]
            has_radial = len(bore_pts) >= 1

            pat = _region_pattern(part_pattern, area, Lu, Lv, rib_px, has_radial)
            used_[pat] = used_.get(pat, 0) + 1
            if pat == "none":
                continue

            # hot_dense, not hot: identical unless the saturation guard above
            # fired, in which case this is all-False and every rib stays single
            # width.
            hot_sub = hot_dense[slc]
            short = min(Lu, Lv)
            # Cell sized off the region itself AND off the whole part, then
            # scaled by the coverage loop. The part term is what fixes big
            # plates: `short / 5.5` alone is scale-free, so a 450 mm side plate
            # gets the same five-bays-across treatment as a 90 mm gusset and
            # ends up with fifty small pockets. Nobody machines a plate that
            # way -- a pocket costs a tool path and a rib costs weight, so real
            # 6061 parts hold roughly a dozen large bays however big they get.
            # Tying the floor to the part's own span keeps bay COUNT roughly
            # constant with size instead of bay LENGTH.
            # A triangulated lattice of spacing c drawn with ribs of width t
            # spends roughly 3.5*t/c of its area on rib. At c = 8*t that is
            # 43% -- nearly half the pocket region eaten by rib, which is what
            # made the first big-plate attempt read as a spider web. Real 6061
            # runs a 3 mm rib against a 50-80 mm bay, about 1:20, so the floor
            # here is 14 rib widths and the part term is a quarter of the span:
            # three or four bays across, whatever size the plate is.
            # (These three factors are the ones tools/pocket_ref.py fits.)
            #
            # THERE IS DELIBERATELY NO STOCK-THICKNESS MULTIPLIER HERE. There
            # was one for about a day, and taking it out again is the single
            # most useful thing in this file's history, so here is the whole
            # story rather than a shrug.
            #
            # The argument for it was clean. A rib between two nodes is a strut;
            # in compression it buckles at sigma_cr proportional to (t/L)^2,
            # where t is the stock and L the unsupported length, which is
            # exactly `cell`. Hold the allowable stress fixed and L goes
            # linearly with t: halve the thickness, halve the bays. So the line
            # read `... * cell_f * scale * thick_m`, with thick_m linear in t,
            # and the docs said the theory was so clean it did not need fitting.
            #
            # Ten of team 2813's parts say otherwise. Regressing their real bay
            # size on span and stock gives bay ~ span^0.35 * t^0.21, and the
            # standard error on that 0.21 is 0.30 -- a 95% interval of roughly
            # [-0.4, +0.8], which contains zero comfortably and EXCLUDES the 1.0
            # the formula predicted. Drop any single part and the exponent walks
            # between -0.21 and +0.49 depending on which of the three 1/8" parts
            # you happened to keep. A permutation test on bay/span between the
            # thin and thick groups returns p = 0.69. There is no bay-size
            # signal here to implement.
            #
            # Why the theory misses: buckling is not what sets bay size on a
            # part like this. The end mill sets the rib width, the rib width and
            # the removal target set the pitch, and a designer looking at 1/8"
            # plate reaches for "take less out" long before "take the same out
            # in smaller squares". The parts show exactly that -- the thin ones
            # carry FATTER ribs relative to their bays (0.32 vs 0.26) and remove
            # less overall. Thickness acts on how much comes out, not on how it
            # is divided up.
            #
            # So the term moved to the one place the evidence puts it: the
            # removal band, ~180 lines down. Bay size still responds to stock,
            # because a lower removal target with the same cutter gives a
            # tighter pitch -- about t^0.33 through that route, which is inside
            # the measured interval instead of five times outside it. The effect
            # is a consequence now rather than an assertion, which is the right
            # shape for something this weakly measured.
            cell = max(rib_px * _c["cell_rib_f"],
                       short * _c["cell_short_f"],
                       span_px * _c["cell_span_f"]) * cell_f * scale
            # Idle material earns bigger bays. Applied as a per-region nudge
            # rather than per-pixel, because a cell size has to be constant
            # across a lattice or the ribs stop meeting at nodes.
            if cool[slc][reg].mean() > 0.25:
                cell *= 1.25
            cell = float(np.clip(cell, rib_px * 4.0, short * 0.95))
            cell_log.setdefault(round(float(scale), 6), []).append((area, cell))

            if pat == "xbrace":
                U, V = _uv_grid(reg.shape, c, u, v)
                base = dense = _ribs_xbrace(U, V, rib_px, Lu, Lv)
            else:
                # One triangulator for truss, waffle and radial alike: only the
                # node placement differs. Every rib ends on a node and every
                # bay is a triangle, which is what the reference 6061 parts
                # look like and what the old crossed-lattice approach could not
                # produce.
                base, dense = _ribs_web(reg, hole_mask[slc], c, u, v, Lu, Lv,
                                        cell, rib_px, pat,
                                        bore_pts if pat == "radial" else ())
                if base is None:            # degenerate region: leave it whole
                    continue

            # Two tiers, not a solid block. Hot material gets twice the ribs;
            # because the dense lattice is a superset of the base one, members
            # run straight through the boundary instead of terminating on it.
            ribs = np.where(hot_sub, dense, base)
            reg_out = reg & ~ribs

            sub = core[slc]
            sub[reg] = reg_out[reg]
            core[slc] = sub

        # cutter radius: rounds every convex corner of the pocket, which is
        # what a 4 mm end mill physically leaves behind
        core = ndi.binary_opening(core, _disk(cr))
        # ...and a fillet at every rib junction, which is the concave corner
        # the cutter cannot reach and the designer therefore draws. Closing the
        # rib material adds exactly that: a radius in the crotch of every node
        # where three or four ribs meet.
        if fillet > cr:
            core = core & ~ndi.binary_closing(~core, _disk(fillet))
            core = ndi.binary_opening(core, _disk(max(1, cr // 2)))

        # drop slivers no sane machinist would cut
        lab_, n_ = ndi.label(core)
        if n_:
            sz = np.asarray(ndi.sum(np.ones_like(lab_), lab_, index=range(1, n_ + 1)))
            for i in np.nonzero(sz < min_area)[0] + 1:
                core[lab_ == i] = False
        return core, used_

    # --------------------------------------------------------------------
    # Coverage feedback. "The entire part should be pocketed" is a statement
    # about the ANSWER, not about a threshold, so it is enforced on the answer:
    # cut, measure what came off, resize the cells, cut again. Bounded to four
    # passes -- this is a search for a plausible plan, not an optimiser.
    # --------------------------------------------------------------------
    # THIS is where stock thickness acts, and the only place it acts.
    #
    # Thin plate is lighter-duty plate: it keeps more of its area. Ten of team
    # 2813's parts, seven on 1/4" and three on 1/8", give removal ~ t^0.45 once
    # span is controlled for -- the thin three average 38% removal against the
    # thick seven's 49%. That is not overwhelming evidence (standard error 0.23,
    # permutation p = 0.09) but it is the only thickness effect in the set that
    # survives looking at, and the sign and rough size are stable however the
    # parts are sliced. See CAL_DEFAULTS["thick_exp"] for why the default is the
    # rounder 0.5.
    #
    # Applying it here rather than on `cell` is not a style choice, and the
    # reason is the one genuinely non-obvious thing about this file. The loop
    # below searches `scale` until the removed fraction lands in the band. For a
    # lattice of ribs of width w on a pitch of `cell`, the surviving rib area is
    # about 3.5*w/cell, so removal is about 1 - 3.5*w/cell. Rib width is an
    # input -- it is the cutter -- so FIXING REMOVAL FIXES `cell`. The band is
    # not one of several influences on bay size; downstream of this loop it is
    # the whole of it. Anything multiplied onto `cell` upstream is not a change
    # to the answer at all, only to the loop's starting point, and the loop
    # spends its four passes undoing it: measured on a synthetic plate, halving
    # the thickness upstream moved `cell_scale` 0.452 -> 0.820 and the actual
    # bay only 24.2 -> 21.4 mm. A 2x change in stock bought 12% of lattice.
    #
    # So bay size does still respond to stock -- it just responds THROUGH here.
    # A lower removal target with the same cutter gives a tighter pitch, working
    # out near t^0.33, which sits inside the [-0.4, +0.8] interval the parts
    # actually measure for bay size. That is the correct relationship between a
    # weakly-measured effect and the code: a consequence of the thing there is
    # evidence for, not a second assertion competing with it.
    #
    # Clamped rather than trusted at the extremes: 0.08 keeps a very thin plate
    # from being told to remove nothing at all (at which point the loop has no
    # target to steer by), and 0.86 keeps thick plate from chasing a removal
    # fraction that leaves no closed rib network behind.
    #
    # Known resolution limit, worth stating so nobody reports it as a bug. The
    # loop below steps `scale` by fixed factors of 1.25 / 0.82 and stops at the
    # first result INSIDE the band, not at its centre. So two bands less than
    # about 20% of removal apart can land on the same plan: on a plain
    # rectangular test plate, 1/8" and 3/16" come out byte-identical, while 1/8"
    # and 1/4" differ properly (59% -> 42% removal, 80 -> 54 mm bays). The
    # engine resolves stock in roughly half-steps of the gauge, which is honest
    # for an effect measured to +/- 0.23 in the exponent. Aiming at the band
    # centre instead would sharpen it, but it would also move every part that
    # has nothing to do with thickness, including the ones the constants were
    # calibrated against -- so it is a separate change, not a footnote to this
    # one.
    lo, hi = _t["target"]
    if abs(thick_m - 1.0) > 1e-9:
        lo = float(np.clip(lo * thick_m, 0.08, 0.85))
        hi = float(np.clip(hi * thick_m, lo + 0.04, 0.86))
    scale = 1.0
    pocket_core, used = _build(scale)
    best = (pocket_core, used, scale)
    for _ in range(4):
        frac = float(pocket_core.sum()) / max(1, part_px)
        if lo <= frac <= hi:
            break
        scale = scale * (1.25 if frac < lo else 0.82)
        if not (0.25 <= scale <= 4.0):
            break
        cand, used_c = _build(scale)
        cf = float(cand.sum()) / max(1, part_px)
        # Only adopt the new cut if it is closer to the band; a bigger cell can
        # overshoot into one giant pocket, which is worse than being 5% light.
        if abs(cf - np.clip(cf, lo, hi)) < abs(frac - np.clip(frac, lo, hi)):
            pocket_core, used, best = cand, used_c, (cand, used_c, scale)
        else:
            break
    pocket_core, used, scale = best

    lab, n_pockets = ndi.label(pocket_core)
    removable = float(pocket_core.sum()) / max(1, part_px)

    # Area-weighted, not a plain mean: a bellypan is one big region and a dozen
    # offcuts, and averaging those as equals would report the offcuts' bay size
    # as the part's.
    _cl = cell_log.get(round(float(scale), 6)) or []
    _wa = sum(a for a, _ in _cl)
    cell_px_ = (sum(a * cv for a, cv in _cl) / _wa) if _wa else 0.0
    return pocket_core, keep, {
        "n_pockets": int(n_pockets),
        "removable_frac": removable,
        "rib_px": rib_px,
        "cell_scale": round(float(scale), 3),
        "cell_mm": round(cell_px_ / max(px_per_mm, 1e-9), 2),
        # Reported even when it is 1.0. A silent multiplier on the removal
        # target is the kind of thing someone spends an afternoon on when two
        # runs of the "same" part disagree and only the stock dropdown moved.
        "thick_mm": (None if thick_mm is None else round(float(thick_mm), 2)),
        "thick_factor": round(thick_m, 3),
        # The band the loop above actually steered by, which is NOT the one
        # thresholds() returns once thickness is in play. The API hands the
        # caller thresholds(density) alongside these stats, so without this the
        # two would disagree on screen with nothing to explain the difference.
        "target_band": [round(lo, 3), round(hi, 3)],
        "solid_frac": float(solid.sum()) / max(1, part_px),
        # Reported always, not only when it fires. "Why did this plan change
        # when I only moved the rib width" is answerable from the payload only
        # if the doubling state is in the payload.
        "hot_frac": round(hot_frac, 3),
        "hot_saturated": hot_saturated,
        "pattern": part_pattern,
        "region_patterns": used,
        # What was actually DRAWN, as opposed to how the part was classified.
        # These disagree more often than you would think -- a broad panel
        # classifies as `waffle`, then its one region turns out to be organised
        # around a bore and comes back `radial` -- and when they do, captioning
        # from `pattern` describes a lattice that is not on the screen. The
        # reader is looking at radial ribs while the text explains an even
        # field of near-equilateral bays.
        #
        # Area would be the better weight than count; region count is what is
        # already to hand here and the two only differ when a part has several
        # regions of genuinely different sizes, which is the case where the
        # caption is hedged anyway.
        "region_pattern": _dominant_region(used),
        "region_note": _PATTERN_NOTE.get(_dominant_region(used), ""),
        "calibrated": bool(cal is None and is_calibrated()),
        # The full provenance, not just the flag. `cal is not None` means the
        # fitter is driving with trial constants it has not written, so the
        # honest answer there is "no calibration is in effect for the app".
        "calibration": (calibration_meta() if cal is None
                        else {"active": False, "n_parts": 0,
                              "span_mm": None, "changed": []}),
        "geometry": desc,
        "note": _PATTERN_NOTE.get(part_pattern, ""),
        "pockets": _pocket_records(lab, n_pockets, field, part_px),
        "hole_notes": _hole_records(hole_mask, field, rib_px),
        "keep_note": _keep_note(keep, part_mask, hole_mask, field),
    }


# --------------------------------------------------------------------------
# annotations — the plan in words, anchored to where it applies
# --------------------------------------------------------------------------
def _label_anchor(reg_slice, reg):
    """A point INSIDE the region to hang a label on.

    The centroid is wrong for anything C-shaped or crescent — a rib pattern
    makes plenty of those — and a label floating in the rib wall next to the
    pocket points at nothing. The pixel furthest from the region's own edge is
    always inside it and is also the roomiest spot for text.
    """
    d = ndi.distance_transform_edt(np.pad(reg, 1))
    iy, ix = np.unravel_index(int(np.argmax(d)), d.shape)
    return (float(ix - 1 + reg_slice[1].start),
            float(iy - 1 + reg_slice[0].start),
            float(d.max()))


def _pocket_records(lab, n, field, part_px):
    """One record per pocket: where to label it, and how hard it works.

    The percentage is the region's mean stress as a share of the scale peak —
    the number that decides whether it may be pocketed at all. Area alone
    would look like a machining estimate and say nothing about why the pocket
    is allowed to exist there.
    """
    out = []
    boxes = ndi.find_objects(lab)
    for i in range(1, n + 1):
        slc = boxes[i - 1]
        if slc is None:
            continue
        reg = lab[slc] == i
        area = int(reg.sum())
        if area <= 0:
            continue
        x, y, room = _label_anchor(slc, reg)
        out.append({"id": f"P{len(out) + 1}", "x": x, "y": y,
                    "room": room, "area_frac": area / max(1, part_px),
                    "stress": float(np.clip(field[slc][reg].mean(), 0, 1))})
    # biggest pockets first: if the view can only fit a few labels, they should
    # be the ones a machinist would actually cut first
    out.sort(key=lambda r: -r["area_frac"])
    for k, r in enumerate(out):
        r["id"] = f"P{k + 1}"
    return out


def _hole_records(hole_mask, field, rib_px):
    """Every bore, with the two things that constrain a pocket near it: how big
    it is, and how hard the material around its rim is working.

    Size matters on its own. A small mount hole needs a solid boss ring so the
    bolt has something to clamp; a large bore is a hard boundary the pocket
    pattern may not cross at all. They are different rules, so the view has to
    tell them apart rather than drawing one ring around everything.
    """
    if not hole_mask.any():
        return []
    H, W = hole_mask.shape
    lab, n = ndi.label(hole_mask)
    boxes = ndi.find_objects(lab)
    pad = int(max(2, rib_px))
    out = []
    for i in range(1, n + 1):
        slc = boxes[i - 1]
        if slc is None:
            continue
        # Grow the bounding box before dilating. find_objects returns the bore's
        # tight box, so a dilation computed inside it is clipped at the edge --
        # the ring would survive only in the box corners and "rim stress" would
        # be read from four diagonal specks instead of all the way round.
        s0 = slice(max(0, slc[0].start - pad), min(H, slc[0].stop + pad))
        s1 = slice(max(0, slc[1].start - pad), min(W, slc[1].stop + pad))
        sub_lab = lab[s0, s1]
        reg = sub_lab == i
        area = int(reg.sum())
        if area < 6:
            continue
        ys, xs = np.nonzero(reg)
        cx = float(xs.mean() + s1.start)
        cy = float(ys.mean() + s0.start)
        r = float(np.sqrt(area / np.pi))
        # Rim stress: the ring of material just outside the bore, which is where
        # a hole concentration actually shows up. Neighbouring bores are cut out
        # of the ring -- on a tight bolt circle the next hole falls inside it,
        # and the field there is a value sampled from empty space.
        ring = ndi.binary_dilation(reg, _disk(pad)) & (sub_lab == 0)
        sub = field[s0, s1]
        rim = float(sub[ring].max()) if ring.any() and sub.size else 0.0
        out.append({"x": cx, "y": cy, "r": r,
                    "rim": float(np.clip(rim, 0, 1))})

    # Which bores count as structural, judged against this part's own bolt
    # pattern rather than a fixed size. A fastener hole is whatever size repeats
    # across the plate; a bore is what stands out from it. Keying the cut to the
    # rib width alone made the class flip when the same plate was rasterised at
    # a different scale, which is exactly the wrong sensitivity.
    if out:
        med = float(np.median([h["r"] for h in out])) or 1.0
        cut = max(2.0 * med, 1.2 * rib_px)
        for h in out:
            h["kind"] = "large" if h["r"] >= cut else "mount"
    return out


def _keep_note(keep, part_mask, hole_mask, field):
    """The single loudest 'do not touch this' on the part.

    One callout, on the largest connected block of keep material rather than
    on the hottest pixel: the hottest pixel is usually a bore rim already
    ringed in red, and repeating it there teaches nothing.
    """
    m = keep & part_mask & ~hole_mask
    if not m.any():
        return None
    lab, n = ndi.label(m)
    if n < 1:
        return None
    sizes = ndi.sum(np.ones_like(lab), lab, index=range(1, n + 1))
    # Boss collars are keep material too, and on a bolt-heavy plate the biggest
    # of them can outweigh a real load-bearing block. A callout on a 6 mm ring
    # says nothing a machinist did not already know from the red circle, so
    # anything under ~1.5% of the plate does not earn a label.
    if float(np.max(sizes)) < 0.015 * float(max(1, int(part_mask.sum()))):
        return None
    i = int(np.argmax(sizes)) + 1
    reg_slices = ndi.find_objects(lab)
    slc = reg_slices[i - 1]
    reg = lab[slc] == i
    x, y, _ = _label_anchor(slc, reg)
    # How hard the block works on average, not at its worst pixel. The clamp
    # line is a stress singularity: the display field saturates over a wide band
    # around it, so a max -- or even an upper quartile -- prints "100%" on
    # essentially every part and says nothing. The mean is the one summary that
    # still separates a block that is uniformly critical from one that merely
    # touches the threshold.
    return {"x": x, "y": y,
            "stress": float(np.clip(field[slc][reg].mean(), 0, 1))}


def _dominant_region(used):
    """The pattern most regions actually got, ignoring the ones left whole.

    `"none"` is excluded because it is not a lattice -- it means the region was
    too narrow or too small to subdivide and stayed one continuous pocket. A
    part whose only subdivided region is a truss should be captioned as a truss
    even if three slivers came back as `none`.
    """
    real = {k: v for k, v in (used or {}).items() if k != "none"}
    if not real:
        return ""
    return max(real.items(), key=lambda kv: kv[1])[0]


_PATTERN_NOTE = {
    "truss": "Slender member — Warren truss: nodes zig-zag between the two long "
             "edges and the ribs are the triangulation of them, so every bay is "
             "a shear-stable triangle and both edges stay continuous chords.",
    "waffle": "Broad panel — a triangulated node lattice on the panel's own "
              "axes. Panels fail by buckling rather than bending about one "
              "axis, so an even field of near-equilateral bays is right here.",
    "radial": "Bore-driven plate — every bearing is a node in the web, so ribs "
              "radiate from the bores and run bore-to-bore where two can see "
              "each other. Bays between those load paths are pocketed.",
    "xbrace": "Small gusset — a single X-brace. Below roughly a dozen rib "
              "widths there is no room for a lattice, and one brace carries "
              "more than a chopped-up grid.",
}


def _hatch(shape, pitch, thick=2, slope=1):
    """Diagonal machinist hatching.

    Hatching is not decoration here. Green-on-grey alone reads as two flat
    colours and gives you no sense of which way a region runs; ruled lines
    make the pocket read as a *surface to cut* and keep pocket and rib wall
    distinguishable when the image is scaled down to a thumbnail or printed
    without colour.
    """
    H, W = shape
    y, x = np.mgrid[0:H, 0:W]
    return ((x + slope * y) % int(max(3, pitch))) < thick


def render_png(part_mask, hole_mask, pocket_core, keep, path, rib_px=6):
    """The machining plan: what stays, what goes, and where the hard edges are.

    Deliberately NOT a heat map. The stress map answers "how hard is this
    working"; this answers "may I put a cutter here", which is a yes/no per
    region, and mixing the two encodings into one picture is how a caution
    zone gets cut by mistake.
    """
    H, W = part_mask.shape
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (7, 10, 18)

    body = part_mask & ~hole_mask
    hatch_fine = _hatch((H, W), max(4, rib_px // 2), 1, 1)
    hatch_wide = _hatch((H, W), max(7, rib_px), 2, -1)

    img[body] = (168, 178, 200)                       # rib wall / solid stock
    img[body & hatch_fine] = (150, 161, 186)

    kp = keep & body
    img[kp] = (196, 52, 44)                           # keep — carries the load
    img[kp & hatch_wide] = (236, 92, 78)

    pk = pocket_core & body
    img[pk] = (30, 178, 88)                           # pocket — free to remove
    img[pk & hatch_wide] = (16, 128, 62)

    img[hole_mask] = (6, 9, 16)

    # A bright rim on every boundary: the silhouette, each bore, and the walls
    # of each pocket. Without it the pocket floor and the rib wall touch at a
    # colour seam, and a seam is exactly the line a machinist needs to see.
    for m, col in ((part_mask, (232, 240, 255)),
                   (pocket_core, (186, 255, 214))):
        if m.any():
            img[m & ~ndi.binary_erosion(m, _disk(1))] = col
    Image.fromarray(img).save(path)
