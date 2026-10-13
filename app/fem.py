"""
2D plane-stress linear-elastic FEM solver, built on scikit-fem.

Accuracy notes (why this is not the textbook-minimum version):

* **Quadratic (P2) displacement elements.** Linear CST triangles have a
  *constant* strain over each element, so stress is piecewise-constant and the
  elements are artificially stiff in bending -- exactly the load case this app
  cares about. P2 gives linear strain per element, which converges an order
  faster and resolves bending gradients across the plate thickness instead of
  smearing them.

* **Tractions, not lumped nodal point loads.** The old code split the load
  equally over every node in a band. Where the mesh happens to be dense (around
  small holes) that dumps far more force per unit length than where it is
  coarse, which fabricates hot spots that have nothing to do with the geometry.
  Loads are now real Neumann boundary integrals: force per unit length, so
  refining the mesh does not change the loading.

* **L2-projected stress recovery.** Von Mises is evaluated at the quadrature
  points -- where the finite-element stress is actually most accurate -- and
  projected onto the nodes through the mass matrix, rather than averaging
  element constants into whichever nodes they touch.

* **Percentile normalisation.** A linear-elastic model has genuine stress
  *singularities* at a clamped edge and at any sharp re-entrant corner: the
  computed value there does not converge, it just grows as the mesh refines.
  Dividing the whole field by that number is what turned every real part green
  with a couple of red specks at the clamp. The colour scale is now referenced
  to a high percentile of the non-singular field and clipped.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse.linalg import spsolve
from skfem import (
    MeshTri, Basis, FacetBasis, ElementTriP1, ElementTriP2, ElementVector,
    BilinearForm, LinearForm, Functional, asm, condense, solve,
)
from skfem.helpers import ddot, sym_grad, trace

# Colour-scale reference percentile. 98 keeps a real hot spot red while
# refusing to let one singular node at the clamp own the whole scale.
REF_PCTL = 98.0
# A boundary node whose material wedge exceeds this multiple of pi is a
# re-entrant corner, where linear elasticity gives unbounded stress. The exact
# value does not matter -- a true 270-degree corner reads 1.5, a straight edge
# reads 1.0, and nothing real sits between -- which was confirmed by sweeping
# it: 1.12, 1.25 and 1.35 give byte-identical results.
REENTRANT_WEDGE = 1.25
# How far from such a corner to stop reading, as a fraction of the span.
#
# This one DOES matter, because the field is still steep near a singularity
# and the reported peak is whatever survives just outside the exclusion.
# Measured on an L-bracket across a 4x range of element size:
#
#     radius      reported peak (MPa)        spread
#     0.025      35.9  36.9  43.9  40.8      22.2%
#     0.040      35.9  34.4  35.4  36.2       5.5%
#     0.060      32.4  32.6  32.4  32.2       1.0%
#
# 0.04 is the knee: it buys most of the stability for a fraction of the
# material thrown away. A plate with a hole reads 80.3/80.1/80.4 at every one
# of them, because a hole rim is not re-entrant and is never excluded.
REENTRANT_R = 0.04
# Percentile used for the COLOUR SCALE only; see REF_PCTL above. The reported
# peak is no longer a percentile -- see where peak_clean is computed.
PEAK_PCTL = 99.5


def _lame(E: float, nu: float):
    """Plane-stress Lamé-like parameters for scikit-fem's linear elasticity."""
    lam = E * nu / (1.0 - nu * nu)
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


def _facet_length(m, facets):
    """Total length of a facet set, in mesh units."""
    if facets is None or len(facets) == 0:
        return 0.0

    @Functional
    def one(w):
        return 1.0 + 0.0 * w.x[0]

    try:
        return float(asm(one, FacetBasis(m, ElementTriP1(), facets=facets)))
    except Exception:
        return 0.0


def _facet_mids(m, facets):
    """Midpoint of each facet, as a (2, k) array in mesh units."""
    fac = np.asarray(facets, dtype=np.int64)
    if fac.size == 0:
        return np.zeros((2, 0))
    return np.asarray(m.p)[:, np.asarray(m.facets)[:, fac]].mean(axis=1)


def _pick_facets(m, m_per_px, pick, boundary=None):
    """Boundary facets inside a clicked disc.

    One primitive serves both halves of "click where it is held and where it
    is loaded", because from the solver's point of view a hole and a stretch
    of outline are the same thing: a patch of boundary. The caller sends a
    centre and a radius in the pixel coordinates the picture is drawn in --
    for a bore, its centre and a little over its radius; for an edge click,
    the click point and a capture radius -- and gets back the facets that
    actually carry the condition.

    Nothing here knows what a "hole" is, which is the point. A UI that
    identifies features differently later does not need this to change.
    """
    if boundary is None:
        boundary = m.boundary_facets()
    boundary = np.asarray(boundary, dtype=np.int64)
    if boundary.size == 0:
        return boundary
    mids = _facet_mids(m, boundary)
    cx = float(pick.get("x", 0.0)) * m_per_px
    cy = float(pick.get("y", 0.0)) * m_per_px
    r = max(float(pick.get("r", 0.0)), 1.0) * m_per_px
    d = np.hypot(mids[0] - cx, mids[1] - cy)
    return boundary[d <= r]


def _facet_loops(m):
    """Split the boundary into closed loops of FACETS, outer one first.

    Named for facets on purpose. There is already a _boundary_loops(points,
    tris) further down that walks the same curves in NODE space, for reporting
    hole geometry; this one is needed because a traction has to be integrated
    over facets, and node loops cannot be handed to a FacetBasis. The first
    version of this was also called _boundary_loops, which silently redefined
    the older one -- Python kept the last definition, so the hub lookup called
    a function expecting different arguments and the whole case raised. Two
    jobs, two names.

    Needed because "apply a torque at the hub" means applying it around a
    bore, and a bore is a boundary loop that is not the outside of the part.
    The mesher does not label them, so they are recovered by walking the
    facet graph: every boundary vertex has exactly two boundary facets, so
    following one from the next traces a loop and stops where it started.

    Returns a list of facet-index arrays. The loop with the largest bounding
    box is the outside and is returned first; everything after it is a hole.
    """
    bf = np.asarray(m.boundary_facets(), dtype=np.int64)
    if bf.size == 0:
        return []
    fv = np.asarray(m.facets)[:, bf]                 # (2, k) vertex ids
    adj = {}
    for i in range(bf.size):
        for a, b in ((fv[0, i], fv[1, i]), (fv[1, i], fv[0, i])):
            adj.setdefault(int(a), []).append((int(b), i))
    seen = np.zeros(bf.size, bool)
    loops = []
    for start in range(bf.size):
        if seen[start]:
            continue
        chain = [start]
        seen[start] = True
        head = int(fv[1, start])
        tail = int(fv[0, start])
        while True:
            nxt = None
            for v, fi in adj.get(head, ()):
                if not seen[fi]:
                    nxt = (v, fi)
                    break
            if nxt is None:
                break
            head = nxt[0]
            seen[nxt[1]] = True
            chain.append(nxt[1])
            if head == tail:
                break
        loops.append(bf[np.asarray(chain, dtype=np.int64)])
    if not loops:
        return []
    p = np.asarray(m.p)

    def extent(fac):
        v = np.unique(np.asarray(m.facets)[:, fac])
        return float((np.ptp(p[0, v]) + 1e-18) * (np.ptp(p[1, v]) + 1e-18))

    loops.sort(key=extent, reverse=True)
    return loops


def _wedge_angles(points_m, tris, N):
    """Total material angle at every node, in radians.

    The sum of the triangle corner angles meeting at a node. An interior node
    reads 2*pi; a node on a straight edge reads pi; the corner of a rectangle
    reads pi/2; the inside corner of an L reads 3*pi/2.

    This is how a genuine singularity can be told from a genuine stress
    concentration without refining the mesh twice. Both are small hot
    regions, but one is a property of the part and one is a property of the
    idealisation: elasticity gives unbounded stress wherever the material
    wedge exceeds pi, and nowhere else. A hole rim is a smooth boundary
    reading pi, so its concentration is finite, converged and real; a
    re-entrant corner reads more than pi and its "peak" is whatever the mesh
    last happened to resolve.

    Measured off the mesh rather than the outline on purpose -- it needs no
    loop orientation, works the same on an outer boundary and a bore, and
    cannot disagree with the elements the solve actually used.
    """
    p = np.asarray(points_m, float)
    t = np.asarray(tris, np.int64)
    ang = np.zeros(N)
    for k in range(3):
        i = t[:, k]
        j = t[:, (k + 1) % 3]
        m = t[:, (k + 2) % 3]
        u = p[j] - p[i]
        v = p[m] - p[i]
        nu = np.linalg.norm(u, axis=1)
        nv = np.linalg.norm(v, axis=1)
        good = (nu > 0) & (nv > 0)
        c = np.zeros(len(i))
        c[good] = np.clip((u[good] * v[good]).sum(axis=1)
                          / (nu[good] * nv[good]), -1.0, 1.0)
        np.add.at(ang, i, np.arccos(c))
    return ang


def _nodal_areas(points_m, tris, N):
    """Each node's share of the plate (a third of every triangle it touches)."""
    p = points_m[tris]
    a = 0.5 * np.abs(
        (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) -
        (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    w = np.zeros(N)
    np.add.at(w, tris.ravel(), np.repeat(a / 3.0, 3))
    w[w <= 0] = w[w > 0].mean() if (w > 0).any() else 1.0
    return w


def _wpercentile(vals, weights, pct):
    """Weighted percentile — the value below which `pct`% of the AREA lies."""
    vals = np.asarray(vals, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()
    if vals.size == 0:
        return 0.0
    if weights.size != vals.size:
        return float(np.percentile(vals, pct))
    o = np.argsort(vals)
    v, w = vals[o], np.maximum(weights[o], 0.0)
    tot = w.sum()
    if tot <= 0:
        return float(np.percentile(vals, pct))
    # midpoint rule: cumulative weight at the centre of each node's share
    cw = (np.cumsum(w) - 0.5 * w) / tot
    return float(np.interp(pct / 100.0, cw, v))


def _boundary_nodes(tris, N):
    """Nodes on the silhouette or on a bore: an edge used by exactly one
    triangle is a free edge, and both its ends sit on a boundary."""
    t = np.asarray(tris)
    e = np.sort(np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    out = np.zeros(N, bool)
    if uniq.size:
        out[uniq[cnt == 1].ravel()] = True
    return out


# Stress bands, as fractions of the hottest *converged* node. Every label on
# the map is bucketed with these, and the pocketing engine is handed the same
# numbers, so a badge that reads "caution" is sitting on material the engine
# also treats as caution rather than on a separate opinion.
BANDS = ((0.80, "critical"), (0.55, "high"), (0.32, "caution"))


def _band(p):
    for cut, name in BANDS:
        if p >= cut:
            return name
    return "low"


def _free_edges(tris):
    """Edges used by exactly one triangle — i.e. edges on some boundary."""
    t = np.asarray(tris)
    if t.size == 0:
        return np.zeros((0, 2), int)
    e = np.sort(np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    return uniq[cnt == 1] if uniq.size else np.zeros((0, 2), int)


def _boundary_loops(points, tris):
    """Every closed boundary curve, largest enclosed area first.

    The free edges of a meshed plate form exactly one loop per closed curve:
    the silhouette, then one per bore. Walking them is what lets this module
    talk about "every hole on the part" without anyone passing it the hole
    list -- the mesh already encodes where they are, and the loop it recovers
    is the true rim, not a circle fitted to a centroid.
    """
    e = _free_edges(tris)
    if e.shape[0] < 3:
        return []
    adj = {}
    for a, b in e:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))

    seen, loops = set(), []
    for start in adj:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = None
            for cand in adj.get(cur, ()):
                if cand != prev and cand not in seen:
                    nxt = cand
                    break
            if nxt is None:
                break
            loop.append(nxt)
            seen.add(nxt)
            prev, cur = cur, nxt
        if len(loop) >= 6:
            loops.append(np.asarray(loop, int))

    p = np.asarray(points, float)

    def _area(idx):
        q = p[idx]
        return 0.5 * abs(float(np.dot(q[:, 0], np.roll(q[:, 1], -1)) -
                               np.dot(q[:, 1], np.roll(q[:, 0], -1))))

    loops.sort(key=_area, reverse=True)
    return loops


def _hole_stress(points, tris, vm_pct, keep, limit=160):
    """One record per bore: how hard the ring of material around it works.

    A bore is where a plate is loaded and it is where a plate cracks, so
    "which of these forty holes is a problem" is the question the map exists
    to answer. Reporting only the worst one answers it for a single hole and
    leaves the other thirty-nine as unlabelled circles.

    Read on the rim loop itself, at the highest node on it -- a hole
    concentration is a peak at one point of the rim (the two ends of the
    diameter across the load), never an average round the ring. Nodes swallowed
    by the support collar are skipped, because the value there is the clamp
    singularity rather than the hole's own Kt; if the whole rim is inside the
    collar the record is flagged instead of quietly reporting a fiction.
    """
    loops = _boundary_loops(points, tris)
    if len(loops) < 2:
        return []
    p = np.asarray(points, float)
    pct = np.asarray(vm_pct, float)
    kp = np.asarray(keep, bool)
    out = []
    for idx in loops[1:]:
        q = p[idx]
        c = q.mean(axis=0)
        r = float(np.hypot(*(q - c).T).mean())
        good = idx[kp[idx]] if kp.size == p.shape[0] else idx
        singular = good.size == 0
        use = idx if singular else good
        v = pct[use]
        pk = float(v.max()) if v.size else 0.0
        out.append({"x": float(c[0]), "y": float(c[1]), "r": r,
                    "pct": pk, "mean_pct": float(v.mean()) if v.size else 0.0,
                    "band": _band(pk), "singular": bool(singular)})
    out.sort(key=lambda h: -h["pct"])
    return out[:limit]


def _nms(pts, cand, rank, gap, limit):
    """Non-maximum suppression: strongest first, then veto its neighbourhood.

    This is what keeps one fillet from collecting six badges. Candidates are
    capped before the loop because on a refined mesh a single concentration
    can contribute tens of thousands of nodes, and every one of them past the
    first few thousand is a neighbour of something already vetoed.
    """
    cand = np.asarray(cand)
    if cand.size == 0:
        return []
    order = cand[np.argsort(-np.asarray(rank)[cand])][:6000]
    chosen, cx, cy = [], [], []
    g2 = float(gap) * float(gap)
    for i in order:
        if chosen:
            dx = np.asarray(cx) - pts[i, 0]
            dy = np.asarray(cy) - pts[i, 1]
            if float((dx * dx + dy * dy).min()) < g2:
                continue
        chosen.append(int(i))
        cx.append(pts[i, 0])
        cy.append(pts[i, 1])
        if len(chosen) >= limit:
            break
    return chosen


def _callouts(points, tris, vm_pct, vm_norm, keep, hot_cut=0.32,
              safe_cut=0.38, n_hot=14, n_cool=8):
    """Every distinct weak and strong point on the part, not just the worst.

    Three markers on a forty-hole plate is a summary, not a map: it names one
    fillet and leaves the reader to guess about everything else. What the
    pocketing engine needs is the *set* of concentrations -- each one is a
    place ribs have to run to -- and the set of genuinely idle areas, which is
    where the material actually comes off. So both are enumerated here and
    both are handed downstream.

    Picked here rather than in the browser because this is the only place that
    knows `keep`. The clamp collar holds the numerically largest stresses in
    the model and they are singular: they grow without bound as the mesh
    refines and mean nothing physically. A client-side picker would stack
    every "weakest point" badge on the bolt line and miss the real
    concentrations around the bores.
    """
    pts = np.asarray(points, float)
    N = pts.shape[0]
    if N == 0 or not np.any(keep):
        return []
    span = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))) or 1.0
    pct = np.asarray(vm_pct, float)
    nrm = np.asarray(vm_norm, float)
    kp = np.asarray(keep, bool)

    out = []
    # Weak points: local maxima of the converged field. The suppression radius
    # is a fraction of the part's own size, so the same rule gives a gusset
    # three badges and a bellypan a dozen.
    for i in _nms(pts, np.flatnonzero(kp & (pct >= hot_cut)), pct,
                  0.065 * span, n_hot):
        out.append({"x": float(pts[i, 0]), "y": float(pts[i, 1]),
                    "pct": float(pct[i]), "kind": "hot",
                    "band": _band(float(pct[i]))})

    # Strong points -- material that is along for the ride. Ranked by depth
    # into clear stock as well as by how little it carries: the lowest node on
    # the part is nearly always a sliver on the rim, where a badge reads as
    # pointing at the edge rather than at the area behind it.
    bnd = _boundary_nodes(tris, N)
    if bnd.any():
        try:
            from scipy.spatial import cKDTree
            depth = cKDTree(pts[bnd]).query(pts)[0]
        except Exception:
            depth = np.ones(N)
    else:
        depth = np.ones(N)
    depth = depth / (float(depth.max()) or 1.0)
    rank = depth * (1.0 - np.clip(nrm, 0.0, 1.0))
    for i in _nms(pts, np.flatnonzero(kp & (nrm <= safe_cut)), rank,
                  0.11 * span, n_cool):
        out.append({"x": float(pts[i, 0]), "y": float(pts[i, 1]),
                    "pct": float(pct[i]), "kind": "safe",
                    "band": _band(float(pct[i]))})
    return out


def _facet_pairs(m, facets, N, limit=800):
    """A facet set as (n, 2) mesh-vertex index pairs, boundary facets only.

    The drawing code wants line segments, not a cloud of nodes: a support is a
    stretch of edge that is held, and scattering dots along it reads as noise.
    Intersecting with the boundary is belt-and-braces -- these facet sets are
    already boundary sets by construction -- but it costs nothing and it means
    an interior facet can never end up drawn as a clamp.
    """
    fac = np.asarray(facets, dtype=np.int64).ravel()
    if fac.size == 0:
        return np.zeros((0, 2), int)
    try:
        bnd = np.asarray(m.boundary_facets(), dtype=np.int64).ravel()
        if bnd.size:
            fac = np.intersect1d(fac, bnd)
    except Exception:
        pass
    if fac.size == 0:
        return np.zeros((0, 2), int)
    try:
        seg = np.asarray(m.facets)[:, fac].T
    except Exception:
        return np.zeros((0, 2), int)
    seg = np.asarray(seg, dtype=np.int64)
    if seg.ndim != 2 or seg.shape[1] != 2:
        return np.zeros((0, 2), int)
    ok = (seg >= 0).all(axis=1) & (seg < N).all(axis=1)
    seg = seg[ok]
    if seg.shape[0] > limit:
        seg = seg[np.unique(np.linspace(0, seg.shape[0] - 1,
                                        limit).astype(int))]
    return seg


def _spread(xy, n):
    """`n` points spread along a set of points by position, not by index.

    Facets come out of the mesh in whatever order the triangulator built them,
    so taking every k-th one clumps the glyphs at one end. Sorting by position
    along the set's own long axis first makes them land evenly on the edge.
    """
    xy = np.asarray(xy, float)
    if xy.shape[0] <= n:
        return xy
    long_ax = 0 if np.ptp(xy[:, 0]) >= np.ptp(xy[:, 1]) else 1
    o = np.argsort(xy[:, long_ax])
    pick = np.unique(np.linspace(0, o.size - 1, n).astype(int))
    return xy[o[pick]]


def _bc_payload(m, points, tris, N, ax, trans_dof, load_case, load,
                fixed_facets, fixed_nodes, load_facets, load_nodes,
                applied, span_px, size_px, min_px, picked=False):
    """Where the part is held and where the force goes, in node coordinates.

    Everything here is read back out of the boundary conditions the solve
    actually used -- the same facet set that was condensed out of the stiffness
    matrix, and the same one the traction was integrated over. It is not a
    second drawing of the load case from the dropdown, because that is the
    version that can quietly disagree with the maths. If the solver fell back
    to lumped nodal forces, that fallback is what gets drawn, and
    `load_applied_as` says so on the picture.

    Coordinates are the raw pixel-space `points` array -- the same frame the
    stress map paints in -- so the front end can draw straight onto the canvas
    with no transform of its own.
    """
    p = np.asarray(points, float)

    def _segs(pairs):
        if pairs.shape[0] == 0:
            return []
        a, b = p[pairs[:, 0]], p[pairs[:, 1]]
        return [[[float(a[i, 0]), float(a[i, 1])],
                 [float(b[i, 0]), float(b[i, 1])]] for i in range(a.shape[0])]

    sup_pairs = _facet_pairs(m, fixed_facets, N)
    sup_segs = _segs(sup_pairs)
    if sup_pairs.shape[0]:
        sup_xy = 0.5 * (p[sup_pairs[:, 0]] + p[sup_pairs[:, 1]])
    else:
        # Degenerate geometry: the solve constrained nodes directly, so draw
        # the nodes it constrained rather than drawing nothing at all.
        fx = np.asarray(fixed_nodes, bool)
        sup_xy = p[fx] if fx.any() else np.zeros((0, 2))

    ld_pairs = _facet_pairs(m, load_facets, N)
    ld_segs = _segs(ld_pairs)
    if ld_pairs.shape[0]:
        ld_xy = 0.5 * (p[ld_pairs[:, 0]] + p[ld_pairs[:, 1]])
    else:
        ln = np.asarray(load_nodes, bool)
        ld_xy = p[ln] if ln.any() else np.zeros((0, 2))

    # The load form is `-trac * v[trans_dof]`, so the force points along the
    # negative transverse axis. Reporting the sign instead of assuming it in
    # the browser means the arrow follows the form if the form ever changes.
    d = [0.0, 0.0]
    d[trans_dof] = -1.0

    body = (load_case == "ss_dist")
    if body:
        # Uniform pressure: there is no loaded edge, so the arrows spread over
        # the part itself. Interior nodes only -- an arrow rooted on the rim
        # reads as an edge load, which is exactly what this case is not.
        try:
            inner = ~_boundary_nodes(np.asarray(tris), p.shape[0])
        except Exception:
            inner = np.ones(p.shape[0], bool)
        ld_xy = p[inner] if inner.any() else p
        ld_segs = []

    return {
        "span_axis": int(ax),
        "load_axis": int(trans_dof),
        "load_dir": d,
        "load_case": load_case,
        "load_n": float(load),
        "load_applied_as": applied,
        # A couple has no single direction to draw an arrow along, so the map
        # is told which it is rather than left to infer it from the case name.
        "load_kind": ("moment" if load_case in ("bending", "torque")
                      else "force"),
        "load_units": "N*m" if load_case in ("bending", "torque") else "N",
        "picked": bool(picked),
        "distributed": bool(body),
        "two_ends": bool(load_case in ("ss_center", "ss_dist", "fixed_fixed")),
        # stretches of edge that are held, and where to put the hatch glyphs
        "support_segs": sup_segs,
        "support_pts": [[float(x), float(y)] for x, y in _spread(sup_xy, 9)],
        # the loaded edge, and where to root the force arrows
        "load_segs": ld_segs,
        "load_pts": [[float(x), float(y)]
                     for x, y in _spread(ld_xy, 10 if body else 7)],
        # A sensible glyph size for whoever draws this, in the same units.
        #
        # Scaled off the part's SHORTEST dimension, because neither obvious
        # choice works. The span axis draws 6-pixel stubs on a vertically
        # spanned part. The longest dimension draws 120-pixel arrows across a
        # 240-pixel-tall bar, where they run straight off the picture -- the
        # canvas is the part's bounding box, so there is no margin to spill
        # into. The short side is what actually bounds the room a glyph has,
        # whichever way the load points, so it is what sets the size.
        "glyph_px": float(max(min_px * 0.08, 6.0)),
        "span_px": float(span_px),
        "size_px": float(size_px),
        "min_px": float(min_px),
    }


def solve_plane_stress(points, tris, E=69e9, nu=0.33, thickness=0.00635,
                       load_case="cantilever", orientation="horizontal",
                       load=500.0, m_per_px=1.0, picks=None):
    """
    points    : Nx2 float  mesh nodes in IMAGE PIXELS (returned unchanged)
    tris      : Mx3 int    triangle connectivity
    thickness : plate thickness in METRES
    load      : total applied force in NEWTONS
    m_per_px  : metres per pixel. The solve runs in SI so von Mises comes out in
                real pascals.

    returns dict with per-node von Mises, displacement magnitude, and metadata.
    """
    points = np.asarray(points, dtype=np.float64)
    points_m = points * float(m_per_px)
    m = MeshTri(points_m.T.copy(), np.asarray(tris).T.copy())

    e = ElementVector(ElementTriP2())        # quadratic displacement
    basis = Basis(m, e, intorder=4)
    sbasis = Basis(m, ElementTriP1(), intorder=4)   # nodal output field

    lam, mu = _lame(E, nu)
    thk = float(thickness)

    @BilinearForm
    def stiffness(u, v, w):
        return thk * (2.0 * mu * ddot(sym_grad(u), sym_grad(v)) +
                      lam * trace(sym_grad(u)) * trace(sym_grad(v)))

    K = asm(stiffness, basis)

    # ---- geometry of the load case -------------------------------------
    ax = 0 if orientation == "horizontal" else 1     # span axis
    trans_dof = 1 if orientation == "horizontal" else 0
    coord = points_m[:, ax]
    s0, s1 = float(coord.min()), float(coord.max())
    span = max(s1 - s0, 1e-12)
    band = span * 0.06          # load band: a tip load is never a knife edge
    # The support band is kept thin on purpose. The old 6% slab clamped the
    # top and bottom edges for 6% of the length as well, which braces the part
    # like a socket and shortens its effective span -- a rectangular test beam
    # came out 20% too stiff. 2% grips the end face and little else.
    fband = span * 0.02
    N = points.shape[0]

    two_ends = load_case in ("ss_center", "ss_dist", "fixed_fixed")

    # What the number in the box means. A force for the shear cases, a moment
    # for the couple cases. Stating it rather than assuming Newtons everywhere
    # is the difference between an answer and a number: 200 N on the end of a
    # 300 mm arm and 200 N*m applied as a couple are not the same load, and a
    # solver that silently treats one as the other is wrong by the span.
    MOMENT_CASES = ("bending", "torque")
    load_units = "N*m" if load_case in MOMENT_CASES else "N"

    def facets_where(fn):
        try:
            f = m.facets_satisfying(fn)
        except Exception:
            return np.array([], dtype=np.int64)
        return np.asarray(f, dtype=np.int64)

    fix_lo = facets_where(lambda x: x[ax] <= s0 + fband)
    if fix_lo.size == 0:                       # coarse mesh: widen once
        fband = span * 0.05
        fix_lo = facets_where(lambda x: x[ax] <= s0 + fband)
    fix_hi = facets_where(lambda x: x[ax] >= s1 - fband) if two_ends \
        else np.array([], dtype=np.int64)
    fixed_facets = np.unique(np.concatenate([fix_lo, fix_hi])) \
        if fix_hi.size else fix_lo

    # ---- picked supports override the banded ones ------------------------
    #
    # "Which of these thirty holes is bolted down" is a question a dropdown
    # cannot answer and a sentence typed at an assistant answers badly. When
    # the caller has pointed at the actual features, those win outright over
    # the guessed end bands -- not merged with them, because a part held by
    # one bore in the middle is not also clamped at its left edge, and quietly
    # adding a support the user did not ask for makes it stiffer than it is.
    picks = [q for q in (picks or []) if isinstance(q, dict)]
    sup_picks = [q for q in picks if q.get("role") in ("fixed", "support")]
    load_picks = [q for q in picks if q.get("role") == "load"]
    pin_picks = [q for q in picks if q.get("role") == "pin"]
    bnd = np.asarray(m.boundary_facets(), dtype=np.int64)
    picked_support = False
    if sup_picks:
        got = [_pick_facets(m, m_per_px, q, bnd) for q in sup_picks]
        got = [g for g in got if g.size]
        if got:
            fixed_facets = np.unique(np.concatenate(got))
            picked_support = True

    # Constraining boundary facets (rather than every node inside a 6%-wide
    # slab of material) keeps the support where a real clamp actually is.
    if fixed_facets.size:
        D = np.asarray(basis.get_dofs(fixed_facets).flatten(), dtype=np.int64)
        if picked_support:
            fixed_nodes = np.zeros(N, bool)
            vids = np.unique(np.asarray(m.facets)[:, fixed_facets])
            vids = vids[(vids >= 0) & (vids < N)]
            fixed_nodes[vids] = True
        else:
            fixed_nodes = (coord <= s0 + fband)
            if two_ends:
                fixed_nodes |= (coord >= s1 - fband)
    else:                                     # degenerate geometry fallback
        fixed_nodes = coord <= s0 + fband
        nd = basis.nodal_dofs
        D = np.concatenate([nd[0][fixed_nodes], nd[1][fixed_nodes]])

    # ---- pins: hold the part still without holding it back ---------------
    #
    # A self-equilibrated load has no reaction to find. A ring under internal
    # pressure pushes outward everywhere and the forces cancel exactly, so the
    # stiffness matrix is singular by three: the part can drift in x, drift in
    # y, and spin, all at zero energy, and the solve fails or returns garbage.
    #
    # Clamping an edge to fix that would be worse than the disease -- it would
    # resist the expansion the problem is about and change the answer. A pin
    # constrains ONE displacement component at ONE node, so three of them
    # placed where that component is not the direction the part wants to move
    # remove the drift and the spin and leave the stress field untouched. For
    # a ring: hold y at the two ends of a horizontal diameter, and x at the
    # top. Neither point is restrained radially, which is the only direction
    # pressure pushes it.
    #
    # This is how the Lame benchmark is run, and the check that it is harmless
    # is the benchmark itself: if a pin were taking load, the answer would not
    # match the closed form.
    if pin_picks:
        nd_pin = basis.nodal_dofs
        extra = []
        for q in pin_picks:
            px_ = float(q.get("x", 0.0)) * m_per_px
            py_ = float(q.get("y", 0.0)) * m_per_px
            i = int(np.argmin((points_m[:, 0] - px_) ** 2 +
                              (points_m[:, 1] - py_) ** 2))
            which = str(q.get("dofs") or "xy").lower()
            if "x" in which:
                extra.append(nd_pin[0][i])
            if "y" in which:
                extra.append(nd_pin[1][i])
        if extra:
            D = np.unique(np.concatenate([D, np.asarray(extra, np.int64)]))

    # ---- Neumann loading ------------------------------------------------
    # Distribute by *length* (or area), never by node count: the total force
    # must be independent of how finely the mesh happens to be cut.
    f = np.zeros(basis.N)
    applied = "traction"
    load_nodes = np.zeros(N, bool)     # nodes the load is attached to
    # Kept past the branch below so the load arrows are drawn from the facets
    # the traction was actually integrated over, not from a second guess at
    # where the load case "should" have put them.
    load_facets = np.array([], dtype=np.int64)

    def _facet_nodes(fac):
        """Mesh-vertex indices touched by a set of boundary facets."""
        sel = np.zeros(N, bool)
        if fac is None or np.asarray(fac).size == 0:
            return sel
        try:
            vids = np.unique(np.asarray(m.facets)[:, np.asarray(fac)])
        except Exception:
            return sel
        vids = vids[(vids >= 0) & (vids < N)]
        sel[vids] = True
        return sel

    def _couple_about(fac, moment, cx0=None, cy0=None):
        """A pure couple of `moment` N*m applied over a patch of boundary.

        Tangential traction, scaled so the moments come to exactly what was
        asked for, and then corrected so no net FORCE comes with them.

        That correction is not theoretical tidiness. The first version of this
        asserted in its own docstring that "no net force comes out of it --
        the tractions cancel round the loop", which is true of a closed loop
        like a bore and false of every other patch. A click near the corner of
        an end face selects a patch that wraps onto the adjacent edge, the
        tangential directions there no longer oppose each other, and what gets
        applied is a couple plus a sideways shove. On a cantilever that shove
        bends the part, and the benchmark measured the result: a nominal
        50 N*m delivered 2.88 times the bending stress it should have, and the
        error grew with how much of the corner the patch caught.

        So the resultant of the tangential field is measured and subtracted as
        a uniform traction. Taken about the patch's own length-weighted
        centroid, a uniform traction contributes no moment at all -- the
        integral of r over the patch is zero there by definition -- so the
        couple survives the correction untouched and the net force goes to
        zero. On a closed bore the correction evaluates to nothing, which is
        the old behaviour, correctly.
        """
        fb1 = FacetBasis(m, ElementTriP1(), facets=fac)

        @Functional
        def one(w):
            return 1.0 + 0.0 * w.x[0]

        L = float(asm(one, fb1))
        if not np.isfinite(L) or L <= 0:
            return None

        # The length-weighted centroid, not the mean of the midpoints. They
        # differ wherever the mesh is graded, and this is the one point about
        # which the correction below is moment-free.
        @Functional
        def mx(w):
            return w.x[0]

        @Functional
        def my(w):
            return w.x[1]

        cx0 = float(asm(mx, fb1)) / L
        cy0 = float(asm(my, fb1)) / L

        @Functional
        def arm(w):
            rx = w.x[0] - cx0
            ry = w.x[1] - cy0
            return np.sqrt(rx * rx + ry * ry)

        A = float(asm(arm, fb1))
        if not np.isfinite(A) or A <= 0:
            return None
        tau = float(moment) / (A * thk)

        # Resultant of the unit tangential field, per component.
        @Functional
        def tgx(w):
            rx = w.x[0] - cx0
            ry = w.x[1] - cy0
            return -ry / (np.sqrt(rx * rx + ry * ry) + 1e-15)

        @Functional
        def tgy(w):
            rx = w.x[0] - cx0
            ry = w.x[1] - cy0
            return rx / (np.sqrt(rx * rx + ry * ry) + 1e-15)

        corr_x = -tau * float(asm(tgx, fb1)) / L
        corr_y = -tau * float(asm(tgy, fb1)) / L

        @LinearForm
        def torq(v, w):
            rx = w.x[0] - cx0
            ry = w.x[1] - cy0
            rr = np.sqrt(rx * rx + ry * ry) + 1e-15
            return thk * ((tau * (-ry / rr) + corr_x) * v[0] +
                          (tau * (rx / rr) + corr_y) * v[1])

        return asm(torq, FacetBasis(m, e, facets=fac))

    def _end_moment(fac, moment):
        """A pure bending moment on an end face, as linearly varying pull.

        What makes this different from the cantilever case, and worth having
        separately: a tip load puts a moment on the root that grows along the
        span and carries shear with it, so the stress depends on where you
        look. A couple on the end puts the SAME moment everywhere with no
        shear at all, which is the textbook pure-bending state and the right
        model for a link loaded through its ends. The traction is axial and
        proportional to distance from the face's own centreline -- tension on
        one side, compression on the other, no net force.
        """
        mids = _facet_mids(m, fac)
        c = float(mids[trans_dof].mean())

        @Functional
        def second(w):
            d = w.x[trans_dof] - c
            return d * d

        I = float(asm(second, FacetBasis(m, ElementTriP1(), facets=fac)))
        if not np.isfinite(I) or I <= 0:
            return None
        k = float(moment) / (I * thk)

        @LinearForm
        def bend(v, w):
            return thk * k * (w.x[trans_dof] - c) * v[ax]

        return asm(bend, FacetBasis(m, e, facets=fac))

    # ---- picked loads override the load case ----------------------------
    #
    # Same reasoning as the picked supports: if the caller has said which bore
    # takes the force, that is the load, and the dropdown's guess is not also
    # applied. Each pick carries its force in newtons as components, and/or a
    # torque in newton-metres, so one click can be "500 N downward on this
    # hole" and another "12 N*m about this hub".
    if load_picks:
        parts, used = [], []
        # A click that carries no direction is the common case and the point of
        # the whole feature: "the load goes HERE" is a complete thought, and
        # making someone also pick a compass arrow before anything happens
        # turns placing a load into filling in a form. So a bare pick inherits
        # the load case's own direction -- the same transverse axis the
        # dropdown would have used -- and the total from the load box is split
        # evenly between however many load points were placed. Two clicks at
        # 500 N is 250 N each, not 500 N each, because the number in the box
        # says what the part carries.
        bare = [q for q in load_picks
                if not (q.get("fx") or q.get("fy") or q.get("torque")
                        or q.get("pressure"))]
        if bare:
            share = float(load) / len(bare)
            for q in load_picks:
                if not (q.get("fx") or q.get("fy") or q.get("torque")
                        or q.get("pressure")):
                    d = [0.0, 0.0]
                    d[trans_dof] = -share
                    q["fx"], q["fy"] = d[0], d[1]
                    q["auto_dir"] = True
        for q in load_picks:
            sel = _pick_facets(m, m_per_px, q, bnd)
            if sel.size == 0:
                continue
            L = _facet_length(m, sel)
            if L <= 0:
                continue
            fx, fy = float(q.get("fx") or 0.0), float(q.get("fy") or 0.0)
            if fx or fy:
                tx, ty = fx / (L * thk), fy / (L * thk)

                @LinearForm
                def edgeload(v, w, tx=tx, ty=ty):
                    return thk * (tx * v[0] + ty * v[1])

                parts.append(asm(edgeload, FacetBasis(m, e, facets=sel)))
                used.append(sel)
            # Pressure: a normal traction, the one load a direction arrow
            # cannot express. The traction on a surface under pressure p is
            # -p*n with n the material's OUTWARD normal, so on the wall of a
            # bore -- where the outward normal points into the hole -- it
            # pushes the material away from the hole axis, which is what
            # internal pressure does. Needed for the Lame thick-cylinder
            # check, and the right model for a bore under hydraulic pressure
            # or a press fit.
            pr = float(q.get("pressure") or 0.0)
            if pr:
                @LinearForm
                def pressload(v, w, pr=pr):
                    return -pr * thk * (w.n[0] * v[0] + w.n[1] * v[1])

                parts.append(asm(pressload, FacetBasis(m, e, facets=sel)))
                used.append(sel)
            T = float(q.get("torque") or 0.0)
            if T:
                mids = _facet_mids(m, sel)
                vec = _couple_about(sel, T, float(mids[0].mean()),
                                    float(mids[1].mean()))
                if vec is not None:
                    parts.append(vec)
                    used.append(sel)
        if parts:
            for vec in parts:
                f = f + vec
            load_facets = np.unique(np.concatenate(used))
            load_nodes = _facet_nodes(load_facets)
            applied = "picked features"
        else:
            applied = "picked features matched no edge"
    elif load_case == "axial":
        # Straight pull (or push) along the span. The simplest case there is,
        # and the one every other case is measured against -- a part that
        # cannot be checked in pure tension is not telling you much in bending.
        lf = facets_where(lambda x: x[ax] >= s1 - band)
        load_facets = np.asarray(lf, dtype=np.int64)
        L = _facet_length(m, lf)
        if lf.size and L > 0:
            trac = load / (L * thk)

            @LinearForm
            def pull(v, w):
                return trac * thk * v[ax]

            f = asm(pull, FacetBasis(m, e, facets=lf))
            load_nodes = _facet_nodes(lf)
            applied = "axial traction"
    elif load_case == "bending":
        lf = facets_where(lambda x: x[ax] >= s1 - band)
        load_facets = np.asarray(lf, dtype=np.int64)
        vec = _end_moment(lf, load) if lf.size else None
        if vec is not None:
            f = vec
            load_nodes = _facet_nodes(lf)
            applied = "end couple (pure bending)"
        else:
            applied = "end couple failed - no end face found"
    elif load_case == "torque":
        # Torque about the plate's normal, applied where a torque actually
        # goes in: the hub. The largest bore is the hub on the overwhelming
        # majority of parts that have one -- a sprocket boss, a bearing seat,
        # a shaft clamp -- and when a part has no holes at all there is
        # nothing to twist through, so the couple goes on the end face and the
        # result says so rather than pretending a hub was found.
        loops = _facet_loops(m)
        hub = None
        if len(loops) > 1:
            hub = max(loops[1:], key=lambda fac: _facet_length(m, fac))
        if hub is not None and hub.size:
            mids = _facet_mids(m, hub)
            vec = _couple_about(hub, load, float(mids[0].mean()),
                                float(mids[1].mean()))
            if vec is not None:
                f = vec
                load_facets = hub
                load_nodes = _facet_nodes(hub)
                applied = "torque about the largest bore"
        if load_facets.size == 0:
            lf = facets_where(lambda x: x[ax] >= s1 - band)
            vec = _end_moment(lf, load) if lf.size else None
            if vec is not None:
                f = vec
                load_facets = np.asarray(lf, dtype=np.int64)
                load_nodes = _facet_nodes(lf)
                applied = ("no bore to twist through - couple applied to the "
                           "end face instead")
    elif load_case == "ss_dist":
        # Uniform pressure over the whole plate (e.g. its own supported load).
        @Functional
        def area(w):
            return 1.0 + 0.0 * w.x[0]
        A = float(asm(area, sbasis)) or 1.0
        body = load / (A * thk)               # N/m^3

        @LinearForm
        def bodyload(v, w):
            return -body * thk * v[trans_dof]
        f = asm(bodyload, basis)
        applied = "body force"
    else:
        if load_case in ("ss_center", "fixed_fixed"):
            mid = 0.5 * (s0 + s1)
            lf = facets_where(lambda x: np.abs(x[ax] - mid) <= band)
        else:                                  # cantilever → free end
            lf = facets_where(lambda x: x[ax] >= s1 - band)
        load_facets = np.asarray(lf, dtype=np.int64)
        L = _facet_length(m, lf)
        if lf.size and L > 0:
            trac = load / (L * thk)            # N/m^2 on the edge face

            @LinearForm
            def edgeload(v, w):
                return -trac * thk * v[trans_dof]
            f = asm(edgeload, FacetBasis(m, e, facets=lf))
            load_nodes = _facet_nodes(lf)
        else:
            # No boundary facets in the band (very odd geometry): fall back to
            # lumped nodal forces so the solve still produces something.
            load_facets = np.array([], dtype=np.int64)
            sel = (coord >= s1 - band) & (~fixed_nodes)
            idx = np.where(sel)[0]
            if idx.size:
                f[basis.nodal_dofs[trans_dof][idx]] += -load / idx.size
                load_nodes[idx] = True
            applied = "lumped nodal (no facets found)"

    # ---- solve ----------------------------------------------------------
    u = solve(*condense(K, f, D=D))

    nd = basis.nodal_dofs                      # (2, n_vertices)
    ux = u[nd[0]]
    uy = u[nd[1]]
    disp = np.sqrt(ux * ux + uy * uy)

    # ---- stress recovery: von Mises at quadrature points, L2 → nodes ----
    w = basis.interpolate(u)
    g = w.grad                                 # (2, 2, n_elem, n_qp)
    exx = g[0, 0]
    eyy = g[1, 1]
    exy = 0.5 * (g[0, 1] + g[1, 0])
    c = E / (1.0 - nu * nu)
    sxx = c * (exx + nu * eyy)
    syy = c * (eyy + nu * exx)
    sxy = (E / (1.0 + nu)) * exy               # G * gamma_xy, gamma = 2*exy
    vmq = np.sqrt(np.maximum(
        sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0))

    @BilinearForm
    def mass(a, b, w):
        return a * b

    @LinearForm
    def rhs(v, w):
        return w["vm"] * v

    try:
        M = asm(mass, sbasis)
        F = asm(rhs, sbasis, vm=vmq)
        vm_node = spsolve(M.tocsr(), F)
        # An L2 projection can ring slightly past the data it is fitting;
        # clamping to the quadrature range keeps it physical.
        vm_node = np.clip(vm_node, 0.0, float(vmq.max()) if vmq.size else 0.0)
    except Exception:
        vm_node = np.zeros(N)
        cnt = np.zeros(N)
        cell_avg = vmq.mean(axis=1) if vmq.size else np.zeros(len(tris))
        for ti, t in enumerate(np.asarray(tris)):
            for n_ in t:
                vm_node[n_] += cell_avg[ti]
                cnt[n_] += 1
        cnt[cnt == 0] = 1
        vm_node /= cnt

    vm_node = np.nan_to_num(vm_node, nan=0.0, posinf=0.0, neginf=0.0)
    vm_node = np.asarray(vm_node, dtype=np.float64).ravel()[:N]
    if vm_node.size < N:                       # paranoia: pad, never crash
        vm_node = np.pad(vm_node, (0, N - vm_node.size))

    # ---- the individual components, projected the same way ---------------
    #
    # von Mises is a scalar summary and it is the right thing to PAINT, but it
    # is the wrong thing to CHECK against a textbook. Every closed-form result
    # in mechanics names a component: Mc/I is sigma_xx, Winkler-Bach is
    # sigma_theta, Lame gives sigma_r and sigma_theta separately. Comparing von
    # Mises to any of them happens to work where one component dominates and
    # quietly does not elsewhere -- near a cantilever's neutral axis the
    # bending stress passes through zero while von Mises does not, because
    # shear is carrying the section there.
    #
    # Signs survive, which is the other half of the point: a curved beam in
    # bending is in tension on one fibre and compression on the other, and a
    # magnitude cannot tell you which.
    def _project(field):
        """Quadrature-point field -> nodal values, by L2 projection."""
        try:
            @LinearForm
            def _rhs(v, w):
                return w["q"] * v
            out = spsolve(M.tocsr(), asm(_rhs, sbasis, q=field))
            out = np.nan_to_num(np.asarray(out, float).ravel()[:N],
                                nan=0.0, posinf=0.0, neginf=0.0)
            if out.size < N:
                out = np.pad(out, (0, N - out.size))
            # Clamp to the range the data actually spans. The projection can
            # ring past it at a re-entrant corner, and a component that
            # overshoots its own quadrature values is an artefact of the fit.
            return np.clip(out, float(field.min()), float(field.max()))
        except Exception:
            cnt = np.zeros(N)
            acc = np.zeros(N)
            cell = field.mean(axis=1) if field.size else np.zeros(len(tris))
            for ti, t in enumerate(np.asarray(tris)):
                for n_ in t:
                    acc[n_] += cell[ti]
                    cnt[n_] += 1
            cnt[cnt == 0] = 1
            return acc / cnt

    try:
        sxx_node = _project(sxx)
        syy_node = _project(syy)
        sxy_node = _project(sxy)
    except Exception:
        sxx_node = syy_node = sxy_node = np.zeros(N)

    # ---- colour-scale reference ----------------------------------------
    # Stress at a clamped edge is singular in linear elasticity: it rises
    # without limit as the mesh refines, so it is a meaningless number to
    # normalise by. Exclude a thin collar around the supports and reference
    # the scale to a high percentile of what is left.
    #
    # The collar is measured as a true distance from the boundary conditions,
    # not as a slab in the span direction. On the pentagon plate that matters:
    # its support is a narrow pointed tip, so a span-wise band either misses
    # the singular corner or swallows a tenth of the part. Saint-Venant says
    # the disturbance from how a load or support is attached dies out within
    # roughly one attachment-width, so that is the radius used.
    collar = np.zeros(N, bool)

    def _ball(seed_mask, radius):
        """Nodes within `radius` of any seed node."""
        out = np.zeros(N, bool)
        if not seed_mask.any() or radius <= 0:
            return out
        try:
            from scipy.spatial import cKDTree
            return cKDTree(points_m[seed_mask]).query(points_m)[0] <= radius
        except Exception:
            # No KD-tree: fall back to a span-wise slab around the seeds.
            lo = float(coord[seed_mask].min()) - radius
            hi = float(coord[seed_mask].max()) + radius
            return (coord >= lo) & (coord <= hi)

    if fixed_nodes.any() and (~fixed_nodes).any():
        # Saint-Venant radius: the disturbance from *how* a part is held dies
        # out within roughly one attachment width, so that is the radius.
        # The 6%-of-span floor covers the case this was written for -- a
        # narrow attachment (a pointed tip, a single bolt) whose own width is
        # far too small to contain the singularity it creates. Going wider
        # than one width would start discarding the genuine peak bending
        # stress at the wall, which is the number the part is designed by.
        bc_xy = points_m[fixed_nodes]
        width = float(np.ptp(bc_xy[:, 1 - ax])) if bc_xy.shape[0] > 1 else 0.0
        r_sv = float(np.clip(max(width, 0.06 * span),
                             0.05 * span, 0.20 * span))
        collar = _ball(fixed_nodes, r_sv)

    # The loaded end gets a smaller collar: a distributed traction is a far
    # milder artefact than a clamp, and on a cantilever the tip is exactly
    # where the user most wants to see honest (low) stress.
    if load_nodes.any() and (~load_nodes).any():
        lo_xy = points_m[load_nodes]
        lw = float(np.ptp(lo_xy[:, 1 - ax])) if lo_xy.shape[0] > 1 else 0.0
        r_ld = float(np.clip(max(lw, 0.03 * span), 0.02 * span, 0.08 * span))
        collar |= _ball(load_nodes, r_ld)

    # Re-entrant corners, and any node held by a single pin.
    #
    # These are the other two places the field is singular, and neither is
    # anywhere near the supports, so the Saint-Venant collar above misses
    # both. Until they were excluded the reported peak had to be an area
    # percentile to stay stable -- and that percentile is what made the app
    # report 35.8 MPa for a plate whose hole rim genuinely reaches 78.3. A
    # stress concentration is localised by definition, so it occupies far
    # less than the top half-percent of a plate's AREA and an area percentile
    # steps straight over it. Excluding the singularities by geometry instead
    # lets the honest maximum be reported.
    try:
        wedge = _wedge_angles(points_m, np.asarray(tris), N)
        # BOUNDARY nodes only. An interior node is surrounded by material and
        # reads 2*pi by construction, so testing the whole array against pi
        # marks the entire inside of the part as re-entrant -- the collar then
        # covers everything, trips the "collar ate the part" guard, and
        # switches itself off completely. That is what the first version did,
        # and the symptom was the L-bracket's reported peak going back to
        # tracking the raw singular maximum with 0 nodes excluded.
        on_edge = _boundary_nodes(np.asarray(tris), N)
        # Beyond pi the wedge is re-entrant and the stress is unbounded. The
        # margin keeps an ordinary straight edge, whose nodes read pi to
        # within meshing noise, out of it.
        reentrant = on_edge & (wedge > (np.pi * REENTRANT_WEDGE))
        if reentrant.any():
            collar |= _ball(reentrant, REENTRANT_R * span)
    except Exception:
        pass
    if pin_picks:
        pin_nodes = np.zeros(N, bool)
        for q in pin_picks:
            px_ = float(q.get("x", 0.0)) * m_per_px
            py_ = float(q.get("y", 0.0)) * m_per_px
            pin_nodes[int(np.argmin((points_m[:, 0] - px_) ** 2 +
                                    (points_m[:, 1] - py_) ** 2))] = True
        collar |= _ball(pin_nodes, 0.04 * span)

    keep = ~collar
    if int(keep.sum()) < max(12, int(0.05 * N)):   # collar ate the part
        keep = np.ones(N, bool)
        singular_excluded = 0
    else:
        singular_excluded = int(collar.sum())

    # Percentiles are taken over AREA, not over node count. A plain nodal
    # percentile moves every time the mesh changes -- refine near a hole and
    # you add a cloud of low-stress nodes that drags the percentile down, so
    # the colour scale would shift for reasons that have nothing to do with
    # the physics. Weighting by each node's share of the plate makes the
    # reference mesh-independent.
    clean = vm_node[keep]
    wts = _nodal_areas(points_m, np.asarray(tris), N)[keep]
    ref = _wpercentile(clean, wts, REF_PCTL)
    peak_true = float(vm_node.max()) if vm_node.size else 0.0
    # The reported peak is the MAXIMUM of the non-singular field, not a
    # percentile of it. Once the singular regions are excluded by geometry
    # there is nothing left for a percentile to protect against, and a
    # percentile actively hides the thing the tool exists to find: a hole's
    # concentration is a tiny patch of area, so any area percentile reports
    # the plate around it instead of the rim.
    peak_clean = float(clean.max()) if clean.size else 0.0
    if ref <= 0:
        ref = peak_true
    if ref <= 0:
        ref = 1.0

    vm_lin = np.clip(vm_node / ref, 0.0, 1.0)

    # ---- the strong/weak field ------------------------------------------
    # ONE field does two jobs: it is what the stress map paints, and it is
    # what the pocketing engine thresholds against. They must be the same
    # array or the picture stops explaining the pocket plan -- red means
    # "this carries load, keep material here" and green means "this is along
    # for the ride, pocket it".
    #
    # A raw sigma/sigma_ref ratio is a bad field for that job. Stress is
    # heavily skewed: a cantilever puts most of its AREA at a small fraction
    # of its peak, so 80% of the part lands under 0.25 -- one flat colour on
    # the map, and everything below the pocket threshold at once, so the
    # engine can't tell a lightly-loaded web from genuinely dead material.
    # A single monotonic gamma, chosen so the area-median sits mid-scale,
    # spreads the part across the full range. It never reorders two nodes --
    # if A is redder than B, A really does carry more stress than B -- so the
    # ranking the pocketing depends on is untouched, and the true MPa behind
    # each scale stop is reported so the legend stays honest.
    gamma = 1.0
    med = _wpercentile(clean, wts, 50.0) / ref if ref > 0 else 0.5
    if 1e-6 < med < 0.35:
        gamma = float(np.clip(np.log(0.35) / np.log(med), 0.40, 1.0))
    vm_norm = np.power(vm_lin, gamma)
    ramp_mpa = [float(ref * (s ** (1.0 / gamma))) for s in
                (0.0, 0.25, 0.5, 0.75, 1.0)]

    # "% of peak" for the callout badges. The divisor is the hottest node the
    # part actually converges to -- the highest stress OUTSIDE the collar --
    # not the singular clamp value, which grows without bound as you refine
    # and would make every real hot spot read as a rounding error. Uncapped by
    # construction: exactly one node reads 100, so three badges on three
    # different concentrations get three different numbers instead of all
    # saturating at 100%.
    hot_ref = float(clean.max()) if clean.size else peak_true
    if hot_ref <= 0:
        hot_ref = ref
    vm_pct = np.clip(vm_node / hot_ref, 0.0, 1.0)

    callouts = _callouts(points, np.asarray(tris), vm_pct, vm_norm, keep)
    hole_stress = _hole_stress(points, np.asarray(tris), vm_pct, keep)

    # Where it is held and where the force goes, handed back so the map can
    # say so out loud. A stress picture with no boundary conditions drawn on
    # it is not wrong, it is unreadable: the reader cannot tell a hot clamp
    # from a hot fillet, and cannot check that the load went on the end they
    # meant. Built from the solve's own facet sets -- see _bc_payload.
    try:
        bc = _bc_payload(m, points, np.asarray(tris), N, ax, trans_dof,
                         load_case, load, fixed_facets, fixed_nodes,
                         load_facets, load_nodes, applied,
                         float(np.ptp(points[:, ax])),
                         float(max(np.ptp(points[:, 0]),
                                   np.ptp(points[:, 1]))),
                         float(min(np.ptp(points[:, 0]),
                                   np.ptp(points[:, 1]))),
                         picked=bool(picked_support or load_picks))
    except Exception as ex:
        bc = {"error": str(ex)}

    return {
        # supports and load arrows, in the same coordinates as "nodes"
        "bc": bc,
        "nodes": points.tolist(),
        "tris": np.asarray(tris).tolist(),
        "von_mises": vm_node.tolist(),
        # Signed Cartesian components at the same nodes, in pascals. These are
        # what a closed-form check needs; see _project above.
        "sigma_xx": sxx_node.tolist(),
        "sigma_yy": syy_node.tolist(),
        "sigma_xy": sxy_node.tolist(),
        # shared by the stress map AND the pocketing engine
        "von_mises_norm": vm_norm.tolist(),
        # straight sigma/sigma_ref, kept for anyone who needs true ratios
        "von_mises_linear": vm_lin.tolist(),
        # fraction of the hottest converged node, for the "% peak" callouts
        "von_mises_pct": vm_pct.tolist(),
        "hot_ref_vm": hot_ref,
        # labelled points, picked where the collar mask is known
        "callouts": callouts,
        # every bore on the part, with the stress on its own rim
        "hole_stress": hole_stress,
        # the cut lines those labels were bucketed with, so the legend and the
        # pocketing engine can quote the same numbers instead of two guesses
        "bands": {name: cut for cut, name in BANDS},
        "disp_gamma": gamma,
        "ramp_vm": ramp_mpa,
        "disp": disp.tolist(),
        # peak_vm is the value the safety factor should use: the highest
        # *converged* stress. peak_vm_raw is the singular maximum, kept for
        # reference but not trustworthy as an engineering number.
        "peak_vm": peak_clean if peak_clean > 0 else peak_true,
        "peak_vm_raw": peak_true,
        "scale_ref_vm": ref,
        "singular_nodes_excluded": singular_excluded,
        "max_disp_m": float(disp.max()) if disp.size else 0.0,
        "element": "P2 (quadratic)",
        "load_applied_as": applied,
        "n_nodes": int(N),
        "n_dofs": int(basis.N),
        "n_elems": int(np.asarray(tris).shape[0]),
        "load_case": load_case,
        "load_units": load_units,
        "picked_features": int(len(picks)),
    }
