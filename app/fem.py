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
# Percentile used for the *reported* peak / safety factor.
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


def solve_plane_stress(points, tris, E=69e9, nu=0.33, thickness=0.00635,
                       load_case="cantilever", orientation="horizontal",
                       load=500.0, m_per_px=1.0):
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

    # Constraining boundary facets (rather than every node inside a 6%-wide
    # slab of material) keeps the support where a real clamp actually is.
    if fixed_facets.size:
        D = np.asarray(basis.get_dofs(fixed_facets).flatten(), dtype=np.int64)
        fixed_nodes = (coord <= s0 + fband)
        if two_ends:
            fixed_nodes |= (coord >= s1 - fband)
    else:                                     # degenerate geometry fallback
        fixed_nodes = coord <= s0 + fband
        nd = basis.nodal_dofs
        D = np.concatenate([nd[0][fixed_nodes], nd[1][fixed_nodes]])

    # ---- Neumann loading ------------------------------------------------
    # Distribute by *length* (or area), never by node count: the total force
    # must be independent of how finely the mesh happens to be cut.
    f = np.zeros(basis.N)
    applied = "traction"
    load_nodes = np.zeros(N, bool)     # nodes the load is attached to

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

    if load_case == "ss_dist":
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
    peak_clean = _wpercentile(clean, wts, PEAK_PCTL)
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

    return {
        "nodes": points.tolist(),
        "tris": np.asarray(tris).tolist(),
        "von_mises": vm_node.tolist(),
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
    }
