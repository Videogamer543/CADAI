"""
3D tetrahedral linear-elastic FEM — the solver behind the solid stress map.

This is a real volume solve, not the flat one painted onto a 3D shape. Every
node has three displacement degrees of freedom, the stress tensor has all six
components, and the von Mises field is recovered from the full tensor. That
matters most for exactly the parts you'd uncheck "2D plate" for: a bracket with
a bent flange, a gearbox plate with a boss, a welded tube. Plane stress assumes
the geometry is a constant-thickness slice loaded in its own plane, and none of
those are.

Conventions are deliberately shared with the 2D solver in fem.py so the two
pictures mean the same thing:

* the colour scale is referenced to a high percentile of the field with a
  Saint-Venant collar around the supports removed, because clamp stress in
  linear elasticity is singular and does not converge;
* percentiles are weighted by each node's share of the VOLUME, so refining the
  mesh near a fillet does not slide the colour scale;
* the same gamma stretch puts the volume-median at mid-scale, so a part is not
  one flat green with a red speck;
* the same four bands (critical / high / caution / low) label the callouts.

P2 (10-node) tetrahedra are used for the same reason the 2D solver uses P2
triangles: a 4-node tet has constant strain, is severely over-stiff in bending,
and reports a stress field that is piecewise-flat. In 3D that costs about seven
times the degrees of freedom of P1 on the same mesh, so the mesh is sized to
keep the total solvable rather than the element count impressive.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse.linalg import spsolve
from skfem import (
    MeshTet, Basis, FacetBasis, ElementTetP1, ElementTetP2, ElementVector,
    BilinearForm, LinearForm, Functional, asm, condense, solve,
)
from skfem.helpers import ddot, sym_grad, trace

from .fem import REF_PCTL, PEAK_PCTL, BANDS, _band, _wpercentile


def _lame3(E: float, nu: float):
    """True 3D Lamé constants.

    Not the plane-stress pair fem.py uses. lambda = E*nu/((1+nu)(1-2nu)) is the
    one that belongs in a volume solve; feeding the plane-stress value to a
    tetrahedral model makes the part measurably too soft in hydrostatic
    compression, which is precisely where a thick section differs from a plate.
    """
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


def boundary_faces(tets):
    """The triangles that bound a tet mesh: every face used by exactly one tet.

    Generic on purpose. Both mesh sources -- gmsh's own surface elements and the
    extruded 2D triangulation -- go through this, so the surface can never
    disagree with the volume it is supposed to wrap. Extracting it from the tets
    themselves also means no bookkeeping about which way a prism was split.
    """
    t = np.asarray(tets, np.int64)
    f = np.vstack([t[:, [0, 1, 2]], t[:, [0, 1, 3]],
                   t[:, [0, 2, 3]], t[:, [1, 2, 3]]])
    key = np.sort(f, axis=1)
    _, first, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    return f[first[cnt == 1]]


def extrude(pts2d, tris2d, thickness_m, m_per_px, layers=4):
    """A flat triangulation -> a tet mesh of the real plate.

    Used when the upload is an image rather than a STEP file: there is no solid
    to mesh, but there is an outline and a thickness, and those define one. The
    prism over each triangle is split into three tets by SORTED GLOBAL INDEX,
    which is what makes the split consistent -- two neighbouring prisms share a
    quad face, and if they disagree about which way its diagonal runs the mesh
    is torn along that face and the solve is meaningless.

    Four layers, not one. The whole reason to run this in 3D is to let the
    section bend, and bending needs at least a couple of elements through the
    thickness to represent the linear strain that does the work.
    """
    p = np.asarray(pts2d, float) * float(m_per_px)
    t = np.asarray(tris2d, np.int64)
    n = p.shape[0]
    L = max(2, int(layers))
    z = np.linspace(-0.5 * thickness_m, 0.5 * thickness_m, L + 1)
    nodes = np.concatenate(
        [np.column_stack([p, np.full(n, zk)]) for zk in z], axis=0)

    s = np.sort(t, axis=1)                  # a<b<c, globally consistent
    a, b, c = s[:, 0], s[:, 1], s[:, 2]
    tets = []
    for k in range(L):
        lo, hi = k * n, (k + 1) * n
        a0, b0, c0 = a + lo, b + lo, c + lo
        a1, b1, c1 = a + hi, b + hi, c + hi
        tets.append(np.column_stack([a0, b0, c0, c1]))
        tets.append(np.column_stack([a0, b0, c1, b1]))
        tets.append(np.column_stack([a0, b1, c1, a1]))
    return nodes, np.vstack(tets)


def _fix_orientation(nodes, tets):
    """Make every tet positively oriented. A negative Jacobian is a mesh with a
    hole punched in its stiffness matrix, and skfem will not warn about it."""
    p = np.asarray(nodes, float)
    t = np.asarray(tets, np.int64).copy()
    d = p[t[:, 1:]] - p[t[:, [0]]]
    bad = np.linalg.det(d) < 0
    if bad.any():
        t[bad] = t[bad][:, [0, 2, 1, 3]]
    return t


def _pca_frame(p):
    """The part's own axes: longest, mid, thinnest.

    Taken from the node cloud rather than the world bounding box, because a
    STEP file is exported in whatever frame the modeller happened to use. A rail
    drawn on the diagonal has a near-cubic world bbox and no obvious span
    direction; its own principal axes have one.
    """
    c = p.mean(axis=0)
    _, sv, vt = np.linalg.svd(p - c, full_matrices=False)
    order = np.argsort(-sv)
    ax = vt[order]
    # Extent along each axis, which is what the load case actually cares about
    # -- a singular value is a spread, not a length.
    ext = np.array([np.ptp((p - c) @ ax[i]) for i in range(3)])
    order2 = np.argsort(-ext)
    return c, ax[order2], ext[order2]


def _callouts(xyz, vm_pct, vm_norm, keep, on_surface, vm_pa=None,
              hot_cut=0.32, n_hot=14, n_cool=8):
    """Named weak and strong points, picked on the SURFACE only.

    A callout buried in the middle of a solid labels something nobody can see,
    and on a thick part most of the volume is interior. Suppression is by true
    3D distance so two badges never land on opposite faces of the same thin wall
    and read as two separate findings.
    """
    idx = np.where(keep & on_surface)[0]
    if idx.size == 0:
        return []
    span = float(np.ptp(xyz, axis=0).max()) or 1.0
    gap = span * 0.10

    def pick(cand, rank, limit):
        out = []
        for i in cand[np.argsort(-rank[cand])]:
            if len(out) >= limit:
                break
            if all(np.linalg.norm(xyz[i] - xyz[j]) > gap for j in out):
                out.append(int(i))
        return out

    def mpa(i):
        # The real stress at the node, so a hover can quote a number instead of
        # only a colour. Absent for a caller that has no field to hand.
        return None if vm_pa is None else float(vm_pa[i])

    hot = idx[vm_pct[idx] >= hot_cut]
    cool = idx[vm_norm[idx] <= 0.38]
    res = []
    for i in pick(hot, vm_pct, n_hot):
        res.append({"x": float(xyz[i, 0]), "y": float(xyz[i, 1]),
                    "z": float(xyz[i, 2]), "kind": "hot",
                    "pct": float(vm_pct[i]), "vm": mpa(i),
                    "band": _band(float(vm_norm[i]))})
    for i in pick(cool, -vm_norm, n_cool):
        res.append({"x": float(xyz[i, 0]), "y": float(xyz[i, 1]),
                    "z": float(xyz[i, 2]), "kind": "safe",
                    "pct": float(vm_pct[i]), "vm": mpa(i), "band": "low"})
    return res


def _mkl_spsolve():
    """scipy's spsolve (SuperLU) or Intel PARDISO if it can be reached.

    SuperLU factorises a 3D elasticity matrix by brute force: on the 180k-dof
    test part it took 237 s, which is not a web request. PARDISO does the same
    job on the same matrix in a few seconds because it is multithreaded and
    knows the matrix is symmetric. It is an optional dependency, so everything
    here degrades quietly -- a missing pypardiso, or an MKL runtime pip did not
    put where the loader looks, just means the slower path.
    """
    import os
    import ctypes.util
    if not os.environ.get("PYPARDISO_MKL_RT") and \
            not ctypes.util.find_library("mkl_rt"):
        for cand in ("/usr/local/lib/libmkl_rt.so.3",
                     "/usr/lib/x86_64-linux-gnu/libmkl_rt.so.3"):
            if os.path.exists(cand):
                os.environ["PYPARDISO_MKL_RT"] = cand
                break
    try:
        from pypardiso import spsolve as _ps
        return _ps, "PARDISO (MKL)"
    except Exception:
        return None, None


def _linsolve(K, f, D):
    """Solve K u = f with the essential dofs in D held at zero.

    Ordered fastest-first with a fallback at every step, because the machine
    this ships to is a Windows laptop with whatever wheels happened to install.
    """
    sysm = condense(K, f, D=D)
    ps, name = _mkl_spsolve()
    if ps is not None:
        try:
            A = sysm[0].tocsr()
            u = np.zeros(K.shape[0])
            u[sysm[3]] = ps(A, np.asarray(sysm[1], float))
            if np.all(np.isfinite(u)):
                return u, name
        except Exception:
            pass
    return solve(*sysm), "SuperLU (scipy)"


def solve_solid(nodes_mm, tets, *, E=69e9, nu=0.33, load_case="cantilever",
                orientation="horizontal", load=500.0, quadratic=True):
    """
    nodes_mm : (N,3) node coordinates in MILLIMETRES (STEP's native unit)
    tets     : (M,4) tetrahedra
    load     : total applied force, newtons

    Returns a payload shaped for the 3D viewer: surface geometry, a normalised
    von Mises value per surface vertex, banded callouts, and the same scale
    metadata the 2D map reports.
    """
    p_mm = np.asarray(nodes_mm, float)
    p = p_mm * 1e-3                                   # solve in SI
    t = _fix_orientation(p, tets)
    m = MeshTet(p.T.copy(), t.T.copy())

    e = ElementVector(ElementTetP2() if quadratic else ElementTetP1())
    basis = Basis(m, e, intorder=2)
    sbasis = Basis(m, ElementTetP1(), intorder=2)
    N = p.shape[0]

    lam, mu = _lame3(E, nu)

    @BilinearForm
    def stiffness(u, v, w):
        return (2.0 * mu * ddot(sym_grad(u), sym_grad(v)) +
                lam * trace(sym_grad(u)) * trace(sym_grad(v)))

    K = asm(stiffness, basis)

    # ---- load case, in the part's own frame -----------------------------
    ctr, ax, ext = _pca_frame(p)
    span_ax = ax[0]
    # "Horizontal" loads the part edgewise (along its wider cross-section
    # direction); "vertical" loads it flatwise, across the thin direction. On a
    # plate that is the difference between a beam on edge and a beam laid flat,
    # which is most of the answer, so it is reported back rather than assumed
    # silently.
    load_ax = ax[1] if orientation == "horizontal" else ax[2]
    s = (p - ctr) @ span_ax
    s0, s1 = float(s.min()), float(s.max())
    span = max(s1 - s0, 1e-12)
    band, fband = span * 0.06, span * 0.02
    two_ends = load_case in ("ss_center", "ss_dist", "fixed_fixed")

    def coord_of(x):
        return (np.asarray(x).T - ctr) @ span_ax

    def facets_where(fn):
        try:
            return np.asarray(m.facets_satisfying(fn), np.int64)
        except Exception:
            return np.array([], np.int64)

    fix_lo = facets_where(lambda x: coord_of(x) <= s0 + fband)
    if fix_lo.size == 0:
        fband = span * 0.05
        fix_lo = facets_where(lambda x: coord_of(x) <= s0 + fband)
    fix_hi = facets_where(lambda x: coord_of(x) >= s1 - fband) if two_ends \
        else np.array([], np.int64)
    fixed_facets = np.unique(np.concatenate([fix_lo, fix_hi])) \
        if fix_hi.size else fix_lo

    fixed_nodes = (s <= s0 + fband)
    if two_ends:
        fixed_nodes |= (s >= s1 - fband)
    if fixed_facets.size:
        D = np.asarray(basis.get_dofs(fixed_facets).flatten(), np.int64)
    else:
        nd = basis.nodal_dofs
        D = np.concatenate([nd[i][fixed_nodes] for i in range(3)])

    # ---- Neumann traction -----------------------------------------------
    f = np.zeros(basis.N)
    applied = "traction"
    load_nodes = np.zeros(N, bool)

    def facet_area(fac):
        if fac is None or np.asarray(fac).size == 0:
            return 0.0

        @Functional
        def one(w):
            return 1.0 + 0.0 * w.x[0]
        try:
            return float(asm(one, FacetBasis(m, ElementTetP1(),
                                             facets=fac, intorder=2)))
        except Exception:
            return 0.0

    if load_case == "ss_dist":
        @Functional
        def volume(w):
            return 1.0 + 0.0 * w.x[0]
        V = float(asm(volume, sbasis)) or 1.0
        body = load / V                                   # N/m^3

        @LinearForm
        def bodyload(v, w):
            d = np.zeros_like(v[0])
            for i in range(3):
                d = d + load_ax[i] * v[i]
            return -body * d
        f = asm(bodyload, basis)
        applied = "body force"
    else:
        if load_case in ("ss_center", "fixed_fixed"):
            mid = 0.5 * (s0 + s1)
            lf = facets_where(lambda x: np.abs(coord_of(x) - mid) <= band)
        else:
            lf = facets_where(lambda x: coord_of(x) >= s1 - band)
        A = facet_area(lf)
        if lf.size and A > 0:
            trac = load / A                               # N/m^2

            @LinearForm
            def edgeload(v, w):
                d = np.zeros_like(v[0])
                for i in range(3):
                    d = d + load_ax[i] * v[i]
                return -trac * d
            f = asm(edgeload, FacetBasis(m, e, facets=lf, intorder=2))
            try:
                vids = np.unique(np.asarray(m.facets)[:, lf])
                load_nodes[vids[(vids >= 0) & (vids < N)]] = True
            except Exception:
                pass
        else:
            sel = (s >= s1 - band) & (~fixed_nodes)
            idx = np.where(sel)[0]
            if idx.size:
                for i in range(3):
                    f[basis.nodal_dofs[i][idx]] += -load * load_ax[i] / idx.size
                load_nodes[idx] = True
            applied = "lumped nodal (no facets found)"

    u, solver_name = _linsolve(K, f, D)

    nd = basis.nodal_dofs
    disp = np.sqrt(sum(u[nd[i]] ** 2 for i in range(3)))

    # ---- full 3D stress tensor at the quadrature points ------------------
    g = basis.interpolate(u).grad                  # (3,3,nelem,nqp)
    eps = [[0.5 * (g[i, j] + g[j, i]) for j in range(3)] for i in range(3)]
    tr = eps[0][0] + eps[1][1] + eps[2][2]
    sig = [[2.0 * mu * eps[i][j] + (lam * tr if i == j else 0.0)
            for j in range(3)] for i in range(3)]
    sxx, syy, szz = sig[0][0], sig[1][1], sig[2][2]
    sxy, syz, szx = sig[0][1], sig[1][2], sig[2][0]
    vmq = np.sqrt(np.maximum(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) +
        3.0 * (sxy ** 2 + syz ** 2 + szx ** 2), 0.0))

    @BilinearForm
    def mass(a, b, w):
        return a * b

    @LinearForm
    def rhs(v, w):
        return w["vm"] * v

    @LinearForm
    def unit(v, w):
        return v + 0.0 * w.x[0]

    M = asm(mass, sbasis)
    vm_node = spsolve(M.tocsr(), asm(rhs, sbasis, vm=vmq))
    vm_node = np.clip(np.nan_to_num(vm_node), 0.0,
                      float(vmq.max()) if vmq.size else 0.0)
    vm_node = np.asarray(vm_node, float).ravel()[:N]
    if vm_node.size < N:
        vm_node = np.pad(vm_node, (0, N - vm_node.size))
    # Each node's share of the volume: the 3D analogue of the nodal areas the
    # plane solver weights its percentiles by, and it comes straight out of the
    # basis rather than from a hand-rolled sum over elements.
    wts_all = np.abs(np.asarray(asm(unit, sbasis), float).ravel()[:N])

    # ---- Saint-Venant collar --------------------------------------------
    collar = np.zeros(N, bool)

    def ball(seed, radius):
        if not seed.any() or radius <= 0:
            return np.zeros(N, bool)
        try:
            from scipy.spatial import cKDTree
            return cKDTree(p[seed]).query(p)[0] <= radius
        except Exception:
            lo, hi = s[seed].min() - radius, s[seed].max() + radius
            return (s >= lo) & (s <= hi)

    if fixed_nodes.any() and (~fixed_nodes).any():
        w_bc = float(max(ext[1], 1e-9)) if fixed_nodes.sum() > 1 else 0.0
        collar = ball(fixed_nodes,
                      float(np.clip(max(w_bc, 0.06 * span),
                                    0.05 * span, 0.20 * span)))
    if load_nodes.any() and (~load_nodes).any():
        collar |= ball(load_nodes, float(np.clip(0.03 * span,
                                                 0.02 * span, 0.08 * span)))

    keep = ~collar
    if int(keep.sum()) < max(12, int(0.05 * N)):
        keep, singular = np.ones(N, bool), 0
    else:
        singular = int(collar.sum())

    # The colour scale is fitted to the SURFACE, not to the whole volume, and
    # this is the one place where the 3D map has to depart from the flat one.
    # A plate in bending is near zero stress along its neutral plane, and in a
    # solid mesh most nodes are interior -- so a volume-wide median sits far
    # below anything you can actually see. Fitting the gamma stretch to that
    # median lifts the whole visible skin into orange and the map reports a part
    # as hot everywhere. Only the surface is ever painted, so only the surface
    # gets a vote on the scale.
    surf = boundary_faces(t)
    sv = np.unique(surf)
    on_surface = np.zeros(N, bool)
    on_surface[sv] = True

    scale_set = keep & on_surface
    if int(scale_set.sum()) < max(12, int(0.02 * N)):
        scale_set = keep
    clean, wts = vm_node[scale_set], wts_all[scale_set]
    ref = _wpercentile(clean, wts, REF_PCTL)
    peak_true = float(vm_node.max()) if vm_node.size else 0.0
    peak_clean = _wpercentile(clean, wts, PEAK_PCTL)
    ref = ref if ref > 0 else (peak_true if peak_true > 0 else 1.0)

    vm_lin = np.clip(vm_node / ref, 0.0, 1.0)
    gamma = 1.0
    med = _wpercentile(clean, wts, 50.0) / ref
    if 1e-6 < med < 0.35:
        gamma = float(np.clip(np.log(0.35) / np.log(med), 0.40, 1.0))
    vm_norm = np.power(vm_lin, gamma)
    ramp = [float(ref * (x ** (1.0 / gamma))) for x in (0, .25, .5, .75, 1.)]

    hot_ref = float(clean.max()) if clean.size else peak_true
    vm_pct = np.clip(vm_node / (hot_ref if hot_ref > 0 else ref), 0.0, 1.0)

    # ---- surface payload -------------------------------------------------
    remap = np.full(N, -1, np.int64)
    remap[sv] = np.arange(sv.size)

    callouts = _callouts(p_mm, vm_pct, vm_norm, keep, on_surface, vm_pa=vm_node)

    mn, mx = p_mm.min(axis=0), p_mm.max(axis=0)
    return {
        "verts": np.round(p_mm[sv], 3).tolist(),
        "tris": remap[surf].astype(int).tolist(),
        "vnorm": np.round(vm_norm[sv], 4).tolist(),
        "vpct": np.round(vm_pct[sv], 4).tolist(),
        "callouts": callouts,
        "bands": {name: cut for cut, name in BANDS},
        "bbox": {"min": mn.tolist(), "max": mx.tolist(),
                 "spans": (mx - mn).tolist()},
        "axes": {"span": span_ax.tolist(), "load": load_ax.tolist(),
                 "extents_mm": (ext * 1e3).tolist(),
                 # Whether "the long axis" is a real feature of the part or an
                 # artefact of the mesh. On a round part the two large extents
                 # are equal to within meshing noise, so the PCA that picks the
                 # span axis is choosing between directions that differ by a
                 # fraction of a percent -- and it lands differently on every
                 # remesh. Measured on a 36-tooth pulley: the span axis flipped
                 # between -X and -Y across three mesh densities and the peak
                 # stress moved 8.5 / 22.8 / 1.6 MPa with it. That is not
                 # convergence, it is a coin toss, and a stress tool that hides
                 # it is worse than one that admits it.
                 "span_ambiguous": bool(ext[1] > 0.85 * ext[0]),
                 "span_margin": (float(ext[0] / ext[1])
                                 if ext[1] > 0 else 0.0)},
        "load_dir_name": ("edgewise (in the wide direction)"
                          if orientation == "horizontal"
                          else "flatwise (across the thin direction)"),
        "peak_vm": peak_clean if peak_clean > 0 else peak_true,
        "peak_vm_raw": peak_true,
        "scale_ref_vm": ref,
        "ramp_vm": ramp,
        "disp_gamma": gamma,
        "max_disp_m": float(disp.max()) if disp.size else 0.0,
        "singular_nodes_excluded": singular,
        "element": "P2 tet (quadratic)" if quadratic else "P1 tet (linear)",
        "load_applied_as": applied,
        "n_nodes": int(N),
        "n_dofs": int(basis.N),
        "n_elems": int(t.shape[0]),
        "n_surf_tris": int(surf.shape[0]),
        "load_case": load_case,
        "solver": "3D tetrahedral",
        "linear_solver": solver_name,
    }
