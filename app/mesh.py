"""
Quality unstructured meshing — the accuracy upgrade over the JS staircase grid.

Takes a part outline (exterior polygon + hole polygons) and produces a
conforming, quality-constrained Delaunay triangle mesh via Jonathan Shewchuk's
Triangle (through the `triangle` Python wrapper). Triangles follow the true
boundary and refine automatically, which is the single biggest driver of FEM
accuracy — far better than sampling a pixel grid.
"""
from __future__ import annotations
import numpy as np
import triangle as tr


def _ring_to_pslg(points, segments, ring, start_index):
    """Append a closed ring (list of (x, y)) as PSLG vertices + segments."""
    n = len(ring)
    for p in ring:
        points.append([float(p[0]), float(p[1])])
    for i in range(n):
        a = start_index + i
        b = start_index + (i + 1) % n
        segments.append([a, b])
    return start_index + n


# Mesh density presets. Each is (first-pass triangle target, ceiling on the
# refined mesh). The first number sets the max triangle area from the part's
# own bounding box, so it means the same thing on a 50 mm bracket and a 750 mm
# rail; the second is what the adaptive pass is allowed to grow to.
#
# "standard" is deliberately finer than the old fixed default (5000 / 26000),
# because the fair complaint about the first release was that hole rims read
# as polygons. Everything above it costs real solve time: the elements are
# quadratic, so doubling the triangle count roughly doubles the degrees of
# freedom and more than doubles the factorisation, and this runs on one shared
# vCPU. That is why the element count is reported rather than hidden.
DENSITY_PRESETS = {
    "draft":    (2500, 12000),
    "standard": (6500, 26000),
    "fine":     (13000, 36000),
    "ultra":    (18000, 45000),
}
DEFAULT_DENSITY = "standard"


def density_preset(name):
    """(first-pass triangle target, refined-mesh ceiling) for a preset name."""
    return DENSITY_PRESETS.get(str(name or "").lower(),
                               DENSITY_PRESETS[DEFAULT_DENSITY])


def build_mesh(exterior, holes, max_area=None, min_angle=30.0,
               return_pslg=False, target_tris=None):
    """
    exterior : list[(x, y)]           outer boundary, CCW or CW (any)
    holes    : list[list[(x, y)]]     inner boundaries (each a closed ring)
    max_area : float | None           max triangle area (auto if None)
    target_tris : roughly how many triangles the first pass should aim for,
                  used only when max_area is None. The cap is derived from the
                  part's own bounding box, so one number gives comparable
                  resolution across parts of very different size.
    return_pslg : also return the PSLG dict + min_angle so the mesh can be
                  adaptively refined later without rebuilding the geometry.
    returns  : (points Nx2 float64, tris Mx3 int)  — the FEM mesh
    """
    points, segments = [], []
    idx = 0
    idx = _ring_to_pslg(points, segments, exterior, idx)
    hole_seeds = []
    for h in holes:
        if len(h) < 3:
            continue
        idx = _ring_to_pslg(points, segments, h, idx)
        # a seed point strictly inside the hole so Triangle carves it out
        hx = float(np.mean([p[0] for p in h]))
        hy = float(np.mean([p[1] for p in h]))
        hole_seeds.append([hx, hy])

    pts = np.array(points, dtype=np.float64)
    if max_area is None:
        # Aim for `target_tris` triangles, sized off the exterior bounding box.
        #
        # QUALITY_OVERSHOOT is measured, not guessed: Triangle's 30-degree
        # quality constraint inserts roughly 60% more triangles than the area
        # cap alone would give, so dividing the box straight by the target
        # overshoots it by that much every time. Pre-dividing means the preset
        # names mean what they say -- "6500" delivers about 6500.
        #
        # The 1.5 px^2 floor is a guard against a pathological outline asking
        # for sub-pixel triangles and taking the mesher down with it.
        QUALITY_OVERSHOOT = 1.6
        tgt = float(np.clip(float(target_tris or 5000.0), 500.0, 60000.0))
        bb = (pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min())
        max_area = max(1.5, bb / (tgt / QUALITY_OVERSHOOT))

    A = {"vertices": pts, "segments": np.array(segments, dtype=np.int32)}
    if hole_seeds:
        A["holes"] = np.array(hole_seeds, dtype=np.float64)

    # p = respect PSLG, q = quality (min angle), a = max area, D = conforming Delaunay
    flags = f"pq{min_angle:g}a{max_area:g}D"
    B = tr.triangulate(A, flags)
    verts = np.asarray(B["vertices"], dtype=np.float64)
    tris = np.asarray(B["triangles"], dtype=np.int64)
    if not return_pslg:
        return verts, tris
    return verts, tris, A, min_angle


def refine_by_field(verts, tris, pslg, node_field, min_angle=30.0,
                    frac=0.30, shrink=0.28, max_tris=32000):
    """Cut the elements that matter smaller, then re-solve.

    A uniform mesh spends most of its triangles on the calm middle of the
    plate and too few where the stress actually turns -- hole edges, fillets,
    the clamp. This takes the first solve's stress field, picks the top
    `frac` of elements by (stress x element size), and re-triangulates with a
    per-triangle area cap only on those. Everything else keeps its original
    size, so the mesh grows where accuracy is limited and nowhere else.

    Returns (verts, tris) -- the input mesh unchanged if refinement fails.
    """
    verts = np.asarray(verts, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64)
    field = np.asarray(node_field, dtype=np.float64).ravel()
    if field.size < verts.shape[0] or tris.shape[0] >= max_tris:
        return verts, tris

    p = verts[tris]
    area = 0.5 * np.abs(
        (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) -
        (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    f = field[tris]
    # Error indicator: how much the field swings across the element, scaled by
    # the element's own size. Flat regions score ~0 no matter how stressed.
    swing = f.max(axis=1) - f.min(axis=1)
    score = (swing + 0.15 * f.mean(axis=1)) * np.sqrt(np.maximum(area, 0.0))
    if not np.isfinite(score).any() or score.max() <= 0:
        return verts, tris

    # Refine the hottest `frac` of elements; if that overshoots the ceiling,
    # refine a smaller share rather than giving up.
    #
    # Bailing out on overshoot is the wrong failure: it hands back a mesh with
    # NO refinement at all, so asking for a finer setting can leave hole rims
    # coarser than the setting below it. Halving the share twice costs two
    # cheap re-triangulations (no solve) and lands under the ceiling with the
    # refinement concentrated on the elements that needed it most.
    def _try(fr):
        caps = np.full(tris.shape[0], -1.0)        # -1 = no constraint
        cut = np.percentile(score, 100.0 * (1.0 - fr))
        hot = score >= max(cut, 1e-30)
        if not hot.any():
            return None
        caps[hot] = np.maximum(area[hot] * shrink, 1e-9)
        A = {"vertices": verts, "triangles": tris,
             "triangle_max_area": caps.reshape(-1, 1)}
        for k in ("segments", "holes"):
            if pslg and k in pslg:
                A[k] = pslg[k]
        try:
            # r = refine an existing triangulation, a (bare) = honour the
            # per-triangle area attribute, q = keep quality during refinement.
            B = tr.triangulate(A, f"prq{min_angle:g}aD")
            return (np.asarray(B["vertices"], dtype=np.float64),
                    np.asarray(B["triangles"], dtype=np.int64))
        except Exception:
            return None

    for fr in (frac, frac * 0.5, frac * 0.25, frac * 0.12):
        got = _try(fr)
        if got is None:
            continue
        v2, t2 = got
        if t2.shape[0] < tris.shape[0]:
            continue
        if t2.shape[0] <= max_tris:
            return v2, t2
    return verts, tris
