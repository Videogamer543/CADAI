"""
STEP → 3D solid tessellation.

Delegates to a subprocess (app/step_worker.py) so gmsh — which is not
thread-safe and installs signal handlers — always runs as the main thread of a
clean interpreter. This is the robust way to embed a CAD kernel in a web server:
no global-state clashes, no event-loop blocking, and a crash can't take down the
server. Falls back cleanly if gmsh isn't available.
"""
from __future__ import annotations
import os
import sys
import json
import tempfile
import subprocess


def tessellate(step_bytes: bytes, size_factor: float = 1.0, max_tris: int = 60000):
    """Returns dict: tris, normals, bbox{min,max,spans}, n_tris. Raises on failure."""
    with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
        f.write(step_bytes)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "app.step_worker", path,
             str(size_factor), str(max_tris)],
            capture_output=True, text=True, timeout=120,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or "gmsh worker failed").strip()[-300:])
        # worker prints the JSON as the last stdout line
        out = proc.stdout.strip().splitlines()
        if not out:
            raise RuntimeError("empty tessellation output")
        return json.loads(out[-1])
    finally:
        # Best-effort on purpose. On Windows a scanner or indexer can still
        # hold the file for a moment after the worker exits, and a stray
        # kilobyte in Temp is not worth failing a completed tessellation over.
        try:
            os.unlink(path)
        except Exception:
            pass


def min_width_axis(verts, iters=60):
    """(unit axis, width) of the narrowest direction of a point cloud.

    PCA alone is not good enough for this number, and the reason is worth
    keeping. On team 2813's parts the PCA thin axis comes out tilted from the
    true plate normal by around 0.04 degrees -- nothing, visually -- because
    the principal axes are weighted by where the mesher happened to put
    vertices, and a plate with a dense cluster of bolt holes at one end pulls
    them slightly. But width along a tilted axis picks up

        thickness * cos(theta) + span * sin(theta)

    and `span` here is up to 600 mm. 0.04 degrees of tilt against 250 mm of
    plate is 0.18 mm of phantom thickness, so a 1/4" plate measured 6.53 and a
    3/16" plate would have been within a whisker of reading as 1/4".

    So: start from the PCA guess and the three global axes, take whichever is
    already narrowest, then walk downhill over the sphere with a halving step.
    Width along an axis can never be less than the cloud's true minimal width,
    so every candidate is an upper bound and taking the smallest is always the
    better estimate -- there is no way for this to undershoot into a wrong
    answer, only to stop early at a slightly high one.
    """
    import numpy as np
    v = np.asarray(verts, dtype=np.float64)
    c = v.mean(axis=0)
    V = v - c
    _, _, vt = np.linalg.svd(V, full_matrices=False)
    cands = [np.asarray(a, float) for a in vt] + [
        np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]
    best = min(cands, key=lambda a: float(np.ptp(V @ a)))
    best = best / (np.linalg.norm(best) or 1.0)
    bw = float(np.ptp(V @ best))

    # Two directions perpendicular to `best`, to tilt within.
    seed = np.array([1.0, 0, 0]) if abs(best[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(best, seed); e1 /= (np.linalg.norm(e1) or 1.0)
    e2 = np.cross(best, e1)

    step = 0.02                      # radians-ish; ~1.1 degrees to start
    for _ in range(iters):
        moved = False
        for d in (e1, -e1, e2, -e2):
            a = best + step * d
            a /= (np.linalg.norm(a) or 1.0)
            w = float(np.ptp(V @ a))
            if w < bw - 1e-12:
                best, bw, moved = a, w, True
        if not moved:
            step *= 0.5
            if step < 1e-7:
                break
    return best, bw


def faces(tess):
    """The tessellation as an (M, 3, 3) array of triangle corner coordinates.

    app/step_worker.py ships INDEXED geometry -- a shared vertex table plus
    integer triangles -- because repeating every vertex six times turned a
    finer mesh into a double-digit-megabyte download. Everything in this module
    was written against the expanded form, and expanding is one fancy-index, so
    the conversion lives here rather than in five places.

    The old un-indexed payload is still accepted. That matters for more than
    tidiness: a tessellation can arrive from a cached result or an older worker,
    and a shape check is cheaper than a class of bug where the silhouette is
    silently computed from integers.
    """
    import numpy as np
    t = np.asarray(tess.get("tris"), dtype=np.float64)
    if t.size == 0:
        return np.zeros((0, 3, 3), dtype=np.float64)
    if t.ndim == 3 and t.shape[1:] == (3, 3):
        return t                                   # already expanded
    v = tess.get("verts")
    if v is None:
        raise ValueError("tessellation has neither expanded tris nor verts")
    v = np.asarray(v, dtype=np.float64)
    return v[np.asarray(tess["tris"], dtype=np.int64)]


def plate_frame(tess):
    """(centroid, u, v, normal, thickness_mm) for a tessellated plate.

    One function so the silhouette and the thickness readout can never
    disagree: they are the same principal axes, computed once.

    The thin direction comes from a PCA of the vertices, not from the bounding
    box. That distinction is the whole point. A plate exported at an angle --
    which is most of them, since nobody re-orients a part before exporting --
    has three large global bbox spans and no thin one, so a bbox reading of
    "thickness" on a tilted 3 mm plate can come back as 40 mm. The smallest
    principal axis is the plate normal whatever attitude the file is in, and
    the spread of the vertices along it is the stock thickness in millimetres,
    since STEP is always in mm.

    Note that the returned FRAME is the raw PCA one but the returned THICKNESS
    is measured along the refined minimum-width axis from min_width_axis(),
    which is a hair different. That is deliberate rather than sloppy. The frame
    is what silhouette_png rasterizes in, and every pocketing constant in this
    project was fitted against silhouettes drawn in the PCA frame, so rotating
    it -- even by the 0.04 degrees the refinement would move it -- puts every
    calibrated part at risk to fix a number that is not used for drawing. The
    thickness, by contrast, is a pure readout: it feeds the FEM and the
    pocketing removal target and nothing that gets rasterized, so it is free to
    be as accurate as it can be.
    """
    import numpy as np
    tris = faces(tess)
    if tris.size == 0:
        raise ValueError("empty tessellation")
    verts = tris.reshape(-1, 3)
    c = verts.mean(axis=0)
    # Rows of vt are in descending singular value, so vt[2] is the thin axis.
    _, _, vt = np.linalg.svd(verts - c, full_matrices=False)
    u_ax, v_ax, n_ax = vt[0], vt[1], vt[2]
    _, thickness_mm = min_width_axis(verts)
    return c, u_ax, v_ax, n_ax, float(thickness_mm)


def plate_aspect(verts):
    """(thickness, in-plane diameter) of a point cloud, both in mm.

    "Diameter" is the widest caliper measurement across the part in the plane
    perpendicular to its thin axis -- the max over a fan of directions, not the
    ptp along two chosen axes. That sounds fussy and isn't: a 300 x 300 plate
    has degenerate in-plane principal axes, so PCA is free to return them
    rotated 45 degrees, and the "width" then reads 424 mm instead of 300. Any
    aspect-ratio gate built on that would flip its answer depending on which
    way a numerically arbitrary tie broke. A caliper max is the same number
    whatever attitude the file is in.
    """
    import numpy as np
    v = np.asarray(verts, dtype=np.float64)
    axis, thin = min_width_axis(v)
    c = v.mean(axis=0)
    rel = v - c
    perp = rel - np.outer(rel @ axis, axis)
    seed = np.array([1.0, 0, 0]) if abs(axis[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(axis, seed); e1 /= (np.linalg.norm(e1) or 1.0)
    e2 = np.cross(axis, e1)
    ang = np.linspace(0.0, np.pi, 24, endpoint=False)
    dirs = np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2
    proj = perp @ dirs.T                                  # (N, 24)
    diameter = float((proj.max(axis=0) - proj.min(axis=0)).max())
    return float(thin), diameter


def measure_thickness(tess, plate_ratio=0.18):
    """Stock thickness in mm from a tessellation, or None if it isn't a plate.

    Deliberately returns None rather than a plausible-looking number for a
    genuine 3D body. "The narrowest dimension of this bellcrank" is not a
    thickness, and a tool that quietly types one into a box marked Thickness
    (mm) is worse than one that leaves the box alone.
    """
    import numpy as np
    try:
        verts = faces(tess).reshape(-1, 3)
        thin, diameter = plate_aspect(verts)
    except Exception:
        return None
    if thin <= 0 or diameter <= 0 or (thin / diameter) > plate_ratio:
        return None
    return round(thin, 3)


def silhouette_png(tess, out_px=1500):
    """Project a STEP tessellation onto the plate's own plane and rasterize a
    filled silhouette, so a STEP plate can flow through the same
    image -> mesh -> FEM pipeline with every hole intact.

    Two things make small holes (countersinks, #10 clearance, rivet holes)
    survive here where the naive version lost them:

      1. PCA plane fit, not the global bbox. A plate that sits tilted in the
         STEP file has no thin *global* axis, so projecting along X/Y/Z shears
         the top face against the bottom face by thickness*tan(tilt) - enough
         to seal any hole near the plate thickness in size. Fitting the actual
         plate plane removes that shear entirely.

      2. Only near-planar faces are rasterized. Side walls and countersink
         cones project straight into the bore and fill it in; dropping every
         triangle whose normal isn't parallel to the plate normal keeps bores
         open at their true diameter.
    """
    import numpy as np
    import cv2
    from PIL import Image
    import io

    tris = faces(tess)                                      # (M, 3, 3)
    if tris.size == 0:
        raise ValueError("empty tessellation")
    verts = tris.reshape(-1, 3)

    # --- 1. plate frame from PCA (thin direction = smallest singular value) ---
    c, u_ax, v_ax, n_ax, thickness_mm = plate_frame(tess)

    # --- 2. keep only faces lying in the plate plane ---
    nrm = np.asarray(tess.get("normals") or [], dtype=np.float64)
    if nrm.ndim != 2 or nrm.shape[0] != tris.shape[0]:
        e1 = tris[:, 1] - tris[:, 0]
        e2 = tris[:, 2] - tris[:, 0]
        nrm = np.cross(e1, e2)
        ln = np.linalg.norm(nrm, axis=1)
        ln[ln == 0] = 1.0
        nrm = nrm / ln[:, None]
    keep = np.abs(nrm @ n_ax) > 0.80
    if keep.sum() < 0.10 * len(tris):
        keep = np.ones(len(tris), bool)      # not really a plate - use everything
    tris = tris[keep]

    # --- 3. project onto the plate plane and rasterize ---
    rel = tris.reshape(-1, 3) - c
    p2 = np.stack([rel @ u_ax, rel @ v_ax], axis=1).reshape(-1, 3, 2)
    flat = p2.reshape(-1, 2)
    mn2 = flat.min(0)
    span2 = flat.max(0) - mn2
    span2[span2 == 0] = 1.0
    pad = 24

    # --- resolution cap -----------------------------------------------------
    # `scale` below IS pixels per millimetre, because STEP is always in mm.
    # With a FIXED out_px that number is inversely proportional to how big the
    # part is, which is exactly backwards: the smallest parts get the finest
    # raster and therefore the most work.
    #
    # Measured per part, in isolated processes, through the full analyze path:
    #
    #     part      span     px/mm    nodes   peak RSS   solve
    #     P4008    134 mm    10.82     3328    3726 MiB   126 s
    #     P5009    189 mm     7.69     4106    3787 MiB   173 s
    #     P4010    442 mm     3.29     6887     296 MiB    58 s
    #
    # A 134 mm plate costs twelve times the memory of a 442 mm one while having
    # half the nodes. The pocketing engine works in pixels -- rib_px is
    # rib_mm * px_per_mm and the morphological disks scale with it, so at
    # 10.8 px/mm the fillet disk carries a 65 px radius instead of 20, and both
    # the time and the intermediates explode. On a 2 GiB container the small
    # parts are simply killed, which reaches the browser as HTTP 503 with an
    # EMPTY body and no application error -- nothing survives to write one.
    #
    # So cap the resolution rather than the part. 5 px/mm puts 24 pixels across
    # a #10 clearance hole and 15 across an M3, far more than the silhouette
    # needs. Parts large enough that 5 px/mm would exceed the old 1500 px
    # budget are untouched: their scale is still set by out_px exactly as
    # before, so every large reference part rasterizes identically and its
    # fitted constants stay valid.
    #
    # MIN_OUT_PX floors genuinely tiny parts. A 40 mm bracket climbs back above
    # 5 px/mm, but on a 520 px raster instead of a 1500 px one -- and it is
    # raster AREA times disk radius that hurts, so it stays cheap anyway.
    TARGET_PPM = 5.0
    MIN_OUT_PX = 520
    out_px = int(np.clip(float(span2.max()) * TARGET_PPM + 2 * pad,
                         MIN_OUT_PX, out_px))

    scale = (out_px - 2 * pad) / span2.max()
    W = int(span2[0] * scale) + 2 * pad
    H = int(span2[1] * scale) + 2 * pad
    mask = np.zeros((H, W), np.uint8)
    polys = np.round((p2 - mn2) * scale + pad).astype(np.int32)
    # One fillConvexPoly per triangle. A single fillPoly over the whole set is
    # NOT equivalent: OpenCV scan-converts the batch with a parity rule, so
    # adjacent triangles cancel each other and the plate comes out as a wireframe.
    for poly in polys:
        cv2.fillConvexPoly(mask, poly, 255, cv2.LINE_8)
    # Stroke the edges too: rounding vertices to the pixel grid leaves 1 px
    # cracks between neighbouring triangles, and those cracks otherwise come
    # back as thousands of phantom "holes" during contour extraction.
    cv2.polylines(mask, list(polys), True, 255, 2, cv2.LINE_8)

    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[mask > 0] = (120, 120, 130, 255)   # part opaque; background transparent
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")

    # Real scale + real plate thickness, straight off the CAD model (STEP is in
    # mm), so the FEM can run in physical units instead of guessing from pixels.
    # thickness_mm came from plate_frame() above, measured on the FULL vertex
    # set -- before the near-planar filter at step 2 threw the side walls away.
    # Measuring it after that filter would read the thickness of the top face,
    # which is zero.
    return buf.getvalue(), float(scale), thickness_mm
