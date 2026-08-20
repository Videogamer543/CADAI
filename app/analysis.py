"""
High-level analysis orchestration: geometry → mesh → FEM → (modes, safety
factor) → optional pocketing. One call the API can use.
"""
from __future__ import annotations
import numpy as np
from .geometry import extract_polygons
from .mesh import build_mesh, refine_by_field
from .fem import solve_plane_stress
from . import materials, raster, pocketing

# analysis-mode stress-concentration multipliers (match the browser app)
MODE_KT = {"full": 1.0, "structural": 1.0, "fatigue": 1.35, "manufacturing": 1.15}


def run(data: bytes, *, material="Aluminum 6061-T6", mode="structural",
        load_case="cantilever", orientation="horizontal", load=500.0,
        do_pocketing=False, px_per_mm=None, rib_mm=3.0, density="normal",
        max_area=None, thickness_mm=6.35, adapt=True):
    exterior, holes, size = extract_polygons(data)
    W, H = size
    mat = materials.get(material)
    pts, tris, pslg, min_angle = build_mesh(exterior, holes, max_area=max_area,
                                            return_pslg=True)

    # px/mm — STEP supplies the true scale; for images assume long side ~150 mm
    if px_per_mm is None:
        px_per_mm = max(W, H) / 150.0

    kw = dict(E=mat["E"], nu=mat["nu"],
              thickness=max(0.0005, thickness_mm / 1000.0),
              m_per_px=1.0 / (px_per_mm * 1000.0),
              load_case=load_case, orientation=orientation, load=load)

    res = solve_plane_stress(pts, tris, **kw)

    # Adaptive pass: the first solve tells us where the stress field is
    # actually turning, so the second mesh puts its triangles there. This is
    # what makes hole edges and fillets read as real concentrations instead of
    # blurring into the background.
    res["adaptive_pass"] = False
    if adapt:
        try:
            pts2, tris2 = refine_by_field(pts, tris, pslg,
                                          res["von_mises_norm"],
                                          min_angle=min_angle)
            if tris2.shape[0] > tris.shape[0]:
                res2 = solve_plane_stress(pts2, tris2, **kw)
                res2["adaptive_pass"] = True
                res2["mesh_pass1_elems"] = int(np.asarray(tris).shape[0])
                res, pts, tris = res2, pts2, tris2
        except Exception as ex:
            res["adaptive_error"] = str(ex)

    kt = MODE_KT.get(mode, 1.0)
    peak = res["peak_vm"] * kt
    # Not simply the yield strength: allowable() falls back to ultimate where a
    # material has no yield point, refuses to believe a library yield that sits
    # above ultimate, and knocks printed plastics down for the across-layer
    # weakness an isotropic solver cannot see. See app/materials.py.
    allow, allow_note = materials.allowable(mat)
    res["material"] = material
    res["material_info"] = materials.info(material)
    res["mode"] = mode
    res["peak_vm"] = peak
    res["allowable_pa"] = allow
    res["allowable_note"] = allow_note
    res["safety_factor"] = (allow / peak) if (allow > 0 and peak > 0) else None
    res["image_size"] = size
    res["holes"] = len(holes)
    res["px_per_mm"] = px_per_mm
    res["thickness_mm"] = thickness_mm

    # How the part splits between "keep", "watch" and "free to remove", by
    # AREA rather than by node count. A node census answers a question nobody
    # asked -- the adaptive pass piles nodes into the hot spots, so counting
    # nodes would report a part as mostly critical purely because that is
    # where the mesh is finest.
    res["zone_frac"] = _zone_fracs(res["nodes"], res["tris"],
                                   res["von_mises_norm"],
                                   pocketing.thresholds(density))

    # Why this part is stressed where it is. Stated as short phrases next to the
    # picture because the map alone cannot distinguish "this fillet is hot
    # because the bore is next to it" from "this whole limb is hot because it is
    # the thinnest thing carrying the moment", and the two lead to opposite
    # decisions: relieve the hole, or leave the limb alone.
    try:
        res["drivers"] = _drivers(res, exterior, holes, W, H)
    except Exception as ex:
        res["drivers"] = []
        res["drivers_error"] = str(ex)

    if do_pocketing:
        part_mask, hole_mask = raster.masks_from_polygons(exterior, holes, W, H)
        # Same array the stress map paints. The picture IS the input to the
        # pocketing engine, so red really does mean "keep material here".
        field = raster.field_from_mesh(res["nodes"], res["tris"],
                                       np.asarray(res["von_mises_norm"]), W, H)
        # The solver's own weak/strong points go in as well, not just the
        # raster field. A concentration at a fillet is a handful of pixels and
        # can sit just under the colour-scale cut line once rasterised; the
        # callout list carries it at full strength, so the ribs land on every
        # point the map badges rather than on a blurred version of them.
        core, keep, stats = pocketing.generate(
            field, part_mask, hole_mask, px_per_mm=px_per_mm,
            rib_mm=rib_mm, density=density,
            stress_pts=res.get("callouts"),
            # The stock thickness this solve already ran on. It was sitting in
            # this function the whole time and never reached the pocketing, so
            # a 1/8" plate was given a 1/4" plate's lattice.
            thick_mm=thickness_mm)
        res["pocketing"] = stats
        res["_pocket_layers"] = (part_mask, hole_mask, core, keep)  # for PNG render
        # Outlines of what gets cut away, so the stress map can draw the pocket
        # plan on top of the field that produced it.
        res["pocket_outlines"] = _outlines(core)
        res["pocket_thresholds"] = pocketing.thresholds(density)
    return res


def _drivers(res, exterior, holes, W, H):
    """Name the features that are setting the peak stress, in plain words.

    Everything here is measured off this part, never assumed from the load
    case. "Holes (Kirsch Kt≈3)" is only claimed when there are bores; "thin
    sections" is only claimed when the part genuinely has a narrow ligament
    relative to its own size; and the dominant driver is decided by asking
    where the hot callouts actually landed, not by which load case was picked.
    """
    out = []
    if holes:
        out.append("Holes (Kirsch Kt≈3)")

    span = float(max(W, H)) or 1.0

    # Thin sections: measure the ligament width along the part's medial ridge.
    # Distance-to-boundary at a ridge pixel is half the local wall thickness, so
    # a low percentile of 2*dt is the narrowest real neck -- as opposed to the
    # distance at an arbitrary pixel, which is near zero everywhere along every
    # edge and would call every part thin.
    try:
        import cv2
        import scipy.ndimage as ndi
        part_mask, hole_mask = raster.masks_from_polygons(exterior, holes, W, H)
        solid = (part_mask & ~hole_mask).astype(np.uint8)
        dt = cv2.distanceTransform(solid, cv2.DIST_L2, 3)
        ridge = (dt >= ndi.maximum_filter(dt, size=3) - 1e-6) & (dt > 1.0)
        if ridge.any():
            neck = float(np.percentile(2.0 * dt[ridge], 10))
            res["min_ligament_px"] = neck
            res["min_ligament_mm"] = neck / (res.get("px_per_mm") or 1.0)
            if neck < 0.09 * span:
                out.append("Thin sections")
    except Exception:
        pass

    # Dominant driver: is the hot end of the field sitting on bore rims, or out
    # in open material? A callout inside roughly one radius of a rim is a hole
    # concentration; one in clear stock is the section itself working.
    hot = [c for c in (res.get("callouts") or []) if c.get("kind") == "hot"]
    if hot and holes:
        near = 0
        for c in hot:
            for h in holes:
                p = np.asarray(h, float)
                if p.shape[0] < 3:
                    continue
                cen = p.mean(axis=0)
                r = float(np.hypot(*(p - cen).T).max()) or 1.0
                if np.hypot(c["x"] - cen[0], c["y"] - cen[1]) < 2.0 * r:
                    near += 1
                    break
        if near * 2 >= len(hot):
            out.append("Hole-proximity dominant")
        else:
            out.append("Section-dominant — stress is in open material")
    return out


def _zone_fracs(nodes, tris, vm, t):
    """Area shares of the three bands the pocketing engine works in.

    Each triangle votes with its own area, using the mean of its three nodal
    values -- the same quantity the engine thresholds -- so the split you read
    at the bottom of the screen is the split that produced the pocket plan.
    """
    p = np.asarray(nodes, float)
    tr = np.asarray(tris)
    v = np.asarray(vm, float)
    if tr.size == 0:
        return {"safe": 0.0, "caution": 0.0, "critical": 0.0}
    a, b, c = p[tr[:, 0]], p[tr[:, 1]], p[tr[:, 2]]
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                        (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    m = v[tr].mean(axis=1)
    tot = float(area.sum()) or 1.0
    safe = float(area[m <= t["pocket"]].sum()) / tot
    crit = float(area[m >= t["keep"]].sum()) / tot
    return {"safe": safe, "critical": crit,
            "caution": max(0.0, 1.0 - safe - crit)}


def _outlines(mask, max_pts=4500):
    """Simplified polygon outlines of a boolean mask, for canvas overlay."""
    try:
        import cv2
        m = np.ascontiguousarray(np.asarray(mask).astype(np.uint8))
        cs, _ = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    except Exception:
        return []
    out, budget = [], int(max_pts)
    for c in sorted(cs, key=lambda a: -a.shape[0]):
        if budget <= 0:
            break
        # 0.9 px tolerance keeps real corners and drops rasteriser stair-steps
        c = cv2.approxPolyDP(c, 0.9, True).reshape(-1, 2)
        if c.shape[0] < 3:
            continue
        c = c[:budget]
        budget -= c.shape[0]
        out.append(c.astype(int).tolist())
    return out
