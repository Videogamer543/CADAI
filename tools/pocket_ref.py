"""
Reference parts -> calibrated pocketing constants.

What this is not
----------------
It is not training. There is no model in the pocketing engine to train:
app/pocketing.py places nodes, triangulates them, and calls the edges ribs. It
is geometry with constants in it, and every one of those constants was chosen by
looking at photographs of machined 6061 and picking a number that looked right.
"Looked right" was judged almost entirely on parts in the 200-400 mm range,
which is why the engine's behaviour on a 600 mm bellypan is a guess rather than
a decision.

What this is
------------
A way to make those constants answer to parts you have actually machined. Drop
photos, screenshots or STEP files of real pocketed parts into reference_parts/,
and:

    python tools/pocket_ref.py list      what it found, and what it needs
    python tools/pocket_ref.py report    real vs ours, part by part
    python tools/pocket_ref.py fit       search the constants, write the file
    python tools/pocket_ref.py revert    delete the file, back to defaults

`fit` writes data/pocket_cal.json. app/pocketing.py reads it on the next run.
Deleting it restores the shipped numbers exactly -- there is no second copy of
the defaults anywhere, so a reverted install and a fresh one are identical.

Why ratios and not millimetres
------------------------------
Every measurement here is divided by the part's own span before anything is
compared. A 40 mm bay is generous on a gusset and a rounding error on a
bellypan; 0.19 of the span is the same design decision at both sizes. Since the
complaint that started this was specifically about LARGE parts, comparing
absolute sizes would have measured the thing that is already known -- big parts
have big pockets -- instead of the thing in question, which is whether the
engine keeps the same proportions as it scales up. It does not, and the
scale-invariant form is what makes that visible.

Naming your files
-----------------
A photo carries no scale, so the part's span has to come from somewhere:

    bellypan_610mm.png          <- span in the filename, easiest
    armplate.jpg + armplate.json   {"span_mm": 330, "rib_mm": 3.0}

"Span" is the part's longest dimension, measured along its own long axis --
the same thing the engine calls span, so the two are comparable by
construction. STEP files need none of this: they are already in millimetres.

A file with no span is still measured and still listed. It just cannot be
compared against the engine, because without a scale the engine cannot be run
on it, so it sits out of the report and the fit rather than being silently
counted with a made-up number.
"""
from __future__ import annotations
import argparse
import io
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                # noqa: E402
import cv2                                                        # noqa: E402
from scipy import ndimage as ndi                                  # noqa: E402
from PIL import Image                                             # noqa: E402

from app import pocketing                                         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR = os.path.join(ROOT, "reference_parts")
CAL_PATH = pocketing.CAL_PATH

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
STEP_EXT = {".step", ".stp"}

# Exit codes, so a launcher can tell "wrote it" from "declined to write it".
# `fit` has four ways to finish without writing -- too few usable parts, no
# improvement over the defaults, --dry-run, and a crash -- and only the last of
# those is an error. Printing the reason is not enough on its own: a batch file
# cannot read its own scrollback, and neither can a user who has just watched
# 300 engine runs scroll past. EXIT_NOT_WRITTEN is deliberately not 1, so it
# stays distinguishable from a traceback.
EXIT_WROTE = 0
EXIT_NOT_WRITTEN = 2

# Telling a bore from a pocket. Size alone does not do it: an FRC plate has
# 1.125" (28.6 mm) bearing bores, which are larger than plenty of legitimate
# lightening pockets on a gusset. Roundness does, because a bore is turned and
# a pocket is milled to a rounded triangle or hex -- so a hole is a bore if it
# is round, or if it is simply too small to be anything else.
BORE_CIRCULARITY = 0.80
BORE_MAX_MM = 45.0        # ...but nothing this big is a bore, however round
BORE_ALWAYS_MM = 12.0     # ...and nothing this small is a pocket, whatever shape

# How big the working raster is. Pocketing cost is roughly quadratic in this and
# the measurements are ratios, so there is nothing to gain from full resolution.
WORK_PX = 640

RENDER_HELP = "reference_parts"


# ==========================================================================
# part specs -- where a photo's scale comes from
# ==========================================================================
_MM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mm\b", re.I)


def _spec_for(path):
    """Span/rib/thickness for one reference file, from sidecar then filename.

    The sidecar wins because it was typed on purpose; the filename is the
    convenience path. Both are optional and their absence is not an error --
    see the module docstring on why an unscaled part is listed rather than
    dropped.
    """
    stem, _ = os.path.splitext(path)
    spec = {}
    for cand in (stem + ".json", stem + ".txt"):
        if not os.path.exists(cand):
            continue
        try:
            with open(cand, "r", encoding="utf-8") as fh:
                txt = fh.read()
            if cand.endswith(".json"):
                spec.update(json.loads(txt) or {})
            else:
                for line in txt.splitlines():
                    if "=" in line or ":" in line:
                        sep = "=" if "=" in line else ":"
                        k, v = line.split(sep, 1)
                        try:
                            spec[k.strip()] = float(v.strip().rstrip("m").strip())
                        except ValueError:
                            spec[k.strip()] = v.strip()
        except Exception as e:
            print("  ! ignoring %s (%s)" % (os.path.basename(cand), e))
        break
    if "span_mm" not in spec:
        m = _MM_RE.search(os.path.basename(stem))
        if m:
            spec["span_mm"] = float(m.group(1))
    return spec


def _part_name(path):
    base = os.path.splitext(os.path.basename(path))[0]
    base = _MM_RE.sub("", base).strip(" _-")
    return base or os.path.splitext(os.path.basename(path))[0]


def find_parts(ref_dir=REF_DIR):
    """Every reference file in the folder, newest naming rules applied."""
    out = []
    if not os.path.isdir(ref_dir):
        return out
    for name in sorted(os.listdir(ref_dir)):
        path = os.path.join(ref_dir, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXT:
            kind = "image"
        elif ext in STEP_EXT:
            kind = "step"
        else:
            continue
        out.append({"path": path, "kind": kind, "name": _part_name(path),
                    "spec": _spec_for(path)})
    return out


# ==========================================================================
# measurement -- ONE metric function, used on real parts and on ours alike
# ==========================================================================
def _metrics(part_mask, pocket_mask, bore_mask, px_per_mm):
    """Span-normalised description of a pocketed plate.

    Both sides of the comparison go through this function, and that is
    deliberate: every estimate in here is biased somehow -- the rib width is
    pulled up by the rim, the bay size is a median over pockets that are not
    all the same -- and a bias that lands on the real part and on ours in the
    same way cancels out of the difference. A separate, tidier measurement of
    our own output would be more accurate and less useful.
    """
    part_mask = part_mask.astype(bool)
    part_px = int(part_mask.sum())
    if part_px < 100:
        return None
    mm = 1.0 / max(px_per_mm, 1e-9)
    _c, _u, _v, Lu, Lv = pocketing._frame(part_mask)
    span_mm = float(max(Lu, Lv)) * mm
    if span_mm <= 0:
        return None

    pockets = pocket_mask.astype(bool) & part_mask
    bores = bore_mask.astype(bool) & part_mask
    solid = part_mask & ~pockets & ~bores

    lab, n = ndi.label(pockets)
    areas_px = []
    if n:
        areas_px = list(np.asarray(
            ndi.sum(np.ones_like(lab, dtype=np.float64), lab,
                    index=range(1, n + 1))))
    # Slivers are not bays. They are what a fillet or a JPEG edge leaves
    # behind, and letting them into the median drags the bay size towards
    # nothing on exactly the parts with the most rib junctions.
    floor_px = max(9.0, 0.00035 * part_px)
    areas_px = [a for a in areas_px if a >= floor_px]

    # Bay size as the square root of a pocket's area, not its bbox or its
    # longest chord: a triangular bay and a hexagonal one of the same area are
    # the same amount of removed material and the same distance between ribs,
    # and sqrt(area) says so while a diagonal measure does not.
    bay_mm = float(np.sqrt(np.median(areas_px)) * mm) if areas_px else 0.0

    return {
        "span_mm": span_mm,
        "removal": float(pockets.sum()) / part_px,
        "bay_span": bay_mm / span_mm,
        "bay_mm": bay_mm,
        "rib_mm": _rib_mm(solid, px_per_mm),
        "n_pockets": len(areas_px),
        "n_bores": int(ndi.label(bores)[1]),
        "pattern": _pattern_guess(part_mask, pockets, bores, px_per_mm),
    }


def _rib_mm(solid, px_per_mm):
    """Rib width, from the ridge of the material's distance transform.

    Twice the distance to the nearest edge, at the points furthest from any
    edge, is the local thickness of whatever you are standing on. Taking the
    MEDIAN of that over the ridge is what makes it a rib measurement rather
    than an average of everything: the perimeter wall and the bearing bosses
    are genuinely thicker than the ribs, but on a part worth using as a
    reference they are also a small minority of the skeleton, so they move the
    median hardly at all while they would drag a mean noticeably.
    """
    m = solid.astype(np.uint8)
    if m.sum() < 50:
        return 0.0
    d = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    ridge = (d >= ndi.maximum_filter(d, size=3) - 1e-6) & (d > 0.75)
    vals = d[ridge]
    if vals.size < 8:
        return 0.0
    return float(2.0 * np.median(vals) / max(px_per_mm, 1e-9))


def _pattern_guess(part_mask, pockets, bores, px_per_mm):
    """Which archetype the real part looks like, from the shape of its bays.

    Diagnostic only. Nothing in `fit` optimises this -- it is here so that a
    disagreement between what we chose and what the part actually is shows up
    in the report as something to think about, rather than hiding inside a
    removal fraction that happens to match for the wrong reason.
    """
    lab, n = ndi.label(pockets)
    if n == 0:
        return "solid"
    _c, _u, _v, Lu, Lv = pocketing._frame(part_mask)
    slender = max(Lu, Lv) / max(min(Lu, Lv), 1e-6)
    cnts, _h = cv2.findContours(pockets.astype(np.uint8), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    verts = []
    for c in cnts:
        if cv2.contourArea(c) < 40:
            continue
        peri = cv2.arcLength(c, True)
        verts.append(len(cv2.approxPolyDP(c, 0.045 * peri, True)))
    if not verts:
        return "solid"
    v = float(np.median(verts))
    n_big = len(verts)
    n_bore = int(ndi.label(bores)[1])
    if n_big <= 2:
        return "xbrace"
    if slender >= 2.6:
        return "truss"
    if n_bore >= 2 and v >= 4.5:
        return "radial"
    if v <= 3.6:
        return "truss" if slender >= 1.9 else "waffle"
    return "waffle"


# ==========================================================================
# measuring a photo or screenshot
# ==========================================================================
def _binarize(path):
    """Part-vs-background mask, tolerant of both screenshots and photographs.

    Screenshots of CAD are a dark part on white and threshold cleanly. Photos
    of a plate on a bench do not: the background is not white, so the rule is
    "different from the border", which is the one assumption a photo of a part
    reliably satisfies -- whatever is at the edge of the frame is not the part.
    """
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)
    alpha = arr[:, :, 3:4] / 255.0
    rgb = arr[:, :, :3] * alpha + 255.0 * (1 - alpha)
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    bg = float(np.percentile(border, 75))
    if bg > 200:
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    else:
        mask = (np.abs(gray.astype(int) - bg) > 35).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask


def measure_image(path, spec=None):
    """(part_mask, pocket_mask, bore_mask, px_per_mm) for a photo.

    The part comes back FILLED -- footprint including everything cut out of it
    -- with the pockets and the bores as separate masks over it. That split is
    what makes the file useful twice: measured as it stands it says what the
    part is, and filling the pockets back in recovers the blank the engine can
    be asked to pocket from scratch.

    The voids are found by subtracting the material from the footprint rather
    than by filling the inner contours, and the difference matters. Filling a
    contour ignores what is inside it, so a rib crossing a pocket -- or the
    little island a fillet leaves in a junction -- gets counted as removed
    material. Our own output is measured from a mask where those islands are
    plainly material, so filling contours on the real part and not on ours
    would put a thumb on the scale in a way that varies with how busy the
    lattice is: exactly the quantity being fitted.
    """
    spec = spec or {}
    mat = _binarize(path) > 0
    cnts, hier = cv2.findContours(mat.astype(np.uint8), cv2.RETR_CCOMP,
                                  cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("no part found in the image")
    hier = hier[0]
    areas = [cv2.contourArea(c) for c in cnts]
    outer = max((i for i in range(len(cnts)) if hier[i][3] == -1),
                key=lambda i: areas[i])

    H, W = mat.shape
    part = np.zeros((H, W), np.uint8)
    cv2.drawContours(part, cnts, outer, 255, -1)
    part = part > 0
    # Only the material inside the chosen outline counts; a second part in the
    # corner of the photo, or a caption, is not this part.
    mat = mat & part

    span_mm = float(spec.get("span_mm") or 0.0)
    _c, _u, _v, Lu, Lv = pocketing._frame(part)
    span_px = float(max(Lu, Lv))
    px_per_mm = (span_px / span_mm) if span_mm > 0 else 0.0

    pockets, bores = _sort_voids(part & ~mat, px_per_mm)
    part, pockets, bores, px_per_mm = _downscale(part, pockets, bores, px_per_mm)
    return part, pockets, bores, px_per_mm


def _sort_voids(voids, px_per_mm):
    """Split everything cut out of the plate into pockets and bores."""
    pockets = np.zeros(voids.shape, bool)
    bores = np.zeros(voids.shape, bool)
    lab, n = ndi.label(voids)
    for i in range(1, n + 1):
        reg = lab == i
        a = int(reg.sum())
        if a < 25:
            continue                      # antialiasing speckle
        cnts, _h = cv2.findContours(reg.astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        if _is_bore(cnts[0], float(a), px_per_mm):
            bores |= reg
        else:
            pockets |= reg
    return pockets, bores


def _is_bore(contour, area_px, px_per_mm):
    """Bore or pocket. Unscaled parts fall back to roundness alone."""
    peri = max(cv2.arcLength(contour, True), 1e-6)
    circ = 4.0 * np.pi * area_px / (peri * peri)
    if px_per_mm <= 0:
        return circ > BORE_CIRCULARITY
    d_mm = 2.0 * np.sqrt(max(area_px, 0.0) / np.pi) / px_per_mm
    if d_mm <= BORE_ALWAYS_MM:
        return True
    if d_mm >= BORE_MAX_MM:
        return False
    return circ > BORE_CIRCULARITY


def _downscale(part, pockets, bores, px_per_mm, target=WORK_PX):
    """Shrink to the working size, carrying the scale factor along."""
    H, W = part.shape
    long_side = max(H, W)
    if long_side <= target:
        return part, pockets, bores, px_per_mm
    f = target / float(long_side)
    size = (max(2, int(round(W * f))), max(2, int(round(H * f))))

    def rs(m):
        return cv2.resize(m.astype(np.uint8), size,
                          interpolation=cv2.INTER_NEAREST) > 0

    return rs(part), rs(pockets), rs(bores), px_per_mm * f


# ==========================================================================
# measuring a STEP file
# ==========================================================================
def measure_step(path, spec=None, max_tris=90000):
    data = open(path, "rb").read()
    from app import step3d
    tess = step3d.tessellate(data, size_factor=1.0, max_tris=max_tris)
    return measure_tess(tess, spec=spec)


def measure_tess(tess, spec=None, out_px=WORK_PX):
    """(part, pockets, bores, px_per_mm, extras) from a tessellated solid.

    A blind pocket is invisible to a silhouette. app/step3d.silhouette_png
    projects the near-planar faces of a plate onto its own plane, and a pocket
    FLOOR is a near-planar face, so a plate milled 3 mm deep into 6 mm stock
    projects as a solid rectangle with no pockets in it at all. That is the
    right answer for the stress pipeline, which wants the outline, and useless
    here, where the pockets are the entire subject.

    So the faces are sorted by height along the plate normal instead of merged:

      the level with the most area is the BACK face -- the side that was never
      cut, so its outline is the blank with the through-holes in it;
      levels above it are pocket FLOORS, and each one is a pocket;
      whatever the back face covers and no planar face at all does is a hole
      cut clean through, sorted into pocket or bore by size.

    The thinnest floor above the back face is the part's floor thickness, which
    is the one measurement a photo can never give and the one that decides
    whether a pocketing plan is machinable in the stock you actually have.
    """
    spec = spec or {}
    tris = np.asarray(tess["tris"], dtype=np.float64)
    if tris.size == 0:
        raise ValueError("empty tessellation")
    verts = tris.reshape(-1, 3)
    c = verts.mean(axis=0)

    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    cr = np.cross(e1, e2)
    a2 = np.linalg.norm(cr, axis=1)                  # twice the triangle area
    nrm = cr / np.where(a2 == 0, 1.0, a2)[:, None]

    # The plate normal comes from the FACES, not from an SVD of the vertex
    # positions.
    #
    # Fitting the normal to the point cloud looks equivalent and is not. The
    # vertices are not spread evenly over the solid -- bore walls and pocket
    # corners are tessellated far more finely than a flat face -- so the third
    # singular vector comes out a tenth of a degree off true. A tenth of a
    # degree is nothing until it is multiplied by the length of the part: over
    # 300 mm it slides one end of a flat face 0.3 mm relative to the other,
    # which is larger than the tolerance that decides whether two faces are the
    # same face. A single top face then shatters into a stack of levels and the
    # whole plate reads as one enormous pocket. Measured on a 300 mm test plate:
    # 8 levels found where there were 3, and 96% removal reported on a part with
    # five pockets in it.
    #
    # The area-weighted scatter of the unit face normals has no such problem.
    # Two big opposed faces dominate it, n and -n contribute identically to
    # n.nT, and the dominant eigenvector is the plate normal to machine
    # precision no matter how the mesh was refined.
    scat = (nrm * a2[:, None]).T @ nrm
    evals, evecs = np.linalg.eigh(scat)
    n_ax = evecs[:, int(np.argmax(evals))]
    n_ax = n_ax / max(float(np.linalg.norm(n_ax)), 1e-12)

    # In-plane axes still come from an SVD, of the vertices projected onto the
    # plane, so the raster stays tight around a part that is not axis-aligned.
    # Getting these slightly wrong only costs a few pixels of margin.
    rel_v = verts - c
    inplane = rel_v - np.outer(rel_v @ n_ax, n_ax)
    _u, _s, vt = np.linalg.svd(inplane, full_matrices=False)
    u_ax = vt[0] - (vt[0] @ n_ax) * n_ax
    u_ax = u_ax / max(float(np.linalg.norm(u_ax)), 1e-12)
    v_ax = np.cross(n_ax, u_ax)

    rel = tris.reshape(-1, 3) - c
    p2 = np.stack([rel @ u_ax, rel @ v_ax], axis=1).reshape(-1, 3, 2)
    hgt = ((tris.mean(axis=1) - c) @ n_ax)
    planar = np.abs(nrm @ n_ax) > 0.85

    flat = p2.reshape(-1, 2)
    mn2 = flat.min(0)
    span2 = flat.max(0) - mn2
    span2[span2 == 0] = 1.0
    pad = 12
    scale = (out_px - 2 * pad) / span2.max()          # px per mm (STEP is mm)
    W = int(span2[0] * scale) + 2 * pad
    H = int(span2[1] * scale) + 2 * pad
    polys = np.round((p2 - mn2) * scale + pad).astype(np.int32)

    def raster(sel, stroke=1):
        m = np.zeros((H, W), np.uint8)
        idx = np.nonzero(sel)[0]
        for i in idx:
            cv2.fillConvexPoly(m, polys[i], 255, cv2.LINE_8)
        if stroke and idx.size:
            # Rounding vertices to the pixel grid opens 1 px cracks between
            # neighbouring triangles; unstroked, those cracks come back as
            # thousands of phantom pockets.
            cv2.polylines(m, [polys[i] for i in idx], True, 255, stroke,
                          cv2.LINE_8)
        return m > 0

    sil = raster(np.ones(len(polys), bool))
    if not planar.any():
        raise ValueError("no plate-like faces in this solid")

    # Cluster the planar faces into levels. The tolerance is a real machining
    # quantity -- two faces less than a quarter millimetre apart are one face
    # with a tessellation wobble, not two pocket depths.
    tol = 0.25
    order = np.argsort(hgt)
    levels = []
    for i in order:
        if not planar[i]:
            continue
        if levels and abs(hgt[i] - levels[-1][0]) <= tol:
            levels[-1][1].append(i)
        else:
            levels.append([float(hgt[i]), [i]])
    if not levels:
        raise ValueError("no plate-like faces in this solid")

    masks, hs, areas = [], [], []
    for h, idxs in levels:
        sel = np.zeros(len(polys), bool)
        sel[idxs] = True
        m = raster(sel)
        masks.append(m)
        hs.append(h)
        areas.append(int(m.sum()))
    back = int(np.argmax(areas))

    part = masks[back].copy()
    # A back face with a bite out of it -- a plate pocketed from both sides, or
    # one whose largest planar level happens to be a pocket floor -- would
    # understate the footprint, so the full silhouette stands in when the back
    # face is clearly not the whole plate.
    if areas[back] < 0.55 * int(sil.sum()):
        part = sil.copy()
    part = ndi.binary_fill_holes(part)

    all_planar = np.zeros_like(sil)
    for m in masks:
        all_planar |= m
    faceless = ndi.binary_opening(part & ~all_planar, np.ones((3, 3), bool))

    pockets = np.zeros_like(sil)
    bores = np.zeros_like(sil)
    # Levels are grouped by which SIDE of the back face they sit on, and the
    # outermost level on each side is that side's untouched face. Everything
    # between an outer face and the back face is standing on material that was
    # milled away, which is what a pocket floor is. Sorting by side rather than
    # by raw height is what lets a plate pocketed from both sides be read: with
    # a single ordering, the far side's outer face looks like a very deep pocket.
    floor_depths = []
    for side in (+1, -1):
        idxs = [i for i in range(len(masks))
                if i != back and np.sign(hs[i] - hs[back]) == side
                and abs(hs[i] - hs[back]) > tol]
        if not idxs:
            continue
        outer = max(abs(hs[i] - hs[back]) for i in idxs)
        for i in idxs:
            d = abs(hs[i] - hs[back])
            if d < outer - tol:
                pockets |= masks[i] & part
                floor_depths.append(d)
    floor_mm = float(min(floor_depths)) if floor_depths else 0.0

    lab, n = ndi.label(faceless)
    px_per_mm = float(scale)
    if n:
        for i in range(1, n + 1):
            reg = lab == i
            a = int(reg.sum())
            if a < 12:
                continue
            d_mm = 2.0 * np.sqrt(a / np.pi) / px_per_mm
            cnts, _h = cv2.findContours(reg.astype(np.uint8),
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            circ = 0.0
            if cnts:
                peri = max(cv2.arcLength(cnts[0], True), 1e-6)
                circ = 4.0 * np.pi * a / (peri * peri)
            if d_mm <= BORE_ALWAYS_MM or (d_mm < BORE_MAX_MM
                                          and circ > BORE_CIRCULARITY):
                bores |= reg
            else:
                pockets |= reg

    pockets &= part
    bores &= part & ~pockets
    thickness = float(np.ptp((verts - c) @ n_ax))
    extras = {"floor_mm": round(floor_mm, 2),
              "thickness_mm": round(thickness, 2),
              "n_levels": len(levels)}
    if spec.get("thickness_mm"):
        extras["thickness_mm"] = float(spec["thickness_mm"])
    return part, pockets, bores, px_per_mm, extras


# ==========================================================================
# one reference part, measured and compared
# ==========================================================================
def load_part(rec, verbose=False):
    """Measure one file. Returns the record with `real` filled in, or an error."""
    out = dict(rec)
    try:
        if rec["kind"] == "step":
            part, pockets, bores, ppm, extras = measure_step(rec["path"],
                                                             rec["spec"])
            out["extras"] = extras
        else:
            part, pockets, bores, ppm = measure_image(rec["path"], rec["spec"])
            out["extras"] = {}
        if ppm <= 0:
            out["error"] = ("no span. Rename it like %s_300mm%s, or put "
                            '{"span_mm": 300} beside it.'
                            % (rec["name"],
                               os.path.splitext(rec["path"])[1]))
            out["real"] = _metrics(part, pockets, bores, 1.0)
            out["masks"] = None
            return out
        m = _metrics(part, pockets, bores, ppm)
        if not m:
            out["error"] = "could not measure it (part too small in frame?)"
            return out
        if rec["spec"].get("rib_mm"):
            m["rib_mm"] = float(rec["spec"]["rib_mm"])
        out["real"] = m
        out["masks"] = (part, pockets, bores, ppm)
    except Exception as e:
        out["error"] = "%s: %s" % (type(e).__name__, e)
    return out


# The FEM field is held FLAT on purpose. The constants being calibrated are
# geometric, and the reference parts arrive with no load case attached, so any
# field would be a load case invented per part -- folding a guess about how each
# plate is loaded into a fit about how big its bays should be. Flat also means
# no solid island and no hot/cool zones, which is the fairest possible baseline:
# what the engine does on shape alone.
FLAT_FIELD = 0.5


def run_ours(masks, rib_mm=3.0, cal=None, density="normal", thick_mm=None):
    """Pocket the same blank with the engine, and measure the result the same way.

    The blank is the reference part with its pockets filled back in -- which is
    the one thing a photo of a finished part can be turned into that the engine
    can be fairly asked to work on. Its bores stay, because a bore is a mounting
    feature the engine is supposed to design around, not lightening it is
    supposed to invent.

    Note that ours' rib width is MEASURED off the generated result, not read
    back from the input. They are not the same number: fillets at the junctions
    and the perimeter wall both add material, so the geometry the engine
    actually produces is a little heavier than the rib it was asked for. Since
    the real part is measured the same way, that difference is visible in the
    report instead of being defined away.

    `thick_mm` is handled the same way as rib width, and for the same reason:
    stock thickness is an INPUT to pocketing rather than something the engine
    decides, so the engine is handed the reference part's own stock. Withhold
    it and every comparison against non-1/4" material becomes partly a
    comparison of thicknesses -- and `fit`, seeing a residual it can only
    reach with the geometric constants, would spend `cell_span_f` trying to
    absorb a thickness effect that constant cannot represent, making the fit
    worse on every other part in the set to chase it.

    A photo carries no thickness, so it passes None and the term goes inert.
    That is the correct answer rather than a fallback: an unstated thickness
    must not quietly become an assumed one.
    """
    part, pockets, bores, ppm = masks
    blank = part.copy()                       # pockets filled back in
    field = np.full(part.shape, FLAT_FIELD, dtype=np.float64)
    core, keep, stats = pocketing.generate(
        field, blank, bores, px_per_mm=ppm, rib_mm=rib_mm,
        density=density, stress_pts=None, cal=cal, thick_mm=thick_mm)
    m = _metrics(blank, core, bores, ppm)
    if m:
        m["engine_pattern"] = stats.get("pattern", "")
        m["n_pockets_engine"] = stats.get("n_pockets", 0)
        m["rib_in_mm"] = rib_mm
        m["thick_factor"] = stats.get("thick_factor", 1.0)
    return m, stats


def prepare(parts, rib_from_real=True, verbose=False):
    """Measure every part once. The expensive half; do it before any search."""
    done = []
    for rec in parts:
        t0 = time.time()
        r = load_part(rec, verbose=verbose)
        if r.get("real") and r.get("masks") and rib_from_real:
            # Feed the real part's own rib width back into the engine. Rib
            # width is an INPUT to pocketing, not something it decides, so
            # letting the two differ would make the bay comparison partly a
            # comparison of rib widths and the fit would chase it.
            rib = r["real"].get("rib_mm") or 0.0
            r["rib_mm"] = float(np.clip(rib, 1.5, 12.0)) if rib > 0 else 3.0
        if verbose and not r.get("error"):
            print("    measured %-18s %.1fs" % (r["name"], time.time() - t0))
        done.append(r)
    return done


def evaluate(prepped, cal=None, density="normal"):
    """Run the engine on every usable part and pair it with the measurement."""
    rows = []
    for r in prepped:
        if r.get("error") or not r.get("masks"):
            continue
        try:
            # Thickness comes from the STEP measurement (`extras`), so it is
            # present for CAD and absent for photos, per part rather than per
            # run. A mixed reference set therefore compares each part at its
            # own stock without any of them having to be told twice.
            ours, stats = run_ours(r["masks"], rib_mm=r.get("rib_mm", 3.0),
                                   cal=cal, density=density,
                                   thick_mm=(r.get("extras") or {})
                                   .get("thickness_mm"))
        except Exception as e:
            rows.append({"name": r["name"], "real": r["real"], "ours": None,
                         "error": "%s: %s" % (type(e).__name__, e)})
            continue
        if not ours:
            continue
        rows.append({"name": r["name"], "real": r["real"], "ours": ours,
                     "extras": r.get("extras", {}), "stats": stats})
    return rows


# ==========================================================================
# the objective
# ==========================================================================
W_REMOVAL = 0.5
W_BAY = 0.5


def gap_of(row):
    """How wrong we are about one part, in one number.

    Removal enters as an absolute difference because it is already a fraction
    of the same thing on both sides -- 47% against 61% is fourteen points of
    material whether the plate is small or large. Bay size enters as a RELATIVE
    difference because the same absolute error means completely different
    things at either end of its range: 0.06 against 0.19 is a bay a third the
    size it should be, while 0.30 against 0.43 is a fair approximation of a big
    open panel. An absolute measure would rank those two as identical mistakes.
    """
    real, ours = row.get("real"), row.get("ours")
    if not real or not ours:
        return None
    d_rem = abs(ours["removal"] - real["removal"])
    rb = max(real["bay_span"], 0.02)
    d_bay = abs(ours["bay_span"] - real["bay_span"]) / rb
    return W_REMOVAL * d_rem + W_BAY * min(d_bay, 3.0)


def aggregate(rows):
    gs = [g for g in (gap_of(r) for r in rows) if g is not None]
    return float(np.mean(gs)) if gs else float("inf")


# ==========================================================================
# commands
# ==========================================================================
def ensure_dir():
    if not os.path.isdir(REF_DIR):
        os.makedirs(REF_DIR, exist_ok=True)
    readme = os.path.join(REF_DIR, "README.txt")
    if not os.path.exists(readme):
        with open(readme, "w", encoding="utf-8") as fh:
            fh.write(REF_README)


REF_README = """\
Reference parts
===============

Put pocketed parts you have actually machined in this folder. Photos,
screenshots and STEP files all work.

  photos / screenshots   .png .jpg .jpeg .bmp .webp .tif
  CAD                    .step .stp

A photo carries no scale, so tell it the part's span (its longest dimension,
in mm) one of two ways:

  1. in the filename       bellypan_610mm.png
  2. in a sidecar file     armplate.jpg  +  armplate.json
                           {"span_mm": 330, "rib_mm": 3.0}

STEP files need neither: they are already in millimetres.

What makes a good reference part:

  * one part per file, filling most of the frame, shot square-on
  * a plain background, or a screenshot on white
  * the pockets clearly visible -- a part photographed at an angle measures
    as a smaller, differently-shaped part
  * parts at the sizes you actually care about. Six 150 mm gussets will
    calibrate the engine for 150 mm gussets.

Then, from the project folder:

  python tools/pocket_ref.py list      what was found
  python tools/pocket_ref.py report    real vs ours, part by part
  python tools/pocket_ref.py fit       tune the constants to match
  python tools/pocket_ref.py revert    undo it

Nothing in this folder is uploaded anywhere. It is read locally by that
script and by nothing else.
"""


def cmd_list(args):
    ensure_dir()
    parts = find_parts()
    if not parts:
        print("\nNo reference parts yet.\n")
        print("  Drop photos, screenshots or STEP files into:")
        print("    %s\n" % REF_DIR)
        print("  and name a photo with its span, e.g. bellypan_610mm.png")
        print("  (see README.txt in that folder)\n")
        return
    print("\n%d file(s) in %s\n" % (len(parts), RENDER_HELP))
    print("  %-22s %-6s %8s  %s" % ("NAME", "KIND", "SPAN", "STATUS"))
    n_ok = 0
    for rec in parts:
        span = rec["spec"].get("span_mm")
        if rec["kind"] == "step":
            span_s, status = "(CAD)", "ready"
            n_ok += 1
        elif span:
            span_s, status = "%.0fmm" % span, "ready"
            n_ok += 1
        else:
            span_s, status = "-", "needs a span (rename or add a sidecar)"
        print("  %-22s %-6s %8s  %s"
              % (rec["name"][:22], rec["kind"], span_s, status))
    print("\n  %d of %d usable for report/fit\n" % (n_ok, len(parts)))


def _fmt(v, spec="%.2f"):
    return "-" if v is None else spec % v


def cmd_report(args):
    ensure_dir()
    parts = find_parts()
    if not parts:
        return cmd_list(args)
    print("\nmeasuring %d reference part(s)..." % len(parts))
    prepped = prepare(parts, verbose=args.verbose)
    bad = [p for p in prepped if p.get("error")]
    for p in bad:
        print("  skipped %-18s %s" % (p["name"], p["error"]))
    rows = evaluate(prepped, density=args.density)
    if args.min_span > 0:
        rows = [r for r in rows if r["real"]["span_mm"] >= args.min_span]
    if not rows:
        print("\nNothing to compare. Add a span to the parts above, or lower"
              " --min-span.\n")
        return

    cal = pocketing.calibration()
    tag = " (calibrated)" if pocketing.is_calibrated() else ""
    # STOCK earns a column because bay size now depends on it. Without it a
    # mixed-thickness set reads as the engine being erratic -- two parts of
    # similar span given visibly different lattices, for a reason not on screen.
    print("\n  PART                     SPAN  STOCK  REMOVAL  BAY/SPAN"
          "  RIB/SPAN  PATTERN")
    for r in sorted(rows, key=lambda r: -r["real"]["span_mm"]):
        ex = r.get("extras") or {}
        t_mm = ex.get("thickness_mm") or 0.0
        for who in ("real", "ours"):
            m = r[who]
            pat = m.get("engine_pattern") or m.get("pattern") or "-"
            print("  %-5s %-18s %5.0f  %5s  %6.0f%%  %8.3f  %8.4f  %s"
                  % (who + ":", r["name"][:18], r["real"]["span_mm"],
                     ("%.2f" % t_mm) if t_mm else "  -",
                     100.0 * m["removal"], m["bay_span"],
                     (m.get("rib_mm") or 0.0) / max(r["real"]["span_mm"], 1e-6),
                     pat))
        tf = (r["ours"] or {}).get("thick_factor", 1.0)
        if abs(tf - 1.0) > 0.005:
            print("        %-18s bays x%.2f for %.2f mm stock"
                  % ("", tf, t_mm))
        if ex.get("floor_mm"):
            print("        %-18s floor %.1f mm in %.1f mm stock"
                  % ("", ex["floor_mm"], t_mm))

    print("\n  Across %d part(s)%s%s:"
          % (len(rows),
             " over %.0f mm span" % args.min_span if args.min_span else "",
             tag))
    r_real = np.mean([r["real"]["removal"] for r in rows])
    r_ours = np.mean([r["ours"]["removal"] for r in rows])
    b_real = np.mean([r["real"]["bay_span"] for r in rows])
    b_ours = np.mean([r["ours"]["bay_span"] for r in rows])
    ratio = b_ours / max(b_real, 1e-6)
    print("    bay/span    ours %.3f vs real %.3f   (%.1fx too %s)"
          % (b_ours, b_real, (1.0 / ratio) if ratio < 1 else ratio,
             "fine" if ratio < 1 else "coarse"))
    print("    removal     ours %.1f%% vs real %.1f%%   (%.1f pts %s)"
          % (100 * r_ours, 100 * r_real, abs(100 * (r_ours - r_real)),
             "low" if r_ours < r_real else "high"))
    print("    gap         %.3f  (0 = we match the reference parts)"
          % aggregate(rows))

    _size_trend(rows)
    sug = _suggest(rows, cal, ratio, r_ours, r_real)
    for line in sug:
        print("    %s" % line)
    print("\n  `python tools/pocket_ref.py fit` searches for the values"
          " instead of guessing.\n")


def _size_trend(rows):
    """Split the parts by size and compare the halves.

    The single averaged number above can be near 1.0 while the engine is twice
    too fine on the big parts and twice too coarse on the small ones -- the two
    errors cancel in the mean and the average says everything is fine. Since
    "it gets worse on larger parts" is the specific complaint this whole tool
    exists to answer, the average is exactly the wrong statistic to answer it
    with, and the halves are printed whether or not they disagree.
    """
    if len(rows) < 4:
        return
    srt = sorted(rows, key=lambda r: r["real"]["span_mm"])
    mid = len(srt) // 2
    halves = [("small", srt[:mid]), ("large", srt[len(srt) - mid:])]
    ratios = {}
    print("")
    for label, grp in halves:
        b_o = np.mean([r["ours"]["bay_span"] for r in grp])
        b_r = np.mean([r["real"]["bay_span"] for r in grp])
        ratios[label] = b_o / max(b_r, 1e-6)
        print("    %-5s parts (%.0f-%.0f mm): bay/span ours %.3f vs real %.3f"
              % (label, grp[0]["real"]["span_mm"], grp[-1]["real"]["span_mm"],
                 b_o, b_r))
    lo, hi = ratios["small"], ratios["large"]
    if max(lo, hi) / max(min(lo, hi), 1e-6) > 1.35:
        worse = "larger" if hi < lo else "smaller"
        print("    -> the bays drift %s as the part grows; %s parts are the"
              % ("finer" if hi < lo else "coarser", worse))
        print("       ones we get wrong. cell_span_f is the term that ties bay")
        print("       size to the part, and it is the first thing fit moves.")
        print("       `--min-span 300` fits only the parts you care about.")


def _fit_meta():
    """What the calibration file records about how it was made, or {}."""
    try:
        with open(CAL_PATH, "r", encoding="utf-8") as fh:
            got = json.load(fh)
        return got if isinstance(got, dict) else {}
    except Exception:
        return {}


def _suggest(rows, cal, ratio, r_ours, r_real):
    """One arithmetic guess per knob -- the thing `fit` does properly.

    Deliberately naive: it assumes each constant acts alone and linearly, which
    neither does. It is here so the report says something actionable on its own,
    and so there is an independent number to sanity-check the fitted one
    against. When the two disagree wildly, one of them is wrong and it is worth
    knowing that before either is trusted.

    Where `fit` has already moved a constant, the naive guess for that same
    constant is downgraded to a note. The search tried values either side of
    where it stopped; a one-line division that contradicts it is not a second
    opinion, it is the weaker method arguing with the stronger one, and a reader
    given both as equals has no way to tell which is which.
    """
    out = []
    meta = _fit_meta()
    fitted = meta.get("constants") or {}
    if ratio > 0 and abs(np.log(max(ratio, 1e-6))) > 0.12:
        if "cell_span_f" in fitted:
            out.append("Bays are %.1fx too %s, but cell_span_f is already"
                       " fitted (%.2f)."
                       % (max(ratio, 1.0 / max(ratio, 1e-6)),
                          "fine" if ratio < 1 else "coarse",
                          cal["cell_span_f"]))
            out.append("      Add parts at the sizes it still gets wrong and"
                       " re-run fit, rather")
            out.append("      than hand-editing it -- what is left is where the"
                       " parts disagree.")
        else:
            out.append("Suggested: cell_span_f %.2f -> %.2f"
                       % (cal["cell_span_f"],
                          float(np.clip(cal["cell_span_f"] / ratio, 0.05, 1.0))))
    if abs(r_ours - r_real) > 0.03:
        d = r_real - r_ours
        if "target_lo" in fitted:
            # The search has already been down this axis. Printing the naive
            # guess anyway would tell the reader to hand-edit a constant back
            # towards a value the fitter tested and rejected, and the reader has
            # no way to know that the one-line suggestion is the weaker of the
            # two numbers. Say which one looked harder instead.
            out.append("Removal is still %.0f pts %s, but `fit` searched the"
                       " target band" % (abs(d) * 100.0,
                                         "low" if d > 0 else "high"))
            out.append("      and settled at [%.2f,%.2f] on %d part(s). Asking"
                       " for more removal"
                       % (cal["target_lo"], cal["target_hi"],
                          int(meta.get("n_parts") or 0)))
            out.append("      does not produce it here -- the binding limit is"
                       " rib width, minimum")
            out.append("      pocket area and hole clearance, none of which the"
                       " band can move.")
        else:
            out.append("Suggested: target band [%.2f,%.2f] -> [%.2f,%.2f]"
                       % (cal["target_lo"], cal["target_hi"],
                          float(np.clip(cal["target_lo"] + d, 0.05, 0.90)),
                          float(np.clip(cal["target_hi"] + d, 0.10, 0.94))))
    # Compared coarsely on purpose. Whether a triangulated web has its nodes on
    # the bores (radial), between two chords (truss) or hex-packed (waffle) is
    # not recoverable from the shape of the resulting bays -- they are triangles
    # either way -- so demanding an exact match would flag disagreements that
    # are really just the limits of measuring a pattern from a photograph. What
    # IS visible is the difference between a lattice and no lattice, and that
    # one is worth reporting.
    def coarse(p):
        return "lattice" if p in ("truss", "waffle", "radial") else (p or "-")

    mism = [r for r in rows
            if coarse(r["ours"].get("engine_pattern")) !=
               coarse(r["real"].get("pattern"))]
    if len(mism) >= max(2, len(rows) // 2):
        out.append("Note: %d of %d parts differ on whether there is a lattice at"
                   " all (%s)."
                   % (len(mism), len(rows),
                      ", ".join(sorted(r["name"] for r in mism))[:56]))
        out.append("      That is classify_part's area thresholds, not a bay"
                   " size, and fit does not touch them.")
    if not out:
        out.append("No change suggested: the engine is already inside the"
                   " noise of these parts.")
    return out


# --------------------------------------------------------------------------
# the fit
# --------------------------------------------------------------------------
# Which constants get a vote, and how far they are allowed to move. The bounds
# are not decoration: a coordinate search on six data points will happily run a
# constant to an absurd value that happens to fit, and the bounds are where
# "fits the data" stops being evidence and starts being an artefact.
FIT_PARAMS = [
    ("cell_span_f", 0.08, 0.90),      # the large-part knob
    ("cell_short_f", 0.10, 0.70),
    ("cell_rib_f", 5.0, 34.0),
    ("target_center", 0.25, 0.80),    # band centre; width is held at its default
    # The thin-stock knob. 0 would mean bay size does not care about thickness
    # at all (true if in-plane bending governs); 1.0 is weak-axis Euler buckling
    # and is the shipped default. The ceiling is above 1.0 rather than at it so
    # a genuine measurement slightly over the theory is not silently clipped
    # into agreeing with the theory -- but far enough below 2.0 that the search
    # cannot buy a fit with an exponent no mechanism produces.
    ("thick_exp", 0.0, 1.6),
]

# How much thickness variation a reference set needs before `thick_exp` is
# worth searching, as a ratio of the thickest usable part to the thinnest.
#
# With every part on the same stock the multiplier is (t/t_ref)**e with t/t_ref
# identical for all of them, so the objective is exactly FLAT along this axis:
# every trial value scores the same and the search learns nothing while paying
# full price for it (one grid of engine runs per pass, which on this objective
# is minutes). MIN_GAIN would hold the constant anyway, so the result is
# already correct -- this only stops it being slow, and makes the report say
# WHY the constant held, which "(held)" on its own does not.
#
# 1.4 is a little under the 1/8"-to-1/4" step, so the commonest real mix of
# aluminium plate qualifies and a set that merely spans mill tolerance does not.
THICK_SPREAD_MIN = 1.4

# How many parts have to be off the majority stock before the exponent is a
# measurement rather than one part's excuse.
#
# Spread alone is not enough, and this was found the hard way. Team 2813's set
# has a 1.99x spread and passes THICK_SPREAD_MIN comfortably -- but the spread
# comes from ONE part. Vary thick_exp on that set and exactly one part's score
# moves, which makes it not an exponent but a per-part fudge factor: the search
# will drive it to whatever cancels that part's residual, from whatever cause.
# It duly did, proposing 0.72 and buying 12% of the gap, against a theoretical
# 1.0 that two independent measurements on that very part (bay ratio 1.05,
# removal ratio 1.10) both agree with. The 0.72 is P3003's aspect-4.0 outline
# and its fifteen through-cuts being blamed on its thickness.
#
# Two is the smallest number at which the exponent has to explain more than one
# part at once, and therefore the smallest number at which a wrong value has
# somewhere to show up. It is a low bar on purpose -- the point is to exclude
# n=1, not to demand a study.
THICK_MIN_OFF_STOCK = 2

# What counts as "the same stock" when grouping. 1/4" plate measures 6.35-6.45
# across their parts -- mill tolerance and tessellation, not a design decision --
# and treating a 1.6% difference as a thickness signal is fitting noise.
THICK_SAME_TOL = 0.12

# How much a constant has to buy before the fit is allowed to move it, as a
# fraction of the gap still on the table.
#
# Without this the search accepts any improvement at all, and on six parts "any
# improvement" includes improvements that are really just which pixels happened
# to land where. The first run of this fitter moved cell_rib_f from 14.0 to 8.9
# -- a third of its range -- to buy 0.0015 of gap, and then wrote 8.925 to a
# file as though that third decimal place had been measured. A constant whose
# effect is down at the noise floor should keep its shipped value, which was at
# least chosen by looking at real parts.
MIN_GAIN = 0.02


def _expand(vec):
    """Fit vector -> calibration overrides."""
    cal = {}
    base = pocketing.CAL_DEFAULTS
    width = base["target_hi"] - base["target_lo"]
    for (name, _lo, _hi), val in zip(FIT_PARAMS, vec):
        if name == "target_center":
            cal["target_lo"] = float(np.clip(val - width / 2.0, 0.05, 0.88))
            cal["target_hi"] = float(np.clip(val + width / 2.0, 0.10, 0.94))
        else:
            cal[name] = float(val)
    return cal


def _start_vec():
    base = pocketing.calibration()
    out = []
    for name, lo, hi in FIT_PARAMS:
        if name == "target_center":
            out.append((base["target_lo"] + base["target_hi"]) / 2.0)
        else:
            out.append(base[name])
    return out


def cmd_fit(args):
    ensure_dir()
    parts = find_parts()
    if not parts:
        return cmd_list(args)
    print("\nmeasuring %d reference part(s)..." % len(parts))
    prepped = prepare(parts, verbose=args.verbose)
    for p in prepped:
        if p.get("error"):
            print("  skipped %-18s %s" % (p["name"], p["error"]))
    usable = [p for p in prepped if not p.get("error") and p.get("masks")]
    if args.min_span > 0:
        usable = [p for p in usable
                  if p["real"]["span_mm"] >= args.min_span]
    if len(usable) < 2:
        print("\nfit needs at least 2 usable reference parts; found %d."
              % len(usable))
        print("Run `list` to see what is missing.\n")
        return EXIT_NOT_WRITTEN

    print("\nfitting on %d reference part%s: %s"
          % (len(usable), "" if len(usable) == 1 else "s",
             ", ".join(p["name"] for p in usable)))

    # Which constants this particular reference set is entitled to an opinion
    # about. Only thickness is conditional so far, and it is conditional on the
    # set containing more than one stock -- see THICK_SPREAD_MIN.
    ts = [float((p.get("extras") or {}).get("thickness_mm") or 0.0)
          for p in usable]
    ts = [t for t in ts if t > 0]
    spread = (max(ts) / min(ts)) if len(ts) >= 2 else 1.0

    # Group by stock, within tolerance, and count how many parts are NOT on the
    # commonest one. That count -- not the spread -- is how many parts actually
    # constrain the exponent. See THICK_MIN_OFF_STOCK.
    groups = []
    for t in sorted(ts):
        for g in groups:
            if abs(t - g[0]) <= THICK_SAME_TOL * g[0]:
                g.append(t)
                break
        else:
            groups.append([t])
    groups.sort(key=len, reverse=True)
    n_off = sum(len(g) for g in groups[1:])

    fit_thick = spread >= THICK_SPREAD_MIN and n_off >= THICK_MIN_OFF_STOCK
    if not fit_thick:
        if not ts:
            why = "no part in this set states its thickness (photos cannot)"
            add = ("A thickness exponent is only measurable against parts cut"
                   " from different stock.")
        elif len(ts) < 2:
            why = "only one part states its thickness"
            add = ("A thickness exponent is only measurable against parts cut"
                   " from different stock.")
        elif spread < THICK_SPREAD_MIN:
            why = ("every part is %.1f-%.1f mm stock (%.2fx spread)"
                   % (min(ts), max(ts), spread))
            add = ("Fitting it here would search an axis the objective is"
                   " flat along.")
        else:
            off = sorted(t for g in groups[1:] for t in g)
            why = ("%d of %d parts are %.2f mm stock and only %s off it"
                   % (len(groups[0]), len(ts), groups[0][0],
                      ", ".join("%.2f" % t for t in off)))
            add = ("With one part off the majority stock, this exponent moves"
                   " one part's score and\n  nothing else -- so the search"
                   " would set it to cancel that part's residual whatever\n"
                   "  the residual is actually from, and call the result a"
                   " measurement.")
        print("  holding thick_exp at %.2f: %s."
              % (pocketing.CAL_DEFAULTS["thick_exp"], why))
        print("  %s" % add)
        print("  Add a second 1/8\" part (or a 3/8\" one) and it becomes"
              " fittable.")
    active = [i for i, (n, _lo, _hi) in enumerate(FIT_PARAMS)
              if fit_thick or n != "thick_exp"]

    cache = {}
    calls = [0]

    def score(vec):
        key = tuple(round(float(v), 5) for v in vec)
        if key in cache:
            return cache[key]
        rows = evaluate(usable, cal=_expand(vec), density=args.density)
        g = aggregate(rows)
        cache[key] = g
        calls[0] += 1
        return g

    vec = _start_vec()
    g0 = score(vec)
    print("  starting gap %.4f" % g0)
    best_g = g0

    # Coordinate descent, coarse then narrow. Not a global optimiser and not
    # pretending to be: with a handful of parts and an objective this noisy, a
    # cleverer search would mostly find a better fit to the noise.
    for sweep in range(args.passes):
        span_f = 1.0 if sweep == 0 else 0.35 ** sweep
        for i in active:
            name, lo, hi = FIT_PARAMS[i]
            cur = vec[i]
            width = (hi - lo) * span_f
            grid = np.unique(np.clip(
                np.linspace(cur - width / 2.0, cur + width / 2.0, args.grid),
                lo, hi))
            local_best, local_g = cur, best_g
            for val in grid:
                trial = list(vec)
                trial[i] = float(val)
                g = score(trial)
                if g < local_g - 1e-9:
                    local_best, local_g = float(val), g
            need = max(best_g * MIN_GAIN, 1e-4)
            took = local_best != cur and local_g <= best_g - need
            if took:
                vec[i] = local_best
                best_g = local_g
            print("  pass %d  %-14s %8.3f  gap %.4f%s"
                  % (sweep + 1, name, vec[i], best_g,
                     "" if took else "   (held)"))

    if best_g >= g0 - 1e-4:
        print("\nNo improvement found. The engine already fits these parts as"
              " well as these constants can.")
        print("Nothing written; %s left as it was.\n"
              % os.path.relpath(CAL_PATH, ROOT))
        return EXIT_NOT_WRITTEN

    new = _expand(vec)
    old = pocketing.calibration()
    print("\n  %-14s %10s    %-10s" % ("CONSTANT", "FROM", "TO"))
    for k in sorted(new):
        if abs(new[k] - old[k]) < 1e-9:
            continue
        print("  %-14s %10.3f -> %-10.3f" % (k, old[k], new[k]))
    print("\n  gap %.4f -> %.4f  (%.0f%% closer)"
          % (g0, best_g, 100.0 * (1.0 - best_g / max(g0, 1e-9))))
    print("  %d engine runs" % (calls[0] * len(usable)))

    if args.dry_run:
        print("\n--dry-run: nothing written.\n")
        return EXIT_NOT_WRITTEN

    payload = {
        "_comment": ("Written by tools/pocket_ref.py fit. Delete this file to "
                     "restore the defaults in app/pocketing.py CAL_DEFAULTS. "
                     "Hand-editing is fine; unknown keys are ignored."),
        "fitted_on": [p["name"] for p in usable],
        "n_parts": len(usable),
        # The size range these constants were fitted over, so the app can show
        # it. A fit is a better guess than the default only for parts the size
        # of the ones it saw, and that caveat belongs in the file rather than
        # in a console warning the user reads once and closes.
        "span_mm": [round(min(p["real"]["span_mm"] for p in usable), 1),
                    round(max(p["real"]["span_mm"] for p in usable), 1)],
        "gap_before": round(g0, 4),
        "gap_after": round(best_g, 4),
        "density": args.density,
        "constants": {k: round(v, 4) for k, v in new.items()},
    }
    os.makedirs(os.path.dirname(CAL_PATH), exist_ok=True)
    with open(CAL_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("\n  wrote %s" % os.path.relpath(CAL_PATH, ROOT))
    print("  the engine reads it on the next run;"
          " `pocket_ref.py revert` undoes it")
    print("\n  WARNING: fitted on %d part%s. A constant fitted to %d examples"
          % (len(usable), "" if len(usable) == 1 else "s", len(usable)))
    print("  is a guess with a decimal point on it. It is a better guess than")
    print("  the one it replaced only for parts like the ones you gave it --")
    if usable:
        spans = [p["real"]["span_mm"] for p in usable]
        print("  here, %.0f-%.0f mm. Check the result on a part you did not fit."
              % (min(spans), max(spans)))
    print("")
    return EXIT_WROTE


def cmd_revert(args):
    if not os.path.exists(CAL_PATH):
        print("\nNo calibration file; already running the shipped defaults.\n")
        return
    if args.keep:
        bak = CAL_PATH + ".bak"
        os.replace(CAL_PATH, bak)
        print("\nmoved to %s -- rename it back to restore.\n"
              % os.path.relpath(bak, ROOT))
    else:
        os.remove(CAL_PATH)
        print("\ndeleted %s; back to the defaults in app/pocketing.py.\n"
              % os.path.relpath(CAL_PATH, ROOT))


def cmd_show(args):
    cal = pocketing.calibration()
    base = pocketing.CAL_DEFAULTS
    meta = {}
    if os.path.exists(CAL_PATH):
        try:
            with open(CAL_PATH, "r", encoding="utf-8") as fh:
                meta = json.load(fh) or {}
        except Exception:
            pass
    if meta:
        print("\ncalibrated from %d part(s): %s"
              % (meta.get("n_parts", 0),
                 ", ".join(meta.get("fitted_on") or []) or "?"))
        print("gap %s -> %s" % (meta.get("gap_before", "?"),
                                meta.get("gap_after", "?")))
    else:
        print("\nno calibration file: these are the shipped defaults")
    print("\n  %-16s %10s %10s" % ("CONSTANT", "ACTIVE", "DEFAULT"))
    for k in sorted(base):
        mark = "  <-" if abs(cal[k] - base[k]) > 1e-12 else ""
        print("  %-16s %10.4f %10.4f%s" % (k, cal[k], base[k], mark))
    print("")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="pocket_ref",
        description="Calibrate the pocketing engine against parts you machined.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Naming your files")[0])
    sub = p.add_subparsers(dest="cmd")

    def common(sp, span=True):
        sp.add_argument("--density", default="normal",
                        choices=["conservative", "normal", "aggressive"],
                        help="which preset to compare against (default normal)")
        sp.add_argument("-v", "--verbose", action="store_true")
        if span:
            sp.add_argument("--min-span", type=float, default=0.0,
                            metavar="MM",
                            help="ignore parts smaller than this. Use it to "
                                 "calibrate large parts on large parts only.")

    sp = sub.add_parser("list", help="what is in reference_parts/")
    common(sp, span=False)
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("report", help="real vs ours, part by part")
    common(sp)
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("fit", help="search the constants and write the file")
    common(sp)
    sp.add_argument("--passes", type=int, default=2,
                    help="coordinate sweeps (default 2)")
    sp.add_argument("--grid", type=int, default=5,
                    help="candidates per constant per sweep (default 5)")
    sp.add_argument("--dry-run", action="store_true",
                    help="search, print, write nothing")
    sp.set_defaults(func=cmd_fit)

    sp = sub.add_parser("show", help="the constants in force right now")
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser("revert", help="delete the calibration file")
    sp.add_argument("--keep", action="store_true",
                    help="rename to .bak instead of deleting")
    sp.set_defaults(func=cmd_revert)

    args = p.parse_args(argv)
    if not getattr(args, "cmd", None):
        p.print_help()
        print("")
        return EXIT_WROTE
    return args.func(args) or EXIT_WROTE


if __name__ == "__main__":
    sys.exit(main() or EXIT_WROTE)
