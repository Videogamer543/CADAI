"""
Image / silhouette → polygon-with-holes extraction.

Robust replacement for the browser app's pixel-threshold flood fill: uses
OpenCV contour detection with hierarchy so holes (bearing bores, mounting
holes, lightening pockets) come through as inner rings the mesher can carve.
"""
from __future__ import annotations
import io
import numpy as np
import cv2
from PIL import Image


def _load_gray(data: bytes):
    img = Image.open(io.BytesIO(data)).convert("RGBA")
    arr = np.array(img)
    # composite onto white using alpha, then grayscale
    alpha = arr[:, :, 3:4] / 255.0
    rgb = arr[:, :, :3] * alpha + 255.0 * (1 - alpha)
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return gray, arr


def extract_polygons(data: bytes, simplify=1.5, min_area_frac=0.02):
    """
    Returns (exterior, holes, size) where:
      exterior : list[(x, y)] largest part outline
      holes    : list[list[(x, y)]] inner boundaries
      size     : (W, H)
    Coordinates are in image pixels (y down).
    """
    gray, arr = _load_gray(data)
    H, W = gray.shape

    # binarize: part = anything sufficiently different from the (light) border
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    bg = np.percentile(border, 75)
    if bg > 200:
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    else:
        mask = (np.abs(gray.astype(int) - bg) > 35).astype(np.uint8) * 255
    # 3x3 close: bridges antialiasing seams without swallowing small bores.
    # (The old 5x5 was wide enough to erase 4-6 px countersink holes outright.)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise ValueError("No part detected in image")
    hier = hier[0]

    # largest outer contour = the part
    areas = [cv2.contourArea(c) for c in cnts]
    outer_idx = max(
        (i for i in range(len(cnts)) if hier[i][3] == -1),
        key=lambda i: areas[i],
    )
    part_area = areas[outer_idx]

    def simplify_contour(c):
        # scale eps to the ring: a fixed 1.5 px tolerance flattens a 12 px
        # countersink into a triangle (or nothing), so small rings get a
        # proportionally tighter tolerance and keep their round shape.
        peri = cv2.arcLength(c, True)
        eps = min(simplify, max(0.35, peri * 0.012))
        ap = cv2.approxPolyDP(c, eps, True)
        return [(float(p[0][0]), float(p[0][1])) for p in ap]

    exterior = simplify_contour(cnts[outer_idx])

    # Speckle floor, in PIXELS, not a fraction of the part.
    # The old `part_area * 0.0008` scaled with the plate: on a 300 mm plate it
    # demanded a hole bigger than ~9 mm, which silently dropped every #10
    # clearance / countersunk hole on the part. Anything above ~4 px across is
    # real geometry; JPEG/antialias speckle is smaller than that.
    min_hole_px = max(14.0, part_area * 2e-5)
    holes = []
    child = hier[outer_idx][2]
    while child != -1:
        a = areas[child]
        if a > min_hole_px:
            ring = simplify_contour(cnts[child])
            if len(ring) >= 3:
                holes.append(ring)
        child = hier[child][0]

    return exterior, holes, (W, H)
