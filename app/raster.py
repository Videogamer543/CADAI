"""
Rasterization helpers: turn the FEM mesh + nodal von Mises into pixel grids
(part mask, hole mask, stress field) so the pocketing engine can run on a
regular raster with SciPy/NumPy.
"""
from __future__ import annotations
import numpy as np
import cv2


def masks_from_polygons(exterior, holes, W, H):
    part = np.zeros((H, W), np.uint8)
    cv2.fillPoly(part, [np.array(exterior, np.int32)], 1)
    hole = np.zeros((H, W), np.uint8)
    for h in holes:
        if len(h) >= 3:
            cv2.fillPoly(hole, [np.array(h, np.int32)], 1)
    part = (part.astype(bool)) & (~hole.astype(bool))
    return part, hole.astype(bool)


def field_from_mesh(nodes, tris, vm_norm, W, H):
    """Rasterize per-triangle averaged von Mises into a W×H float grid."""
    field = np.zeros((H, W), np.float32)
    nodes = np.asarray(nodes)
    for t in tris:
        v = (vm_norm[t[0]] + vm_norm[t[1]] + vm_norm[t[2]]) / 3.0
        poly = nodes[t].astype(np.int32)
        cv2.fillConvexPoly(field, poly, float(v))
    return field
