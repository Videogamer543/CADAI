"""
Orchestration for the SOLID (3D) stress map — the path taken when the part is
not being treated as a 2D plate.

Two ways in, one way out:

  STEP file   → gmsh meshes the real solid into tetrahedra (app/vol_worker.py,
                in a subprocess, because gmsh is not thread-safe).
  image file  → the same silhouette + triangulation the flat solver uses, then
                extruded through the stated thickness into tets.

Both then go through app/fem3d.solve_solid, so a photographed bracket and a
CAD bracket are judged by exactly the same physics. The difference is only in
how honest the geometry is: an extruded silhouette is a true prism, so it has
no chamfers, no bosses and no varying thickness, and the payload says so via
`geometry_source` rather than letting the picture imply more than it knows.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from . import materials
from .analysis import MODE_KT

# A stress picture, not a certification run. 16k tets at P2 is mesh-converged
# for peak von Mises to a couple of percent on the parts this tool sees (12k
# and 36k agreed to 0.07% on tip deflection), and it solves in seconds instead
# of minutes.
TARGET_TETS = 16000


def _mesh_step(data: bytes, target_tets: int = TARGET_TETS):
    """STEP bytes → (nodes_mm, tets). Runs gmsh out of process.

    The scratch directory is created and removed by hand rather than with
    `tempfile.TemporaryDirectory()`, and the .npz is read inside a `with`. Both
    are Windows fixes, and they are worth a note because the failure they cause
    is remote from its cause:

        [WinError 32] The process cannot access the file because it is being
        used by another process: 'C:\\Users\\...\\AppData\\Local\\Temp\\tmp...'

    `np.load` on an .npz does NOT read the archive. It returns a lazy NpzFile
    holding the zip open, and each `d["nodes"]` decompresses a member on
    demand. So `return d["nodes"], d["tets"]` hands back two arrays with the
    file still open behind them, and TemporaryDirectory's cleanup then tries to
    delete a file the same process has a live handle on. POSIX allows that --
    unlink drops the directory entry and the open handle keeps working -- so
    this ran clean on Linux for as long as it existed and only ever failed for
    someone on Windows, where a delete of an open file is refused outright.

    Two independent guards, because one is a correctness fix and the other is
    insurance: reading inside `with np.load(...)` closes the handle before
    cleanup runs, and `shutil.rmtree(ignore_errors=True)` means that if some
    other holder ever appears (an antivirus scanner mid-scan is the usual
    Windows culprit) the user gets their stress map and the OS reaps a few
    kilobytes of Temp later, instead of the whole analysis dying at the finish
    line with the answer already computed.
    """
    td = tempfile.mkdtemp(prefix="stressviz_vol_")
    try:
        sp = os.path.join(td, "part.step")
        out = os.path.join(td, "vol.npz")
        with open(sp, "wb") as fh:
            fh.write(data)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ)
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        r = subprocess.run(
            [sys.executable, "-m", "app.vol_worker", sp, out, str(int(target_tets))],
            cwd=root, env=env, capture_output=True, timeout=300)
        if r.returncode != 0 or not os.path.exists(out):
            err = (r.stderr or b"").decode("utf8", "replace").strip()
            raise RuntimeError(err.splitlines()[-1] if err else
                               "volume mesher failed with no message")
        # np.copy, not the bare member: the array a lazy NpzFile hands out is
        # already materialised, but being explicit here is what documents that
        # nothing may outlive the `with`.
        with np.load(out) as d:
            nodes = np.array(d["nodes"], dtype=np.float64)
            tets = np.array(d["tets"]).astype(np.int64)
        return nodes, tets
    finally:
        shutil.rmtree(td, ignore_errors=True)


def plate_thickness_of(nodes_mm, plate_ratio=0.18):
    """Measured thickness of a meshed solid, or None if it isn't plate-like.

    Same PCA trick as the 2D path (see step3d.plate_frame): the thin direction
    is the smallest principal axis, and the thickness is the spread along it.
    Fitting the part's own axes rather than reading the bounding box matters
    for exactly one common case -- a plate exported at an angle, where every
    global bbox span is large and none of them is the thickness.

    Returns None rather than a number for a genuine 3D body. A bellcrank with a
    boss on it has no single thickness, and reporting the narrowest thing about
    it as "the thickness" would be worse than admitting there isn't one.
    """
    from .step3d import plate_aspect
    v = np.asarray(nodes_mm, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] < 4:
        return None
    try:
        thin, diameter = plate_aspect(v)
    except Exception:
        return None
    if thin <= 0 or diameter <= 0 or (thin / diameter) > plate_ratio:
        return None
    return float(thin)


def _mesh_image(data: bytes, thickness_mm: float, px_per_mm=None,
                layers: int = 3, target_tets: int = TARGET_TETS):
    """Image bytes → (nodes_mm, tets) by extruding the flat triangulation."""
    from .geometry import extract_polygons
    from .mesh import build_mesh
    from .fem3d import extrude

    exterior, holes, (W, H) = extract_polygons(data)

    # The flat solver's default triangulation is ~5000 elements, which is right
    # for a plane problem and catastrophic here: extrusion multiplies it by
    # layers x 3, so 5000 triangles becomes 60000 tets and a 76-second solve.
    # Aim the 2D pass at a budget that lands the extruded count near TARGET_TETS
    # instead. Accuracy through the thickness comes from the elements being
    # quadratic, not from stacking more of them.
    budget = max(400, int(target_tets / max(1, 3 * layers)))
    bb = ((max(p[0] for p in exterior) - min(p[0] for p in exterior)) *
          (max(p[1] for p in exterior) - min(p[1] for p in exterior)))
    pts, tris = build_mesh(exterior, holes,
                           max_area=max(1.0, bb / float(budget)))[:2]
    if px_per_mm is None:
        px_per_mm = max(W, H) / 150.0

    # extrude() works in metres, so hand it metres and scale the result back to
    # the millimetres solve_solid expects. Keeping one unit convention at the
    # boundary is cheaper than tracking two through the solver.
    m_per_px = 1.0 / (px_per_mm * 1000.0)
    t_m = max(0.0005, thickness_mm / 1000.0)
    nodes_m, tets = extrude(np.asarray(pts, float), np.asarray(tris, np.int64),
                            t_m, m_per_px, layers=layers)
    return nodes_m * 1000.0, tets, (W, H), len(holes), px_per_mm


def run(data: bytes, *, filename="part", material="Aluminum 6061-T6",
        mode="structural", load_case="cantilever", orientation="horizontal",
        load=500.0, thickness_mm=6.35, px_per_mm=None,
        target_tets=TARGET_TETS):
    """Solid stress map. Returns the payload the 3D viewer paints."""
    from .fem3d import solve_solid

    from .fem3d import _mkl_spsolve

    mat = materials.get(material)
    name = (filename or "").lower()
    extra = {}

    # Direct factorisation cost grows much faster than the element count, so a
    # machine without PARDISO gets a coarser mesh rather than a four-minute
    # wait. 9k tets still resolves the bores and the neutral plane; it just
    # reports peak stress a couple of percent differently.
    if _mkl_spsolve()[0] is None:
        target_tets = min(target_tets, 9000)

    if name.endswith(".step") or name.endswith(".stp"):
        nodes_mm, tets = _mesh_step(data, target_tets)
        extra["geometry_source"] = "STEP solid (true 3D geometry)"
        # The typed thickness has no effect on this path -- the solver is
        # chewing the real solid, so the number is a readout, not an input. It
        # still gets measured and returned, because leaving the box showing the
        # 6.35 mm default next to a solve of a 1/8" plate reads as "the tool
        # ignored my file", and that is the one impression a stress tool cannot
        # afford to give.
        _meas = plate_thickness_of(nodes_mm)
        if _meas is not None:
            thickness_mm = round(_meas, 3)
            extra["thickness_source"] = "measured from the STEP solid"
        else:
            thickness_mm = None
            extra["thickness_source"] = (
                "not applicable — this is a solid, not a constant-thickness plate")
    else:
        nodes_mm, tets, size, nholes, px_per_mm = _mesh_image(
            data, thickness_mm, px_per_mm, target_tets=target_tets)
        extra["geometry_source"] = (
            f"silhouette extruded {thickness_mm:g} mm "
            f"(constant thickness assumed)")
        extra["thickness_source"] = "as entered — a photo carries no thickness"
        extra["image_size"] = list(size)
        extra["holes"] = nholes

    res = solve_solid(nodes_mm, tets, E=mat["E"], nu=mat["nu"],
                      load_case=load_case, orientation=orientation,
                      load=float(load))

    # Same stress-concentration multiplier the flat path applies, so a part
    # analysed both ways reports one safety factor, not two.
    kt = MODE_KT.get(mode, 1.0)
    peak = res["peak_vm"] * kt
    allow, allow_note = materials.allowable(mat)
    res.update(extra)
    res["material"] = material
    res["material_info"] = materials.info(material)
    res["mode"] = mode
    res["peak_vm"] = peak
    res["allowable_pa"] = allow
    res["allowable_note"] = allow_note
    res["safety_factor"] = (allow / peak) if (allow > 0 and peak > 0) else None
    res["thickness_mm"] = thickness_mm
    res["px_per_mm"] = px_per_mm
    res["is_3d"] = True
    return res
