"""
Subprocess worker: STEP file -> tetrahedral VOLUME mesh, written as an .npz.

Same reasoning as app/step_worker.py -- gmsh is not thread-safe, keeps global
state and installs signal handlers, so it only ever runs as the main thread of
a throwaway interpreter. A crash in the CAD kernel takes down this process and
nothing else.

Why .npz instead of the JSON that step_worker prints: a surface tessellation is
tens of thousands of numbers, but a volume mesh at solver density is millions.
Encoding that as JSON text costs more time than the mesh generation itself and
several hundred megabytes of transient string. The worker writes a binary array
file and prints only its path.

Why there is a ladder of strategies
-----------------------------------
Volume meshing fails on real CAD in ways that have nothing to do with how the
mesh was asked for. The one that brought this path down was:

    Invalid boundary mesh (overlapping facets) on surface 233

That is gmsh's 3D Delaunay boundary-recovery step reporting that the SURFACE
mesh it was handed intersects itself -- two facets of the same face crossing.
It is provoked by tight curvature on small features (gear teeth, a bolt circle
of small bores) meeting a size field that is coarse elsewhere, and it is a
property of one particular solid: the same settings mesh a hundred other parts
without complaint. So there is nothing to "fix" in the settings, and a single
attempt is the wrong shape of code.

What works is trying again differently. HXT is a separate 3D implementation
with its own boundary recovery and frequently succeeds where Delaunay fails.
AngleToleranceFacetOverlap is the threshold the failing test uses, and relaxing
it stops gmsh rejecting facets that are merely thin. Turning curvature
refinement down removes the slivers at source. Each of those is a rung, they
run coarsest-last, and the rung that succeeded is recorded so a degraded mesh
can never pass itself off as a clean one.

OCC healing (OCCFixSmallEdges / OCCFixSmallFaces / OCCSewFaces) is deliberately
NOT in the ladder. Measured on a 40-tooth pulley it took 104 seconds and
produced zero tetrahedra -- it removes the small faces the part is made of.

Usage:  python -m app.vol_worker <step_path> <out_npz> <target_tets>
"""
import sys
import numpy as np


# (label, curvature segments, Algorithm3D, min-size factor, facet-overlap tol)
#
# Algorithm3D: 1 = Delaunay (fast, the default), 10 = HXT (different boundary
# recovery). Facet tolerance default is 0.1; smaller means gmsh is slower to
# declare two facets overlapping.
STRATEGIES = (
    ("standard",               12,  1, 0.22, 0.1),
    ("HXT boundary recovery",  12, 10, 0.22, 0.1),
    ("relaxed overlap test",   12, 10, 0.35, 0.02),
    ("reduced curvature",       6, 10, 0.35, 0.02),
    ("curvature off",           0, 10, 0.50, 0.02),
    ("curvature off, coarse",   0,  1, 0.60, 0.02),
)


class BudgetExceeded(RuntimeError):
    """Too many elements at these settings -- NOT a meshing failure.

    Kept distinct because the two escalate for different reasons and deserve
    different words. A part with forty gear teeth legitimately overshoots the
    element budget at full curvature and is meshed coarser; that is routine
    and telling the user their mesh is "degraded" for it is crying wolf. An
    overlapping-facet error is a real failure of this solid at these settings
    and does deserve to be said out loud.
    """


def _count_tets(gmsh):
    ets, _, ens = gmsh.model.mesh.getElements(3)
    for et, en in zip(ets, ens):
        if et == 4:
            return len(en) // 4
    return 0


def _attempt(gmsh, path, target_tets, curv, alg3d, minf, facet_tol):
    """One rung. Opens the model fresh, meshes to budget, returns the arrays.

    Re-opening rather than re-meshing the loaded model is deliberate: a failed
    boundary recovery leaves gmsh's internal state partly built, and reusing it
    makes the next rung fail for reasons belonging to the previous one.
    """
    gmsh.clear()
    gmsh.open(path)
    gmsh.model.occ.synchronize()

    try:
        xa, ya, za, xb, yb, zb = gmsh.model.getBoundingBox(-1, -1)
        spans = sorted([xb - xa, yb - ya, zb - za])
        vol = max(1e-9, spans[0] * spans[1] * spans[2])
    except Exception:
        spans, vol = [1.0, 1.0, 1.0], 1.0
    try:
        vol = sum(max(0.0, gmsh.model.occ.getMass(3, t))
                  for _, t in gmsh.model.getEntities(3)) or vol
    except Exception:
        pass

    # 1.5 rather than the textbook 6: a Delaunay tet is nothing like a regular
    # one, and measured against gmsh's own output the mean element comes out
    # about four times h^3/6. The textbook figure asked for 42k and got 9k.
    target = max(1000, int(target_tets))
    h = (1.5 * vol / target) ** (1.0 / 3.0)
    # Never coarser than a third of the thin direction: a plate meshed with one
    # tet through its thickness cannot bend, and would report a part as several
    # times stiffer than it is.
    h = min(h, max(spans[0] / 3.0, 1e-6))

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.Algorithm3D", alg3d)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curv)
    gmsh.option.setNumber("Mesh.AngleToleranceFacetOverlap", facet_tol)

    # The target is a BUDGET, not a wish. A 3D elasticity matrix factorised by
    # SuperLU costs memory far faster than linearly in the element count, so a
    # mesh that overshoots does not merely run slow -- the kernel kills the
    # process mid-factorisation, which surfaces as a 503 with an empty body and
    # no traceback anywhere. Nothing downstream can catch that.
    cap = int(max(target * 1.6, target + 2000))
    best_n = None
    for _ in range(4):
        gmsh.option.setNumber("Mesh.MeshSizeMin", h * minf)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h * 1.6)
        gmsh.model.mesh.clear()
        gmsh.model.mesh.generate(3)
        n_tets = _count_tets(gmsh)
        if not n_tets:
            raise RuntimeError("no tetrahedra produced")
        if best_n is None or n_tets < best_n:
            best_n = n_tets
        if n_tets <= cap and n_tets >= 0.35 * target:
            break
        h *= float(np.clip((n_tets / float(target)) ** (1.0 / 3.0), 0.45, 2.2))

    n_tets = _count_tets(gmsh)
    if n_tets > cap * 2:
        raise BudgetExceeded(
            "%d tetrahedra against a budget of %d" % (n_tets, cap))

    tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.asarray(coords, float).reshape(-1, 3)
    idx = np.full(int(tags.max()) + 2, -1, np.int64)
    idx[np.asarray(tags, np.int64)] = np.arange(len(tags))

    def conn(dim, etype, k):
        ets, _, ens = gmsh.model.mesh.getElements(dim)
        for et, en in zip(ets, ens):
            if et == etype:
                return idx[np.asarray(en, np.int64).reshape(-1, k)]
        return np.zeros((0, k), np.int64)

    tets = conn(3, 4, 4)          # 4-node tetrahedron
    surf = conn(2, 2, 3)          # 3-node triangle
    if tets.shape[0] == 0:
        raise RuntimeError("gmsh produced no tetrahedra")
    return coords, tets, surf


def build(path, target_tets=16000):
    """STEP path -> (nodes, tets, surface tris, note). Tries the ladder."""
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        errs = []
        budget_only = True          # has every escalation so far been size?
        for i, (label, curv, alg3d, minf, tol) in enumerate(STRATEGIES):
            try:
                coords, tets, surf = _attempt(gmsh, path, target_tets,
                                              curv, alg3d, minf, tol)
            except BudgetExceeded as e:
                errs.append("%s: %s" % (label, e))
                budget_only = budget_only and True
                continue
            except Exception as e:
                errs.append("%s: %s" % (label, str(e)[:110]))
                budget_only = False
                continue
            # Drop nodes no tet references (gmsh emits vertices for 0D and 1D
            # entities too) and renumber, so the solver never sees a zero row
            # in its stiffness matrix.
            used = np.zeros(coords.shape[0], bool)
            used[tets.ravel()] = True
            ren = np.full(coords.shape[0], -1, np.int64)
            ren[used] = np.arange(int(used.sum()))
            surf = surf[(ren[surf] >= 0).all(axis=1)] if surf.size else surf
            if i == 0:
                note = ""
            elif budget_only:
                # Routine: the part is simply dense at full curvature.
                note = ("meshed coarser than the default ('%s') to stay inside "
                        "the element budget -- %s. Normal on a part with many "
                        "small features." % (label, errs[0] if errs else ""))
            else:
                note = ("meshed with the '%s' fallback: the standard settings "
                        "could not mesh this solid (%s). The stress field is "
                        "real; treat fine detail near small features with a "
                        "little caution." % (label, errs[0] if errs else ""))
            return coords[used], ren[tets], (ren[surf] if surf.size else surf), note
        raise RuntimeError(
            "volume meshing failed at every setting. Attempts -- "
            + " | ".join(errs))
    finally:
        gmsh.finalize()


def main():
    path, out = sys.argv[1], sys.argv[2]
    target = int(sys.argv[3]) if len(sys.argv) > 3 else 16000
    nodes, tets, surf, note = build(path, target)
    np.savez_compressed(out, nodes=nodes.astype(np.float64),
                        tets=tets.astype(np.int32),
                        surf=surf.astype(np.int32),
                        note=np.array(note))
    print(out)


if __name__ == "__main__":
    main()
