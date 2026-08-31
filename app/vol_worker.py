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

Usage:  python -m app.vol_worker <step_path> <out_npz> <target_tets>
"""
import sys
import numpy as np


def build(path, target_tets=16000):
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(path)

        # Element size from the target count rather than a fixed factor. A
        # regular tet of edge h occupies about h^3/6, so h = (6V/N)^(1/3) puts
        # roughly N tets in the solid whatever its size -- a 40 mm bushing and a
        # 900 mm bellypan rail both come back at a density the solver can carry.
        try:
            gmsh.model.occ.synchronize()
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
        # 1.5 rather than the textbook 6: a Delaunay tet is nothing like a
        # regular one, and measured against gmsh's own output the mean element
        # comes out about four times h^3/6. Using the textbook figure asked for
        # 42k and got 9k.
        target = max(1000, int(target_tets))
        h = (1.5 * vol / target) ** (1.0 / 3.0)
        # Never coarser than a third of the thin direction: a plate meshed with
        # one tet through its thickness cannot bend, and would report a part as
        # several times stiffer than it is.
        h = min(h, max(spans[0] / 3.0, 1e-6))

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)

        # The target is a BUDGET, not a wish. Everything downstream is sized
        # from it: a 3D elasticity matrix factorised by SuperLU costs memory
        # far faster than linearly in the element count, so a mesh that
        # overshoots does not merely run slow -- the kernel kills the process
        # mid-factorisation, which surfaces as a 503 with an empty body and no
        # traceback anywhere. Nothing in the app can catch that, so it has to
        # be prevented here.
        #
        # The old code did ONE corrective pass and then accepted whatever came
        # back regardless. On a part with many small round features that is not
        # close to enough: measured on a 36-tooth pulley asking for 9,000 tets,
        # pass one gave 115,128 and pass two gave 76,541 -- and 76,541 was
        # handed to the solver, which is roughly 336,000 quadratic degrees of
        # freedom and certain death. Iterating properly reaches 13,712.
        #
        # Curvature refinement is what fights the budget: MeshSizeFromCurvature
        # asks for a fixed number of elements around every arc, so forty small
        # bores can outvote the size field no matter how large h grows. It is
        # worth having -- it is what keeps a bore round -- so it is relaxed in
        # steps rather than abandoned, and only when the size field alone has
        # failed to get under the cap.
        cap = int(max(target * 1.6, target + 2000))
        best = None                      # (n_tets, h, curvature) actually met
        for curv in (12, 6, 0):
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", curv)
            for _ in range(4):
                gmsh.option.setNumber("Mesh.MeshSizeMin", h * 0.22)
                gmsh.option.setNumber("Mesh.MeshSizeMax", h * 1.6)
                gmsh.model.mesh.clear()
                gmsh.model.mesh.generate(3)
                n_tets = 0
                ets, _, ens = gmsh.model.mesh.getElements(3)
                for et, en in zip(ets, ens):
                    if et == 4:
                        n_tets = len(en) // 4
                if not n_tets:
                    break
                if best is None or n_tets < best[0]:
                    best = (n_tets, h, curv)
                if n_tets <= cap and n_tets >= 0.35 * target:
                    break
                h *= float(np.clip((n_tets / float(target)) ** (1.0 / 3.0),
                                   0.45, 2.2))
            if best and best[0] <= cap:
                break

        # Last resort: the size field and the curvature setting together could
        # not get under the budget. Rebuild at the coarsest thing tried, with
        # curvature off entirely, and if THAT still overshoots, say so plainly
        # rather than returning a mesh that kills the process that receives it.
        if best is None or best[0] > cap:
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.MeshSizeMin", h * 0.5)
            gmsh.option.setNumber("Mesh.MeshSizeMax", h * 3.0)
            gmsh.model.mesh.clear()
            gmsh.model.mesh.generate(3)
            n_tets = 0
            ets, _, ens = gmsh.model.mesh.getElements(3)
            for et, en in zip(ets, ens):
                if et == 4:
                    n_tets = len(en) // 4
            if not n_tets or n_tets > cap * 2:
                raise RuntimeError(
                    "this solid could not be meshed inside the element budget "
                    "(%d tetrahedra, budget %d). It usually means a great many "
                    "small rounded features -- gear teeth, a bolt circle of "
                    "tiny holes -- each of which forces its own local "
                    "refinement." % (n_tets, cap))

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
    finally:
        gmsh.finalize()

    # Drop nodes no tet references (gmsh emits vertices for 0D/1D entities too)
    # and renumber, so the solver never sees a zero row in its stiffness matrix.
    used = np.zeros(coords.shape[0], bool)
    used[tets.ravel()] = True
    ren = np.full(coords.shape[0], -1, np.int64)
    ren[used] = np.arange(int(used.sum()))
    surf = surf[(ren[surf] >= 0).all(axis=1)] if surf.size else surf
    return coords[used], ren[tets], (ren[surf] if surf.size else surf)


def main():
    path, out = sys.argv[1], sys.argv[2]
    target = int(sys.argv[3]) if len(sys.argv) > 3 else 16000
    nodes, tets, surf = build(path, target)
    np.savez_compressed(out, nodes=nodes.astype(np.float64),
                        tets=tets.astype(np.int32),
                        surf=surf.astype(np.int32))
    print(out)


if __name__ == "__main__":
    main()
