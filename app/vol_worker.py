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

        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)

        # One corrective pass. Element count scales as h^-3, so a single
        # cube-root rescale lands close; the guard rails matter more than
        # precision, because too few elements means a wrong answer and too many
        # means the solve never finishes.
        n_tets = 0
        for attempt in range(2):
            gmsh.option.setNumber("Mesh.MeshSizeMin", h * 0.22)
            gmsh.option.setNumber("Mesh.MeshSizeMax", h * 1.6)
            gmsh.model.mesh.clear()
            gmsh.model.mesh.generate(3)
            ets, _, _ = gmsh.model.mesh.getElements(3)
            n_tets = 0
            for et, en in zip(*[gmsh.model.mesh.getElements(3)[i] for i in (0, 2)]):
                if et == 4:
                    n_tets = len(en) // 4
            if attempt or not n_tets:
                break
            ratio = n_tets / float(target)
            if 0.5 <= ratio <= 2.2:
                break
            h *= float(np.clip(ratio ** (1.0 / 3.0), 0.45, 2.2))

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
