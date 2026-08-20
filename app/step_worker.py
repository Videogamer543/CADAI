"""
Subprocess worker: tessellate a STEP file with gmsh and print JSON to stdout.

Run in its own process so gmsh (not thread-safe, installs signal handlers, holds
global state) is always the main thread of a clean interpreter — the robust way
to embed it in a web server.

Usage:  python -m app.step_worker <step_path> <size_factor> <max_tris>
"""
import sys
import json
import numpy as np


def main():
    path = sys.argv[1]
    size_factor = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    max_tris = int(sys.argv[3]) if len(sys.argv) > 3 else 60000

    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(path)
        gmsh.option.setNumber("Mesh.MeshSizeFactor", size_factor)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(coords).reshape(-1, 3)
        tag_index = {int(t): i for i, t in enumerate(node_tags)}

        etypes, _, enodes = gmsh.model.mesh.getElements(2)
        tri_conn = None
        for et, en in zip(etypes, enodes):
            if et == 2:
                tri_conn = np.array(en, dtype=np.int64).reshape(-1, 3)
                break
        if tri_conn is None or len(tri_conn) == 0:
            raise RuntimeError("no surface triangles produced")
        if len(tri_conn) > max_tris:
            tri_conn = tri_conn[:: int(np.ceil(len(tri_conn) / max_tris))]

        tris, normals = [], []
        for a, b, c in tri_conn:
            pa = coords[tag_index[int(a)]]
            pb = coords[tag_index[int(b)]]
            pc = coords[tag_index[int(c)]]
            n = np.cross(pb - pa, pc - pa)
            ln = np.linalg.norm(n) or 1.0
            n = n / ln
            tris.append([pa.tolist(), pb.tolist(), pc.tolist()])
            normals.append(n.tolist())

        mn = coords.min(axis=0).tolist()
        mx = coords.max(axis=0).tolist()
        spans = [mx[i] - mn[i] for i in range(3)]
        print(json.dumps({
            "tris": tris, "normals": normals,
            "bbox": {"min": mn, "max": mx, "spans": spans},
            "n_tris": len(tris),
        }))
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
