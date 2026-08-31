"""
Subprocess worker: tessellate a STEP file with gmsh and print JSON to stdout.

Run in its own process so gmsh (not thread-safe, installs signal handlers, holds
global state) is always the main thread of a clean interpreter — the robust way
to embed it in a web server.

Two things this does that the first version did not, both of which were visible
on screen:

* **Curvature-based sizing.** gmsh's default size field knows nothing about
  curvature, so it puts the same element on a flat face and around a 4 mm bore
  — which is why every hole in the viewer came out as an octagon. Asking for a
  fixed number of segments per turn is what makes a bore look like a bore.
  Measured on a 36-tooth pulley: 4,402 triangles before, 84,004 after, with the
  extra triangles going where the curvature is rather than over the flat webs.

* **Indexed output.** The old payload stored three full vertices per triangle,
  so every vertex was repeated about six times and a finer mesh became a
  double-digit-megabyte JSON download. Sharing vertices costs one renumbering
  pass here and cuts the wire format about fourfold (10.4 MB -> 2.5 MB at 73k
  triangles). Coordinates are rounded to a micron, which is far finer than any
  STEP this tool will see and saves another third.

Normals are no longer sent. Every consumer either computes its own face normal
(the browser does, per frame, from the rotated vertices) or has always had a
fallback that derives them from the triangle — so shipping them was a quarter
of the payload spent on a number nobody read.

Usage:  python -m app.step_worker <step_path> <size_factor> <max_tris>
"""
import sys
import json
import numpy as np

# Segments per full turn on a curved edge. The octagons this replaced were
# effectively 8. 18 is enough to make a 3 mm bore look round but NOT enough for
# a 40 mm outer rim -- the setting is per turn, so a large radius gets the same
# segment count spread over a much longer arc and still reads as a polygon.
# 26 makes both look right on the parts this tool sees. Going much past it buys
# detail below one screen pixel and costs triangles on every fillet at once.
CURVATURE_SEGMENTS = 26

# Ceiling on what gets shipped. Not a memory limit -- this is display geometry,
# there is no factorisation behind it -- but the browser paints every facet on
# every frame, so an unbounded mesh is bought at the cost of the drag being
# smooth. See the decimation note in main().
DEFAULT_MAX_TRIS = 95000


def main():
    path = sys.argv[1]
    size_factor = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    max_tris = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_MAX_TRIS

    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(path)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", CURVATURE_SEGMENTS)

        # Curvature sizing says nothing about a face that is not curved, so a
        # large flat web can still come back as two enormous triangles. Cap the
        # element at a fraction of the part's own diagonal, which is scale-free:
        # the same divisor behaves the same on a 40 mm bushing and a 900 mm rail.
        # Measured cost on the pulley: 84,004 triangles to 88,304 -- about 5%
        # for a guarantee that no face is drawn as a single facet.
        try:
            gmsh.model.occ.synchronize()
            xa, ya, za, xb, yb, zb = gmsh.model.getBoundingBox(-1, -1)
            diag = float(np.linalg.norm([xb - xa, yb - ya, zb - za]))
            if diag > 0:
                gmsh.option.setNumber("Mesh.MeshSizeMax", diag / 45.0)
        except Exception:
            pass                                # a sizing hint, never fatal

        # Stay under the budget by MESHING COARSER, never by throwing triangles
        # away. The old code did `tri_conn[::n]`, which deletes every n-th
        # triangle from a closed surface -- that does not coarsen a mesh, it
        # punches holes in it, and the holes then read as gaps in the part.
        # Triangle count on a surface goes as 1/h^2, so one square-root rescale
        # of the size factor lands close and a second pass finishes the job.
        sf = float(size_factor)
        tri = None
        coords = None
        for _ in range(3):
            gmsh.option.setNumber("Mesh.MeshSizeFactor", sf)
            gmsh.model.mesh.clear()
            gmsh.model.mesh.generate(2)

            tags, xyz, _ = gmsh.model.mesh.getNodes()
            coords = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
            idx = np.full(int(tags.max()) + 2, -1, np.int64)
            idx[np.asarray(tags, np.int64)] = np.arange(len(tags))

            tri = None
            etypes, _, enodes = gmsh.model.mesh.getElements(2)
            for et, en in zip(etypes, enodes):
                if et == 2:                      # 3-node triangle
                    tri = idx[np.asarray(en, np.int64).reshape(-1, 3)]
                    break
            if tri is None or len(tri) == 0:
                raise RuntimeError("no surface triangles produced")
            if len(tri) <= max_tris:
                break
            sf *= float(np.clip((len(tri) / float(max_tris)) ** 0.5, 1.05, 4.0))

        # Drop vertices no triangle references (gmsh emits nodes for 0D and 1D
        # entities too) and renumber, so the indices the browser receives are
        # dense and every vertex it downloads is one it draws.
        used = np.zeros(coords.shape[0], bool)
        used[tri.ravel()] = True
        ren = np.full(coords.shape[0], -1, np.int64)
        ren[used] = np.arange(int(used.sum()))
        verts = coords[used]
        tri = ren[tri]

        mn = verts.min(axis=0).tolist()
        mx = verts.max(axis=0).tolist()
        print(json.dumps({
            # Indexed geometry. app/step3d.faces() expands this back to the
            # (M, 3, 3) array the silhouette and plate-frame code works in.
            "verts": np.round(verts, 3).tolist(),
            "tris": tri.tolist(),
            "indexed": True,
            "bbox": {"min": mn, "max": mx,
                     "spans": [mx[i] - mn[i] for i in range(3)]},
            "n_tris": int(tri.shape[0]),
            "n_verts": int(verts.shape[0]),
            "size_factor": sf,
        }))
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
