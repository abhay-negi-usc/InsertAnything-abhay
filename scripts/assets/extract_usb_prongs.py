"""Extract the USB-A receptacle's spring prongs (the shell's detent tabs) from McMaster's STEP as convex pieces.

Run in an environment with gmsh + numpy + scipy (cable-assembly's `./run.sh --extra cad`):

    python scripts/assets/extract_usb_prongs.py <4574N11_panel_mount_usb_adapter.step> <out.json>

The STEP is in the socket frame [mm], the frame the cable-assembly usb_socket.usd is authored in. Each prong
is a thin bent tab; as a collider it becomes the solid between the tab and its wall. A tab is long and
curved, so it is split along the insertion axis (z) into overlapping slabs, and each slab is the convex hull
of the tab's surface points there plus their projections onto the wall. Only the part of a tab that comes
within MARGIN of the plug's path is kept: the rest never touches the plug, and convex slabs over it would
only add false interference.
"""

import json
import sys

import gmsh
import numpy as np
from scipy.spatial import ConvexHull

# Pocket walls (the shell's inside, USB-A 12.50 x 5.12 mm) and the plug shell's outside when seated, socket frame [mm].
WALL = {"top": 3.031, "bottom": -2.089, "left": -6.25, "right": 6.25}
PLUG = {"top": 2.706, "bottom": -1.794, "left": -6.00, "right": 6.00}
MARGIN = 0.10
"""Keep tab points that come within this of the plug shell's outside."""
SLAB, OVERLAP = 0.40, 0.10

# Where each prong is, from the STEP's faces (x, y, z ranges, mm) and which wall it hangs from.
PRONGS = {
    "top_left": ("top", (-4.2, -1.8), (2.0, 3.03), (-2.3, 3.75)),
    "top_right": ("top", (1.8, 4.2), (2.0, 3.03), (-2.3, 3.75)),
    "top_centre": ("top", (-0.45, 0.45), (2.4, 3.03), (-0.3, 2.1)),
    "bottom_left": ("bottom", (-4.0, -1.8), (-2.088, -1.1), (-2.3, 3.75)),
    "bottom_right": ("bottom", (1.8, 4.0), (-2.088, -1.1), (-2.3, 3.75)),
    "side_left": ("left", (-6.249, -5.8), (0.0, 0.95), (2.35, 4.3)),
    "side_right": ("right", (5.8, 6.249), (0.0, 0.95), (2.35, 4.3)),
}
TONGUE = ((-5.55, 5.55), (0.471, 2.311), (-4.84, 5.31))


def surface_points(step, size=0.05):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(step)
    gmsh.model.occ.synchronize()
    f = gmsh.model.mesh.field
    box = f.add("Box")
    for k, v in dict(VIn=size, VOut=2.0, XMin=-6.5, XMax=6.5, YMin=-2.2, YMax=3.2, ZMin=-5.0, ZMax=5.6).items():
        f.setNumber(box, k, v)
    f.setAsBackgroundMesh(box)
    for k in ("Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature"):
        gmsh.option.setNumber(k, 0)
    gmsh.model.mesh.generate(2)
    _, coords, _ = gmsh.model.mesh.getNodes()
    gmsh.finalize()
    return coords.reshape(-1, 3)


def wall_axis(wall):
    return 1 if wall in ("top", "bottom") else 0


def protrusion(p, wall):
    a = wall_axis(wall)
    return np.abs(p[:, a] - WALL[wall])


def main(step, out):
    pts = surface_points(step)
    in_tongue = np.all([(pts[:, i] > lo - 1e-3) & (pts[:, i] < hi + 1e-3) for i, (lo, hi) in enumerate(TONGUE)], axis=0)
    pts = pts[~in_tongue]
    result = {"frame": "socket, metres", "margin_mm": MARGIN, "prongs": {}}
    for name, (wall, xr, yr, zr) in PRONGS.items():
        a = wall_axis(wall)
        sel = np.all([(pts[:, i] >= lo) & (pts[:, i] <= hi) for i, (lo, hi) in enumerate((xr, yr, zr))], axis=0)
        p = pts[sel]
        clearance = abs(WALL[wall] - PLUG[wall])
        p = p[protrusion(p, wall) >= clearance - MARGIN]
        if len(p) < 4:
            print(f"{name}: no points reach the plug path", file=sys.stderr)
            continue
        z0, z1 = p[:, 2].min(), p[:, 2].max()
        pieces = []
        z = z0
        while z < z1:
            s = p[(p[:, 2] >= z - 1e-6) & (p[:, 2] <= z + SLAB + 1e-6)]
            z += SLAB - OVERLAP
            if len(s) < 3:
                continue
            base = s.copy()
            base[:, a] = WALL[wall] + (0.3 if WALL[wall] > 0 else -0.3)  # bury the base 0.3 mm into the wall
            cloud = np.vstack((s, base))
            try:
                hull = ConvexHull(cloud)
            except Exception:
                continue
            v = cloud[hull.vertices]
            pieces.append((v / 1000.0).round(7).tolist())
        reach = protrusion(p, wall).max()
        result["prongs"][name] = {
            "wall": wall,
            "max_protrusion_mm": round(float(reach), 3),
            "interference_mm": round(float(reach - clearance), 3),
            "z_range_mm": [round(float(z0), 3), round(float(z1), 3)],
            "pieces": pieces,
        }
        print(f"{name:13s} wall={wall:6s} reach {reach:.3f} mm, into plug path {reach - clearance:+.3f} mm, "
              f"z {z0:.2f}..{z1:.2f}, {len(pieces)} convex pieces")
    with open(out, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
