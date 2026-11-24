"""Build InsertAnything's USB-A assets from cable-assembly's USB colliders.

    ./scripts/run.sh python scripts/assets/build_usb_assets.py [--prong_stiffness 5000] [--prong_damping 10]

Inputs (``assets/usb/cad``, from cable-assembly's sim/scripts/build_usb.py; ``USB_CAD_DIR`` overrides):
    usb_plug.usd         the plug: a solid shell envelope with the tongue cavity (SDF), overmold, cable
    usb_socket.usd       the receptacle: the pocket cut clean at the USB-A shell's inside, i.e. with the
                         shell's spring prongs taken out, and the adapter CAD for drawing
and ``assets/usb/usb_prongs.json`` (scripts/assets/extract_usb_prongs.py): the seven prongs as convex pieces.

Outputs (``assets/usb``), laid out as the other InsertAnything assets (a root rigid body; the hole fixed
to the world by a FixedJoint articulation root):
    usb_plug.usd                 held asset. Origin = the socket origin when seated; +z from the tip to the cable.
    usb_socket_noprongs.usd      fixed asset, prongs removed (cable-assembly's receptacle collider as it is).
    usb_socket_compliant.usd     fixed asset, the same receptacle plus the prongs as compliant-contact colliders:
                                 each pushes back k * depth, like a spring tab deflected by the plug's shell.
"""

import argparse
import json
import os

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "source/InsertAnything/InsertAnything/tasks/direct/insertanything/assets/usb")
SRC = os.environ.get("USB_CAD_DIR", os.path.join(OUT, "cad"))
"""cable-assembly's usb/usb_{plug,socket}.usd (its sim/scripts/build_usb.py), copied into assets/usb/cad."""


def mesh_data(stage, path):
    m = UsdGeom.Mesh(stage.GetPrimAtPath(path))
    xf = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(m.GetPrim())).T
    pts = np.array(m.GetPointsAttr().Get(), float)
    pts = (np.c_[pts, np.ones(len(pts))] @ xf.T)[:, :3]
    return pts, list(m.GetFaceVertexCountsAttr().Get()), list(m.GetFaceVertexIndicesAttr().Get())


def add_mesh(stage, path, pts, counts, idx, color=None, collider=None, sdf_resolution=256):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in pts])
    m.CreateFaceVertexCountsAttr(counts)
    m.CreateFaceVertexIndicesAttr(idx)
    m.CreateSubdivisionSchemeAttr("none")
    if color is not None:
        m.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collider:
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr(collider)
        if collider == "sdf":  # PhysX needs its SDF schema, or it falls back to a convex hull
            m.GetPrim().AddAppliedSchema("PhysxSDFMeshCollisionAPI")
            m.GetPrim().CreateAttribute("physxSDFMeshCollision:sdfResolution", Sdf.ValueTypeNames.UInt).Set(sdf_resolution)
        elif collider == "convexHull":
            m.GetPrim().AddAppliedSchema("PhysxConvexHullCollisionAPI")
    return m


def hull_mesh(points):
    from scipy.spatial import ConvexHull

    p = np.asarray(points)
    h = ConvexHull(p)
    tris = h.simplices.copy()
    centre = p.mean(0)
    for t in tris:  # outward winding
        n = np.cross(p[t[1]] - p[t[0]], p[t[2]] - p[t[0]])
        if np.dot(n, p[t[0]] - centre) < 0:
            t[1], t[2] = t[2], t[1]
    return p, [3] * len(tris), tris.reshape(-1).tolist()


def new_stage(path):
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Root")
    stage.SetDefaultPrim(root.GetPrim())
    return stage


def physics_material(stage, path, friction, stiffness=None, damping=None):
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(friction)
    api.CreateDynamicFrictionAttr(friction)
    api.CreateRestitutionAttr(0.0)
    if stiffness is not None:
        prim = mat.GetPrim()
        prim.AddAppliedSchema("PhysxMaterialAPI")
        prim.CreateAttribute("physxMaterial:compliantContactStiffness", Sdf.ValueTypeNames.Float).Set(float(stiffness))
        prim.CreateAttribute("physxMaterial:compliantContactDamping", Sdf.ValueTypeNames.Float).Set(float(damping))
        prim.CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("average")
    return mat


def bind_physics(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def build_plug():
    src = Usd.Stage.Open(os.path.join(SRC, "usb_plug.usd"))
    stage = new_stage(os.path.join(OUT, "usb_plug.usd"))
    body = UsdGeom.Xform.Define(stage, "/Root/usb_plug")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim())
    mat = physics_material(stage, "/Root/usb_plug/physics_material", 0.3)
    colors = {"plug": (0.75, 0.75, 0.78), "overmold": (0.1, 0.1, 0.1), "cable": (0.12, 0.12, 0.12), "contacts": (0.85, 0.7, 0.3)}
    for name in ("plug", "overmold", "contacts", "cable"):
        pts, c, i = mesh_data(src, f"/UsbPlug/body/{name}")
        # The shell is an SDF; the overmold, where the gripper holds the plug, a convex hull.
        collider = {"plug": "sdf", "overmold": "convexHull"}.get(name)
        m = add_mesh(stage, f"/Root/usb_plug/{name}", pts, c, i, colors[name], collider)
        if collider:
            bind_physics(m.GetPrim(), mat)
    stage.GetRootLayer().Save()


def build_socket(variant, prongs, stiffness, damping):
    src = Usd.Stage.Open(os.path.join(SRC, "usb_socket.usd"))
    stage = new_stage(os.path.join(OUT, f"usb_socket_{variant}.usd"))
    holder = UsdGeom.Xform.Define(stage, "/Root/usb_socket")
    body = UsdGeom.Xform.Define(stage, "/Root/usb_socket/usb_socket")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.MassAPI.Apply(body.GetPrim())
    mat = physics_material(stage, "/Root/usb_socket/usb_socket/physics_material", 0.3)
    pts, c, i = mesh_data(src, "/UsbSocket/socket/receptacle")
    m = add_mesh(stage, "/Root/usb_socket/usb_socket/receptacle", pts, c, i, (0.7, 0.7, 0.72), "sdf", 512)
    bind_physics(m.GetPrim(), mat)
    pts, c, i = mesh_data(src, "/UsbSocket/socket/adapter")
    add_mesh(stage, "/Root/usb_socket/usb_socket/adapter", pts, c, i, (0.2, 0.2, 0.22))
    if variant == "compliant":
        pmat = physics_material(stage, "/Root/usb_socket/usb_socket/prong_material", 0.3, stiffness, damping)
        UsdGeom.Scope.Define(stage, "/Root/usb_socket/usb_socket/prongs")
        for name, prong in prongs["prongs"].items():
            for k, piece in enumerate(prong["pieces"]):
                p, cc, ii = hull_mesh(piece)
                pm = add_mesh(stage, f"/Root/usb_socket/usb_socket/prongs/{name}_{k}", p, cc, ii, (0.9, 0.5, 0.2), "convexHull")
                bind_physics(pm.GetPrim(), pmat)
    joint = UsdPhysics.FixedJoint.Define(stage, "/Root/usb_socket/FixedJoint")
    joint.CreateBody1Rel().SetTargets([body.GetPath()])
    UsdPhysics.ArticulationRootAPI.Apply(joint.GetPrim())
    stage.GetRootLayer().Save()
    return holder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prong_stiffness", type=float, default=5000.0, help="N/m per prong contact.")
    ap.add_argument("--prong_damping", type=float, default=10.0, help="N s/m per prong contact.")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "usb_prongs.json")) as f:
        prongs = json.load(f)
    build_plug()
    build_socket("noprongs", prongs, None, None)
    build_socket("compliant", prongs, args.prong_stiffness, args.prong_damping)
    print("wrote", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
