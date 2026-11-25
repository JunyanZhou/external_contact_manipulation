"""Create a drawer-task G1 asset with a wider, high-friction left palm."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


def build(source: Path, output: Path, contact_width: float, friction: float,
          disable_right_hand_collisions: bool = False):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.Open(str(source.resolve()))
    if stage is None:
        raise FileNotFoundError(source)
    hand = "/g1_29dof_rev_1_0/left_hand_mimic"
    cube = stage.GetPrimAtPath(f"{hand}/Cube")
    sphere = stage.GetPrimAtPath(f"{hand}/Sphere")
    if not cube or not sphere:
        raise RuntimeError("Expected G1 left-hand collision cube and sphere")
    size = UsdGeom.Cube(cube).GetSizeAttr().Get()
    UsdGeom.Xformable(cube).GetOrderedXformOps()[-1].Set(
        Gf.Vec3f(1., contact_width / size, contact_width / size))
    UsdPhysics.CollisionAPI(sphere).GetCollisionEnabledAttr().Set(False)
    if disable_right_hand_collisions:
        for shape in ("Cube", "Sphere"):
            right = stage.GetPrimAtPath(
                f"/g1_29dof_rev_1_0/right_hand_mimic/{shape}")
            if not right:
                raise RuntimeError(f"Missing right-hand {shape} collider")
            UsdPhysics.CollisionAPI(right).GetCollisionEnabledAttr().Set(False)

    material = UsdShade.Material.Define(
        stage, "/g1_29dof_rev_1_0/drawer_left_palm_physics")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(friction)
    physics.CreateDynamicFrictionAttr(friction)
    physics.CreateRestitutionAttr(0.)
    material.GetPrim().CreateAttribute(
        "physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("max")
    cube.CreateRelationship("material:binding:physics").SetTargets(
        [material.GetPath()])
    output.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(output.resolve()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--contact-width", type=float, default=.12)
    p.add_argument("--friction", type=float, default=12.)
    p.add_argument("--disable-right-hand-collisions", action="store_true")
    args = p.parse_args()
    app = AppLauncher({"headless": True}).app
    try:
        build(args.source, args.output, args.contact_width, args.friction,
              args.disable_right_hand_collisions)
    finally:
        app.close()
