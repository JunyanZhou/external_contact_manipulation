"""Build a fixed-base, spring-return vertical pedal for the external-support task."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher


def build(path: Path, stiffness: float, damping: float):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    root = UsdGeom.Xform.Define(stage, "/SpringPedal")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    base = UsdGeom.Xform.Define(stage, "/SpringPedal/Base")
    plate = UsdGeom.Xform.Define(stage, "/SpringPedal/Plate")
    for body, mass in ((base, 25.), (plate, 1.5)):
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(mass)
    def material(name, color, friction):
        visual = UsdShade.Material.Define(stage, f"/SpringPedal/{name}Visual")
        shader = UsdShade.Shader.Define(stage, f"/SpringPedal/{name}Visual/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        visual.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        physics = UsdShade.Material.Define(stage, f"/SpringPedal/{name}Physics")
        api = UsdPhysics.MaterialAPI.Apply(physics.GetPrim())
        api.CreateStaticFrictionAttr(friction)
        api.CreateDynamicFrictionAttr(friction)
        api.CreateRestitutionAttr(0.)
        return visual, physics
    base_mat = material("Base", (.20, .25, .29), .9)
    plate_mat = material("Plate", (.90, .57, .12), 1.4)
    def cube(parent, name, center, size, mats):
        prim = UsdGeom.Cube.Define(stage, str(parent.GetPath()) + "/" + name)
        prim.CreateSizeAttr(1.)
        prim.AddTranslateOp().Set(Gf.Vec3d(*center))
        prim.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mats[0])
        prim.GetPrim().CreateRelationship("material:binding:physics").SetTargets([mats[1].GetPath()])
    # Robot stands near x=0, y=0; pressing foot is right (negative y).
    cube(base, "Frame", (.27, -.12, .010), (.31, .26, .020), base_mat)
    cube(plate, "Pad", (.27, -.12, .0625), (.27, .22, .035), plate_mat)
    fixed = UsdPhysics.FixedJoint.Define(stage, "/SpringPedal/BaseToWorld")
    fixed.CreateBody1Rel().SetTargets([base.GetPath()])
    slide = UsdPhysics.PrismaticJoint.Define(stage, "/SpringPedal/Slide")
    slide.CreateBody0Rel().SetTargets([base.GetPath()])
    slide.CreateBody1Rel().SetTargets([plate.GetPath()])
    slide.CreateAxisAttr("Z")
    slide.CreateLowerLimitAttr(-.050)
    slide.CreateUpperLimitAttr(0.)
    drive = UsdPhysics.DriveAPI.Apply(slide.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(stiffness)
    drive.CreateDampingAttr(damping)
    drive.CreateTargetPositionAttr(0.)
    drive.CreateMaxForceAttr(250.)
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stiffness", type=float, default=1150.)
    parser.add_argument("--damping", type=float, default=25.)
    args = parser.parse_args()
    app = AppLauncher({"headless": True}).app
    try:
        build(args.output.resolve(), args.stiffness, args.damping)
    finally:
        app.close()
