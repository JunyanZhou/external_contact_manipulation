"""Build a conventional two-drawer cabinet with a resisted lower drawer."""

from pathlib import Path
import argparse

from isaaclab.app import AppLauncher


def build(path: Path):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/TwoDrawerCabinet")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    base = UsdGeom.Xform.Define(stage, "/TwoDrawerCabinet/Base")
    lower = UsdGeom.Xform.Define(stage, "/TwoDrawerCabinet/LowerDrawer")
    upper = UsdGeom.Xform.Define(stage, "/TwoDrawerCabinet/UpperDrawer")
    for body, mass in ((base, 55.0), (lower, 5.0), (upper, 4.0)):
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(mass)

    def material(name, rgb, friction):
        visual = UsdShade.Material.Define(stage, f"/TwoDrawerCabinet/{name}Visual")
        shader = UsdShade.Shader.Define(stage, f"/TwoDrawerCabinet/{name}Visual/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        visual.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        physics = UsdShade.Material.Define(stage, f"/TwoDrawerCabinet/{name}Physics")
        p = UsdPhysics.MaterialAPI.Apply(physics.GetPrim())
        p.CreateStaticFrictionAttr(friction)
        p.CreateDynamicFrictionAttr(friction)
        p.CreateRestitutionAttr(0.0)
        return visual, physics

    case_mat = material("Case", (.43, .32, .24), 0.85)
    upper_mat = material("UpperFrontSupport", (.68, .49, .33), 2.0)
    lower_mat = material("LowerDrawer", (.67, .47, .31), 0.9)
    handle_mat = material("Handle", (.10, .10, .12), 1.25)

    def cube(parent, name, center, size, mats):
        prim = UsdGeom.Cube.Define(stage, f"{parent.GetPath()}/{name}")
        prim.CreateSizeAttr(1.0)
        prim.AddTranslateOp().Set(Gf.Vec3d(*center))
        prim.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mats[0])
        prim.GetPrim().CreateRelationship("material:binding:physics").SetTargets([mats[1].GetPath()])

    # The front of both drawers faces -X, toward the robot. The upper panel is
    # a broad, vertical high-friction surface for the left palm; there is no
    # side shelf or horizontal support extension.
    cube(base, "LeftSide", (.60, .315, .60), (.38, .03, .96), case_mat)
    cube(base, "RightSide", (.60, -.315, .60), (.38, .03, .96), case_mat)
    cube(base, "Top", (.60, 0, 1.08), (.38, .66, .035), case_mat)
    cube(base, "Bottom", (.60, 0, .12), (.38, .66, .035), case_mat)
    cube(base, "MiddleDivider", (.60, 0, .60), (.38, .63, .026), case_mat)
    cube(base, "Back", (.78, 0, .60), (.035, .63, .94), case_mat)
    cube(upper, "UpperDrawerFront", (.408, 0, .835), (.045, .58, .40), upper_mat)
    cube(upper, "UpperTrayBottom", (.60, 0, .655), (.33, .51, .018), upper_mat)
    cube(upper, "UpperHandleBar", (.327, 0, .84), (.027, .19, .026), handle_mat)
    for side, y in (("Left", -.082), ("Right", .082)):
        cube(upper, f"UpperHandle{side}Post", (.365, y, .84), (.075, .018, .018), handle_mat)

    cube(lower, "LowerDrawerFront", (.405, 0, .355), (.05, .58, .40), lower_mat)
    cube(lower, "LowerTrayBottom", (.60, 0, .19), (.33, .51, .018), lower_mat)
    cube(lower, "LowerTrayLeft", (.60, .245, .30), (.33, .018, .20), lower_mat)
    cube(lower, "LowerTrayRight", (.60, -.245, .30), (.33, .018, .20), lower_mat)
    cube(lower, "LowerHandleBar", (.245, -.055, .45), (.027, .20, .032), handle_mat)
    for side, y in (("Left", -.14), ("Right", .03)):
        cube(lower, f"LowerHandle{side}Post", (.315, y, .45), (.135, .018, .019), handle_mat)

    fixed = UsdPhysics.FixedJoint.Define(stage, "/TwoDrawerCabinet/BaseToWorld")
    fixed.CreateBody1Rel().SetTargets([base.GetPath()])
    slide = UsdPhysics.PrismaticJoint.Define(stage, "/TwoDrawerCabinet/Slide")
    slide.CreateBody0Rel().SetTargets([base.GetPath()])
    slide.CreateBody1Rel().SetTargets([lower.GetPath()])
    slide.CreateAxisAttr("X")
    slide.CreateLowerLimitAttr(-.24)
    slide.CreateUpperLimitAttr(0.0)
    drive = UsdPhysics.DriveAPI.Apply(slide.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(55.0)
    drive.CreateDampingAttr(35.0)
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)
    drive.CreateMaxForceAttr(150.0)
    upper_slide = UsdPhysics.PrismaticJoint.Define(stage, "/TwoDrawerCabinet/UpperSlide")
    upper_slide.CreateBody0Rel().SetTargets([base.GetPath()])
    upper_slide.CreateBody1Rel().SetTargets([upper.GetPath()])
    upper_slide.CreateAxisAttr("X")
    upper_slide.CreateLowerLimitAttr(-.18)
    upper_slide.CreateUpperLimitAttr(0.0)
    upper_drive = UsdPhysics.DriveAPI.Apply(upper_slide.GetPrim(), "linear")
    upper_drive.CreateTypeAttr("force")
    upper_drive.CreateStiffnessAttr(140.0)
    upper_drive.CreateDampingAttr(45.0)
    upper_drive.CreateTargetPositionAttr(0.0)
    upper_drive.CreateTargetVelocityAttr(0.0)
    upper_drive.CreateMaxForceAttr(200.0)
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    app = AppLauncher({"headless": True}).app
    try:
        build(parser.parse_args().output.resolve())
    finally:
        app.close()
