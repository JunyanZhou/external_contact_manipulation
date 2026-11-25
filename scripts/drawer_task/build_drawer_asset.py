"""Build a floor-fixed cabinet with a passive, resisted sliding drawer."""
from pathlib import Path
import argparse
from isaaclab.app import AppLauncher


def build(path: Path):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    root = UsdGeom.Xform.Define(stage, "/DrawerCabinet")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    base = UsdGeom.Xform.Define(stage, "/DrawerCabinet/Base")
    drawer = UsdGeom.Xform.Define(stage, "/DrawerCabinet/Drawer")
    for body, mass in ((base, 50.), (drawer, 2.)):
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(mass)

    def material(name, color, friction):
        visual = UsdShade.Material.Define(stage, f"/DrawerCabinet/{name}Visual")
        shader = UsdShade.Shader.Define(stage, f"/DrawerCabinet/{name}Visual/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        visual.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        physics = UsdShade.Material.Define(stage, f"/DrawerCabinet/{name}Physics")
        api = UsdPhysics.MaterialAPI.Apply(physics.GetPrim())
        api.CreateStaticFrictionAttr(friction)
        api.CreateDynamicFrictionAttr(friction)
        api.CreateRestitutionAttr(0.)
        return visual, physics

    cabinet_mat = material("Cabinet", (.55,.45,.34), .8)
    support_mat = material("Support", (.28,.50,.68), 2.0)
    drawer_mat = material("Drawer", (.76,.56,.34), .8)
    def cube(parent, name, center, size, mats):
        obj = UsdGeom.Cube.Define(stage, str(parent.GetPath()) + "/" + name)
        obj.CreateSizeAttr(1.)
        obj.AddTranslateOp().Set(Gf.Vec3d(*center))
        obj.AddScaleOp().Set(Gf.Vec3f(*size))
        UsdPhysics.CollisionAPI.Apply(obj.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(obj.GetPrim()).Bind(mats[0])
        obj.GetPrim().CreateRelationship("material:binding:physics").SetTargets([mats[1].GetPath()])
    # Cabinet opens toward -X. Its broad, high-friction top extension supports
    # the left palm from below without an initial horizontal impact.
    cube(base,"SupportPanel",(.425,.185,.77),(.25,.24,.06),support_mat)
    cube(base,"SupportPillar",(.50,.25,.38),(.08,.08,.76),cabinet_mat)
    cube(base,"RightSide",(.63,-.29,.78),(.32,.035,.70),cabinet_mat)
    cube(base,"Top",(.63,-.05,1.13),(.32,.48,.035),cabinet_mat)
    cube(base,"Bottom",(.63,-.05,.43),(.32,.48,.035),cabinet_mat)
    cube(base,"Back",(.78,-.05,.78),(.035,.48,.70),cabinet_mat)
    # Drawer body and raised pull bar. Reaching behind this bar allows a palm
    # to pull by ordinary contact; no grasp or scripted drawer translation.
    cube(drawer,"FrontLeftRail",(.48,.055,.75),(.035,.025,.18),drawer_mat)
    cube(drawer,"FrontRightRail",(.48,-.195,.75),(.035,.025,.18),drawer_mat)
    # Leave a hand-width gap behind the pull bar so the palm can hook it.
    cube(drawer,"Tray",(.67,-.07,.65),(.23,.22,.025),drawer_mat)
    cube(drawer,"PullBar",(.39,-.07,.72),(.025,.13,.045),drawer_mat)
    cube(drawer,"HandleLeftStrut",(.47,-.13,.72),(.16,.012,.018),drawer_mat)
    cube(drawer,"HandleRightStrut",(.47,-.01,.72),(.16,.012,.018),drawer_mat)
    fixed = UsdPhysics.FixedJoint.Define(stage,"/DrawerCabinet/BaseToWorld")
    fixed.CreateBody1Rel().SetTargets([base.GetPath()])
    slide = UsdPhysics.PrismaticJoint.Define(stage,"/DrawerCabinet/Slide")
    slide.CreateBody0Rel().SetTargets([base.GetPath()])
    slide.CreateBody1Rel().SetTargets([drawer.GetPath()])
    slide.CreateAxisAttr("X")
    slide.CreateLowerLimitAttr(-.20)
    slide.CreateUpperLimitAttr(0.)
    drive = UsdPhysics.DriveAPI.Apply(slide.GetPrim(),"linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.)
    drive.CreateDampingAttr(18.)
    drive.CreateTargetVelocityAttr(0.)
    drive.CreateMaxForceAttr(100.)
    stage.GetRootLayer().Save()

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    app=AppLauncher({"headless": True}).app
    try:
        build(p.parse_args().output.resolve())
    finally:
        app.close()
