"""Pinocchio IK + Crocoddyl waypoint fit for a braced spring-pedal press."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares, linprog
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/references"))
from generate_wall_support import add_frame, load_usd  # noqa: E402
from generate_foot_reach_dataset import body_names_from_loader, quintic_segment  # noqa: E402
from generate_flat_foot_box_push import refine_waypoints_crocoddyl  # noqa: E402


def support_lp(model, q, ids, v=None, a=None, pedal_force=0.):
    """Floating-base balance with left sole, right palm and pedal foot contact."""
    d = model.createData()
    pin.computeJointJacobians(model, d, q)
    pin.updateFramePlacements(model, d)
    g = pin.computeGeneralizedGravity(model, d, q) if v is None else pin.rnea(model, d, q, v, a)
    foot = pin.getFrameJacobian(model, d, ids["left_sole"], pin.LOCAL_WORLD_ALIGNED)
    hand = pin.getFrameJacobian(model, d, ids["right_hand_mimic"], pin.LOCAL_WORLD_ALIGNED)[:3]
    pedal = pin.getFrameJacobian(model, d, ids["right_sole"], pin.LOCAL_WORLD_ALIGNED)[:3]
    jt = np.column_stack((foot.T, hand.T, pedal.T))
    rows = []
    def row(terms):
        r = np.zeros(12)
        for i, value in terms.items(): r[i] = value
        rows.append(r)
    for i in (0, 1):
        row({i: 1., 2: -.7}); row({i: -1., 2: -.7})
    for i, lim in ((3, .04), (5, .025)):
        row({i: 1., 2: -lim}); row({i: -1., 2: -lim})
    row({4: -1., 2: -.11}); row({4: 1., 2: -.045})
    for i in (6, 8):
        row({i: 1., 7: -.7}); row({i: -1., 7: -.7})
    for i in (9, 10):
        row({i: 1., 11: -.9}); row({i: -1., 11: -.9})
    effort = .85 * model.effortLimit[6:]
    A = np.vstack((rows, jt[6:], -jt[6:]))
    b = np.r_[np.zeros(len(rows)), g[6:] + effort, effort - g[6:]]
    objective = np.zeros(12); objective[7] = 1.
    result = linprog(objective, A_ub=A, b_ub=b,
        A_eq=jt[:6], b_eq=g[:6],
        bounds=[(None,None),(None,None),(20,None),
                (None,None),(None,None),(None,None),
                (None,None),(5,None),(None,None),
                (None,None),(None,None),(max(0., pedal_force),None)], method="highs")
    return {"feasible": bool(result.success), "reason": result.message,
            "hand_normal_N": float(result.x[7]) if result.success else None,
            "pedal_down_N": float(result.x[11]) if result.success else None,
            "joint_effort_max_fraction": float(np.max(np.abs((g-jt@result.x)[6:]) / model.effortLimit[6:])) if result.success else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--depth-cm", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--seconds", type=float, nargs="+", default=[1.2, 1.6])
    p.add_argument("--wall-hand-y", type=float, default=-.26)
    p.add_argument("--fps", type=int, default=50)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    model, names, meshes, _ = load_usd(ROOT / "active_adaptation/assets/G1/g1_flat_fullcol.usd")
    initial = np.load(ROOT / "artifacts/table_push/static_poses_hand08.npz")["q_pinocchio"][1].copy()
    sole_z = min(v[:,2].min() for n,v,_ in meshes if n == "left_ankle_roll_link")
    ids = {n: model.getFrameId(n) for n in ("left_hand_mimic", "right_hand_mimic", "torso_link")}
    ids["left_sole"] = add_frame(model,"wall_box_left_sole","left_ankle_roll_link",np.array([0,0,sole_z]))
    ids["right_sole"] = add_frame(model,"wall_box_right_sole","right_ankle_roll_link",np.array([0,0,sole_z]))
    d = model.createData()
    pin.framesForwardKinematics(model,d,initial)
    foot_left = d.oMf[ids["left_sole"]].translation.copy()
    foot_left[2] = 0.0
    root0 = initial[:3].copy()
    root0[1] = .085  # center over the left stance foot
    torso_R = d.oMf[ids["torso_link"]].rotation.copy()
    hand_target = np.array([.29,args.wall_hand_y,1.06])
    left_hand_target = np.array([.21,.15,.87])
    lower = np.r_[[-.15,-.08,.67],model.lowerPositionLimit[7:]+.02]
    upper = np.r_[[.18,.18,.89],model.upperPositionLimit[7:]-.02]
    qids = np.arange(7,model.nq)
    def solve(guess, target, anchor=None):
        def residual(x):
            q=initial.copy(); q[:3]=x[:3];q[7:]=x[3:]
            pin.framesForwardKinematics(model,d,q)
            terms=[22*(d.oMf[ids["right_sole"]].translation-target),
                   25*(d.oMf[ids["left_sole"]].translation-foot_left),
                   25*(d.oMf[ids["right_hand_mimic"]].translation-hand_target),
                   7*(d.oMf[ids["left_hand_mimic"]].translation-left_hand_target),
                   2*pin.log3(torso_R.T@d.oMf[ids["torso_link"]].rotation),
                   8.0*(x[:3]-root0),
                   .10*(x[3:]-(initial[7:] if anchor is None else anchor[7:]))]
            return np.concatenate(terms)
        x=np.r_[guess[:3],guess[7:]]
        result=least_squares(residual,np.clip(x,lower+1e-5,upper-1e-5),
             bounds=(lower,upper),max_nfev=250,ftol=1e-7,xtol=1e-7,gtol=1e-7)
        q=initial.copy();q[:3]=result.x[:3];q[7:]=result.x[3:]
        pin.framesForwardKinematics(model,d,q)
        errors={n:float(np.linalg.norm(d.oMf[ids[n]].translation-t)) for n,t in
            (("right_sole",target),("left_sole",foot_left),("right_hand_mimic",hand_target))}
        return q,errors
    start_target=np.array([.01,-.119,.045])
    start,start_error=solve(initial,start_target)
    print("START",start[:3],start_error,"LP",support_lp(model,start,ids),flush=True)
    root_R=Rotation.from_quat(start[3:7]).as_matrix()
    body_names=body_names_from_loader()
    body_ids=[model.getFrameId(n) for n in body_names]
    clips=[];rejected=[]
    for depth in args.depth_cm:
      for variant,seconds in enumerate(args.seconds):
        name=f"spring_pedal_d{depth}_v{variant}"
        split="val" if depth == max(args.depth_cm) else "train"
        # The plate starts at .08 m; the approach sole is above it. Once the
        # pad is touched, lower the sole depth/100 m below the initial top.
        targets=[start_target, np.array([.22,-.12,.095]),
                 np.array([.27,-.12,.125]),
                 np.array([.27,-.12,.08-depth/100])]
        waypoints=[start]; bad=None
        for target in targets[1:]:
            q,errors=solve(waypoints[-1],target,start)
            lp=support_lp(model,q,ids)
            if max(errors.values())>.03 or not lp["feasible"]:
                bad={"name":name,"target":target.tolist(),"errors":errors,"lp":lp};break
            waypoints.append(q)
        if bad:
            rejected.append(bad);print("REJECT",bad,flush=True);continue
        # Crocoddyl solves the robot's contact-pose waypoints. This is a pose
        # fit; contact-force feasibility is checked separately below.
        refined, croc_report = refine_waypoints_crocoddyl(model, ids, waypoints)
        if any((not r["converged"] or r["configuration_distance"] > .15) for r in croc_report[1:]):
            bad={"name":name,"reason":"crocoddyl_waypoint_fit","report":croc_report};rejected.append(bad);continue
        waypoints=refined
        path=waypoints+waypoints[-2::-1]
        chunks=[]
        for a,b in zip(path[:-1],path[1:]):
            joints,vel,acc=quintic_segment(a[7:],b[7:],seconds,args.fps)
            roots,rv,ra=quintic_segment(a[:3],b[:3],seconds,args.fps)
            qs=np.tile(start,(len(joints)-1,1));qs[:,:3]=roots[:-1];qs[:,7:]=joints[:-1]
            vs=np.zeros((len(qs),model.nv));aa=np.zeros_like(vs)
            vs[:,:3]=rv[:-1];vs[:,6:]=vel[:-1]
            aa[:,:3]=ra[:-1];aa[:,6:]=acc[:-1]
            chunks.append((qs,vs,aa))
            chunks.append((np.tile(b,(25,1)),np.zeros((25,model.nv)),np.zeros((25,model.nv))))
        qs,vs,aa=(np.concatenate([c[k] for c in chunks]) for k in range(3))
        wp=[];wr=[];foot_target=[]
        for i,(q,v,a) in enumerate(zip(qs,vs,aa)):
            pin.framesForwardKinematics(model,d,q)
            sole=d.oMf[ids["right_sole"]].translation.copy()
            # During pedal contact the right foot provides a vertical reaction.
            force = max(0., (.08-sole[2])*1150.) if sole[0]>.23 and sole[2]<.08 else 0.
            if sole[2]<-.004 or (i%10==0 and not support_lp(model,q,ids,v,a,force)["feasible"]):
                bad={"name":name,"frame":i,"reason":"ground_or_support_lp"};break
            wp.append([d.oMf[f].translation.copy() for f in body_ids])
            wr.append([d.oMf[f].rotation.copy() for f in body_ids])
            foot_target.append(Rotation.from_quat(q[3:7]).inv().apply(sole-q[:3]))
        if bad:
            rejected.append(bad);print("REJECT",bad,flush=True);continue
        wp=np.asarray(wp);wr=np.asarray(wr)
        local_pos=(wp-qs[:,None,:3])@root_R
        local_rot=np.einsum("ij,tbjk->tbik",root_R.T,wr)
        body_quat=Rotation.from_matrix(local_rot.reshape(-1,3,3)).as_quat().reshape(len(qs),-1,4)
        dest=args.output/"motions"/split/f"{name}.npz";dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,fps=np.array(args.fps),joint_names=np.array(names),
            body_names=np.array(body_names),root_pos=qs[:,:3].astype("f4"),
            root_rot=qs[:,3:7].astype("f4"),dof_pos=qs[:,7:].astype("f4"),
            local_body_pos=local_pos.astype("f4"),local_body_rot=body_quat.astype("f4"),
            target_foot_pos_b=np.asarray(foot_target,dtype="f4"),
            joint_vel=vs[:,6:].astype("f4"),joint_acc=aa[:,6:].astype("f4"))
        clips.append({"name":name,"split":split,"file":str(dest.relative_to(args.output)),
                      "frames":len(qs),"depth_cm":depth,"seconds":seconds,
                      "crocoddyl_waypoints":croc_report})
        print("ACCEPT",name,len(qs),flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"manifest.json").write_text(json.dumps({"fps":args.fps,"pose_file":"static_poses_hand08.npz",
        "joint_names":names,"body_names":body_names,"sole_offset":[0,0,sole_z],
        "right_hand_target":hand_target.tolist(),"left_sole_target":foot_left.tolist(),
        "mechanics":"Pinocchio IK, Crocoddyl waypoint pose fit, sampled Pinocchio inverse dynamics and contact LP; physical Isaac validation required",
        "clips":clips,"rejected":rejected},indent=2)+"\n")
    if not clips:raise RuntimeError("No feasible clips")


if __name__=="__main__":main()
