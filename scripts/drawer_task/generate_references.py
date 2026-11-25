"""Generate five-XYZ drawer-pull references with Pinocchio whole-body IK."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin
import crocoddyl as croc
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/references"))
from generate_wall_support import load_usd, add_frame
from generate_foot_reach_dataset import body_names_from_loader, quintic_segment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()
    model, joint_names, meshes, _ = load_usd(ROOT / "active_adaptation/assets/G1/g1_flat_fullcol.usd")
    initial = np.load(ROOT / "artifacts/table_push/static_poses_hand08.npz")["q_pinocchio"][0].copy()
    sole_z = min(v[:, 2].min() for n, v, _ in meshes if n == "left_ankle_roll_link")
    ids = {n: model.getFrameId(n) for n in ("left_hand_mimic", "right_hand_mimic", "torso_link")}
    ids["left_sole"] = add_frame(model, "drawer_left_sole", "left_ankle_roll_link", np.array([0.,0.,sole_z]))
    ids["right_sole"] = add_frame(model, "drawer_right_sole", "right_ankle_roll_link", np.array([0.,0.,sole_z]))
    data = model.createData()
    state = croc.StateMultibody(model)
    collector = croc.DataCollectorMultibody(data)
    pin.framesForwardKinematics(model, data, initial)
    anchors = {name: data.oMf[frame].translation.copy() for name, frame in ids.items()}
    torso_R = data.oMf[ids["torso_link"]].rotation.copy()
    lower = np.r_[[-.04,-.08,.72], model.lowerPositionLimit[7:]+.015]
    upper = np.r_[[.12,.08,.86], model.upperPositionLimit[7:]-.015]
    body_names = body_names_from_loader()
    body_ids = [model.getFrameId(n) for n in body_names]
    root_R = Rotation.from_quat(initial[3:7]).as_matrix()

    def solve(guess, left, right, root_x):
        targets=(("left_sole",anchors["left_sole"]),
                 ("right_sole",anchors["right_sole"]),
                 ("left_hand_mimic",left),("right_hand_mimic",right))
        residuals=[croc.ResidualModelFrameTranslation(state,ids[n],t,0)
                   for n,t in targets]
        residual_data=[r.createData(collector) for r in residuals]
        def error(x):
            q=initial.copy();q[:3]=x[:3];q[7:]=x[3:]
            pin.framesForwardKinematics(model,data,q)
            xstate=np.r_[q,np.zeros(model.nv)]
            for residual,rd in zip(residuals,residual_data):
                residual.calc(rd,xstate,np.zeros(0))
            return np.r_[
                *(weight*rd.r for weight,rd in zip((35,35,30,30),residual_data)),
                3*pin.log3(torso_R.T @ data.oMf[ids["torso_link"]].rotation),
                5*(x[0]-root_x),2*(x[1:3]-initial[1:3]),
                .09*(x[3:]-initial[7:])]
        x=np.clip(np.r_[guess[:3],guess[7:]],lower+1e-5,upper-1e-5)
        result=least_squares(error,x,bounds=(lower,upper),max_nfev=200,
                             ftol=1e-6,xtol=1e-6,gtol=1e-6)
        q=initial.copy();q[:3]=result.x[:3];q[7:]=result.x[3:]
        pin.framesForwardKinematics(model,data,q)
        target_err=max(np.linalg.norm(data.oMf[ids[n]].translation-t) for n,t in
                       (("left_hand_mimic",left),("right_hand_mimic",right),
                        ("left_sole",anchors["left_sole"]),("right_sole",anchors["right_sole"])))
        return q,float(target_err)

    clips=[];rejected=[]
    cases=[(f"train_{i:02d}","train",depth,z,side)
           for i,(depth,z,side) in enumerate((d,h,s) for d in (.08,.10,.12) for h in (.70,.72) for s in (-.09,-.07))]
    cases += [(f"val_{i:02d}","val",d,h,s) for i,(d,h,s) in enumerate(
        ((.09,.71,-.08),(.11,.70,-.08),(.10,.72,-.075),(.09,.70,-.085)))]
    for name,split,depth,z,side in cases:
        left=np.array([.38,.155,.80])
        start_right=anchors["right_hand_mimic"].copy()
        points=[(anchors["left_hand_mimic"],start_right,initial[0]),
                (np.array([.33,.155,.84]),start_right,initial[0]),
                (left,start_right,initial[0]),
                (left,np.array([.49,side,z+.25]),initial[0]),
                (left,np.array([.50,side,z+.15]),initial[0]),
                (left,np.array([.50,side,z]),initial[0]),
                (left,np.array([.50-depth,side,z]),initial[0]-.015)]
        poses=[initial];errors=[]
        for l,r,root_x in points[1:]:
            q,err=solve(poses[-1],l,r,root_x)
            poses.append(q);errors.append(err)
        if max(errors)>.027:
            rejected.append({"name":name,"errors_m":errors})
            print("REJECT",name,errors,flush=True)
            continue
        chunks=[]
        for a,b,seconds in zip(poses[:-1],poses[1:],(.5,.7,.8,.6,.9,1.2)):
            joints,vel,acc=quintic_segment(a[7:],b[7:],seconds,args.fps)
            roots,rv,ra=quintic_segment(a[:3],b[:3],seconds,args.fps)
            qs=np.tile(initial,(len(joints)-1,1));qs[:,:3]=roots[:-1];qs[:,7:]=joints[:-1]
            vs=np.zeros((len(qs),model.nv));aa=np.zeros_like(vs)
            vs[:,:3]=rv[:-1];vs[:,6:]=vel[:-1];aa[:,:3]=ra[:-1];aa[:,6:]=acc[:-1]
            chunks.append((qs,vs,aa))
            chunks.append((np.tile(b,(20,1)),np.zeros((20,model.nv)),np.zeros((20,model.nv))))
        qs,vs,aa=[np.concatenate([c[k] for c in chunks]) for k in range(3)]
        world_pos=[];world_rot=[]
        for q in qs:
            pin.framesForwardKinematics(model,data,q)
            world_pos.append([data.oMf[f].translation.copy() for f in body_ids])
            world_rot.append([data.oMf[f].rotation.copy() for f in body_ids])
        world_pos=np.asarray(world_pos);world_rot=np.asarray(world_rot)
        local_pos=(world_pos-qs[:,None,:3])@root_R
        local_rot=np.einsum("ij,tbjk->tbik",root_R.T,world_rot)
        quat=Rotation.from_matrix(local_rot.reshape(-1,3,3)).as_quat().reshape(len(qs),-1,4)
        dest=args.output/"motions"/split/f"{name}.npz";dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,fps=np.array(args.fps),joint_names=np.array(joint_names),
            body_names=np.array(body_names),root_pos=qs[:,:3].astype("f4"),root_rot=qs[:,3:7].astype("f4"),
            dof_pos=qs[:,7:].astype("f4"),local_body_pos=local_pos.astype("f4"),
            local_body_rot=quat.astype("f4"),joint_vel=vs[:,6:].astype("f4"),joint_acc=aa[:,6:].astype("f4"))
        clips.append({"name":name,"split":split,"file":str(dest.relative_to(args.output)),
                      "frames":len(qs),"pull_depth_m":depth,"hand_height_m":z,"hand_y_m":side,
                      "ik_errors_m":errors})
        print("ACCEPT",name,len(qs),errors,flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"manifest.json").write_text(json.dumps({"fps":args.fps,"joint_names":joint_names,
        "body_names":body_names,"sole_offset":[0,0,sole_z],"clips":clips,"rejected":rejected},indent=2)+"\n")
    if len(clips)<8:raise RuntimeError("Too few feasible IK references")

if __name__=="__main__":main()
