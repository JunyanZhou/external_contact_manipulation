"""Evaluate a direct table-push task policy on held-out reference clips."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MEMPATH", str(ROOT / "dataset"))
os.environ.setdefault("ACTIVE_ADAPTATION_DISABLE_TORCH_COMPILE", "1")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--matched-scene-checkpoint", type=Path,
                   help="Use another checkpoint's task scene and observations, retaining this policy")
    p.add_argument("--task-override-yaml", type=Path,
                   help="Merge a task YAML into the checkpoint task for a scene probe")
    p.add_argument("--robot-name-override", type=str,
                   help="Diagnostic: use a task-specific robot actuator configuration")
    p.add_argument("--controller", choices=("task", "zero", "reference_joint", "ceer_joint_reference", "ceer_native"),
                   default="task", help="Policy to run with the task's unchanged scene and motor")
    p.add_argument("--kinematic-preview", action="store_true",
                   help="Storyboard only: write reference robot pose and scripted drawer slide before rendering")
    p.add_argument("--reference-torque-audit", type=Path,
                   help="Optional inverse-dynamics torque feedforward for reference_joint control")
    p.add_argument("--reference-torque-gain", type=float, default=1.,
                   help="Scale the inverse-dynamics feedforward before converting to PD target offset")
    p.add_argument("--preview-drawer-depth", type=float, default=.14,
                   help="Prescribed lower-drawer slide in kinematic storyboard (meters)")
    p.add_argument("--drawer-handle-spring-grasp", action="store_true",
                   help="Diagnostic: attach right hand to lower handle with equal/opposite spring forces")
    p.add_argument("--drawer-grasp-stiffness", type=float, default=1200.,
                   help="Spring grasp stiffness in N/m")
    p.add_argument("--drawer-grasp-damping", type=float, default=35.,
                   help="Spring grasp damping in Ns/m")
    p.add_argument("--drawer-grasp-x-only", action="store_true",
                   help="Diagnostic: apply grasp reaction only along the drawer slide axis")
    p.add_argument("--ceer-checkpoint", type=Path,
                   default=ROOT / ".cache/ceer/checkpoint_final.pt")
    p.add_argument("--ceer-reference-npz", type=Path,
                   help="IK reference fitted to the same external foot target clip")
    p.add_argument("--split", choices=("train", "val"), default="val")
    p.add_argument("--dataset", help="Dataset prefix; defaults to checkpoint's training dataset")
    p.add_argument("--motion-id", type=int, default=0)
    p.add_argument("--external-target-npz", type=Path,
                   help="Per-step root-frame left-sole xyz; bypass motion command for the actor")
    p.add_argument("--explicit-goals", action="store_true",
                   help="Send the reference's five xyz targets through the external command API each step")
    p.add_argument("--external-goals-npz", type=Path,
                   help="Per-step world-frame [root,left hand,right hand,left sole,right sole] xyz")
    p.add_argument("--block", action="store_true", help="Add a light block in the lateral foot sweep")
    p.add_argument("--no-table-contact", action="store_true",
                   help="Move the table collider away as an external-support ablation")
    p.add_argument("--no-wall-contact", dest="no_table_contact", action="store_true",
                   help="Move the support wall away in the wall-box task")
    p.add_argument("--door-back-wall-shift-y", type=float, default=0.,
                   help="Diagnostic: move the two full-height right return wall pieces in y")
    p.add_argument("--door-asset-usd", type=Path,
                   help="Diagnostic: override the spring door USD loaded from a checkpoint")
    p.add_argument("--door-initial-angle-deg", type=float,
                   help="Diagnostic: change the spring door initial joint angle for an opening probe")
    p.add_argument("--door-object-reference", type=Path,
                   help="Show a simplified object transfer, triggered by actual hand proximity")
    p.add_argument("--door-object-open-angle-deg", type=float, default=32.,
                   help="Minimum measured opening before the right hand may collect the object")
    p.add_argument("--door-object-handoff-distance-m", type=float, default=.065,
                   help="Maximum actual hand-center gap before the object can pass between hands")
    p.add_argument("--table-width", type=float,
                   help="Override tabletop width along y for a reachable side edge")
    p.add_argument("--block-x", type=float, default=.25)
    p.add_argument("--block-y", type=float, default=.05)
    p.add_argument("--block-init-x", type=float,
                   help="Override an existing push_block center x for a matched geometry probe")
    p.add_argument("--block-init-y", type=float,
                   help="Override an existing push_block center y for a matched geometry probe")
    p.add_argument("--block-mass-override", type=float,
                   help="Override an existing push_block mass for a robustness probe")
    p.add_argument("--block-friction-override", type=float,
                   help="Override an existing push_block friction for a robustness probe")
    p.add_argument("--block-linear-damping", type=float,
                   help="Override an existing push_block viscous linear damping")
    p.add_argument("--block-size-x-override", type=float)
    p.add_argument("--block-size-y-override", type=float)
    p.add_argument("--block-height-override", type=float)
    p.add_argument("--block-mass", type=float, default=.08)
    p.add_argument("--block-height", type=float, default=.12)
    p.add_argument("--block-size-x", type=float, default=.045)
    p.add_argument("--block-size-y", type=float, default=.045)
    p.add_argument("--block-friction", type=float, default=.3)
    p.add_argument("--video", type=Path, help="Record a 25 fps Isaac scene video")
    p.add_argument("--video-label", type=str, default=None)
    p.add_argument("--video-width", type=int, default=960)
    p.add_argument("--video-height", type=int, default=540)
    p.add_argument("--video-crf", type=int, default=24)
    p.add_argument("--camera-eye", type=float, nargs=3,
                   help="Camera eye offset from robot root for the recorded video")
    p.add_argument("--camera-lookat", type=float, nargs=3,
                   help="Camera target offset from robot root for the recorded video")
    p.add_argument("--compact-video-overlay", action="store_true")
    p.add_argument("--minimal-video-overlay", action="store_true",
                   help="Small time, uphill displacement, and two-hand contact readout")
    p.add_argument("--force-chain-overlay", action="store_true",
                   help="Small readout of palm push, foot normal loads, lean, and foot travel")
    p.add_argument("--show-reach-markers", action="store_true",
                   help="Show 3D goal and actual markers for both hands and feet")
    p.add_argument("--wall-box-close-camera", action="store_true",
                   help="Frame the robot and box more closely in wall-box videos")
    p.add_argument("--foot-force-max", type=float, default=0.,
                   help="Random 3D force disturbance up to this many newtons")
    p.add_argument("--steps", type=int, default=870)
    p.add_argument("--min-uphill-distance-m", type=float, default=.08,
                   help="Minimum measured uphill box displacement for incline success")
    p.add_argument("--stop-on-fall", action="store_true",
                   help="End a failed rollout at root height <0.5m or tilt >0.6rad")
    p.add_argument("--stop-on-instability", action="store_true",
                   help="End rollout at root height <0.65m or tilt >0.25rad")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fixed-foot-friction", action="store_true",
                   help="Use the midpoint of the trained foot-friction ranges in evaluation")
    p.add_argument("--leg-reference-blend", type=float, default=0.,
                   help="Diagnostic only: blend swing-leg action with joint reference")
    p.add_argument("--leg-reference-all", action="store_true",
                   help="Diagnostic only: apply leg blend to both planted legs too")
    p.add_argument("--record-leg-action-trace", action="store_true",
                   help="Record commanded and actual leg joints for actuator tracking diagnostics")
    p.add_argument("--arm-reference-blend", type=float, default=0.,
                   help="Diagnostic only: blend arm action with joint reference")
    p.add_argument("--arm-policy-checkpoint", type=Path,
                   help="Diagnostic only: take arm actions from a second teacher checkpoint")
    p.add_argument("--leg-policy-checkpoint", type=Path,
                   help="Diagnostic only: take leg and waist actions from a second checkpoint")
    p.add_argument("--foot-static-friction", type=float,
                   help="Fix sole static friction for this paired evaluation")
    p.add_argument("--foot-dynamic-friction", type=float,
                   help="Fix sole dynamic friction for this paired evaluation")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/table_push/task_policy_eval.json")
    args = p.parse_args()
    if args.kinematic_preview and args.controller != "reference_joint":
        p.error("--kinematic-preview requires --controller reference_joint")
    if args.drawer_handle_spring_grasp and args.kinematic_preview:
        p.error("--drawer-handle-spring-grasp requires physical replay")
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from isaaclab.app import AppLauncher

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    torque_audit = None
    if args.reference_torque_audit is not None:
        if args.controller != "reference_joint":
            p.error("--reference-torque-audit requires --controller reference_joint")
        torque_audit = json.loads(args.reference_torque_audit.read_text())
        if torque_audit["frames_checked"] < args.steps or any(
                row["frame"] != i or not row["feasible"]
                for i, row in enumerate(torque_audit["rows"][:args.steps])):
            p.error("Torque audit must contain a feasible row for every playback frame")
    external_targets = None
    if args.external_target_npz is not None:
        external_targets = np.load(args.external_target_npz)["target_foot_pos_b"]
        if len(external_targets) < args.steps:
            raise ValueError("External target clip is shorter than requested steps")
    external_goals = None
    if args.external_goals_npz is not None:
        external_goals = np.load(args.external_goals_npz)["goal_xyz_world"]
        if external_goals.shape[1:] != (5, 3) or len(external_goals) < args.steps:
            raise ValueError("External goal clip must have at least --steps frames of [5,3] xyz")
    door_object_reference = None
    if args.door_object_reference is not None:
        door_object_reference = np.load(args.door_object_reference)
        if len(door_object_reference["phase"]) < args.steps:
            raise ValueError("Door object reference is shorter than requested steps")
    if args.controller.startswith("ceer_") and (external_targets is None or args.ceer_reference_npz is None):
        p.error("CEER comparison requires --external-target-npz and --ceer-reference-npz")
    cfg = OmegaConf.create(OmegaConf.to_container(state["cfg"], resolve=True))
    OmegaConf.set_struct(cfg, False)
    if args.matched_scene_checkpoint is not None:
        scene_state = torch.load(args.matched_scene_checkpoint, map_location="cpu", weights_only=False)
        cfg.task = OmegaConf.create(OmegaConf.to_container(scene_state["cfg"].task, resolve=True))
        # The compared actor sees the same policy observation. Preserve its
        # checkpoint's private observation layout for VecNorm and critic loading.
        cfg.task.observation.priv = OmegaConf.create(
            OmegaConf.to_container(state["cfg"].task.observation.priv, resolve=True))
    if args.task_override_yaml is not None:
        override = OmegaConf.load(args.task_override_yaml)
        if "defaults" in override:
            del override["defaults"]
        cfg.task = OmegaConf.merge(cfg.task, override)
    wall_box_task = "WallBox" in str(cfg.task.name)
    if args.robot_name_override is not None:
        cfg.task.robot.name = args.robot_name_override
    heavy_box_task = "HeavyBoxPush" in str(cfg.task.name)
    incline_box_task = "Incline" in str(cfg.task.name) and heavy_box_task
    alternating_heavy_box_task = "HeavyBoxPushAlternating" in str(cfg.task.name)
    right_foot_push_task = "RightFootPush" in str(cfg.task.name)
    pedal_task = "SpringPedal" in str(cfg.task.name)
    extraction_task = "BoxExtraction" in str(cfg.task.name)
    door_task = "Door" in str(cfg.task.name)
    drawer_task = "Drawer" in str(cfg.task.name)
    if door_object_reference is not None and not door_task:
        p.error("--door-object-reference requires a door task")
    if door_task and args.door_back_wall_shift_y:
        for obj in cfg.task.objects:
            if obj.get("name") in ("right_return_wall", "right_return_wall_reveal"):
                obj.pos[1] += args.door_back_wall_shift_y
    if door_task and args.door_asset_usd is not None:
        for obj in cfg.task.objects:
            if obj.get("name") == "spring_door":
                obj.usd_path = str(args.door_asset_usd.resolve())
    if door_task and args.door_initial_angle_deg is not None:
        for obj in cfg.task.objects:
            if obj.get("name") == "spring_door":
                obj.initial_angle_rad = float(np.deg2rad(args.door_initial_angle_deg))
        cfg.task.command.initial_door_angle_rad = float(np.deg2rad(args.door_initial_angle_deg))
    if sum((args.explicit_goals, external_goals is not None, external_targets is not None)) > 1:
        p.error("Choose only one external target mode")
    cfg.checkpoint_path = str(args.checkpoint.resolve())
    cfg.vecnorm = "eval"
    cfg.seed = args.seed
    cfg.task.num_envs = 1
    cfg.task.max_episode_length = max(args.steps + 10, 1000)
    dataset_prefix = args.dataset or str(cfg.task.command.dataset.mem_paths[0]).rsplit("/", 1)[0]
    cfg.task.command.dataset.mem_paths = [f"{dataset_prefix}/{args.split}"]
    cfg.task.command.dataset.path_weights = [1.0]
    cfg.task.command.dataset.fix_ds = 0
    cfg.task.command.dataset.fix_motion_id = args.motion_id
    cfg.task.command.disable_motion_finish = True
    if args.fixed_foot_friction and "perturb_body_materials" in cfg.task.randomization:
        material = cfg.task.randomization.perturb_body_materials
        for key in ("static_friction_range", "dynamic_friction_range", "restitution_range"):
            midpoint = sum(material[key]) / 2
            material[key] = [midpoint, midpoint]
    if args.foot_static_friction is not None or args.foot_dynamic_friction is not None:
        material = cfg.task.randomization.perturb_body_materials
        if args.foot_static_friction is not None:
            material.static_friction_range = [args.foot_static_friction] * 2
        if args.foot_dynamic_friction is not None:
            material.dynamic_friction_range = [args.foot_dynamic_friction] * 2
    if args.table_width is not None and not cfg.task.objects:
        p.error("--table-width requires a tabletop in the task scene")
    if args.table_width is not None:
        cfg.task.objects[0].size[1] = args.table_width
    if args.foot_force_max > 0:
        cfg.task.command.foot_force = {"enabled": True, "train_only": False,
            "force_range": [0., args.foot_force_max], "zero_prob": 0.}
    if args.no_table_contact and not cfg.task.objects:
        p.error("--no-table-contact requires a tabletop in the task scene")
    if args.no_table_contact:
        cfg.task.objects[0].pos[0] = 10.
    block_enabled = args.block or any(o.get("name") == "push_block" for o in cfg.task.objects)
    if block_enabled and not args.block:
        block_cfg = next(o for o in cfg.task.objects if o.get("name") == "push_block")
        args.block_size_x, args.block_size_y, args.block_height = map(float, block_cfg.size)
        if args.block_init_x is not None:
            block_cfg.pos[0] = args.block_init_x
        if args.block_init_y is not None:
            block_cfg.pos[1] = args.block_init_y
        if args.block_mass_override is not None:
            block_cfg.mass = args.block_mass_override
        if args.block_friction_override is not None:
            block_cfg.friction = args.block_friction_override
        if args.block_linear_damping is not None:
            block_cfg.linear_damping = args.block_linear_damping
        for index, override in enumerate((args.block_size_x_override,
                                          args.block_size_y_override,
                                          args.block_height_override)):
            if override is not None:
                block_cfg.size[index] = override
        if args.block_height_override is not None:
            block_cfg.pos[2] = args.block_height_override / 2
        args.block_size_x, args.block_size_y, args.block_height = map(float, block_cfg.size)
    if args.block and not any(o.get("name") == "push_block" for o in cfg.task.objects):
        cfg.task.objects.append({"type": "box_dynamic", "name": "push_block",
            "pos": [args.block_x, args.block_y, args.block_height / 2],
            "size": [args.block_size_x, args.block_size_y, args.block_height],
            "mass": args.block_mass, "friction": args.block_friction,
            "color": [.92, .38, .12]})
    cfg.task.termination = {}
    cfg.app = {"headless": True, "enable_cameras": args.video is not None}
    if args.video is not None:
        cfg.task.viewer.resolution = [args.video_width, args.video_height]
        cfg.task.viewer.eye = ([1.85, 3.55, 1.10] if alternating_heavy_box_task else
                               [-3.5, 1.5, 1.1] if door_task else
                               [1.45, 3.0, 1.05] if heavy_box_task else
                               [3.0, .6, 1.4] if wall_box_task else
                               [2.1, -1.8, 1.3] if extraction_task else [1.45, 2.45, .6])
        cfg.task.viewer.lookat = ([.62, 0., .25] if alternating_heavy_box_task else
                                  [.22, -.03, .18] if door_task else
                                  [.42, 0., .20] if heavy_box_task else
                                  [.20, 0., -.20] if wall_box_task else
                                  [.30, 0., -.15] if extraction_task else [.22, 0., -.24])
        if args.camera_eye is not None:
            cfg.task.viewer.eye = args.camera_eye
        if args.camera_lookat is not None:
            cfg.task.viewer.lookat = args.camera_lookat
        if wall_box_task and args.wall_box_close_camera:
            cfg.task.viewer.eye = [2.55, .55, 1.4]
            cfg.task.viewer.lookat = [.20, 0., .05]
    cfg.wandb.mode = "disabled"
    app = AppLauncher(headless=True, device="cuda:0", enable_cameras=args.video is not None).app
    writer = None
    completed = False
    try:
        if args.video is not None:
            import imageio.v2 as imageio
            from PIL import Image, ImageDraw, ImageFont
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(args.video), fps=25, codec="libx264",
                                        macro_block_size=1,
                                        ffmpeg_params=["-crf", str(args.video_crf), "-pix_fmt", "yuv420p"])
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                      14 if args.minimal_video_overlay else
                                      20 if args.compact_video_overlay else 16)
        from scripts.utils.helpers import make_env_policy
        from active_adaptation.utils.math import quat_apply, quat_apply_inverse
        from torchrl.envs.utils import set_exploration_type, ExplorationType
        env, policy, vecnorm, _ = make_env_policy(cfg)
        env.eval(); policy.eval()
        cmd = env.command_manager
        cmd.zero_init_prob = 1.
        robot = env.robot
        block = env.scene["push_block"] if block_enabled else None
        crate = env.scene["sliding_crate"] if extraction_task else None
        insert = env.scene["removable_insert"] if extraction_task else None
        left = robot.body_names.index("left_ankle_roll_link")
        right = robot.body_names.index("right_ankle_roll_link")
        torso = robot.body_names.index("torso_link")
        rollout = policy.get_rollout_policy("eval")
        arm_rollout = None
        if args.arm_policy_checkpoint is not None:
            arm_policy = copy.deepcopy(policy)
            arm_state = torch.load(args.arm_policy_checkpoint, map_location="cuda:0", weights_only=False)
            arm_policy.load_state_dict(arm_state["policy"])
            arm_policy.eval()
            arm_rollout = arm_policy.get_rollout_policy("eval")
        leg_rollout = None
        if args.leg_policy_checkpoint is not None:
            leg_policy = copy.deepcopy(policy)
            leg_state = torch.load(args.leg_policy_checkpoint, map_location="cuda:0", weights_only=False)
            leg_policy.load_state_dict(leg_state["policy"])
            leg_policy.eval()
            leg_rollout = leg_policy.get_rollout_policy("eval")
        td = env.reset()
        reach_markers = None
        if writer is not None and args.show_reach_markers:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
            reach_markers = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/World/ReachTrackingMarkers",
                markers={
                    "hand_goal": sim_utils.CuboidCfg(
                        size=(.055, .055, .055),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(.04, .88, .95), emissive_color=(.02, .35, .4))),
                    "hand_actual": sim_utils.SphereCfg(
                        radius=.055,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(1., .45, .05), emissive_color=(.35, .1, .01))),
                    "foot_goal": sim_utils.CuboidCfg(
                        size=(.065, .065, .035),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(.95, .08, .8), emissive_color=(.35, .02, .28))),
                    "foot_actual": sim_utils.SphereCfg(
                        radius=.045,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(.35, 1., .08), emissive_color=(.08, .35, .02))),
                }))
        door_object_marker = None
        door_object_state = "waiting"
        door_object_events = {}
        door_object_right_offset = None
        door_object_left_offset = None
        door_object_initial = None
        door_object_shelf = None
        door_object_pos = None
        if door_object_reference is not None:
            door_object_initial = np.asarray(door_object_reference["object_xyz"][0], dtype=float)
            door_object_shelf = np.asarray(door_object_reference["object_xyz"][-1], dtype=float)
            door_object_pos = door_object_initial.copy()
            if writer is not None:
                import isaaclab.sim as sim_utils
                from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
                door_object_marker = VisualizationMarkers(VisualizationMarkersCfg(
                    prim_path="/World/DoorObjectProxy",
                    markers={
                        "object": sim_utils.CuboidCfg(
                            size=(.065, .065, .065),
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(.05, .85, .95), emissive_color=(.02, .45, .55))),
                        "stand": sim_utils.CuboidCfg(
                            size=(.12, .12, .075),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.48, .36, .25))),
                        "stand_post": sim_utils.CuboidCfg(
                            size=(.035, .035, .60),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.37, .29, .22))),
                    }))
                door_object_marker.visualize(
                    translations=np.stack([door_object_pos, door_object_initial - [0., 0., .07],
                                           [door_object_initial[0], door_object_initial[1], .30]]),
                    marker_indices=[0, 1, 2])
        baseline = None
        if args.controller.startswith("ceer_"):
            from scripts.table_push.ceer_task_adapter import CEERTaskAdapter
            baseline = CEERTaskAdapter(env, args.ceer_checkpoint, args.ceer_reference_npz,
                                       external_targets,
                                       mode="native" if args.controller == "ceer_native"
                                       else "joint_reference")
            baseline.reset()
        # Eval-mode reset randomizes the internal episode age; force a complete
        # clip from its first frame for this deterministic trajectory trial.
        env.base_env.episode_length_buf.zero_()
        with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
            student_only = cfg.algo.phase in ("adapt", "finetune") and not cfg.algo.direct_policy
            actor_input = (td.select("policy", "is_init", strict=False).clone()
                           if student_only else td.clone())
            rollout(actor_input)
            if actor_input["action"].shape != (1, 29):
                raise RuntimeError("Actor did not produce a 29-joint action")
        initial_reference_t = int(cmd.t[0])
        initial_reference_length = int(cmd.lengths[0])
        actual_max_episode_length = int(env.max_episode_length)
        termination_names = list(env.base_env.termination_funcs)
        initial_hands = robot.data.body_pos_w[0, [robot.body_names.index("left_hand_mimic"),
                                                  robot.body_names.index("right_hand_mimic")]].clone()
        initial_support_foot = robot.data.body_pos_w[0, left if (right_foot_push_task or pedal_task) else right].clone()
        initial_root_xyz = robot.data.root_pos_w[0].clone()
        initial_root_quat = robot.data.root_quat_w[0].clone()
        initial_joint_pos = robot.data.joint_pos[0].clone()
        initial_goal_xyz = (external_goals[0] if external_goals is not None else
                            cmd.target_positions_world()[0].detach().cpu().numpy().copy()
                            if args.explicit_goals else None)
        initial_block_pos = block.data.root_pos_w[0].clone() if block is not None else None
        intentional_lean = bool(
            (alternating_heavy_box_task or "InclineForceWalk" in str(cfg.task.name))
            and initial_block_pos is not None and float(initial_block_pos[0]) < 5.0)
        fall_tilt_limit = .90 if intentional_lean else .60
        instability_tilt_limit = .80 if intentional_lean else .25
        initial_crate_pos = crate.data.root_pos_w[0].clone() if crate is not None else None
        initial_insert_pos = insert.data.root_pos_w[0].clone() if insert is not None else None
        drawer_joint_id = (env.scene["drawer_cabinet"].joint_names.index("Slide")
                           if drawer_task else None)
        initial_drawer_position = (float(env.scene["drawer_cabinet"].data.joint_pos[0, drawer_joint_id])
                                   if drawer_task else None)
        initial_pedal_position = (float(env.scene["spring_pedal"].data.joint_pos[0, 0])
                                  if pedal_task else None)
        grasp_force = None
        if args.drawer_handle_spring_grasp:
            if not drawer_task:
                p.error("--drawer-handle-spring-grasp requires a drawer task")
            drawer_asset = env.scene["drawer_cabinet"]
            drawer_body_id = drawer_asset.body_names.index("LowerDrawer")
            grasp_hand_id = robot.body_names.index("right_hand_mimic")
            grasp_offset = (robot.data.body_pos_w[0, grasp_hand_id]
                            - drawer_asset.data.body_pos_w[0, drawer_body_id]).clone()
            print("drawer grasp body names", drawer_asset.body_names,
                  "initial offset", grasp_offset.tolist(), flush=True)
            def apply_drawer_grasp(_substep):
                nonlocal grasp_force
                drawer_position_w = drawer_asset.data.body_pos_w[0, drawer_body_id]
                drawer_velocity_w = drawer_asset.data.body_lin_vel_w[0, drawer_body_id]
                hand_position_w = robot.data.body_pos_w[0, grasp_hand_id]
                hand_velocity_w = robot.data.body_lin_vel_w[0, grasp_hand_id]
                anchor = drawer_position_w + grasp_offset
                grasp_force = (args.drawer_grasp_stiffness * (anchor - hand_position_w)
                               + args.drawer_grasp_damping * (drawer_velocity_w - hand_velocity_w))
                grasp_force = grasp_force.clamp(-180., 180.)
                if args.drawer_grasp_x_only:
                    grasp_force[1:] = 0.
                wrench_position = anchor.reshape(1, 1, 3)
                zero_torque = torch.zeros_like(wrench_position)
                robot.set_external_force_and_torque(
                    grasp_force.reshape(1, 1, 3), zero_torque,
                    positions=wrench_position, body_ids=[grasp_hand_id], env_ids=[0], is_global=True)
                drawer_asset.set_external_force_and_torque(
                    -grasp_force.reshape(1, 1, 3), zero_torque,
                    positions=wrench_position, body_ids=[drawer_body_id], env_ids=[0], is_global=True)
            env.base_env._pre_step_callbacks.append(apply_drawer_grasp)
        rows = []
        explicit_goal_hasher = hashlib.sha256() if args.explicit_goals else None
        with torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
            for t in range(args.steps):
                if t % 100 == 0:
                    print(f"evaluation step {t}/{args.steps}", flush=True)
                if external_goals is not None:
                    cmd.set_targets_world(torch.as_tensor(external_goals[t], device=cmd.device)
                                          + env.scene.env_origins[0, None])
                    td["policy"][:, :15] = ((cmd.target_support_xyz_b() - vecnorm.loc["policy"][:15])
                                                 / vecnorm.scale["policy"][:15])
                if args.explicit_goals:
                    cmd.set_targets_world(None)
                    goal_world = cmd.target_positions_world()
                    explicit_goal_hasher.update(goal_world[0].to(torch.float32).cpu().numpy().tobytes())
                    cmd.set_targets_world(goal_world)
                    td["policy"][:, :15] = ((cmd.target_support_xyz_b() - vecnorm.loc["policy"][:15])
                                                 / vecnorm.scale["policy"][:15])
                if external_targets is not None:
                    cmd.set_target_foot_pos_b(external_targets[t])
                    td["policy"][:, :3] = ((cmd.target_foot_pos_b() - vecnorm.loc["policy"][:3])
                                             / vecnorm.scale["policy"][:3])
                if args.controller == "zero":
                    td["action"] = torch.zeros((1, 29), device=cmd.device)
                elif args.controller == "reference_joint":
                    action_manager = env.base_env.action_manager
                    joint_order = [cmd.dataset.joint_names.index(name)
                                   for name in action_manager.joint_names]
                    reference_joint_pos = cmd._motion.joint_pos[:, 0, joint_order]
                    if torque_audit is not None:
                        force_order = [torque_audit["joint_names"].index(name)
                                       for name in action_manager.joint_names]
                        torque = torch.as_tensor(
                            torque_audit["rows"][t]["joint_torque_Nm"],
                            device=reference_joint_pos.device,
                            dtype=reference_joint_pos.dtype)[force_order]
                        stiffness = robot.data.joint_stiffness[:, action_manager.joint_ids]
                        reference_joint_pos = reference_joint_pos + (
                            args.reference_torque_gain * torque[None] / stiffness.clamp_min(1.))
                    td["action"] = ((reference_joint_pos - action_manager.default_joint_pos[:, action_manager.joint_ids])
                                    / action_manager.action_scaling)
                elif baseline is None:
                    if student_only:
                        actor_td = td.select("policy", "is_init", strict=False).clone()
                        rollout(actor_td)
                        td["action"] = actor_td["action"]
                    else:
                        td = rollout(td)
                else:
                    td["action"] = baseline.act(t)
                if arm_rollout is not None:
                    arm_td = td.clone()
                    arm_rollout(arm_td)
                    arm_ids = [i for i, name in enumerate(env.base_env.action_manager.joint_names)
                               if any(part in name for part in ("shoulder", "elbow", "wrist"))]
                    td["action"][:, arm_ids] = arm_td["action"][:, arm_ids]
                if leg_rollout is not None:
                    leg_td = td.clone()
                    leg_rollout(leg_td)
                    leg_ids = [i for i, name in enumerate(env.base_env.action_manager.joint_names)
                               if any(part in name for part in ("hip", "knee", "ankle", "waist"))]
                    td["action"][:, leg_ids] = leg_td["action"][:, leg_ids]
                if args.leg_reference_blend > 0.:
                    action_manager = env.base_env.action_manager
                    joint_order = [cmd.dataset.joint_names.index(name)
                                   for name in action_manager.joint_names]
                    reference_joint_pos = cmd._motion.joint_pos[:, 0, joint_order]
                    reference_action = ((reference_joint_pos
                        - action_manager.default_joint_pos[:, action_manager.joint_ids])
                        / action_manager.action_scaling)
                    swing_goals = cmd.target_positions_world()[0, 3:5, 2]
                    for side, swing_z in zip(("left", "right"), swing_goals):
                        if not args.leg_reference_all and float(swing_z - env.scene.env_origins[0, 2]) <= .015:
                            continue
                        ids = [i for i, name in enumerate(action_manager.joint_names)
                               if name.startswith(side + "_") and any(
                                   part in name for part in ("hip", "knee", "ankle"))]
                        b = args.leg_reference_blend
                        td["action"][:, ids] = ((1. - b) * td["action"][:, ids]
                                                 + b * reference_action[:, ids])
                if args.arm_reference_blend > 0.:
                    action_manager = env.base_env.action_manager
                    joint_order = [cmd.dataset.joint_names.index(name)
                                   for name in action_manager.joint_names]
                    reference_joint_pos = cmd._motion.joint_pos[:, 0, joint_order]
                    reference_action = ((reference_joint_pos
                        - action_manager.default_joint_pos[:, action_manager.joint_ids])
                        / action_manager.action_scaling)
                    ids = [i for i, name in enumerate(action_manager.joint_names)
                           if any(part in name for part in ("shoulder", "elbow", "wrist"))]
                    b = args.arm_reference_blend
                    td["action"][:, ids] = ((1. - b) * td["action"][:, ids]
                                             + b * reference_action[:, ids])
                if args.kinematic_preview:
                    ref_root_state = torch.cat((cmd._motion.root_pos_w[:, 0].clone(),
                                                cmd._motion.root_quat_w[:, 0].clone(),
                                                torch.zeros((1, 6), device=cmd.device)), dim=-1)
                    robot_joint_order = [cmd.dataset.joint_names.index(name) for name in robot.joint_names]
                    ref_robot_joints = cmd._motion.joint_pos[:, 0, robot_joint_order].clone()
                if args.record_leg_action_trace:
                    action_manager = env.base_env.action_manager
                    trace_ids = [i for i, name in enumerate(action_manager.joint_names)
                                 if any(part in name for part in ("hip", "knee", "ankle"))]
                    trace_names = [action_manager.joint_names[i] for i in trace_ids]
                    trace_action = td["action"][0, trace_ids].detach().cpu().tolist()
                    trace_order = [cmd.dataset.joint_names.index(name) for name in trace_names]
                    trace_reference = cmd._motion.joint_pos[0, 0, trace_order].detach().cpu().tolist()
                transition, td = env.step_and_maybe_reset(td)
                if args.kinematic_preview:
                    robot.write_root_state_to_sim(ref_root_state)
                    robot.write_joint_state_to_sim(ref_robot_joints, torch.zeros_like(ref_robot_joints))
                    if drawer_task:
                        u = max(0., min(1., (t - 360) / 100.))
                        smooth = u * u * u * (10. + u * (-15. + 6. * u))
                        prescribed = torch.zeros_like(env.scene["drawer_cabinet"].data.joint_pos)
                        prescribed[:, drawer_joint_id] = -args.preview_drawer_depth * smooth
                        env.scene["drawer_cabinet"].write_joint_state_to_sim(
                            prescribed, torch.zeros_like(prescribed))
                foot = robot.data.body_pos_w[:, left] + quat_apply(
                    robot.data.body_quat_w[:, left], cmd.sole_offset.expand(1, -1))
                target = cmd.target_sole_world()
                forces = [(env.scene[name].data.force_matrix_w.reshape(1, -1, 3).sum(1)[0]
                           if name in env.scene.keys() else torch.zeros(3, device=cmd.device))
                          for name in ("eval_left_hand_wall_force", "eval_right_hand_wall_force")]
                crate_foot_force = (env.scene["eval_left_foot_crate_force"].data.force_matrix_w
                                    .reshape(1, -1, 3).sum(1)[0] if extraction_task else
                                    torch.zeros(3, device=cmd.device))
                right_foot_box_force = (env.scene["eval_right_foot_box_force"].data.force_matrix_w
                                        .reshape(1, -1, 3).sum(1)[0] if right_foot_push_task else
                                        torch.zeros(3, device=cmd.device))
                pedal_foot_force = (env.scene["eval_right_foot_pedal_force"].data.force_matrix_w
                                    .reshape(1, -1, 3).sum(1)[0] if pedal_task else
                                    torch.zeros(3, device=cmd.device))
                pedal_position = (float(env.scene["spring_pedal"].data.joint_pos[0, 0])
                                  if pedal_task else None)
                right_ankle_pitch_box_force = (env.scene["eval_right_ankle_pitch_box_force"].data.force_matrix_w
                                               .reshape(1, -1, 3).sum(1)[0] if right_foot_push_task else
                                               torch.zeros(3, device=cmd.device))
                right_knee_box_force = (env.scene["eval_right_knee_box_force"].data.force_matrix_w
                                        .reshape(1, -1, 3).sum(1)[0] if right_foot_push_task else
                                        torch.zeros(3, device=cmd.device))
                hand_box_forces = [
                    env.scene[f"eval_{side}_hand_box_force"].data.force_matrix_w
                    .reshape(1, -1, 3).sum(1)[0] if heavy_box_task else
                    torch.zeros(3, device=cmd.device)
                    for side in ("left", "right")]
                door_angle = (float(env.scene["spring_door"].data.joint_pos[0, 0])
                              if door_task else None)
                drawer_position = (float(env.scene["drawer_cabinet"].data.joint_pos[0, drawer_joint_id])
                                   if drawer_task else None)
                cabinet_force_vec = (env.scene["eval_left_hand_cabinet_force"].data.net_forces_w[0]
                                      .reshape(-1,3).sum(0) if drawer_task else None)
                drawer_hand_force_vec = (env.scene["eval_right_hand_drawer_force"].data.net_forces_w[0]
                                         .reshape(-1,3).sum(0) if drawer_task else None)
                cabinet_force = float(cabinet_force_vec.norm()) if drawer_task else None
                drawer_hand_force = float(drawer_hand_force_vec.norm()) if drawer_task else None
                drawer_hand_raw_force = drawer_hand_force
                cabinet_raw_force = cabinet_force
                back_wall_force = (sum(float(env.scene[name].data.force_matrix_w
                                             .reshape(1, -1, 3).norm(dim=-1).sum())
                                       for name in ("eval_torso_wall_force",
                                                    "eval_waist_yaw_link_wall_force",
                                                    "eval_waist_roll_link_wall_force",
                                                    "eval_pelvis_wall_force"))
                                   if door_task else None)
                back_wall_force_vec = (sum((env.scene[name].data.force_matrix_w
                                            .reshape(1, -1, 3).sum(dim=1)[0]
                                            for name in ("eval_torso_wall_force",
                                                         "eval_waist_yaw_link_wall_force",
                                                         "eval_waist_roll_link_wall_force",
                                                         "eval_pelvis_wall_force")))
                                       if door_task else None)
                door_foot_force_vec = (env.scene["eval_right_foot_door_force"].data.force_matrix_w
                                       .reshape(1, -1, 3).sum(dim=1)[0]
                                       if door_task else None)
                door_foot_force = (float(door_foot_force_vec.norm()) if door_task else None)
                door_knee_force = (float(env.scene["eval_right_knee_door_force"].data.force_matrix_w
                                         .reshape(1, -1, 3).norm(dim=-1).sum())
                                   if door_task else None)
                ff = env.scene["contact_forces"].data.net_forces_w[0]
                right_sole = robot.data.body_pos_w[:, right] + quat_apply(
                    robot.data.body_quat_w[:, right], cmd.sole_offset.expand(1, -1))
                right_target = cmd.target_positions_world()[:, 4]
                gravity = robot.data.projected_gravity_b[0]
                torso_gravity = quat_apply_inverse(
                    robot.data.body_quat_w[:, torso],
                    torch.tensor([[0., 0., -1.]], device=cmd.device))[0]
                hands = robot.data.body_pos_w[0, [robot.body_names.index("left_hand_mimic"),
                                                   robot.body_names.index("right_hand_mimic")]]
                if drawer_task:
                    left_local = hands[0] - env.scene.env_origins[0]
                    right_local = hands[1] - env.scene.env_origins[0]
                    if "TwoDrawer" in str(cfg.task.name):
                        upright = float(robot.data.root_pos_w[0,2]) > .55 and float(gravity[:2].norm()) < .6
                        cabinet_near = (.29<float(left_local[0])<.48 and
                                        .06<float(left_local[1])<.29 and
                                        .66<float(left_local[2])<.99)
                        drawer_near = (.24+drawer_position<float(right_local[0])<.49+drawer_position and
                                       abs(float(right_local[1])+.055)<.16 and
                                       abs(float(right_local[2])-.45)<.16)
                    else:
                        upright = float(robot.data.root_pos_w[0,2]) > .6 and float(gravity[:2].norm()) < .5
                        cabinet_near = (abs(float(left_local[0])-.38)<.11 and
                                        abs(float(left_local[1])-.155)<.11 and
                                        abs(float(left_local[2])-.80)<.10)
                        drawer_near = (.33+drawer_position<float(right_local[0])<.61+drawer_position and
                                       abs(float(right_local[1])+.07)<.12 and
                                       abs(float(right_local[2])-.72)<.18)
                    cabinet_force = cabinet_force if upright and cabinet_near else 0.
                    drawer_hand_force = drawer_hand_force if upright and drawer_near else 0.
                if door_object_reference is not None:
                    phase = str(door_object_reference["phase"][t])
                    hand_xyz = hands.detach().cpu().numpy()
                    right_distance = float(np.linalg.norm(hand_xyz[1] - door_object_initial))
                    hands_distance = float(np.linalg.norm(hand_xyz[0] - hand_xyz[1]))
                    shelf_distance = float(np.linalg.norm(hand_xyz[0] - door_object_shelf))
                    if (door_object_state == "waiting" and phase == "right_hand_reach_item"
                            and float(door_angle) * 180. / np.pi >= args.door_object_open_angle_deg
                            and right_distance <= .075):
                        door_object_state = "right_hand"
                        door_object_right_offset = door_object_pos - hand_xyz[1]
                        door_object_events["pickup_step"] = t
                    if door_object_state == "right_hand":
                        if phase in ("bring_hands_together", "transfer_to_left_hand"):
                            direction = (hand_xyz[0] - hand_xyz[1]) / max(hands_distance, 1e-9)
                            desired = hand_xyz[1] + direction * min(.045, hands_distance / 2.)
                        else:
                            desired = hand_xyz[1] + door_object_right_offset
                        delta = desired - door_object_pos
                        distance = float(np.linalg.norm(delta))
                        door_object_pos += delta * min(1., .006 / max(distance, 1e-9))
                        left_to_object = float(np.linalg.norm(hand_xyz[0] - door_object_pos))
                        right_to_object = float(np.linalg.norm(hand_xyz[1] - door_object_pos))
                        if (phase == "transfer_to_left_hand"
                                and hands_distance <= args.door_object_handoff_distance_m
                                and left_to_object <= .06 and right_to_object <= .06):
                            door_object_state = "left_hand"
                            door_object_events["handoff_step"] = t
                            door_object_left_offset = door_object_pos - hand_xyz[0]
                    if door_object_state == "left_hand":
                        door_object_pos = hand_xyz[0] + door_object_left_offset
                        on_shelf = np.all(np.abs(door_object_pos[:2] - door_object_shelf[:2]) <= .0875)
                        if (phase == "release_item_while_foot_braces_door"
                                and shelf_distance <= .12 and on_shelf):
                            door_object_state = "shelf"
                            door_object_events["place_step"] = t
                    if door_object_state == "shelf":
                        door_object_pos[2] = max(.6575, door_object_pos[2] - .005)
                    if door_object_marker is not None:
                        door_object_marker.visualize(
                            translations=np.stack([door_object_pos, door_object_initial - [0., 0., .07],
                                                   [door_object_initial[0], door_object_initial[1], .30]]),
                            marker_indices=[0, 1, 2])
                goals_w = cmd.target_positions_world()[0]
                rows.append({"step": t,
                    "leg_action": trace_action if args.record_leg_action_trace else None,
                    "leg_joint_reference_rad": trace_reference if args.record_leg_action_trace else None,
                    "leg_joint_actual_rad": robot.data.joint_pos[0, [robot.joint_names.index(n) for n in trace_names]].detach().cpu().tolist() if args.record_leg_action_trace else None,
                    "foot_error_m": float((right_sole-right_target).norm()) if (right_foot_push_task or pedal_task) else float((foot-target).norm()),
                    "reference_t": int(cmd.t[0]),
                    "drawer_grasp_force_xyz_N": (grasp_force.tolist()
                                                 if grasp_force is not None else None),
                    "reference_finished": bool(cmd.finished[0]),
                    "foot_xyz_m": (right_sole if right_foot_push_task else foot)[0].tolist(), "target_foot_xyz_m": target[0].tolist(),
                    "left_sole_z_m": float(foot[0, 2]), "target_sole_z_m": float(target[0, 2]),
                    "right_sole_z_m": float(right_sole[0, 2]),
                    "right_sole_xyz_m": right_sole[0].tolist(),
                    "right_sole_goal_xyz_m": right_target[0].tolist(),
                    "target_right_sole_z_m": float(right_target[0, 2]),
                    "right_foot_error_m": float((right_sole-right_target).norm()),
                    "door_angle_rad": door_angle,
                    "drawer_position_m": drawer_position,
                    "pedal_position_m": pedal_position,
                    "pedal_foot_force_N": float(pedal_foot_force.norm()),
                    "cabinet_hand_force_N": cabinet_force,
                    "drawer_hand_force_N": drawer_hand_force,
                    "cabinet_hand_raw_force_N": cabinet_raw_force,
                    "drawer_hand_raw_force_N": drawer_hand_raw_force,
                    "cabinet_hand_force_xyz_N": cabinet_force_vec.tolist() if drawer_task else None,
                    "drawer_hand_force_xyz_N": drawer_hand_force_vec.tolist() if drawer_task else None,
                    "back_wall_force_N": back_wall_force,
                    "back_wall_force_xyz_N": (back_wall_force_vec.tolist()
                                               if door_task else None),
                    "right_foot_door_force_N": door_foot_force,
                    "right_foot_door_force_xyz_N": (door_foot_force_vec.tolist()
                                                     if door_task else None),
                    "right_knee_door_force_N": door_knee_force,
                    "root_xyz_m": robot.data.root_pos_w[0].tolist(),
                    "root_quat_wxyz": robot.data.root_quat_w[0].tolist(),
                    "joint_pos_rad": robot.data.joint_pos[0].tolist(),
                    "root_error_m": float((robot.data.root_pos_w[0]-goals_w[0]).norm()),
                    "left_hand_error_m": float((hands[0]-goals_w[1]).norm()),
                    "right_hand_error_m": float((hands[1]-goals_w[2]).norm()),
                    "left_hand_xyz_m": hands[0].tolist(),
                    "left_hand_goal_xyz_m": goals_w[1].tolist(),
                    "right_hand_xyz_m": hands[1].tolist(),
                    "right_hand_goal_xyz_m": goals_w[2].tolist(),
                    "door_object_state": door_object_state if door_object_reference is not None else None,
                    "door_object_xyz_m": door_object_pos.tolist() if door_object_reference is not None else None,
                    "door_object_right_distance_m": right_distance if door_object_reference is not None else None,
                    "door_object_hands_distance_m": hands_distance if door_object_reference is not None else None,
                    "door_object_shelf_distance_m": shelf_distance if door_object_reference is not None else None,
                    "left_foot_force_z_N": float(ff[0, 2]),
                    "left_foot_force_x_N": float(ff[0, 0]),
                    "applied_left_foot_force_N": float(cmd.foot_force_ramp.force[0].norm()),
                    "right_foot_force_z_N": float(ff[1, 2]),
                    "right_foot_force_x_N": float(ff[1, 0]),
                    "left_hand_table_force_z_N": float(forces[0][2]),
                    "right_hand_table_force_z_N": float(forces[1][2]),
                    "left_hand_wall_force_y_N": float(forces[0][1]),
                    "right_hand_wall_force_y_N": float(forces[1][1]),
                    "left_foot_crate_force_z_N": float(crate_foot_force[2].abs()),
                    "right_foot_box_force_N": float(right_foot_box_force.norm()),
                    "right_ankle_pitch_box_force_N": float(right_ankle_pitch_box_force.norm()),
                    "right_knee_box_force_N": float(right_knee_box_force.norm()),
                    "left_hand_box_force_x_N": float(hand_box_forces[0][0]),
                    "right_hand_box_force_x_N": float(hand_box_forces[1][0]),
                    "ground_reaction_x_N": float(ff[0, 0] + ff[1, 0]),
                    "root_height_m": float(robot.data.root_pos_w[0, 2]),
                    "root_tilt_rad": float(torch.acos((-gravity[2]).clamp(-1, 1))),
                    "root_forward_lean_rad": float(torch.atan2(gravity[0], -gravity[2])),
                    "torso_forward_lean_rad": float(torch.atan2(torso_gravity[0], -torso_gravity[2])),
                    "hand_drift_m": (hands-initial_hands).norm(dim=-1).tolist(),
                    "right_foot_drift_m": float((robot.data.body_pos_w[0, right]-initial_support_foot).norm()) if not right_foot_push_task else None,
                    "left_foot_drift_m": float((robot.data.body_pos_w[0, left]-initial_support_foot).norm()) if (right_foot_push_task or pedal_task) else None,
                    "block_xyz_m": block.data.root_pos_w[0].tolist() if block is not None else None,
                    "block_quat_wxyz": block.data.root_quat_w[0].tolist() if block is not None else None,
                    "crate_xyz_m": crate.data.root_pos_w[0].tolist() if crate is not None else None,
                    "insert_xyz_m": insert.data.root_pos_w[0].tolist() if insert is not None else None,
                    "insert_quat_wxyz": insert.data.root_quat_w[0].tolist() if insert is not None else None,
                    "episode_done": bool(transition["next", "done"][0].item()),
                    "terminated": bool(transition["next", "terminated"][0].item()),
                    "truncated": bool(transition["next", "truncated"][0].item())})
                if writer is not None and t % 2 == 0:
                    if reach_markers is not None:
                        reach_markers.visualize(
                            translations=np.stack([
                                goals_w[1].detach().cpu().numpy(), hands[0].detach().cpu().numpy(),
                                goals_w[2].detach().cpu().numpy(), hands[1].detach().cpu().numpy(),
                                goals_w[3].detach().cpu().numpy(), foot[0].detach().cpu().numpy(),
                                goals_w[4].detach().cpu().numpy(), right_sole[0].detach().cpu().numpy(),
                            ]),
                            marker_indices=[0, 1, 0, 1, 2, 3, 2, 3])
                    image = Image.fromarray(np.asarray(env.render("rgb_array"))[..., :3].astype(np.uint8))
                    draw = ImageDraw.Draw(image, "RGBA")
                    displacement = (float(block.data.root_pos_w[0, 0 if wall_box_task or heavy_box_task else 1]-
                                          initial_block_pos[0 if wall_box_task or heavy_box_task else 1])
                                    if block is not None else 0.)
                    if args.force_chain_overlay:
                        line1 = (f"{args.video_label or args.controller}  {(.02*(t+1)):.1f}s"
                                 f"   box uphill {100*displacement:+.1f} cm"
                                 f"   lean {rows[-1]['root_forward_lean_rad']*57.3:+.0f} deg")
                        line2 = (f"palms L/R {abs(hand_box_forces[0][0]):.0f}/"
                                 f"{abs(hand_box_forces[1][0]):.0f} N"
                                 f"   feet load L/R {ff[0,2]:.0f}/{ff[1,2]:.0f} N"
                                 f"   feet advance {100*(foot[0,0]-initial_goal_xyz[3,0]):+.0f}/"
                                 f"{100*(right_sole[0,0]-initial_goal_xyz[4,0]):+.0f} cm")
                        width = max(draw.textbbox((0, 0), line, font=font)[2] for line in (line1, line2))
                        draw.rounded_rectangle((10, 8, width + 24, 52), radius=4,
                                               fill=(9, 15, 24, 140))
                        draw.text((17, 10), line1, font=font, fill="white")
                        draw.text((17, 30), line2, font=font, fill=(170, 230, 255))
                    elif args.minimal_video_overlay:
                        label = ((f"{args.video_label or args.controller}  {(.02*(t+1)):.1f}s"
                                  f"   pedal {100*(-pedal_position):.1f} cm"
                                  f"   foot {pedal_foot_force.norm():.0f} N"
                                  f"   hand {forces[1][1]:.0f} N") if pedal_task else
                                 (f"{args.video_label or args.controller}  {(.02*(t+1)):.1f}s"
                                  f"   box uphill {100*displacement:+.1f} cm"
                                  f"   palms L/R {abs(hand_box_forces[0][0]):.0f}/"
                                  f"{abs(hand_box_forces[1][0]):.0f} N"))
                        width = draw.textbbox((0, 0), label, font=font)[2]
                        draw.rounded_rectangle((10, 8, width + 24, 31), radius=4,
                                               fill=(9, 15, 24, 140))
                        draw.text((17, 11), label, font=font, fill="white")
                    elif args.show_reach_markers:
                        legend = [
                            ("HAND target", (4, 225, 243), "square"),
                            ("HAND actual", (255, 115, 13), "circle"),
                            ("FOOT target", (242, 20, 204), "square"),
                            ("FOOT actual", (89, 255, 20), "circle"),
                        ]
                        draw.rounded_rectangle((12, 12, 210, 108), radius=7,
                                               fill=(9, 15, 24, 150))
                        for i, (label, color, shape) in enumerate(legend):
                            y = 22 + 21*i
                            if shape == "square":
                                draw.rectangle((23, y+2, 34, y+13), fill=color)
                            else:
                                draw.ellipse((23, y+2, 34, y+13), fill=color)
                            draw.text((43, y-3), label, font=font, fill="white")
                    elif args.compact_video_overlay:
                        lines = [f"{args.video_label or args.controller}   t={(t+1)*.02:.1f}s"]
                        if initial_goal_xyz is not None:
                            if heavy_box_task:
                                lines.append(
                                    f"goal/actual cm: root x "
                                    f"{100*(goals_w[0,0]-initial_goal_xyz[0,0]):+.0f}/"
                                    f"{100*(robot.data.root_pos_w[0,0]-initial_goal_xyz[0,0]):+.0f}"
                                    f"   hands x L "
                                    f"{100*(goals_w[1,0]-initial_goal_xyz[1,0]):+.0f}/"
                                    f"{100*(hands[0,0]-initial_goal_xyz[1,0]):+.0f}"
                                    f" R {100*(goals_w[2,0]-initial_goal_xyz[2,0]):+.0f}/"
                                    f"{100*(hands[1,0]-initial_goal_xyz[2,0]):+.0f}")
                            elif wall_box_task:
                                lines.append(
                                    f"goal/actual cm: root y "
                                    f"{100*(goals_w[0,1]-initial_goal_xyz[0,1]):+.0f}/"
                                    f"{100*(robot.data.root_pos_w[0,1]-initial_goal_xyz[0,1]):+.0f}"
                                    f"   right hand y "
                                    f"{100*(goals_w[2,1]-initial_goal_xyz[2,1]):+.0f}/"
                                    f"{100*(hands[1,1]-initial_goal_xyz[2,1]):+.0f}"
                                    f"   {'right' if right_foot_push_task else 'left'} foot x "
                                    f"{100*(goals_w[4 if right_foot_push_task else 3,0]-initial_goal_xyz[4 if right_foot_push_task else 3,0]):+.0f}/"
                                    f"{100*((right_sole[0,0] if right_foot_push_task else foot[0,0])-initial_goal_xyz[4 if right_foot_push_task else 3,0]):+.0f}")
                            elif extraction_task:
                                lines.append(
                                    f"goal/actual cm: left foot z {100*goals_w[3,2]:.1f}/{100*foot[0,2]:.1f}"
                                    f"   right hand z {100*goals_w[2,2]:.1f}/{100*hands[1,2]:.1f}"
                                    f"   root z {100*goals_w[0,2]:.1f}/{100*robot.data.root_pos_w[0,2]:.1f}")
                            else:
                                lines.append(
                                    f"goal/actual cm: root y "
                                    f"{100*(goals_w[0,1]-initial_goal_xyz[0,1]):+.0f}/"
                                    f"{100*(robot.data.root_pos_w[0,1]-initial_goal_xyz[0,1]):+.0f}"
                                    f"   hands x L {100*(goals_w[1,0]-initial_goal_xyz[1,0]):+.0f}/"
                                    f"{100*(hands[0,0]-initial_goal_xyz[1,0]):+.0f}"
                                    f" R {100*(goals_w[2,0]-initial_goal_xyz[2,0]):+.0f}/"
                                    f"{100*(hands[1,0]-initial_goal_xyz[2,0]):+.0f}")
                        if door_task:
                            lines.append(
                                f"door {door_angle*57.3:.1f} deg   foot Fy {door_foot_force_vec[1]:+.0f} N"
                                f"   back Fy {back_wall_force_vec[1]:+.0f} N")
                            if door_object_reference is not None:
                                lines.append(
                                    f"object {door_object_state}   hand gap {100*hands_distance:.0f} cm"
                                    f"   right/item {100*right_distance:.0f} cm")
                            else:
                                lines.append(
                                    f"right foot z goal/actual {100*right_target[0,2]:.0f}/"
                                    f"{100*right_sole[0,2]:.0f} cm   hand errors L/R "
                                    f"{100*rows[-1]['left_hand_error_m']:.0f}/"
                                    f"{100*rows[-1]['right_hand_error_m']:.0f} cm")
                        elif drawer_task:
                            if args.kinematic_preview:
                                lines.append(
                                    f"prescribed slide {100*(initial_drawer_position-drawer_position):+.1f} cm"
                                    f"   torso bend {rows[-1]['torso_forward_lean_rad']*57.3:.0f} deg"
                                    f"   stance 41 cm")
                            else:
                                lines.append(f"drawer pull {100*(initial_drawer_position-drawer_position):+.1f} cm"
                                             f"   cabinet hand {cabinet_force:.0f} N"
                                             f"   drawer hand {drawer_hand_force:.0f} N")
                        elif extraction_task:
                            crate_drift = torch.linalg.vector_norm(
                                crate.data.root_pos_w[0, :2]-initial_crate_pos[:2])
                            lines.append(
                                f"left foot on crate {crate_foot_force[2]:.0f} N"
                                f"   insert lift {100*(insert.data.root_pos_w[0,2]-initial_insert_pos[2]):+.1f} cm"
                                f"   crate drift {100*crate_drift:.1f} cm")
                        elif alternating_heavy_box_task:
                            lines.append(
                                f"feet x/z cm L {100*goals_w[3,0]:+.0f}/{100*foot[0,0]:+.0f}"
                                f" {100*goals_w[3,2]:.0f}/{100*foot[0,2]:.0f}"
                                f"   R {100*goals_w[4,0]:+.0f}/{100*right_sole[0,0]:+.0f}"
                                f" {100*goals_w[4,2]:.0f}/{100*right_sole[0,2]:.0f}")
                            lines.append(
                                f"pelvis/torso lean {rows[-1]['root_forward_lean_rad']*57.3:+.0f}/"
                                f"{rows[-1]['torso_forward_lean_rad']*57.3:+.0f} deg"
                                f"   box x {100*displacement:+.1f} cm"
                                f"   hand-box L/R {hand_box_forces[0][0]:+.0f}/{hand_box_forces[1][0]:+.0f} N")
                        elif heavy_box_task:
                            lines.append(
                                f"left foot x "
                                f"{100*(target[0,0]-(initial_goal_xyz[3,0] if initial_goal_xyz is not None else 0.)):+.0f}/"
                                f"{100*(foot[0,0]-(initial_goal_xyz[3,0] if initial_goal_xyz is not None else 0.)):+.0f}"
                                f"  z {100*target[0,2]:.1f}/{100*foot[0,2]:.1f} cm"
                                f"   lean {rows[-1]['root_forward_lean_rad']*57.3:+.0f} deg"
                                f"   box x {100*displacement:+.1f} cm"
                                f"   hand-box x {hand_box_forces[0][0]+hand_box_forces[1][0]:+.0f} N")
                        elif right_foot_push_task:
                            lines.append(
                                f"left stance {ff[0,2]:.0f} N"
                                f"   right foot-box {right_foot_box_force.norm():.1f} N"
                                f"   right wall {forces[1][1]:.0f} N"
                                f"   box x {100*displacement:+.1f} cm")
                        else:
                            lines.append(
                                f"feet z L {100*target[0,2]:.1f}/{100*foot[0,2]:.1f}"
                                f" R {100*right_target[0,2]:.1f}/{100*right_sole[0,2]:.1f} cm"
                                + (f"   right wall {forces[1][1]:.0f} N" if wall_box_task else
                                   f"   hand force {forces[0][2]:.0f}/{forces[1][2]:.0f} N")
                                + f"   box {'x' if wall_box_task else 'y'} {100*displacement:+.1f} cm")
                        line_height = 25
                        text_width = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines)
                        x = image.width - text_width - 28
                        y = 12
                        draw.rounded_rectangle((x - 10, y - 6, image.width - 8,
                                                y + len(lines)*line_height + 2),
                                               radius=7, fill=(9, 15, 24, 145))
                        for i, line in enumerate(lines):
                            draw.text((x, y + i*line_height), line, font=font,
                                      fill="white" if i == 0 else (160, 230, 255))
                    else:
                        draw.rectangle((0, 0, image.width, 100), fill=(9, 15, 24, 205))
                        draw.text((12, 5), f"{args.video_label or args.controller} | t {(t+1)*.02:.1f}s"
                                  f" | foot error {100*rows[-1]['foot_error_m']:.1f} cm",
                                  font=font, fill="white")
                        if initial_goal_xyz is not None:
                            draw.text((12, 31),
                                f"root y goal/actual {100*(goals_w[0,1]-initial_goal_xyz[0,1]):+.0f}/"
                                f"{100*(robot.data.root_pos_w[0,1]-initial_goal_xyz[0,1]):+.0f} cm"
                                f" | hands x L {100*(goals_w[1,0]-initial_goal_xyz[1,0]):+.0f}/"
                                f"{100*(hands[0,0]-initial_goal_xyz[1,0]):+.0f}"
                                f" R {100*(goals_w[2,0]-initial_goal_xyz[2,0]):+.0f}/"
                                f"{100*(hands[1,0]-initial_goal_xyz[2,0]):+.0f} cm",
                                font=font, fill=(120, 225, 255))
                        draw.text((12, 58),
                            f"feet z goal/actual L {100*target[0,2]:.1f}/{100*foot[0,2]:.1f}"
                            f" R {100*right_target[0,2]:.1f}/{100*right_sole[0,2]:.1f} cm"
                            f" | hand force {forces[0][2]:.0f}/{forces[1][2]:.0f} N"
                            f" | block {100*displacement:+.1f} cm", font=font, fill=(120, 225, 255))
                    writer.append_data(np.asarray(image))
                if args.stop_on_fall and (rows[-1]["root_height_m"] < .5 or
                                          rows[-1]["root_tilt_rad"] > fall_tilt_limit):
                    print(f"first fall at step {t}; stopping rollout", flush=True)
                    break
                if args.stop_on_instability and (rows[-1]["root_height_m"] < .65 or
                                                 rows[-1]["root_tilt_rad"] > instability_tilt_limit):
                    print(f"first instability at step {t}; stopping rollout", flush=True)
                    break
        print(f"rollout loop exited after {len(rows)} steps", flush=True)
        arr = lambda key: np.asarray([r[key] for r in rows])
        block_edge_clearance = None
        block_max_tilt = None
        if block is not None:
            block_max_tilt = max(float(np.arccos(np.clip(1 - 2 * (r["block_quat_wxyz"][1] ** 2 +
                                                                 r["block_quat_wxyz"][2] ** 2), -1, 1)))
                                 for r in rows)
            # Positive means the entire yawed box footprint is beyond the
            # negative-y side edge of the tabletop.
            w, x, y, z = (float(v) for v in block.data.root_quat_w[0])
            half_x, half_y = args.block_size_x / 2, args.block_size_y / 2
            y_extent = abs(2 * (x * y + z * w)) * half_x
            y_extent += abs(1 - 2 * (x * x + z * z)) * half_y
            if not wall_box_task and not heavy_box_task:
                edge_y = float(cfg.task.objects[0].pos[1] - cfg.task.objects[0].size[1] / 2)
                block_edge_clearance = edge_y - (float(block.data.root_pos_w[0, 1]) + y_extent)
        print("building evaluation summary", flush=True)
        foot_material = env.base_env.randomizations.get("perturb_body_materials")
        foot_material_values = (robot.data.body_materials[0, foot_material.shape_ids]
                                if foot_material is not None else None)
        drawer_pull_rows = (rows if dataset_prefix.startswith("two_drawer_dynamic")
                            else rows[275:340]) if drawer_task else []
        summary = {"checkpoint": str(args.checkpoint.resolve()),
            "matched_scene_checkpoint": (str(args.matched_scene_checkpoint.resolve())
                                         if args.matched_scene_checkpoint else None),
            "robot_name": str(cfg.task.robot.name),
            "controller": args.controller,
            "reference_torque_audit": (str(args.reference_torque_audit.resolve())
                                        if args.reference_torque_audit else None),
            "reference_torque_gain": (args.reference_torque_gain
                                       if args.reference_torque_audit else None),
            "policy_phase": str(cfg.algo.phase),
            "policy_obs_only": bool(student_only or cfg.algo.direct_policy),
            "ceer_checkpoint": str(args.ceer_checkpoint.resolve()) if baseline is not None else None,
            "ceer_checkpoint_sha256": baseline.checkpoint_sha256 if baseline is not None else None,
            "ceer_reference_npz": str(args.ceer_reference_npz.resolve()) if baseline is not None else None,
            "split": args.split,
            "dataset": dataset_prefix,
            "seed": args.seed,
            "fixed_foot_friction": args.fixed_foot_friction,
            "leg_reference_blend": args.leg_reference_blend,
            "arm_reference_blend": args.arm_reference_blend,
            "arm_policy_checkpoint": str(args.arm_policy_checkpoint.resolve()) if args.arm_policy_checkpoint else None,
            "leg_policy_checkpoint": str(args.leg_policy_checkpoint.resolve()) if args.leg_policy_checkpoint else None,
            "foot_static_friction_override": args.foot_static_friction,
            "foot_dynamic_friction_override": args.foot_dynamic_friction,
            "actual_foot_material_mean": (foot_material_values.mean(0).tolist()
                                          if foot_material_values is not None else None),
            "no_table_contact": args.no_table_contact,
            "no_wall_contact": bool(args.no_table_contact and wall_box_task),
            "block_init_x_override_m": args.block_init_x,
            "block_init_y_override_m": args.block_init_y,
            "block_mass_override_kg": args.block_mass_override,
            "block_friction_override": args.block_friction_override,
            "block_linear_damping_override": args.block_linear_damping,
            "block_size_override_m": [args.block_size_x_override,
                                       args.block_size_y_override,
                                       args.block_height_override],
            "table_width_m": (float(cfg.task.objects[0].size[1])
                              if cfg.task.objects and "size" in cfg.task.objects[0] else None),
            "external_target_npz": str(args.external_target_npz) if args.external_target_npz else None,
            "external_goals_npz": str(args.external_goals_npz) if args.external_goals_npz else None,
            "external_goals_sha256": (hashlib.sha256(args.external_goals_npz.read_bytes()).hexdigest()
                                      if args.external_goals_npz else
                                      explicit_goal_hasher.hexdigest() if explicit_goal_hasher else None),
            "motion_id": args.motion_id,
            "door_object_reference": str(args.door_object_reference.resolve()) if args.door_object_reference else None,
            "door_object_thresholds": ({"pickup_m": .075, "handoff_m": args.door_object_handoff_distance_m,
                                        "placement_left_hand_m": .12,
                                        "placement_object_footprint_half_width_m": .0875,
                                        "opening_deg": args.door_object_open_angle_deg}
                                       if door_object_reference is not None else None),
            "door_object_initial_xyz_m": door_object_initial.tolist() if door_object_reference is not None else None,
            "door_object_shelf_xyz_m": door_object_shelf.tolist() if door_object_reference is not None else None,
            "door_object_events": door_object_events if door_object_reference is not None else None,
            "door_object_final_state": door_object_state if door_object_reference is not None else None,
            "door_object_transfer_success": door_object_state == "shelf" if door_object_reference is not None else None,
            "initial_reference_t": initial_reference_t,
            "initial_reference_length": initial_reference_length,
            "initial_root_xyz_m": initial_root_xyz.tolist(),
            "initial_root_quat_wxyz": initial_root_quat.tolist(),
            "initial_joint_pos_rad": initial_joint_pos.tolist(),
            "robot_joint_names": list(robot.joint_names),
            "actual_max_episode_length": actual_max_episode_length,
            "termination_names": termination_names,
            "steps": len(rows), "requested_steps": args.steps,
            "first_fall_step": next((r["step"] for r in rows if r["root_height_m"] < .5
                                     or r["root_tilt_rad"] > fall_tilt_limit), None),
            "first_instability_step": next((r["step"] for r in rows if r["root_height_m"] < .65
                                            or r["root_tilt_rad"] > instability_tilt_limit), None),
            "mean_foot_error_m": float(arr("foot_error_m").mean()),
            "mean_root_error_m": float(arr("root_error_m").mean()),
            "mean_left_hand_error_m": float(arr("left_hand_error_m").mean()),
            "mean_right_hand_error_m": float(arr("right_hand_error_m").mean()),
            "mean_right_foot_error_m": float(arr("right_foot_error_m").mean()),
            "left_lift_tracking_fraction": float((arr("left_sole_z_m")[arr("target_sole_z_m") > .012] > .012).mean())
                if (arr("target_sole_z_m") > .012).any() else None,
            "right_lift_tracking_fraction": float((arr("right_sole_z_m")[arr("target_right_sole_z_m") > .012] > .012).mean())
                if (arr("target_right_sole_z_m") > .012).any() else None,
            "max_foot_lift_m": float(arr("left_sole_z_m").max()),
            "max_right_foot_lift_m": float(arr("right_sole_z_m").max()),
            "targeted_max_left_foot_lift_m": (
                float(arr("left_sole_z_m")[arr("target_sole_z_m") > .015].max())
                if (arr("target_sole_z_m") > .015).any() else None),
            "targeted_max_right_foot_lift_m": (
                float(arr("right_sole_z_m")[arr("target_right_sole_z_m") > .015].max())
                if (arr("target_right_sole_z_m") > .015).any() else None),
            "target_max_foot_lift_m": float(arr("target_sole_z_m").max()),
            "right_support_fraction": float((arr("right_foot_force_z_N") > 30).mean()),
            "left_support_fraction": float((arr("left_foot_force_z_N") > 30).mean()),
            "right_foot_box_contact_fraction": float((arr("right_foot_box_force_N") > .1).mean()) if right_foot_push_task else None,
            "right_foot_box_contact_frames": int((arr("right_foot_box_force_N") > .1).sum()) if right_foot_push_task else None,
            "right_foot_box_peak_force_N": float(arr("right_foot_box_force_N").max()) if right_foot_push_task else None,
            "right_ankle_pitch_box_contact_fraction": float((arr("right_ankle_pitch_box_force_N") > 5).mean()) if right_foot_push_task else None,
            "right_knee_box_contact_fraction": float((arr("right_knee_box_force_N") > 5).mean()) if right_foot_push_task else None,
            "left_hand_box_contact_fraction": float((np.abs(arr("left_hand_box_force_x_N")) > 5).mean()) if heavy_box_task else None,
            "right_hand_box_contact_fraction": float((np.abs(arr("right_hand_box_force_x_N")) > 5).mean()) if heavy_box_task else None,
            "both_hands_box_contact_fraction": float(((np.abs(arr("left_hand_box_force_x_N")) > 5) &
                                                        (np.abs(arr("right_hand_box_force_x_N")) > 5)).mean()) if heavy_box_task else None,
            "hand_box_peak_force_x_N": float(np.abs(arr("left_hand_box_force_x_N") + arr("right_hand_box_force_x_N")).max()) if heavy_box_task else None,
            "ground_reaction_peak_x_N": float(arr("ground_reaction_x_N").max()) if heavy_box_task else None,
            "max_forward_lean_rad": float(arr("root_forward_lean_rad").max()) if heavy_box_task else None,
            "max_torso_forward_lean_rad": float(arr("torso_forward_lean_rad").max()) if heavy_box_task else None,
            "max_forward_lean_with_both_hands_rad": (
                float(arr("root_forward_lean_rad")[(np.abs(arr("left_hand_box_force_x_N")) > 5) &
                                                     (np.abs(arr("right_hand_box_force_x_N")) > 5)].max())
                if alternating_heavy_box_task and ((np.abs(arr("left_hand_box_force_x_N")) > 5) &
                                                   (np.abs(arr("right_hand_box_force_x_N")) > 5)).any() else None),
            "left_swing_right_stance_fraction": (
                float((arr("right_foot_force_z_N")[arr("target_sole_z_m") > .015] > 30).mean())
                if alternating_heavy_box_task and (arr("target_sole_z_m") > .015).any() else None),
            "right_swing_left_stance_fraction": (
                float((arr("left_foot_force_z_N")[arr("target_right_sole_z_m") > .015] > 30).mean())
                if alternating_heavy_box_task and (arr("target_right_sole_z_m") > .015).any() else None),
            "both_hands_support_fraction": float(((arr("left_hand_table_force_z_N") > 5) &
                                                  (arr("right_hand_table_force_z_N") > 5)).mean()),
            "right_wall_support_fraction": float((arr("right_hand_wall_force_y_N") > 5).mean()),
            "left_wall_contact_fraction": float((np.abs(arr("left_hand_wall_force_y_N")) > 5).mean()),
            "min_root_height_m": float(arr("root_height_m").min()),
            "max_root_tilt_rad": float(arr("root_tilt_rad").max()),
            "max_hand_drift_m": float(max(max(r["hand_drift_m"]) for r in rows)),
            "max_right_foot_drift_m": float(arr("right_foot_drift_m").max()) if not right_foot_push_task else None,
            "max_left_foot_drift_m": float(arr("left_foot_drift_m").max()) if right_foot_push_task else None,
            "episode_done_count": int(arr("episode_done").sum()),
            "foot_force_max_requested_N": args.foot_force_max,
            "foot_force_max_applied_N": float(arr("applied_left_foot_force_N").max()),
            "block_initial_xyz_m": initial_block_pos.tolist() if block is not None else None,
            "block_final_xyz_m": block.data.root_pos_w[0].tolist() if block is not None else None,
            "block_final_quat_wxyz": block.data.root_quat_w[0].tolist() if block is not None else None,
            "block_table_edge_clearance_m": block_edge_clearance,
            "block_max_tilt_rad": block_max_tilt,
            "block_lateral_displacement_m": (float(block.data.root_pos_w[0, 1]-initial_block_pos[1])
                                              if block is not None else None),
            "block_parallel_displacement_m": (float(block.data.root_pos_w[0, 0]-initial_block_pos[0])
                                              if block is not None and (wall_box_task or heavy_box_task) else None),
            "block_vertical_displacement_m": (float(block.data.root_pos_w[0, 2]-initial_block_pos[2])
                                               if block is not None else None),
            "block_normal_displacement_m": (float(block.data.root_pos_w[0, 1]-initial_block_pos[1])
                                            if block is not None and (wall_box_task or heavy_box_task) else None),
            "crate_initial_xyz_m": initial_crate_pos.tolist() if crate is not None else None,
            "crate_final_xyz_m": crate.data.root_pos_w[0].tolist() if crate is not None else None,
            "insert_initial_xyz_m": initial_insert_pos.tolist() if insert is not None else None,
            "insert_final_xyz_m": insert.data.root_pos_w[0].tolist() if insert is not None else None,
            "insert_lift_m": (float(insert.data.root_pos_w[0, 2]-initial_insert_pos[2])
                              if insert is not None else None),
            "insert_lift_from_settled_m": (float(insert.data.root_pos_w[0, 2] -
                min(r["insert_xyz_m"][2] for r in rows[:min(75,len(rows))]))
                if insert is not None else None),
            "insert_planar_drift_m": (float(torch.linalg.vector_norm(
                insert.data.root_pos_w[0,:2]-initial_insert_pos[:2])) if insert is not None else None),
            "insert_max_tilt_rad": (max(float(np.arccos(np.clip(1-2*(r["insert_quat_wxyz"][1]**2+
                r["insert_quat_wxyz"][2]**2),-1,1))) for r in rows) if insert is not None else None),
            "crate_planar_drift_m": (float(torch.linalg.vector_norm(
                crate.data.root_pos_w[0,:2]-initial_crate_pos[:2])) if crate is not None else None),
            "drawer_initial_position_m": initial_drawer_position,
            "pedal_initial_position_m": initial_pedal_position,
            "pedal_asset_sha256": (hashlib.sha256(Path(next(o.get("usd_path",
                ROOT / "artifacts/spring_pedal/assets/spring_pedal.usd") for o in cfg.task.objects
                if o.get("name") == "spring_pedal")).read_bytes()).hexdigest() if pedal_task else None),
            "pedal_max_stroke_m": float((-arr("pedal_position_m")).max()) if pedal_task else None,
            "pedal_first_frame_stroke_m": float(-rows[0]["pedal_position_m"]) if pedal_task else None,
            "pedal_final_stroke_m": float(-rows[-1]["pedal_position_m"]) if pedal_task else None,
            "pedal_final_foot_force_N": float(rows[-1]["pedal_foot_force_N"]) if pedal_task else None,
            "pedal_hold_frames_3cm": int(((-arr("pedal_position_m")) >= .03).sum()) if pedal_task else None,
            "pedal_foot_contact_frames_10N": int((arr("pedal_foot_force_N") >= 10).sum()) if pedal_task else None,
            "drawer_asset_usd": (str(cfg.task.objects[0].usd_path) if drawer_task else None),
            "drawer_asset_sha256": (hashlib.sha256(Path(cfg.task.objects[0].usd_path).read_bytes()).hexdigest()
                                    if drawer_task else None),
            "drawer_final_position_m": (float(env.scene["drawer_cabinet"].data.joint_pos[0, drawer_joint_id])
                                         if drawer_task else None),
            "drawer_pull_m": (initial_drawer_position-float(env.scene["drawer_cabinet"].data.joint_pos[0, drawer_joint_id])
                              if drawer_task else None),
            "drawer_handle_spring_grasp": args.drawer_handle_spring_grasp,
            "drawer_grasp_stiffness_N_per_m": (args.drawer_grasp_stiffness if args.drawer_handle_spring_grasp else None),
            "drawer_grasp_damping_Ns_per_m": (args.drawer_grasp_damping if args.drawer_handle_spring_grasp else None),
            "drawer_grasp_x_only": args.drawer_grasp_x_only,
            "cabinet_hand_contact_fraction": (float((arr("cabinet_hand_force_N")>5).mean())
                                              if drawer_task else None),
            "drawer_hand_contact_fraction": (float((arr("drawer_hand_force_N")>5).mean())
                                             if drawer_task else None),
            "cabinet_support_during_pull_fraction": (sum(r["cabinet_hand_force_N"] > 5 for r in drawer_pull_rows)
                                                     / len(drawer_pull_rows) if drawer_pull_rows else 0. if drawer_task else None),
            "drawer_hand_contact_during_pull_fraction": (sum(r["drawer_hand_force_N"] > 5 for r in drawer_pull_rows)
                                                        / len(drawer_pull_rows) if drawer_pull_rows else 0. if drawer_task else None),
            "left_foot_crate_support_fraction": (float((arr("left_foot_crate_force_z_N")
                [arr("target_sole_z_m") > .15] > 20).mean())
                if extraction_task and (arr("target_sole_z_m") > .15).any() else None),
            "rows": rows}
        if block is not None and heavy_box_task:
            stable_end = summary["first_instability_step"]
            stable_end = len(rows) - 1 if stable_end is None else max(0, stable_end - 1)
            summary["block_parallel_displacement_before_instability_m"] = float(
                rows[stable_end]["block_xyz_m"][0] - initial_block_pos[0])
        if pedal_task:
            summary["task_success"] = bool(
                len(rows) == args.steps and summary["pedal_max_stroke_m"] >= .03
                and summary["pedal_hold_frames_3cm"] >= 20
                and summary["pedal_final_stroke_m"] <= summary["pedal_first_frame_stroke_m"] + .005
                and summary["pedal_final_foot_force_N"] < 10.
                and summary["pedal_foot_contact_frames_10N"] >= 20
                and summary["right_wall_support_fraction"] > .35
                and summary["left_support_fraction"] > .75
                and summary["min_root_height_m"] > .6
                and summary["max_root_tilt_rad"] < .45)
        elif drawer_task:
            summary["task_success"] = bool(
                len(rows)==args.steps and summary["drawer_pull_m"]>.08
                and summary["cabinet_hand_contact_fraction"]>.35
                and summary["drawer_hand_contact_fraction"]>.08
                and summary["cabinet_support_during_pull_fraction"]>.35
                and summary["drawer_hand_contact_during_pull_fraction"]>.35
                and summary["left_support_fraction"]>.8
                and summary["right_support_fraction"]>.8
                and summary["min_root_height_m"]>.6
                and summary["max_root_tilt_rad"]<.40)
            if args.kinematic_preview:
                summary["task_success"] = None
                summary["preview_kind"] = "kinematic_reference_with_prescribed_drawer_slide"
        elif extraction_task:
            shared_extraction_checks = bool(
                len(rows) == args.steps and summary["insert_lift_from_settled_m"] > .06
                and summary["insert_planar_drift_m"] < .05
                and summary["insert_max_tilt_rad"] < .35
                and summary["left_foot_crate_support_fraction"] > .7
                and summary["left_wall_contact_fraction"] > .7
                and summary["right_support_fraction"] > .8
                and summary["min_root_height_m"] > .6 and summary["max_root_tilt_rad"] < .35)
            summary["task_success_strict_5cm"] = bool(
                shared_extraction_checks and summary["crate_planar_drift_m"] < .05)
            summary["task_success"] = bool(
                shared_extraction_checks and summary["crate_planar_drift_m"] < .07)
        elif heavy_box_task:
            if incline_box_task:
                dx = summary["block_parallel_displacement_m"] or 0.
                dz = summary["block_vertical_displacement_m"] or 0.
                summary["uphill_success"] = bool(
                    len(rows) == args.steps and dx >= args.min_uphill_distance_m
                    and dz > dx * np.tan(np.deg2rad(7.)) - .025
                    and abs(summary["block_lateral_displacement_m"] or 0.) < .06
                    and block_max_tilt < .30
                    and summary["first_fall_step"] is None
                    and summary["min_root_height_m"] > .60
                    and summary["left_support_fraction"] > .7
                    and summary["right_support_fraction"] > .7
                    and summary["left_hand_box_contact_fraction"] > .10
                    and summary["right_hand_box_contact_fraction"] > .10
                    and summary["both_hands_box_contact_fraction"] > .10)
                summary["task_success"] = summary["uphill_success"]
                summary["dual_hand_success"] = summary["uphill_success"]
            elif alternating_heavy_box_task:
                physical_push_checks = bool(
                    block is not None and len(rows) == args.steps
                    and summary["block_parallel_displacement_m"] > .10
                    and abs(summary["block_normal_displacement_m"] or 0.) < .08
                    and block_max_tilt < .25
                    and summary["first_fall_step"] is None
                    and summary["min_root_height_m"] > .55
                    and summary["max_forward_lean_with_both_hands_rad"] is not None
                    and summary["max_forward_lean_with_both_hands_rad"] > .52
                    and (summary["targeted_max_left_foot_lift_m"] or 0.) > .025
                    and (summary["targeted_max_right_foot_lift_m"] or 0.) > .025
                    and (summary["left_lift_tracking_fraction"] or 0.) > .5
                    and (summary["right_lift_tracking_fraction"] or 0.) > .5
                    and (summary["left_swing_right_stance_fraction"] or 0.) > .65
                    and (summary["right_swing_left_stance_fraction"] or 0.) > .65
                    and summary["both_hands_box_contact_fraction"] > .12)
                summary["alternating_push_success"] = physical_push_checks
                summary["task_success"] = physical_push_checks
            else:
                physical_push_checks = bool(
                block is not None and len(rows) == args.steps
                and summary["block_parallel_displacement_m"] > .08
                and abs(summary["block_normal_displacement_m"] or 0.) < .08
                and block_max_tilt < .25
                and summary["min_root_height_m"] > .55
                and summary["max_forward_lean_rad"] > .18
                and summary["left_support_fraction"] > .7
                and summary["right_support_fraction"] > .7)
                left_hand_fraction = summary["left_hand_box_contact_fraction"]
                right_hand_fraction = summary["right_hand_box_contact_fraction"]
                # Preserve the original task's predeclared criterion.
                summary["task_success"] = bool(
                    physical_push_checks and max(left_hand_fraction, right_hand_fraction) > .25
                    and left_hand_fraction + right_hand_fraction > .35)
                summary["dual_hand_success"] = bool(
                    physical_push_checks and left_hand_fraction > .2 and right_hand_fraction > .2)
        elif wall_box_task:
            summary["task_success"] = bool(
                block is not None and len(rows) == args.steps
                and summary["block_parallel_displacement_m"] > .08
                and abs(summary["block_normal_displacement_m"]) < .08
                and block_max_tilt < .35
                and abs(float(block.data.root_pos_w[0, 2]-initial_block_pos[2])) < .025
                and summary["min_root_height_m"] > .6 and summary["max_root_tilt_rad"] < .35
                and summary["left_support_fraction" if right_foot_push_task else "right_support_fraction"] > .95
                and (summary["max_left_foot_drift_m"] < .08 if right_foot_push_task else True)
                and (summary["right_foot_box_contact_frames"] >= 5 if right_foot_push_task else True)
                and summary["right_wall_support_fraction"] > .7
                and summary["left_wall_contact_fraction"] < .05)
        else:
            summary["task_success"] = bool(
                block is not None and len(rows) == args.steps and block_edge_clearance > 0 and block_max_tilt < .35
                and abs(float(block.data.root_pos_w[0, 2]-initial_block_pos[2])) < .02
                and summary["min_root_height_m"] > .6 and summary["max_root_tilt_rad"] < .35
                and summary["right_support_fraction"] > .95
                and summary["both_hands_support_fraction"] > .9)
        ground_reach_scene = bool(block is None or (
            heavy_box_task and initial_block_pos is not None and float(initial_block_pos[0]) > 5.0
            and summary["left_hand_box_contact_fraction"] == 0.0
            and summary["right_hand_box_contact_fraction"] == 0.0))
        summary["ground_reach_success"] = bool(
            ground_reach_scene and summary["first_instability_step"] is None and len(rows) == args.steps
            and summary["left_lift_tracking_fraction"] is not None
            and summary["left_lift_tracking_fraction"] > .8
            and (summary["right_lift_tracking_fraction"] is None
                 or summary["right_lift_tracking_fraction"] > .8)
            and summary["mean_root_error_m"] < .05
            and summary["mean_left_hand_error_m"] < .08
            and summary["mean_right_hand_error_m"] < .08)
        print("saving evaluation summary", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)
        completed = True
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
    finally:
        if writer is not None:
            writer.close()
        if completed:
            # Isaac Sim 4.5 can leave non-daemon Kit threads alive after its
            # close call; process exit releases the single-env CUDA context.
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(0)
        app.close()


if __name__ == "__main__":
    main()
