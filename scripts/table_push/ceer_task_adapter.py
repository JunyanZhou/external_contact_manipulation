"""Run frozen CEER weights through the table task's unchanged motor interface."""
import hashlib
from pathlib import Path
import re

import numpy as np
import torch

from active_adaptation.utils.ceer_residual import FrozenCEER, TARGET_JOINT_NAMES
from active_adaptation.utils.sonic import JOINT_NAMES
from active_adaptation.utils.usd_kinematics import USDKinematics


class CEERTaskAdapter:
    def __init__(self, env, checkpoint: Path, reference: Path, target: np.ndarray,
                 mode: str = "joint_reference"):
        self.env = env
        self.robot = env.robot
        self.action = env.action_manager
        self.mode = mode
        self.checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        data = np.load(reference)
        if len(data["dof_pos"]) != len(target) or not np.allclose(
                data["target_foot_pos_b"], target, atol=1e-6):
            raise ValueError("CEER reference does not encode the exact external target clip")
        names = [str(x) for x in data["joint_names"]]
        self.reference = torch.as_tensor(data["dof_pos"][:, [names.index(n) for n in JOINT_NAMES]].copy(),
                                         dtype=torch.float32, device=env.device)
        self.root_pos = torch.as_tensor(data["root_pos"].copy(), dtype=torch.float32,
                                        device=env.device)
        root_vel = np.gradient(data["root_pos"], .02, axis=0)
        self.root_velocity = torch.as_tensor(root_vel.copy(), dtype=torch.float32, device=env.device)
        self.kinematics = USDKinematics(self.robot.cfg.spawn.usd_path, JOINT_NAMES, env.device)
        self.robot_ids = [self.robot.joint_names.index(n) for n in JOINT_NAMES]
        task_order = [JOINT_NAMES.index(n) for n in self.action.joint_names]
        self.task_order = torch.as_tensor(task_order, device=env.device)
        self.joint_target_order = torch.as_tensor([JOINT_NAMES.index(n) for n in TARGET_JOINT_NAMES],
                                                   device=env.device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        patterns = dict(state["cfg"].task.action.action_scaling)
        source_scale = []
        for name in JOINT_NAMES:
            matches = [float(scale) for pattern, scale in patterns.items() if re.fullmatch(pattern, name)]
            if len(matches) != 1:
                raise ValueError(f"Expected one CEER action scale for {name}: {matches}")
            source_scale.append(matches[0])
        self.source_scale = torch.tensor(source_scale, device=env.device)
        self.task_scale = self.action.action_scaling
        if self.task_scale.shape != (29,):
            raise ValueError("Task must use the same 29-joint motor interface")
        normalization = FrozenCEER(checkpoint, env.device)
        self.policy_loc = normalization.policy_loc
        self.policy_scale = normalization.policy_scale
        if mode == "joint_reference":
            self.model = normalization
        elif mode == "native":
            from scripts.export_whole_body_student import build_policy_state, build_student_rollout_from_state
            native_state = build_policy_state(state)
            self.model = build_student_rollout_from_state(native_state)
            self.model.load_state_dict(native_state, strict=True)
            self.model.to(env.device).eval().requires_grad_(False)
        else:
            raise ValueError(mode)
        self.history = None
        self.previous = None
        self.last_observation = None
        self.last_ceer_action = None
        self.age = 0

    def reset(self):
        q = self.robot.data.joint_pos[:, self.robot_ids]
        self.history = q[:, None].expand(-1, 5, -1).clone()
        self.previous = q.new_zeros((1, 3, 29))
        self.age = 0

    @torch.no_grad()
    def act(self, step: int):
        q = self.robot.data.joint_pos[:, self.robot_ids]
        self.history = self.history.roll(1, dims=1)
        self.history[:, 0] = q
        # CEER's original root/wrist interface has no foot xyz channel. The
        # joint-reference route carries the IK-mapped foot target explicitly.
        ref = self.reference[step:step + 1]
        wrists = self.kinematics.interface(ref)
        root_velocity = self.root_velocity[step:step + 1, :2]
        command = torch.cat((self.root_pos[step:step + 1, 2:3], root_velocity,
                             q.new_ones((1, 1)), q.new_zeros((1, 1)),
                             q.new_full((1, 1), 30.)), dim=-1)
        boot = q.new_full((1, 1), max(0., 1. - self.age / 25.))
        obs = torch.cat((boot, command, wrists, self.robot.data.root_ang_vel_b,
                         self.robot.data.projected_gravity_b, self.history.flatten(1),
                         self.previous.flatten(1)), dim=-1)
        if obs.shape != (1, 257):
            raise RuntimeError(f"CEER observation shape {obs.shape} is incorrect")
        future = torch.as_tensor([min(step + offset, len(self.reference) - 1)
                                  for offset in (0, 2, 4, 8, 16)], device=q.device)
        joint_reference = self.reference[future][:, self.joint_target_order].unsqueeze(0)
        if self.mode == "native":
            raw = self.model((obs - self.policy_loc) / self.policy_scale)
        else:
            raw = self.model(obs, joint_reference)
        if not torch.isfinite(raw).all():
            raise RuntimeError("CEER returned a nonfinite action")
        self.previous = self.previous.roll(1, dims=1)
        self.previous[:, 0] = raw
        self.last_observation = obs
        self.last_ceer_action = raw
        self.age += 1
        # Preserve the *task* motor, filtering, delay, joint order and default
        # pose. Only the policy that proposes its position target changes.
        return (raw * self.source_scale)[:, self.task_order] / self.task_scale
