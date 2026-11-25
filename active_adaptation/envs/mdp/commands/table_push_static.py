"""Deterministic initial poses for frozen-CEER table contact diagnostics."""
from pathlib import Path
import numpy as np
import torch

from .base import Command


class TablePushStaticCommand(Command):
    def __init__(self, env, pose_path, pose_index=0):
        super().__init__(env)
        data = np.load(Path(pose_path))
        self.pose_index = int(pose_index)
        self.poses = torch.as_tensor(data["q_pinocchio"].copy(), device=self.device, dtype=torch.float32)
        self.names = [str(x) for x in data["joint_names"]]
        self.indices = [self.asset.joint_names.index(name) for name in self.names]
        self.joint_pos_boot_protect = self.asset.data.default_joint_pos.clone()

    def sample_init(self, env_ids):
        q = self.poses[self.pose_index]
        root = self.init_root_state[env_ids].clone()
        root[:, :3] = q[:3] + self.env.scene.env_origins[env_ids]
        root[:, 3:7] = q[[6, 3, 4, 5]]  # Pinocchio xyzw -> Isaac wxyz
        root[:, 7:] = 0.
        joints = self.init_joint_pos[env_ids].clone()
        joints[:, self.indices] = q[7:]
        self.joint_pos_boot_protect[env_ids] = joints
        velocity = torch.zeros_like(joints)
        self.asset.write_root_state_to_sim(root, env_ids=env_ids)
        self.asset.write_joint_position_to_sim(joints, env_ids=env_ids)
        self.asset.write_joint_velocity_to_sim(velocity, env_ids=env_ids)
        self.asset.set_joint_position_target(joints, env_ids=env_ids)
        self.asset.write_data_to_sim()
