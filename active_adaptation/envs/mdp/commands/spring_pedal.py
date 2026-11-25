"""Right-hand support and right-foot spring-pedal control."""
import torch
from active_adaptation.envs.mdp import reward
from .wall_box_right_foot import WallBoxRightFootCommand


class SpringPedalCommand(WallBoxRightFootCommand):
    @reward
    def pedal_stroke_tracking(self):
        target_z = self.target_positions_world()[:, 4, 2:3] - self.env.scene.env_origins[:, 2:3]
        desired = (.08 - target_z).clamp(0., .04)
        actual = (-self.env.scene["spring_pedal"].data.joint_pos[:, :1]).clamp(0., .06)
        return torch.exp(-torch.abs(actual - desired) / .015)

    @reward
    def pedal_press_contact(self):
        sensor = self.env.scene["eval_right_foot_pedal_force"]
        force = torch.linalg.vector_norm(sensor.data.force_matrix_w.reshape(self.num_envs, -1, 3).sum(1), dim=-1, keepdim=True)
        target_z = self.target_positions_world()[:, 4, 2:3] - self.env.scene.env_origins[:, 2:3]
        active = (target_z < .075).float()
        return active * torch.clamp(force / 25., 0., 1.)

    @reward
    def braced_press(self):
        hand = self.env.scene["eval_right_hand_wall_force"].data.force_matrix_w.reshape(self.num_envs, -1, 3).sum(1)
        hand_load = torch.clamp(hand[:, 1:2] / 15., 0., 1.)
        pedal = (-self.env.scene["spring_pedal"].data.joint_pos[:, :1]).clamp(0., .06)
        target_z = self.target_positions_world()[:, 4, 2:3] - self.env.scene.env_origins[:, 2:3]
        return (target_z < .075).float() * hand_load * torch.clamp(pedal / .04, 0., 1.)

    @reward
    def pedal_release_tracking(self):
        target_z = self.target_positions_world()[:, 4, 2:3] - self.env.scene.env_origins[:, 2:3]
        stroke = (-self.env.scene["spring_pedal"].data.joint_pos[:, :1]).clamp(0., .06)
        return (target_z > .10).float() * torch.exp(-stroke / .015)
