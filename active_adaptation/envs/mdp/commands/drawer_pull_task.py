"""Cabinet-supported physical drawer pull with five-XYZ actor commands."""
import torch
from active_adaptation.envs.mdp import reward
from .table_push_task import TablePushTaskCommand


class DrawerPullTaskCommand(TablePushTaskCommand):
    supports_symmetry_augmentation = False

    @reward
    def drawer_right_hand_tracking(self):
        target=self.target_positions_world()[:,2]
        actual=self.asset.data.body_pos_w[:,self.right_hand_asset]
        error=torch.linalg.vector_norm(target-actual,dim=-1,keepdim=True)
        return torch.exp(-error/.045)

    @reward
    def cabinet_hand_support(self):
        sensor=self.env.scene["eval_left_hand_cabinet_force"]
        force=sensor.data.net_forces_w.reshape(self.num_envs,-1,3).sum(1)
        hand=self.asset.data.body_pos_w[:,self.hand_asset]
        local=hand-self.env.scene.env_origins
        near=((local[:,0:1]-.38).abs()<.11)&((local[:,1:2]-.155).abs()<.11)&((local[:,2:3]-.80).abs()<.10)
        return near.float()*torch.clamp(force[:,2:3].abs()/18.,0.,1.)

    @reward
    def drawer_hand_contact(self):
        sensor=self.env.scene["eval_right_hand_drawer_force"]
        force=sensor.data.net_forces_w.reshape(self.num_envs,-1,3).sum(1)
        hand=self.asset.data.body_pos_w[:,self.right_hand_asset]
        slide=self.env.scene["drawer_cabinet"].data.joint_pos[:,0:1]
        local=hand-self.env.scene.env_origins
        # The handle includes two struts spanning x=.39..55, not only the bar.
        near=(local[:,0:1]>.33+slide)&(local[:,0:1]<.61+slide)&((local[:,1:2]+.07).abs()<.12)&((local[:,2:3]-.72).abs()<.18)
        return near.float()*torch.clamp(torch.linalg.vector_norm(force,dim=-1,keepdim=True)/12.,0.,1.)

    @reward
    def drawer_progress(self):
        drawer=self.env.scene["drawer_cabinet"]
        displacement=(-drawer.data.joint_pos[:,0:1]).clamp(0.,.20)
        target_x=self.target_positions_world()[:,2,0:1]-self.env.scene.env_origins[:,0:1]
        phase=(self.t[:,None]>=280).float()*((.50-target_x)/.12).clamp(0.,1.)
        return phase*torch.clamp(displacement/.12,0.,1.)

    @reward
    def drawer_target_tracking(self):
        drawer=self.env.scene["drawer_cabinet"]
        displacement=(-drawer.data.joint_pos[:,0:1]).clamp(0.,.20)
        target_x=self.target_positions_world()[:,2,0:1]-self.env.scene.env_origins[:,0:1]
        desired=(self.t[:,None]>=280).float()*(.50-target_x).clamp(0.,.14)
        return torch.exp(-torch.abs(displacement-desired)/.04)

    @reward
    def drawer_open_success(self):
        displacement=-self.env.scene["drawer_cabinet"].data.joint_pos[:,0:1]
        phase=(self.t[:,None]>=310).float()
        upright=(self.asset.data.root_pos_w[:,2:3]>.6).float()
        return phase*upright*(displacement>.08).float()


class TwoDrawerDynamicPullCommand(DrawerPullTaskCommand):
    """Reward the frame-zero lower-handle pull in the two-drawer cabinet."""

    @reward
    def cabinet_hand_support(self):
        sensor = self.env.scene["eval_left_hand_cabinet_force"]
        force = sensor.data.net_forces_w.reshape(self.num_envs, -1, 3).sum(1)
        hand = self.asset.data.body_pos_w[:, self.hand_asset] - self.env.scene.env_origins
        near = ((hand[:, 0:1] - .34).abs() < .09) & ((hand[:, 1:2] - .185).abs() < .10) & ((hand[:, 2:3] - .82).abs() < .10)
        return near.float() * torch.clamp(torch.linalg.vector_norm(force, dim=-1, keepdim=True) / 18., 0., 1.)

    @reward
    def drawer_hand_contact(self):
        sensor = self.env.scene["eval_right_hand_drawer_force"]
        force = sensor.data.net_forces_w.reshape(self.num_envs, -1, 3).sum(1)
        hand = self.asset.data.body_pos_w[:, self.right_hand_asset] - self.env.scene.env_origins
        slide = self.env.scene["drawer_cabinet"].data.joint_pos[:, 0:1]
        near = ((hand[:, 0:1] - (.31 + slide)).abs() < .07) & ((hand[:, 1:2] + .055).abs() < .09) & ((hand[:, 2:3] - .45).abs() < .09)
        return near.float() * torch.clamp(torch.linalg.vector_norm(force, dim=-1, keepdim=True) / 12., 0., 1.)

    @reward
    def drawer_progress(self):
        displacement = (-self.env.scene["drawer_cabinet"].data.joint_pos[:, 0:1]).clamp(0., .24)
        target_x = self.target_positions_world()[:, 2, 0:1] - self.env.scene.env_origins[:, 0:1]
        desired = (.29 - target_x).clamp(0., .10)
        return (desired / .10) * torch.clamp(displacement / .10, 0., 1.)

    @reward
    def drawer_target_tracking(self):
        displacement = (-self.env.scene["drawer_cabinet"].data.joint_pos[:, 0:1]).clamp(0., .24)
        target_x = self.target_positions_world()[:, 2, 0:1] - self.env.scene.env_origins[:, 0:1]
        desired = (.29 - target_x).clamp(0., .10)
        return torch.exp(-torch.abs(displacement - desired) / .04)

    @reward
    def drawer_open_success(self):
        displacement = -self.env.scene["drawer_cabinet"].data.joint_pos[:, 0:1]
        phase = (self.t[:, None] >= 65).float()
        upright = (self.asset.data.root_pos_w[:, 2:3] > .60).float()
        return phase * upright * (displacement > .08).float()
