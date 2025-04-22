from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR

from .perceptive_robot_config import PerceptiveRobotCfg

class PerceptiveRobot(LeggedRobot):
    cfg : PerceptiveRobotCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        self.debug_viz = False  # This would slow down the training
    
    def compute_observations(self):
        """ Computes observations, override in child classes.
        """
        
        # proprioception
        self.obs_buf = torch.cat((
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                ),dim=-1)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        
        if self.cfg.env.num_proprioception_history > 1:
            self.obs_buf = torch.cat((
                self.obs_buf, 
                self.proprioception_history_buf.view(self.num_envs, -1)
            ), dim=-1)
            
            # history proprioception        
            self.proprioception_history_buf = torch.where(
                self.episode_length_buf[:, None, None] <=1,
                torch.stack([self.obs_buf] * self.cfg.env.num_proprioception_history, dim=1),
                torch.cat([
                    self.proprioception_history_buf[:, 1:, :],
                    self.obs_buf.unsqueeze(1)
                ], dim=1)
            )
            
        # # scan dots
        # if self.cfg.terrain.measure_heights:
        #     # TODO: what is the magic number 0.5?
        #     heights = self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights
        #     self.scandots_buf = torch.clip(heights, -1, 1).view(
        #         self.num_envs,
        #         len(self.cfg.terrain.measured_points_x),
        #         len(self.cfg.terrain.measured_points_y)
        #     )
        
        if self.cfg.terrain.measure_heights:
            # TODO: what is the magic number 0.5?
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.33 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            if self.add_noise:
                # TODO: horizontal offsets
                heights += self.height_noise_scale_vec * (2 * torch.rand(self.num_envs, self.num_height_points, device=self.device) - 1)
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        
        assert self.obs_buf.shape[1] == self.cfg.env.num_observations, \
            f"Observation buffer shape {self.obs_buf.shape} does not match expected shape {self.cfg.env.num_observations}"
        
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # Reset history proprioception
        if self.cfg.env.num_proprioception_history > 1:
            self.proprioception_history_buf[env_ids, :, :] = 0.
            
    def _post_physics_step_callback(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        
        super()._post_physics_step_callback()
        
    def _init_buffers(self):
        super()._init_buffers()
        # Initialize history proprioception
        if self.cfg.env.num_proprioception_history > 1:
            self.proprioception_history_buf = torch.zeros(
                self.num_envs, 
                self.cfg.env.num_proprioception_history, 
                self.num_obs, 
                dtype=torch.float, device=self.device, requires_grad=False)
        else:
            self.proprioception_history_buf = None
            
        # scandots
        if self.cfg.terrain.measure_heights:
            self.height_noise_scale_vec = self._get_height_noise_scale_vec()
            
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(self.cfg.env.num_proprioception, device=self.device, dtype=torch.float)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[33:45] = 0. # previous actions
        return noise_vec

    def _get_height_noise_scale_vec(self):
        noise_vec = torch.ones(self.num_height_points, device=self.device, dtype=torch.float)
        noise_vec *= self.cfg.noise.noise_scales.height_measurements * self.cfg.noise.noise_level
        return noise_vec
    
    # --------------------------------------------------------------------------------------------
    # reward functions
    # --------------------------------------------------------------------------------------------
    
    def _reward_feet_clearance(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_footpos_target).view(self.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * foot_leteral_vel, dim=1)
