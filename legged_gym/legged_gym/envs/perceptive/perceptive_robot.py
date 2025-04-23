from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR

from .perceptive_robot_config import PerceptiveRobotCfg

import numpy as np
import cv2

class PerceptiveRobot(LeggedRobot):
    cfg : PerceptiveRobotCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        self.debug_viz = False  # This would slow down the training, only for debugging
        self.debug_depth = False # This would slow down the training, only for debugging
        
    def post_physics_step(self):
        if self.cfg.depth_image.use_depth_image:
            self.update_depth_image()
        
        super().post_physics_step()
        
        if self.viewer and self.enable_viewer_sync and self.cfg.depth_image.use_depth_image and self.debug_depth:
            window_name = "Depth Image of env 0"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            img = self.depth_image_tensors[0].cpu().numpy()  # [120, 160]
            img = -1.0 * img
            img = np.clip(img, 0, 10.0)
            img = img / 10.0 * 255.0
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.imshow(window_name, img)
            cv2.waitKey(1)
            
        if self.viewer and self.enable_viewer_sync and self.debug_viz and self.cfg.depth_image.use_depth_image:
            box_geom = gymutil.WireframeBoxGeometry(0.02, 0.06, 0.02, None, color=(1, 0, 0))
            for i in range(self.num_envs):
                base_pos = (self.root_states[i, :3]).clone()
                cam_x = np.mean(self.cfg.depth_image.camera_pose.x_range)
                cam_y = np.mean(self.cfg.depth_image.camera_pose.y_range)
                cam_z = np.mean(self.cfg.depth_image.camera_pose.z_range)
                cam_rot_pitch = np.random.uniform(self.cfg.depth_image.camera_pose.pitch_range[0],
                                                self.cfg.depth_image.camera_pose.pitch_range[1])
                
                cam_pos = torch.tensor([cam_x, cam_y, cam_z], dtype=torch.float32, device=self.device)
                cam_pos = quat_apply(self.base_quat[i], cam_pos)
                
                cam_quat = self.base_quat[i].clone()
                quat_1 = gymapi.Quat.from_euler_zyx(0, cam_rot_pitch, 0)
                quat_1 = torch.tensor([quat_1.x, quat_1.y, quat_1.z, quat_1.w], dtype=torch.float32, device=self.device)
                cam_quat = quat_mul(cam_quat, quat_1)
                
                box_pose = gymapi.Transform()
                box_pose.p = gymapi.Vec3(base_pos[0]+cam_pos[0], base_pos[1]+cam_pos[1], base_pos[2]+cam_pos[2])
                box_pose.r = gymapi.Quat(cam_quat[0], cam_quat[1], cam_quat[2], cam_quat[3])
                gymutil.draw_lines(box_geom, self.gym, self.viewer, self.envs[i], box_pose)
                
        
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
        
        if self.cfg.terrain.measure_heights:
            # 0.33 is the desired height of the robot base
            # TODO: make it configurable
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.33 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            if self.add_noise:
                # TODO: horizontal offsets
                heights += self.height_noise_scale_vec * (2 * torch.rand(self.num_envs, self.num_height_points, device=self.device) - 1)
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        
        assert self.obs_buf.shape[1] == self.cfg.env.num_observations, \
            f"Observation buffer shape {self.obs_buf.shape} does not match expected shape {self.cfg.env.num_observations}"
    
    def update_depth_image(self):
        self.gym.step_graphics(self.sim)
        # render sensors and refresh camera tensors
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        
        # TODO
        
        self.gym.end_access_image_tensors(self.sim)
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # Reset history proprioception
        if self.cfg.env.num_proprioception_history > 1:
            self.proprioception_history_buf[env_ids, :, :] = 0.
            
    def create_sim(self):
        if self.cfg.depth_image.use_depth_image:
            # ref: https://forums.developer.nvidia.com/t/why-it-returns-1-when-i-tried-to-create-camera-sensor/218083
            self.graphics_device_id = self.sim_device_id
            
        super().create_sim()
            
    def _post_physics_step_callback(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        
        super()._post_physics_step_callback()
        
    def _create_envs(self):
        print("Creating environments ...")
        
        super()._create_envs()
        
        self.camera_handles = []
        self.depth_image_tensors = []
        if self.cfg.depth_image.use_depth_image:
            for i in range(self.num_envs):
                env_handle = self.envs[i]
                actor_handle = self.actor_handles[i]
                root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)
                
                base_handle = self.gym.find_actor_rigid_body_handle(env_handle, actor_handle, "base")
                
                if base_handle != root_handle:
                    raise RuntimeError("Base handle is not the root handle")
                
                # add camera
                cam_props = gymapi.CameraProperties()
                cam_props.width = self.cfg.depth_image.image_width
                cam_props.height = self.cfg.depth_image.image_height
                cam_props.enable_tensors = True
                # cam_props.near_plane = self.cfg.depth_image.near_plane
                # cam_props.far_plane = self.cfg.depth_image.far_plane
                cam_props.horizontal_fov = np.random.uniform(self.cfg.depth_image.horizontal_fov_range[0], 
                                                             self.cfg.depth_image.horizontal_fov_range[1])
                
                camera_handle = self.gym.create_camera_sensor(env_handle, cam_props)
                if camera_handle == -1:
                    raise RuntimeError("Failed to create camera sensor")
                self.camera_handles.append(camera_handle)
                # set camera pose
                cam_pos_x = np.random.uniform(self.cfg.depth_image.camera_pose.x_range[0],
                                             self.cfg.depth_image.camera_pose.x_range[1])
                cam_pos_y = np.random.uniform(self.cfg.depth_image.camera_pose.y_range[0],
                                            self.cfg.depth_image.camera_pose.y_range[1])
                cam_pos_z = np.random.uniform(self.cfg.depth_image.camera_pose.z_range[0],
                                            self.cfg.depth_image.camera_pose.z_range[1])
                
                cam_rot_pitch = np.random.uniform(self.cfg.depth_image.camera_pose.pitch_range[0],
                                                self.cfg.depth_image.camera_pose.pitch_range[1])
                cam_transform = gymapi.Transform()
                cam_transform.p = gymapi.Vec3(cam_pos_x, cam_pos_y, cam_pos_z)
                cam_transform.r = gymapi.Quat.from_euler_zyx(0, cam_rot_pitch, 0)
                self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, cam_transform, gymapi.FOLLOW_TRANSFORM)
                
                # obtain depth image tensor
                depth_image_gym_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_handle, camera_handle, gymapi.IMAGE_DEPTH)
                depth_image_tensor = gymtorch.wrap_tensor(depth_image_gym_tensor)
                self.depth_image_tensors.append(depth_image_tensor)
        
        print(f"Created {self.num_envs} environments")
        
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
        
        if self.cfg.depth_image.use_depth_image:
            pass
        
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
