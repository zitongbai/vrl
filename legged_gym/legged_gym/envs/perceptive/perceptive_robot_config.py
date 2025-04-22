from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np

class PerceptiveRobotCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        
        num_proprioception = 45
        num_proprioception_history = 0
        num_height_measurements = 33 * 21
        num_observations = (1+num_proprioception_history)*num_proprioception + num_height_measurements

        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
    
    
    
    class terrain(LeggedRobotCfg.terrain):
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, gap, pit]
        terrain_proportions = [
            0.1,    # smooth slope
            0.1,    # rough slope
            0.35,   # stairs up
            0.25,   # stairs down
            0.2,    # discrete
            0.0,    # stepping stones
            0.0,    # gap
            0.0,    # pit
        ]
        
        measure_heights = True
        measured_points_x = np.linspace(1.1, -0.5, 33).tolist() # from positive to negative
        measured_points_y = np.linspace(0.5, -0.5, 21).tolist() # from positive to negative
        
        max_init_terrain_level = 5
        
    class commands(LeggedRobotCfg.commands):
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]
        
    class noise(LeggedRobotCfg.noise):
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            pass
            
    class rewards( LeggedRobotCfg.rewards ):
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -20.0
            collision = -10.0
            
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -1.5
            
            torques = -1e-5
            dof_vel = -0.
            dof_acc = -2.5e-7
            action_rate = -0.01
            stand_still = -0.001
            
            feet_air_time =  1.5
            feet_clearance = -0.5
            stumble = -1.0
            
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.33
        max_contact_force = 100. # forces above this value are penalized
        clearance_footpos_target = -0.1 # target foot position in body frame

class PerceptiveRobotCfgPPO(LeggedRobotCfgPPO):
    
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
        # TODO: add checking 
        num_proprioception = 45
        height_measurements_size = (33, 21)
        
        cnn_channels=[16, 32, 32]
        cnn_kernel_sizes=[2, 2, 1]
        cnn_strides=[2, 1, 1]
        cnn_padding=[0, 0, 0]
        cnn_embedding_dim=32
        
        rnn_type='gru'
        rnn_hidden_size=256
        rnn_num_layers=1
        
    class algorithm(LeggedRobotCfgPPO.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
        
    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCriticScandots'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'perceptive'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
    
        