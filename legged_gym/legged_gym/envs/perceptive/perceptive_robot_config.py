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
            0.1,   # stairs up
            0.1,   # stairs down
            0.1,    # discrete
            0.5,    # stepping stones
            0.0,    # gap
            0.0,    # pit
        ]
        
        measure_heights = True
        # 1mx1.6m rectangle (without center line)
        # measured_points_x = [1.2, 1.05, 0.9, 0.75, 0.6, 0.45, 0.3, 0.15, 0., -0.15, -0.3, -0.45]
        # measured_points_y = [0.75, 0.6, 0.45, 0.3, 0.15, 0., -0.15, -0.3, -0.45, -0.6, -0.75]
        measured_points_x = np.linspace(1.1, -0.5, 33).tolist()
        measured_points_y = np.linspace(0.5, -0.5, 21).tolist()
        
        max_init_terrain_level = 10
        
    class noise(LeggedRobotCfg.noise):
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            pass
            

class PerceptiveRobotCfgPPO(LeggedRobotCfgPPO):
    
    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
        # TODO: add checking 
        num_proprioception = 45
        height_measurements_size = (33, 21)
        
        propr_rnn_type = 'gru'
        propr_rnn_hidden_size = 256
        propr_rnn_num_layers = 1
        
        height_encoder_output_dim=32
        height_rnn_type='gru'
        height_rnn_hidden_size=256
        height_rnn_num_layers=1
        
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
    
        