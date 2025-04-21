from legged_gym.envs.perceptive.perceptive_robot_config import PerceptiveRobotCfg, PerceptiveRobotCfgPPO

class Go2PerceptiveCfg(PerceptiveRobotCfg):
    class terrain(PerceptiveRobotCfg.terrain):
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
    
    class commands(PerceptiveRobotCfg.commands):
        class ranges(PerceptiveRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state( PerceptiveRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }
        
    class control( PerceptiveRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 40.}  # [N*m/rad]
        damping = {'joint': 1.0}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( PerceptiveRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
    
    class domain_rand(PerceptiveRobotCfg.domain_rand):
        friction_range = [0.2, 1.25] # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.
        randomize_base_mass = True

    class rewards( PerceptiveRobotCfg.rewards ):
        class scales( PerceptiveRobotCfg.rewards.scales ):
            termination = -20.0
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            torques = -1e-5
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0.5
            feet_air_time =  1.5
            collision = -10.0
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.001

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.33
        max_contact_force = 100. # forces above this value are penalized




class Go2PerceptiveCfgPPO(PerceptiveRobotCfgPPO):
    
    class runner(PerceptiveRobotCfgPPO.runner):
        max_iterations = 1500 # number of policy updates
        experiment_name = 'go2_perceptive'