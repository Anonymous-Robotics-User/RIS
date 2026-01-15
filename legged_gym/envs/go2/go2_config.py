from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2Cfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.34] # x,y,z [m]
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

    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = 42
        num_clock_inputs = 8 # (1: ref), (4: default)
        num_privileged_obs = 9
        num_observation_history = 20
        num_observation_target = num_observations - 12 # actions

    class terrain( LeggedRobotCfg.terrain ):
        border_size = 0.0
        mesh_type = 'trimesh'
        num_cols = 30
        num_rows = 30
        terrain_width = 1.0
        terrain_length = 1.0
        x_init_range = 0.2
        y_init_range = 0.2
        center_robots = True
        center_span = 4
        measure_heights = False
        terrain_noise_magnitude = 0.0
        teleport_thresh = 0.3
        teleport_robots = True
        horizontal_scale = 0.10
        yaw_init_range = 3.14

        terrain_proportions = [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
        curriculum = False

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {'joint': 25.}  # [N*m/rad]
        damping = {'joint': 0.6}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        hip_scale_reduction = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hill_model = False

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base", "hip", "Head"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = True
        fix_base_link = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        soft_torque_limit = 1.05
        base_height_target = 0.30
        use_terminal_foot_height = False
        use_terminal_body_height = True
        terminal_body_height = 0.05
        use_terminal_roll_pitch = True
        terminal_body_ori = 1.6
        kappa_gait_probs = 0.07
        gait_force_sigma = 100.
        gait_vel_sigma = 10.
        only_positive_rewards = False
        only_positive_rewards_ji22_style = True
        sigma_rew_neg = 0.02
        class scales( LeggedRobotCfg.rewards.scales ):
            # termination = -0.0
            # tracking_lin_vel = 1.0
            # tracking_ang_vel = 0.5
            # lin_vel_z = -0.02
            # ang_vel_xy = -0.001
            # torques = 0.0 #-0.0001
            # action_rate = -0.005
            # dof_pos_limits = -1000.0 #- 50.0
            # dof_vel_limits = -0.0
            # torque_limits = -30.0 #-1.5
            # feet_contact_forces = 0.0 #-0.1
            # feet_contact_momentum = -1.0 #-0.5
            # feet_slip = -0.2
            # action_smoothness_1 = -0.1
            # action_smoothness_2 = -0.05
            # dof_pos = -0.0
            # dof_vel = 0.0 # -1e-4
            # dof_acc = -1e-7
            # joint_power = -3e-4 # -2e-5
            # jump = 10.0
            # raibert_heuristic = -10.0
            # feet_impact_vel = -0.0
            # feet_clearance_cmd_linear = -30.0
            # orientation_control = -2.5
            # tracking_contacts_shaped_force = 4.0
            # tracking_contacts_shaped_vel = 4.0
            # collision = -1.0
            feet_clearance_cmd_smooth = -2.5

            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -0.02
            ang_vel_xy = -0.001
            torques = 0.0 #-0.0001
            action_rate = -0.01
            dof_pos_limits = -10.0
            dof_vel_limits = -0.0
            torque_limits = -0.3
            feet_contact_forces = 0.0
            feet_contact_momentum = -1.0
            feet_slip = -0.04
            action_smoothness_1 = -0.1
            action_smoothness_2 = -0.1
            dof_pos = -0.0
            dof_vel = -0.0
            dof_acc = -1e-7
            joint_power = -3e-4
            jump = 10.0
            raibert_heuristic = -10.0
            feet_impact_vel = -0.0
            # feet_clearance_cmd_linear = -30.0
            orientation_control = -2.5  
            tracking_contacts_shaped_force = 4.0
            tracking_contacts_shaped_vel = 4.0
            collision = -5.0

    class commands( LeggedRobotCfg.commands ):
        heading_command = False
        resampling_time = 10.0
        command_curriculum = True

        lin_vel_x = [-1.0, 1.0]
        lin_vel_y = [-0.6, 0.6]
        ang_vel_yaw = [-1.0, 1.0]
        body_height_cmd = [-0.25, 0.15]
        gait_frequency_cmd_range = [2.0, 4.0]
        gait_phase_cmd_range = [0.0, 1.0]
        gait_offset_cmd_range = [0.0, 1.0]
        gait_bound_cmd_range = [0.0, 1.0]
        footswing_height_range = [0.03, 0.35]
        body_pitch_range = [-0.4, 0.4]
        stance_width_range = [0.10, 0.45]
        stance_length_range = [0.35, 0.45]

        limit_vel_x = [-5.0, 5.0]
        limit_vel_y = [-0.6, 0.6]
        limit_vel_yaw = [-5.0, 5.0]
        limit_body_height = [-0.25, 0.15] # base target : 0.3
        limit_gait_frequency = [2.0, 4.0]
        limit_gait_phase = [0.0, 1.0]
        limit_gait_offset = [0.0, 1.0]
        limit_gait_bound = [0.0, 1.0]
        limit_footswing_height = [0.03, 0.35]
        limit_body_pitch = [-0.4, 0.4]
        limit_stance_width = [0.10, 0.45]
        limit_stance_length = [0.35, 0.45]

        num_bins_vel_x = 21
        num_bins_vel_y = 1
        num_bins_vel_yaw = 21
        num_bins_body_height = 1
        num_bins_gait_frequency = 1
        num_bins_gait_phase = 1
        num_bins_gait_offset = 1
        num_bins_gait_bound = 1
        num_bins_footswing_height = 1
        num_bins_body_pitch = 1
        num_bins_stance_width = 1

        exclusive_phase_offset = False
        pacing_offset = False
        binary_phases = False # True
        gaitwise_curricula = True

        distributional_commands = True
        num_commands = 12
    
    class curriculum_thresholds:
        tracking_lin_vel = 0.8
        tracking_ang_vel = 0.7
        tracking_contacts_shaped_vel = 0.90
        tracking_contacts_shaped_force = 0.90

    class domain_rand( LeggedRobotCfg.domain_rand ):
        max_push_vel_xy = 0.5
        restitution = 0.5  # default terrain restitution
        lag_timesteps = 6
        randomize_lag_timesteps = False
        randomize_rigids_after_start = False
        randomize_friction_indep = False
        randomize_friction = True
        friction_range = [0.1, 3.0]
        randomize_restitution = True
        restitution_range = [0.0, 0.4]
        randomize_base_mass = True
        added_mass_range = [-1.0, 3.0]

        randomize_gravity = True
        gravity_range = [-1.0, 1.0]
        gravity_rand_interval_s = 8.0
        gravity_impulse_duration = 0.99

        randomize_com_displacement = False
        com_displacement_range = [-0.15, 0.15]
        
        randomize_ground_friction = True
        ground_friction_range = [0.0, 0.0]
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        randomize_motor_offset = True
        motor_offset_range = [-0.02, 0.02]
        push_robots = False

        randomize_Kp_factor = True
        Kp_factor_range = [0.8, 1.3]
        randomize_Kd_factor = True
        Kd_factor_range = [0.5, 1.3]

        rand_interval_s = 4
        tile_height_range = [-0.0, 0.0]
        tile_height_curriculum = False
        tile_height_update_interval = 1000000
        tile_height_curriculum_step = 0.01
    
    class normalization( LeggedRobotCfg.normalization ):
        friction_range = [0.4, 1]
        restitution_range = [0, 1]
        ground_friction_range = [0, 1]
        body_velocity_range = [-4.5, 4.5]
        clip_actions = 10.0

class GO2CfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        value_loss_coef = 7.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1e-3
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go2'

    class estimator:
        tsteps = 24
        decoder_dims = [512, 256, 128]
        kernel_size = [5, 2]
        stride = [4, 2]
        learning_rate = 1e-3
        beta = 1e-4
