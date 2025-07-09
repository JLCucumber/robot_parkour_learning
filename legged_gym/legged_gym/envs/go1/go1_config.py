# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# ... (copyright header remains the same) ...
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np
import os.path as osp


class Go1RoughCfg( LeggedRobotCfg ):
    class custom:
        shared_path = True
        name = "rough_go1"
        logs_root = osp.join("/mnt/rpl_project", "logs")

    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_observations = None # No use, use obs_components
        num_privileged_obs = None # No use, use privileged_obs_components

        use_lin_vel = False # to be decided
        num_actions = 12
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

        obs_components = [
            "lin_vel",
            "ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos",
            "dof_vel",
            "last_actions",
            "height_measurements",
        ]
    
    class sensor:
        class proprioception:
            obs_components = ["ang_vel", "projected_gravity", "commands", "dof_pos", "dof_vel"]
            latency_range = [0.005, 0.045] # [s]
            latency_resampling_time = 5.0 # [s]
    
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.4] # x,y,z [m], Go1 is slightly shorter than A1
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FR_hip_joint': -0.1,  # [rad]
            'FL_hip_joint': 0.1,   # [rad]
            'RR_hip_joint': -0.1,  # [rad]
            'RL_hip_joint': 0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.0,     # [rad]
            'RR_thigh_joint': 1.0,     # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'FR_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,   # [rad]
            'RR_calf_joint': -1.5,   # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 25.}  # [N*m/rad] # Adjusted for Go1 motors
        damping = {'joint': 0.8}     # [N*m*s/rad] # Adjusted for Go1 motors
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class terrain:
        selected = "TerrainPerlin"
        mesh_type = None
        measure_heights = True
        # x: [-0.5, 1.5], y: [-0.5, 0.5] range for go2
        measured_points_x = [i for i in np.arange(-0.5, 1.51, 0.1)]
        measured_points_y = [i for i in np.arange(-0.5, 0.51, 0.1)]
        horizontal_scale = 0.025 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 5 # [m]
        curriculum = False
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 4.
        terrain_width = 4.
        num_rows= 16 # number of terrain rows (levels)
        num_cols = 16 # number of terrain cols (types)
        slope_treshold = 1.

        TerrainPerlin_kwargs = dict(
            zScale= 0.07,
            frequency= 10,
        )

    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_com = True
        class com_range:
            x = [-0.15, 0.15]
            y = [-0.1, 0.1]
            z = [-0.05, 0.05]

        randomize_motor = True
        leg_motor_strength_range = [0.8, 1.2]

        randomize_base_mass = True
        added_mass_range = [0.5, 2.0] # Go1 is lighter

        randomize_friction = True
        friction_range = [0., 2.]

        init_base_pos_range = dict(
            x= [0.05, 0.6],
            y= [-0.25, 0.25],
        )
        init_base_rot_range = dict(
            roll= [-0.75, 0.75],
            pitch= [-0.75, 0.75],
        )
        init_base_vel_range = dict(
            x= [-0.2, 1.5],
            y= [-0.2, 0.2],
            z= [-0.2, 0.2],
            roll= [-1., 1.],
            pitch= [-1., 1.],
            yaw= [-1., 1.],
        )
        init_dof_vel_range = [-5, 5]

        push_robots = True 
        max_push_vel_xy = 0.5 # [m/s]
        push_interval_s = 2


    class termination:
        termination_terms = [
            "roll",
            "pitch",
        ]

        roll_kwargs = dict(
            threshold= 3.0, # [rad]
        )
        pitch_kwargs = dict(
            threshold= 3.0, # [rad] # for leap, jump
        )

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        sdk_dof_range = dict( # Based on Go1 SDK
            Hip_max= 1.0472,
            Hip_min= -1.0472,
            Thigh_max= 2.9670,
            Thigh_min= -0.6108,
            Calf_max= -0.8377,
            Calf_min= -2.7227,
        )
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0

class Go1RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        clip_min_std = 0.2
        learning_rate = 1e-4
        optimizer_class_name = "AdamW"

    # --- ADD THIS POLICY CLASS ---
    class policy( LeggedRobotCfgPPO.policy ):
        # By setting rnn_type, you are telling the system to use a recurrent policy.
        rnn_type = 'gru'
        # These are common architectural parameters for the RNN.
        rnn_hidden_size = 256
        rnn_num_layers = 1
        init_noise_std = 1.0
    # --------------------------

    class runner( LeggedRobotCfgPPO.runner ):
        # IMPORTANT: Ensure the runner's policy class name matches a recurrent one.
        # The exact name depends on your framework, 'ActorCriticRnn' is a common one for rsl_rl.
        policy_class_name = "ActorCriticRecurrent"
        run_name = 'full'
        experiment_name = 'rough_go1'

        max_iterations = 3000
        save_interval = 2000
        log_interval = 100

#### To train the model with partial observation (flat terrain) ####

# class Go1PlaneCfg( Go1RoughCfg ):
#     class env( Go1RoughCfg.env ):
#         use_lin_vel = False
#         num_observations = 48 # Number of observations without height measurements

#     class control( Go1RoughCfg.control ):
#         # Can override stiffness for simpler terrains if needed
#         stiffness = {'joint': 25.}

#     class domain_rand( Go1RoughCfg.domain_rand ):
#         randomize_base_mass = True

#     class terrain( Go1RoughCfg.terrain ):
#         mesh_type = "plane"
#         measure_heights = False # No height measurements on a plane

#### Config for Teacher-Student training ####
# class Go1RoughCfgTPPO( Go1RoughCfgPPO ):

#     class algorithm( Go1RoughCfgPPO.algorithm ):
#         distillation_loss_coef = 50.

#         # IMPORTANT: This path must be updated to a valid, pre-trained Go1 teacher model
#         teacher_ac_path = "logs/rough_go1/some_pretrained_run/model_xxxx.pt"
#         teacher_policy_class_name = Go1RoughCfgPPO.runner.policy_class_name
        
#         class teacher_policy( Go1RoughCfgPPO.policy ):
#             # Observation spaces must match the teacher model
#             num_actor_obs = 235
#             num_critic_obs = 235
#             num_actions = 12
    
#     class runner( Go1RoughCfgPPO.runner ):
#         algorithm_class_name = "TPPO"
#         run_name = 'distillation_run_1'
#         experiment_name = 'teacher_go1'