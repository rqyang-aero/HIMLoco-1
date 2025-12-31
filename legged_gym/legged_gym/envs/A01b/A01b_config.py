# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class A01bRoughCfg( LeggedRobotCfg ):
    # class env( LeggedRobotCfg.env ):
    #     num_envs = 4096
    #     num_one_step_observations = 45
    #     num_observations = num_one_step_observations * 6
    #     num_one_step_privileged_obs = 45 + 3 + 3 + 187 # additional: base_lin_vel, external_forces, scan_dots
    #     num_privileged_obs = num_one_step_privileged_obs * 1 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
    #     num_actions = 12
    #     env_spacing = 3.  # not used with heightfields/trimeshes 
    #     send_timeouts = True # send time out information to the algorithm
    #     episode_length_s = 20 # episode length in seconds

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.60] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': -0.049,   # [rad]
            'RL_hip_joint': -0.049,   # [rad]
            'FR_hip_joint': 0.049 ,  # [rad]
            'RR_hip_joint': 0.049,   # [rad]

            'FL_thigh_joint': -0.81,     # [rad]
            'RL_thigh_joint': -0.81,   # [rad]
            'FR_thigh_joint': -0.81,     # [rad]
            'RR_thigh_joint': -0.81,   # [rad]

            'FL_calf_joint': 1.44,   # [rad]
            'RL_calf_joint': 1.44,    # [rad]
            'FR_calf_joint': 1.44,  # [rad]
            'RR_calf_joint': 1.44,    # [rad]
        }

    class commands( LeggedRobotCfg.commands ):
        max_curriculum = 2.0

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 300.0}  # [N*m/rad]
        damping = {'joint': 7.0}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10
        hip_reduction = 1.0

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/A01b/urdf/A01b.urdf'
        name = "aliengo"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        privileged_contacts_on = ["base", "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up

    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            dof_acc = -2.5e-7
            joint_power = -7e-6
            base_height = -10
            foot_clearance = -0.01
            action_rate = -0.01
            smoothness = -0.01
            torques = -0.0
            dof_vel = -0.0
            collision = -0.08
            termination = -0.0
            dof_pos_limits = 0.0
            dof_vel_limits = 0.0
            torque_limits = 0.0
            stand_still = -0.8
            feet_contact_forces = -0.0
            feet_slip = -0.08
            feet_air_time =  0.0
            feet_stumble = -0.0
            hip_pos = -1.5
            dof_error = -0.08
            stand_contact = -1.0
            stand_yawvel = -0.5
            delta_torques = -1.0e-7
            contact_center = -0.1

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.95 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 0.95
        base_height_target = 0.61
        max_contact_force = 100. # forces above this value are penalized
        clearance_height_target = -0.48
    
    class sim( LeggedRobotCfg.sim ):
        dt =  0.002

class A01bRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_A01b'

  