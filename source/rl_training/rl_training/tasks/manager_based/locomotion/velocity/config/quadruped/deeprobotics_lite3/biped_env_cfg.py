# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from rl_training.assets.deeprobotics import DEEPROBOTICS_LITE3_CFG
import rl_training.tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class DeeproboticsLite3BipedEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Lite3 双腿站立行走配置"""
    
    base_link_name = "TORSO"
    
    # 只有后腿参与行走
    rear_foot_link_name = "H[LR]_FOOT"  # HL_FOOT 和 HR_FOOT
    front_foot_link_name = "F[LR]_FOOT"  # FL_FOOT 和 FR_FOOT（用于惩罚接触）
    
    # 关节名称
    joint_names = [
        "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
        "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
        "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
        "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
    ]
    
    # 前腿关节（需要锁定）
    front_leg_joints = [
        "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
        "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
    ]
    
    # 后腿关节（主要行走用）
    rear_leg_joints = [
        "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
        "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
    ]

    # 前腿目标姿态（收起/抬起状态）
    front_leg_target_positions = {
        "FL_HipX_joint": 0.0,
        "FL_HipY_joint": -1.2,    # 向上抬起
        "FL_Knee_joint": 2.4,     # 弯曲收起
        "FR_HipX_joint": 0.0,
        "FR_HipY_joint": -1.2,
        "FR_Knee_joint": 2.4,
    }

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Scene------------------------------
        self.scene.robot = DEEPROBOTICS_LITE3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        # 保持与四足相同的观测量，确保 sim2real 兼容性
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        self.actions.joint_pos.scale = {".*_HipX_joint": 0.125, "^(?!.*_HipX_joint).*": 0.25}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Rewards------------------------------
        self._setup_biped_rewards()

        # ------------------------------Commands------------------------------
        # 双足行走速度范围可能需要降低
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)

        # 禁用零权重奖励
        if self.__class__.__name__ == "DeeproboticsLite3BipedEnvCfg":
            self.disable_zero_weight_rewards()

    def _setup_biped_rewards(self):
        """配置双足站立行走的奖励函数"""
        
        # ==================== 禁用/降低四足专用奖励 ====================
        
        # 禁用四足腾空时间奖励（将被双足版本替代）
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time_variance.weight = 0.0
        
        # ==================== 速度跟踪（核心奖励）====================
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.8

        # ==================== 姿态与高度 ====================
        
        # 提高机身高度目标（双足站立更高）
        self.rewards.base_height_l2.weight = -15.0
        self.rewards.base_height_l2.params["target_height"] = 0.55  # 比四足高
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # 保持姿态水平
        self.rewards.flat_orientation_l2.weight = -8.0
        
        # Z轴速度惩罚
        self.rewards.lin_vel_z_l2.weight = -3.0
        
        # XY角速度惩罚
        self.rewards.ang_vel_xy_l2.weight = -0.1

        # ==================== 前腿控制（核心：锁定前腿）====================
        
        # 添加前腿固定姿态奖励
        self.rewards.front_legs_fixed_pose = RewTerm(
            func=mdp.front_legs_fixed_pose,
            weight=-20.0,  # 高权重，强制前腿保持固定
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.front_leg_joints),
                "target_positions": self.front_leg_target_positions,
            },
        )
        
        # 惩罚前腿接触地面
        self.rewards.front_legs_no_contact = RewTerm(
            func=mdp.front_legs_no_contact,
            weight=-10.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.front_foot_link_name]),
                "threshold": 1.0,
            },
        )

        # ==================== 后腿步态（双足行走核心）====================
        
        # 双足腾空时间奖励（只针对后腿）
        self.rewards.biped_feet_air_time = RewTerm(
            func=mdp.biped_feet_air_time,
            weight=1.5,
            params={
                "command_name": "base_velocity",
                "threshold": 0.4,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.rear_foot_link_name]),
            },
        )
        
        # 后腿足端滑动惩罚
        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.rear_foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.rear_foot_link_name]
        
        # 后腿足端高度
        self.rewards.feet_height.weight = -0.3
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.rear_foot_link_name]
        self.rewards.feet_height.params["target_height"] = 0.08
        
        # 后腿足端高度（机身坐标系）
        self.rewards.feet_height_body.weight = -3.0
        self.rewards.feet_height_body.params["target_height"] = -0.55  # 与机身高度对应
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.rear_foot_link_name]

        # ==================== 平衡与稳定 ====================
        
        # 双足平衡奖励
        self.rewards.biped_balance = RewTerm(
            func=mdp.biped_balance,
            weight=-5.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.rear_foot_link_name]),
            },
        )

        # ==================== 能耗与平滑性 ====================
        self.rewards.action_rate_l2.weight = -0.03
        self.rewards.joint_torques_l2.weight = -3e-5
        self.rewards.joint_acc_l2.weight = -2e-8
        self.rewards.joint_power.weight = -3e-5
        
        # 后腿 HipX 关节偏离惩罚
        self.rewards.joint_deviation_l1.weight = -0.8
        self.rewards.joint_deviation_l1.params["asset_cfg"].joint_names = ["H[LR]_HipX.*"]
        
        # 静止时保持姿态
        self.rewards.stand_still.weight = -0.5
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.joint_names

        # ==================== 接触力控制 ====================
        self.rewards.contact_forces.weight = -3e-2
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.rear_foot_link_name]
        
        # 非期望接触（排除后腿足端）
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.rear_foot_link_name}).*"]

