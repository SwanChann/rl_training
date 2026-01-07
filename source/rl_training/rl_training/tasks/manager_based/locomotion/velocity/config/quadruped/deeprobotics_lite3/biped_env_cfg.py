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
    
    # 关节名称（所有12个关节）
    joint_names = [
        "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
        "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
        "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
        "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
    ]
    
    # 所有连杆名称（用于 Events 随机化配置）
    link_names = [
        'TORSO', 
        'FL_HIP', 'FR_HIP', 'HL_HIP', 'HR_HIP', 
        'FL_THIGH', 'FR_THIGH', 'HL_THIGH', 'HR_THIGH', 
        'FL_SHANK', 'FR_SHANK', 'HL_SHANK', 'HR_SHANK', 
        'FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT',
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
    # 注意：根据训练反馈优化，HipY抬高，Knee收紧
    front_leg_target_positions = {
        "FL_HipX_joint": 0.0,
        "FL_HipY_joint": -1.5,    # 向上抬起（增大角度）
        "FL_Knee_joint": 2.6,     # 弯曲收起（增大角度）
        "FR_HipX_joint": 0.0,
        "FR_HipY_joint": -1.5,
        "FR_Knee_joint": 2.6,
    }

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Scene------------------------------
        self.scene.robot = DEEPROBOTICS_LITE3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 重要：height_scanner 的 prim_path 必须指向正确的基座链接
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

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

        # ------------------------------Events------------------------------
        # 重要：必须设置 body_names，否则会导致正则表达式匹配错误
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = self.link_names
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_com_positions.params["asset_cfg"].body_names = self.base_link_name
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_push_robot = None  # 双足平衡更难，禁用外力推动
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Rewards------------------------------
        self._setup_biped_rewards()

        # ------------------------------Commands------------------------------
        # 双足行走速度范围 - 降低以帮助学习站立和平衡
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 0.3)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.15, 0.15)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact = None

        # ------------------------------Curriculums------------------------------
        self.curriculum.command_levels = None
        self.curriculum.terrain_levels = None  # 禁用地形课程，双足站立不考虑复杂地形

        # 禁用零权重奖励
        if self.__class__.__name__ == "DeeproboticsLite3BipedEnvCfg":
            self.disable_zero_weight_rewards()

    def _setup_biped_rewards(self):
        """配置双足站立行走的奖励函数
        
        设计思路：
        1. 首先确保能够站立起来（高度、姿态、前腿抬起）- 最高优先级
        2. 然后学习保持平衡（重心、后腿稳定）
        3. 最后学习行走（速度跟踪、步态）
        """
        
        # ==================== 禁用四足专用奖励 ====================
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time_variance.weight = 0.0
        
        # ==================== 第一阶段：站立能力（最高优先级）====================
        
        # 机身高度 - 核心中的核心
        self.rewards.base_height_l2.weight = -50.0  # 大幅提升！站立是第一要务
        self.rewards.base_height_l2.params["target_height"] = 0.50  # 降低目标高度，更容易达成
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # 保持姿态水平 - 防止摔倒
        self.rewards.flat_orientation_l2.weight = -30.0  # 大幅提升
        
        # 前腿抬起并固定 - 站立的关键
        self.rewards.front_legs_fixed_pose = RewTerm(
            func=mdp.front_legs_fixed_pose,
            weight=-25.0,  # 提升权重
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.front_leg_joints),
                "target_positions": self.front_leg_target_positions,
            },
        )
        
        # 前腿不能接触地面 - 强制抬起
        self.rewards.front_legs_no_contact = RewTerm(
            func=mdp.front_legs_no_contact,
            weight=-20.0,  # 提升权重
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.front_foot_link_name]),
                "threshold": 1.0,
            },
        )
        
        # 前腿抬高奖励（类似 gym 中的 handstand 奖励）
        self.rewards.front_legs_height_exp = RewTerm(
            func=mdp.front_legs_height_exp,
            weight=15.0,  # 正向大奖励
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.front_foot_link_name]),
                "target_height": 0.4,  # 目标高度
                "std": 0.2,
                "lift_threshold": 0.025,
            },
        )

        # ==================== 第二阶段：平衡与稳定 ====================
        
        # Z轴速度惩罚 - 防止跳跃
        self.rewards.lin_vel_z_l2.weight = -4.0
        
        # XY角速度惩罚 - 防止翻滚
        self.rewards.ang_vel_xy_l2.weight = -0.5
        
        # 双足平衡奖励 - 重心控制
        self.rewards.biped_balance = RewTerm(
            func=mdp.biped_balance,
            weight=-8.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.rear_foot_link_name]),
            },
        )
        
        # 后腿 HipX 关节偏离惩罚 - 保持双腿笔直
        self.rewards.joint_deviation_l1.weight = -2.0  # 提升权重
        self.rewards.joint_deviation_l1.params["asset_cfg"].joint_names = ["H[LR]_HipX.*"]
        
        # 非期望接触惩罚（膝盖等）
        self.rewards.undesired_contacts.weight = -2.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["^(?!.*_FOOT).*"]
        
        # 站立时保持姿态稳定
        self.rewards.stand_still.weight = -1.0  # 提升权重
        self.rewards.stand_still.params["command_threshold"] = 0.05  # 降低阈值
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.joint_names

        # ==================== 第三阶段：速度跟踪与步态（低优先级）====================
        
        # 速度跟踪 - 降低权重，先学会站立再学行走
        self.rewards.track_lin_vel_xy_exp.weight = 0.5  # 大幅降低
        self.rewards.track_ang_vel_z_exp.weight = 0.3  # 大幅降低
        
        # 双足腾空时间奖励（只针对后腿）
        self.rewards.biped_feet_air_time = RewTerm(
            func=mdp.biped_feet_air_time,
            weight=0.8,  # 降低权重
            params={
                "command_name": "base_velocity",
                "threshold": 0.3,  # 降低阈值
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.rear_foot_link_name]),
            },
        )
        
        # 后腿足端滑动惩罚
        self.rewards.feet_slide.weight = -0.15
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.rear_foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.rear_foot_link_name]
        
        # 后腿足端高度
        self.rewards.feet_height.weight = -0.4
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.rear_foot_link_name]
        self.rewards.feet_height.params["target_height"] = 0.06
        
        # 后腿足端高度（机身坐标系）
        self.rewards.feet_height_body.weight = -2.0  # 降低权重
        self.rewards.feet_height_body.params["target_height"] = -0.50
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.rear_foot_link_name]

        # ==================== 能耗与平滑性（低优先级）====================
        self.rewards.action_rate_l2.weight = -0.01  # 降低，先学会站立
        self.rewards.joint_torques_l2.weight = -1e-5  # 降低
        self.rewards.joint_acc_l2.weight = -1e-9  # 降低
        self.rewards.joint_power.weight = -1e-5  # 降低

        # ==================== 接触力控制 ====================
        self.rewards.contact_forces.weight = -2e-2
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.rear_foot_link_name]
        
        # 非期望接触（排除后腿足端）- 防止膝盖、大腿接触地面
        self.rewards.undesired_contacts.weight = -2.0  # 提升权重
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.rear_foot_link_name}).*"]