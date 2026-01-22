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
        
        参考 gym_legged_robot.py 和 gym_legged_robot_config.py 的奖励权重配置。
        
        设计思路（按权重大小分层）：
        第一层（核心）：handstand_feet_height_exp=17.5, tracking_lin_vel=10, tracking_ang_vel=5
        第二层（重要）：handstand_feet_on_air=1.5, handstand_feet_air_time=1.5, collision=-1.0
        第三层（辅助）：handstand_orientation_l2=0.8, stand_still=-0.8, ang_vel_xy=-0.3
        第四层（精细）：action_rate=-0.03, torques=-1e-5, dof_acc=-2.5e-7
        """
        
        # ==================== 禁用四足专用奖励 ====================
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time_variance.weight = 0.0
        
        # ==================== 第一层：核心奖励（|scale| > 10）====================
        
        # 前腿抬高奖励 - 对应 gym 的 handstand_feet_height_exp (17.5)
        self.rewards.front_legs_height_exp = RewTerm(
            func=mdp.front_legs_height_exp,
            weight=17.5,  # gym 配置: 17.5
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.front_foot_link_name]),
                "target_height": 0.6,  # gym 配置: 0.6
                "std": 0.4,  # gym 配置: 0.4
                "lift_threshold": 0.025,  # gym 配置: 0.025
            },
        )
        
        # 速度跟踪 - 对应 gym 的 tracking_lin_vel (10.0) 和 tracking_ang_vel (5.0)
        self.rewards.track_lin_vel_xy_exp.weight = 10.0
        self.rewards.track_ang_vel_z_exp.weight = 5.0

        # ==================== 第二层：重要奖励（1 < |scale| ≤ 10）====================
        
        # 前腿离地奖励 - 对应 gym 的 handstand_feet_on_air (1.5)
        self.rewards.front_legs_no_contact = RewTerm(
            func=mdp.front_legs_no_contact,
            weight=-1.5,  # 惩罚前腿接触地面
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.front_foot_link_name]),
                "threshold": 1.0,
            },
        )
        
        # 前腿悬空时间 - 对应 gym 的 handstand_feet_air_time (1.5)
        self.rewards.biped_feet_air_time = RewTerm(
            func=mdp.biped_feet_air_time,
            weight=1.5,
            params={
                "command_name": "base_velocity",
                "threshold": 5.0,  # gym 配置: 5.0
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.rear_foot_link_name]),
            },
        )
        
        # 碰撞惩罚 - 对应 gym 的 collision (-1.0)
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*THIGH.*", ".*SHANK.*"]
        
        # ==================== 第三层：辅助奖励（|scale| ≤ 1）====================
        
        # 姿态控制 - 对应 gym 的 handstand_orientation_l2 (0.8)
        self.rewards.handstand_orientation = RewTerm(
            func=mdp.handstand_orientation_l2,
            weight=-0.8,  # 惩罚偏离目标姿态
            params={
                "target_gravity": [1.0, 0.0, 0.0],  # gym 配置: [1, 0, 0]
            },
        )
        
        # 静止保持 - 对应 gym 的 stand_still (-0.8)
        self.rewards.stand_still.weight = -0.8
        self.rewards.stand_still.params["command_threshold"] = 0.1
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.joint_names
        
        # XY角速度惩罚 - 对应 gym 的 ang_vel_xy (-0.3)
        self.rewards.ang_vel_xy_l2.weight = -0.3
        
        # Z轴速度惩罚（保持稳定）
        self.rewards.lin_vel_z_l2.weight = 0.0  # gym 配置: 0.0
        
        # ==================== 第四层：精细调节（|scale| < 0.01）====================
        
        # 动作平滑性 - 对应 gym 的 action_rate (-0.03)
        self.rewards.action_rate_l2.weight = -0.03
        
        # 关节力矩 - 对应 gym 的 torques (-1e-5)
        self.rewards.joint_torques_l2.weight = -1e-5
        
        # 关节加速度 - 对应 gym 的 dof_acc (-2.5e-7)
        self.rewards.joint_acc_l2.weight = -2.5e-7
        
        # 关节功率（能耗）
        self.rewards.joint_power.weight = 0.0  # gym 配置: 0.0
        
        # ==================== 前腿固定姿态（额外惩罚）====================
        
        # 前腿姿态锁定
        self.rewards.front_legs_fixed_pose = RewTerm(
            func=mdp.front_legs_fixed_pose,
            weight=-5.0,  # 中等惩罚
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.front_leg_joints),
                "target_positions": self.front_leg_target_positions,
            },
        )
        
        # 机身高度（辅助）
        self.rewards.base_height_l2.weight = 0.0  # gym 配置: 0.0
        self.rewards.base_height_l2.params["target_height"] = 0.95  # gym 配置: 0.95
        
        # 姿态水平（辅助，与 handstand_orientation 配合）
        self.rewards.flat_orientation_l2.weight = 0.0  # 使用 handstand_orientation 代替
        
        # 双足平衡（额外）
        self.rewards.biped_balance = RewTerm(
            func=mdp.biped_balance,
            weight=-2.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.rear_foot_link_name]),
            },
        )
        
        # 后腿 HipX 关节偏离惩罚
        self.rewards.joint_deviation_l1.weight = 0.0  # 暂时禁用
        
        # 后腿足端滑动惩罚
        self.rewards.feet_slide.weight = 0.0  # 暂时禁用
        
        # 后腿足端高度
        self.rewards.feet_height.weight = 0.0  # 暂时禁用
        
        # 后腿足端高度（机身坐标系）
        self.rewards.feet_height_body.weight = 0.0  # 暂时禁用
        
        # 接触力控制
        self.rewards.contact_forces.weight = 0.0  # 暂时禁用