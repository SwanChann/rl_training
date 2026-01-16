# Lite3 双腿站立行走训练指南

本文档详细介绍如何基于 Lite3 四足机器人训练**双腿站立行走**（Bipedal Walking）模式。在此模式下，机器人使用后腿（HL、HR）站立并行走，前腿（FL、FR）抬起保持固定姿态。

---

## 目录

1. [机器人关节与连杆说明](#1-机器人关节与连杆说明)
2. [双足站立行走的挑战](#2-双足站立行走的挑战)
3. [Isaac Gym 参考实现分析](#3-isaac-gym-参考实现分析)
4. [Isaac Lab 奖励函数设计](#4-isaac-lab-奖励函数设计)
5. [配置文件详解](#5-配置文件详解)
6. [训练与调试](#6-训练与调试)

---

## 1. 机器人关节与连杆说明

在进行双足训练前，首先需要了解 Lite3 机器人的关节和连杆命名规则，以便正确配置奖励函数。

### 关节名称 (Joints)

Lite3 共有 **12 个关节**，命名格式为 `{Leg}_{Joint}_joint`：

| 腿部代号 | 名称 | 位置 |
|---------|------|------|
| `FL` | Front Left | 前左 |
| `FR` | Front Right | 前右 |
| `HL` | Hind Left | 后左 |
| `HR` | Hind Right | 后右 |

| 关节代号 | 名称 | 功能 |
|---------|------|------|
| `HipX` | Hip X-axis | 侧摆关节（负责左右摆动） |
| `HipY` | Hip Y-axis | 髋关节（负责大腿前后摆动） |
| `Knee` | Knee | 膝关节（负责小腿伸缩） |

**完整关节列表**：

```python
joint_names = [
    "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",  # 前左腿
    "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",  # 前右腿
    "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",  # 后左腿
    "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",  # 后右腿
]
```

### 连杆名称 (Links)

主要连杆命名如下：

| 连杆名称 | 说明 |
|---------|------|
| `TORSO` | 躯干/基座 |
| `{Leg}_HIP` | 髋部连杆 |
| `{Leg}_THIGH` | 大腿连杆 |
| `{Leg}_SHANK` | 小腿连杆 |
| `{Leg}_FOOT` | 足端连杆 |

**刚体索引**（参考 `gym_legged_robot.py`）：

```python
# Isaac Gym 中的刚体索引
TORSO = 0
FL_HIP, FL_THIGH, FL_SHANK, FL_FOOT = 1, 2, 3, 4
FR_HIP, FR_THIGH, FR_SHANK, FR_FOOT = 5, 6, 7, 8
HL_HIP, HL_THIGH, HL_SHANK, HL_FOOT = 9, 10, 11, 12
HR_HIP, HR_THIGH, HR_SHANK, HR_FOOT = 13, 14, 15, 16
```

---

## 2. 双足站立行走的挑战

与四足行走相比，双足站立行走面临以下挑战：

| 挑战 | 描述 | 解决方案 |
|------|------|---------|
| **稳定性困难** | 支撑面积从4个足端减少到2个 | 高权重的高度和姿态奖励 |
| **平衡要求高** | 需要精确控制机身姿态 | 双足平衡奖励 + 膝盖安全检查 |
| **学习阶段性** | 必须先学会站立，再学习行走 | 分层奖励设计（站立 > 平衡 > 行走） |
| **前腿控制** | 前腿需要保持抬起状态 | 前腿姿态锁定 + 接触惩罚 |

---

## 3. Isaac Gym 参考实现分析

`gym_legged_robot.py` 和 `gym_legged_robot_config.py` 是基于 Isaac Gym 的成功实现。以下分层分析其奖励函数设计。

### 3.1 奖励函数权重一览（gym_legged_robot_config.py）

```python
class rewards:
    class scales:
        # ============ 第一层：站立核心（权重最高）============
        handstand_feet_height_exp = 17.5   # 🔥 前腿抬高奖励（核心）
        handstand_feet_on_air = 1.5        # 前腿离地奖励
        handstand_feet_air_time = 1.5      # 前腿腾空时间
        handstand_orientation_l2 = 0.8     # 目标姿态（倒立方向）
        
        # ============ 第二层：平衡与稳定 ============
        tracking_lin_vel = 10.0            # 线速度跟踪
        tracking_ang_vel = 5.0             # 角速度跟踪
        collision = -1.0                   # 碰撞惩罚
        stand_still = -0.8                 # 静止时关节惩罚
        
        # ============ 第三层：动作平滑性（权重较低）============
        ang_vel_xy = -0.3                  # XY角速度惩罚
        action_rate = -0.03                # 动作变化率惩罚
        torques = -0.00001                 # 力矩惩罚
        dof_acc = -2.5e-7                  # 关节加速度惩罚
        joint_smoothness = 2.5e-9          # 关节平滑性
        torque_smoothness = 0.06           # 力矩平滑性
```

### 3.2 核心奖励函数详解

#### (1) `_reward_handstand_feet_height_exp` - 前腿抬高奖励（权重: 17.5）

**这是最重要的奖励函数**，实现了复杂的分阶段抬腿逻辑：

```python
def _reward_handstand_feet_height_exp(self):
    """优化版：基于0.022米阈值的抬腿判断"""
    
    # 1. 膝盖安全检查
    knee_safe_height = 0.05
    knee_heights = self.rigid_body_pos[:, shank_indices, 2]
    knee_height_penalty = torch.sum(torch.where(
        knee_heights < knee_safe_height,
        (knee_safe_height - knee_heights) ** 2, 0.0
    ), dim=1)
    knee_safety_reward = torch.exp(-knee_height_penalty / 0.05)
    
    # 2. 抬腿阈值判断（关键: 0.025m）
    LIFT_THRESHOLD = 0.025  # 高度大于此值才算抬腿
    left_leg_lifted = front_left_height > LIFT_THRESHOLD
    right_leg_lifted = front_right_height > LIFT_THRESHOLD
    
    # 3. 分阶段奖励
    base_lift_reward = any_leg_lifted.float() * 0.3      # 基础抬腿
    single_leg_reward = (max_lift / target) * 0.4        # 单腿抬高
    both_legs_reward = both_legs_lifted.float() * 0.5    # 双腿协调
    min_lift_reward = (min_lift / target) * 0.3          # 最小抬升
    alternation_reward = alternation.float() * 0.4       # 交替模式
    target_reward = torch.exp(-height_error / 0.3) * 0.6 # 目标高度
    
    # 4. 膝盖触地强惩罚
    combined_reward[severe_knee_contact] = 0.0
```

**设计要点**：
- 使用 **0.025m 阈值** 判断是否真正抬腿（避免地面接触被误判）
- **分阶段奖励**：先鼓励抬起 → 再鼓励抬高 → 最后鼓励双腿协调
- **膝盖安全检查**：如果膝盖触地，所有奖励清零

#### (2) `_reward_handstand_feet_on_air` - 前腿离地奖励（权重: 1.5）

```python
def _reward_handstand_feet_on_air(self):
    """检查脚部和膝盖的接触状态"""
    # 脚部接触检查
    feet_contact = torch.norm(self.contact_forces[:, feet_indices, :], dim=-1) > 1.0
    
    # 膝盖接触检查
    knee_contact = torch.norm(self.contact_forces[:, knee_indices, :], dim=-1) > 1.0
    
    # 奖励条件：所有脚部未接触 AND 所有膝盖未接触
    reward = ((~feet_contact).float().prod(dim=1) * 
              (~knee_contact).float().prod(dim=1))
    return reward
```

#### (3) `_reward_handstand_orientation_l2` - 目标姿态奖励（权重: 0.8）

```python
def _reward_handstand_orientation_l2(self):
    """惩罚与目标重力方向的偏差"""
    target_gravity = torch.tensor([1, 0.0, 0.0], device=self.device)  # 倒立方向
    return torch.sum((self.projected_gravity - target_gravity) ** 2, dim=1)
```

### 3.3 参数配置

```python
class params:
    handstand_feet_height_exp = {
        "target_height": 0.6,  # 前腿目标高度
        "std": 0.4             # 标准差
    }
    handstand_orientation_l2 = {
        "target_gravity": [1, 0.0, 0.0]  # 目标重力方向（倒立）
    }
    handstand_feet_air_time = {
        "threshold": 5.0       # 腾空时间阈值
    }
    feet_name_reward = {
        "feet_name": "F.*_FOOT"  # 前腿足端正则表达式
    }
```

---

## 4. Isaac Lab 奖励函数设计

基于 Isaac Gym 的成功经验，我们在 Isaac Lab 中设计了对应的奖励函数。

### 4.1 奖励函数分层设计

| 层次 | 优先级 | 目标 | Isaac Gym 参考 | Isaac Lab 实现 |
|------|-------|------|---------------|----------------|
| **第一层** | ★★★ | 站立起来 | `handstand_feet_height_exp` (17.5) | `front_legs_height_exp` (+15.0) |
|  |  |  | `handstand_orientation_l2` (0.8) | `flat_orientation_l2` (-30.0) |
|  |  |  |  | `base_height_l2` (-50.0) |
|  |  |  |  | `front_legs_fixed_pose` (-25.0) |
|  |  |  | `handstand_feet_on_air` (1.5) | `front_legs_no_contact` (-20.0) |
| **第二层** | ★★ | 保持平衡 | `collision` (-1.0) | `undesired_contacts` (-2.0) |
|  |  |  | `stand_still` (-0.8) | `stand_still` (-1.0) |
|  |  |  |  | `biped_balance` (-8.0) |
|  |  |  |  | `lin_vel_z_l2` (-4.0) |
| **第三层** | ★ | 学习行走 | `tracking_lin_vel` (10.0) | `track_lin_vel_xy_exp` (+0.5) |
|  |  |  | `tracking_ang_vel` (5.0) | `track_ang_vel_z_exp` (+0.3) |
|  |  |  | `handstand_feet_air_time` (1.5) | `biped_feet_air_time` (+0.8) |

### 4.2 关键差异说明

| 差异点 | Isaac Gym | Isaac Lab | 原因 |
|--------|-----------|-----------|------|
| 权重符号 | 正值为奖励 | 负值为惩罚 | 框架惯例不同 |
| 站立方向 | 倒立（前腿朝上） | 后腿站立 | 任务目标不同 |
| 速度跟踪权重 | 10.0 (高) | 0.5 (低) | 优先站立 |
| 姿态目标 | `[1,0,0]` (倒立) | `[0,0,-1]` (水平) | 任务目标不同 |

### 4.3 核心奖励函数实现（rewards.py）

#### (1) `front_legs_height_exp` - 前腿抬高奖励

```python
def front_legs_height_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float = 0.2,
    lift_threshold: float = 0.025,
) -> torch.Tensor:
    """前腿抬高奖励 (高级版)
    
    参考 gym_legged_robot.py 中的 _reward_handstand_feet_height_exp 实现。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    
    # 1. 抬腿状态判断
    legs_lifted = foot_heights > lift_threshold
    any_leg_lifted = torch.any(legs_lifted, dim=1).float()
    both_legs_lifted = torch.all(legs_lifted, dim=1).float()
    
    # 2. 计算有效抬升量
    lift_amounts = torch.clamp(foot_heights - lift_threshold, min=0.0)
    
    # 3. 分阶段奖励
    base_reward = any_leg_lifted * 0.3
    single_leg_reward = (torch.max(lift_amounts, dim=1)[0] / (target_height - lift_threshold)) * 0.4
    both_legs_reward = both_legs_lifted * 0.5
    min_lift_reward = (torch.min(lift_amounts, dim=1)[0] / (target_height - lift_threshold)) * 0.3
    
    # 4. 目标高度精确奖励
    effective_heights = torch.where(legs_lifted, foot_heights, 
                                     torch.tensor(lift_threshold, device=env.device))
    height_error = torch.sum(torch.square(effective_heights - target_height), dim=1)
    target_reward = torch.exp(-height_error / (std ** 2)) * 0.6
    
    return base_reward + single_leg_reward + both_legs_reward + min_lift_reward + target_reward
```

#### (2) `front_legs_fixed_pose` - 前腿姿态锁定

```python
def front_legs_fixed_pose(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_positions: dict[str, float],
) -> torch.Tensor:
    """惩罚前腿偏离目标固定姿态"""
    asset: Articulation = env.scene[asset_cfg.name]
    
    reward = torch.zeros(env.num_envs, device=env.device)
    for joint_name, target_pos in target_positions.items():
        joint_ids = asset.find_joints(joint_name)[0]
        if len(joint_ids) > 0:
            current_pos = asset.data.joint_pos[:, joint_ids]
            reward += torch.sum(torch.square(current_pos - target_pos), dim=-1)
    
    return reward
```

#### (3) `front_legs_no_contact` - 前腿接触惩罚

```python
def front_legs_no_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """惩罚前腿接触地面（参考 gym 的 _reward_handstand_feet_on_air）"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(
        torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), 
        dim=1
    )[0] > threshold
    
    return torch.any(is_contact, dim=1).float()
```

#### (4) `biped_balance` - 双足平衡奖励

```python
def biped_balance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """双足平衡奖励：惩罚重心在支撑多边形外的情况"""
    asset: RigidObject = env.scene[asset_cfg.name]
    
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    root_pos_w = asset.data.root_pos_w.unsqueeze(1)
    
    foot_pos_rel = foot_pos_w - root_pos_w
    foot_pos_body = torch.zeros_like(foot_pos_rel)
    for i in range(len(asset_cfg.body_ids)):
        foot_pos_body[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, foot_pos_rel[:, i, :]
        )
    
    feet_center_y = torch.mean(foot_pos_body[:, :, 1], dim=1)
    return torch.square(feet_center_y)
```

---

## 5. 配置文件详解

### 5.1 biped_env_cfg.py 关键配置

```python
@configclass
class DeeproboticsLite3BipedEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Lite3 双腿站立行走配置"""
    
    base_link_name = "TORSO"
    rear_foot_link_name = "H[LR]_FOOT"   # 后腿足端
    front_foot_link_name = "F[LR]_FOOT"  # 前腿足端（惩罚接触用）
    
    # 前腿关节
    front_leg_joints = [
        "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
        "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
    ]
    
    # 前腿目标姿态
    front_leg_target_positions = {
        "FL_HipX_joint": 0.0,
        "FL_HipY_joint": -1.5,    # 向上抬起
        "FL_Knee_joint": 2.6,     # 弯曲收起
        "FR_HipX_joint": 0.0,
        "FR_HipY_joint": -1.5,
        "FR_Knee_joint": 2.6,
    }
```

### 5.2 奖励函数配置

```python
def _setup_biped_rewards(self):
    """配置双足站立行走的奖励函数"""
    
    # ==================== 第一层：站立能力 ====================
    self.rewards.base_height_l2.weight = -50.0
    self.rewards.base_height_l2.params["target_height"] = 0.50
    
    self.rewards.flat_orientation_l2.weight = -30.0
    
    self.rewards.front_legs_fixed_pose = RewTerm(
        func=mdp.front_legs_fixed_pose,
        weight=-25.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=self.front_leg_joints),
            "target_positions": self.front_leg_target_positions,
        },
    )
    
    self.rewards.front_legs_no_contact = RewTerm(
        func=mdp.front_legs_no_contact,
        weight=-20.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.front_foot_link_name]),
            "threshold": 1.0,
        },
    )
    
    self.rewards.front_legs_height_exp = RewTerm(
        func=mdp.front_legs_height_exp,
        weight=15.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[self.front_foot_link_name]),
            "target_height": 0.4,
            "std": 0.2,
            "lift_threshold": 0.025,
        },
    )

    # ==================== 第二层：平衡与稳定 ====================
    self.rewards.lin_vel_z_l2.weight = -4.0
    self.rewards.ang_vel_xy_l2.weight = -0.5
    
    self.rewards.biped_balance = RewTerm(
        func=mdp.biped_balance,
        weight=-8.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[self.rear_foot_link_name]),
        },
    )
    
    self.rewards.joint_deviation_l1.weight = -2.0
    self.rewards.undesired_contacts.weight = -2.0
    self.rewards.stand_still.weight = -1.0

    # ==================== 第三层：速度跟踪与步态 ====================
    self.rewards.track_lin_vel_xy_exp.weight = 0.5
    self.rewards.track_ang_vel_z_exp.weight = 0.3
    
    self.rewards.biped_feet_air_time = RewTerm(
        func=mdp.biped_feet_air_time,
        weight=0.8,
        params={
            "command_name": "base_velocity",
            "threshold": 0.3,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.rear_foot_link_name]),
        },
    )
```

---

## 6. 训练与调试

### 6.1 训练命令

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Biped-Deeprobotics-Lite3-v0 \
    --headless \
    --num_envs=4096 \
    --max_iterations=30000
```

### 6.2 TensorBoard 监控指标

| 指标 | 期望值 | 说明 |
|------|-------|------|
| `Rewards/base_height_l2` | 从 -100 → -20 | 高度奖励应该快速收敛 |
| `Rewards/front_legs_height_exp` | 从 0 → 10+ | 前腿抬起程度 |
| `Rewards/flat_orientation_l2` | 接近 0 | 姿态应该保持水平 |
| `Episode_Reward` | 持续上升 | 总奖励 |

### 6.3 常见问题与解决方案

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 机器人摔倒 | 高度/姿态奖励权重不够 | 增加 `base_height_l2` 和 `flat_orientation_l2` 权重 |
| 前腿不抬起 | 前腿奖励权重不够 | 增加 `front_legs_height_exp` 权重 |
| 站立但不走 | 速度跟踪权重太低 | 训练后期逐步增加 `track_lin_vel_xy_exp` |
| 膝盖触地 | 缺少膝盖惩罚 | 增加 `undesired_contacts` 权重 |

### 6.4 调参建议

1. **先站立后行走**：初期训练时将速度跟踪权重降到 0，只关注站立
2. **渐进式训练**：训练 10000 步后逐步增加速度跟踪权重
3. **降低随机化**：禁用 `randomize_push_robot`，双足平衡更难

---

## 附录：权重对照表

| Isaac Gym 奖励 | 权重 | Isaac Lab 奖励 | 权重 |
|---------------|------|---------------|------|
| `handstand_feet_height_exp` | +17.5 | `front_legs_height_exp` | +15.0 |
| `handstand_feet_on_air` | +1.5 | `front_legs_no_contact` | -20.0 |
| `handstand_orientation_l2` | +0.8 | `flat_orientation_l2` | -30.0 |
| `tracking_lin_vel` | +10.0 | `track_lin_vel_xy_exp` | +0.5 |
| `tracking_ang_vel` | +5.0 | `track_ang_vel_z_exp` | +0.3 |
| `stand_still` | -0.8 | `stand_still` | -1.0 |
| `collision` | -1.0 | `undesired_contacts` | -2.0 |
| - | - | `base_height_l2` | -50.0 |
| - | - | `front_legs_fixed_pose` | -25.0 |
| - | - | `biped_balance` | -8.0 |

> **注意**：Isaac Lab 中负权重表示惩罚，正权重表示奖励。两个框架的权重不能直接对比数值大小，应关注相对优先级。
