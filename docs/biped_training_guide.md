# 四足机器人双腿站立行走训练指南

本文档详细介绍如何基于 Isaac Lab 框架训练 Lite3 四足机器人实现**双腿站立行走**（Bipedal Walking）。

> 参考实现：`gym_legged_robot.py` 和 `gym_legged_robot_config.py` (Isaac Gym 成功案例)

---

## 目录

1. [奖励函数层次总览](#1-奖励函数层次总览)
   - [1.1 Gym 版本奖励权重配置](#11-gym-版本奖励权重配置)
   - [1.2 Isaac Lab 版本奖励权重配置](#12-isaac-lab-版本奖励权重配置)
2. [关节索引对比](#2-关节索引对比)
   - [2.1 Gym 版本关节/刚体索引](#21-gym-版本关节刚体索引)
   - [2.2 Isaac Lab 版本关节配置](#22-isaac-lab-版本关节配置)
3. [关节限位配置](#3-关节限位配置)
4. [核心奖励函数详解](#4-核心奖励函数详解)
5. [Isaac Lab 配置详解](#5-isaac-lab-配置详解)
6. [训练命令与监控](#6-训练命令与监控)
7. [常见问题](#7-常见问题)

---

## 1. 奖励函数层次总览

### 1.1 Gym 版本奖励权重配置

基于 `gym_legged_robot_config.py` 中的 `class rewards.scales` 配置，奖励函数按权重大小分为以下层次：

#### 第一层：核心奖励（权重 |scale| > 10）

| 奖励函数 | 权重 | 类型 | 说明 |
|---------|------|------|------|
| `handstand_feet_height_exp` | **+17.5** | 正向奖励 | 前腿抬高到目标高度 |
| `tracking_lin_vel` | +10.0 | 正向奖励 | 线速度跟踪 |
| `tracking_ang_vel` | +5.0 | 正向奖励 | 角速度跟踪 |

#### 第二层：重要奖励（权重 1 < |scale| ≤ 10）

| 奖励函数 | 权重 | 类型 | 说明 |
|---------|------|------|------|
| `handstand_feet_on_air` | +1.5 | 正向奖励 | 前腿完全离地 |
| `handstand_feet_air_time` | +1.5 | 正向奖励 | 前腿悬空时间 |
| `collision` | -1.0 | 惩罚 | 膝盖/大腿碰撞 |

#### 第三层：辅助奖励（权重 |scale| ≤ 1）

| 奖励函数 | 权重 | 类型 | 说明 |
|---------|------|------|------|
| `handstand_orientation_l2` | +0.8 | 姿态控制 | 惩罚姿态偏离目标 |
| `stand_still` | -0.8 | 惩罚 | 静止时关节抖动 |
| `ang_vel_xy` | -0.3 | 惩罚 | XY 角速度 |
| `action_rate` | -0.03 | 惩罚 | 动作变化率 |

#### 第四层：精细调节（权重 |scale| < 0.01）

| 奖励函数 | 权重 | 类型 | 说明 |
|---------|------|------|------|
| `torques` | -1e-5 | 惩罚 | 能耗优化 |
| `dof_acc` | -2.5e-7 | 惩罚 | 关节加速度 |
| `joint_smoothness` | -2.5e-9 | 惩罚 | 高阶平滑性 |

#### Gym 参数配置

```python
class params:  # gym_legged_robot_config.py
    handstand_feet_height_exp = {
        "target_height": 0.6,  # 前腿目标高度（米）
        "std": 0.4             # 高斯核标准差
    }
    handstand_orientation_l2 = {
        "target_gravity": [1, 0.0, 0.0]  # 目标重力方向（双足站立）
    }
    handstand_feet_air_time = {
        "threshold": 5.0  # 悬空时间阈值（秒）
    }
```

---

### 1.2 Isaac Lab 版本奖励权重配置

基于 `biped_env_cfg.py` 中的 `_setup_biped_rewards()` 方法配置：

#### 第一层：核心奖励（权重 |scale| > 10）

| 奖励函数 (Lab) | 权重 | 对应 Gym 奖励 | 说明 |
|---------------|------|--------------|------|
| `front_legs_height_exp` | **+17.5** | `handstand_feet_height_exp` | 前腿抬高到目标高度 |
| `track_lin_vel_xy_exp` | +10.0 | `tracking_lin_vel` | 线速度跟踪 |
| `track_ang_vel_z_exp` | +5.0 | `tracking_ang_vel` | 角速度跟踪 |

#### 第二层：重要奖励（权重 1 < |scale| ≤ 10）

| 奖励函数 (Lab) | 权重 | 对应 Gym 奖励 | 说明 |
|---------------|------|--------------|------|
| `front_legs_no_contact` | -1.5 | `handstand_feet_on_air` | 惩罚前腿接触地面 |
| `biped_feet_air_time` | +1.5 | `handstand_feet_air_time` | 后腿悬空时间 |
| `undesired_contacts` | -1.0 | `collision` | 膝盖/大腿碰撞 |
| `biped_balance` | -2.0 | - | 双足平衡（新增） |
| `front_legs_fixed_pose` | -5.0 | - | 前腿姿态锁定（新增） |

#### 第三层：辅助奖励（权重 |scale| ≤ 1）

| 奖励函数 (Lab) | 权重 | 对应 Gym 奖励 | 说明 |
|---------------|------|--------------|------|
| `handstand_orientation` | -0.8 | `handstand_orientation_l2` | 姿态偏离惩罚 |
| `stand_still` | -0.8 | `stand_still` | 静止关节抖动 |
| `ang_vel_xy_l2` | -0.3 | `ang_vel_xy` | XY 角速度 |

#### 第四层：精细调节（权重 |scale| < 0.01）

| 奖励函数 (Lab) | 权重 | 对应 Gym 奖励 | 说明 |
|---------------|------|--------------|------|
| `action_rate_l2` | -0.03 | `action_rate` | 动作变化率 |
| `joint_torques_l2` | -1e-5 | `torques` | 力矩惩罚 |
| `joint_acc_l2` | -2.5e-7 | `dof_acc` | 关节加速度 |

---

## 2. 关节索引对比

### 2.1 Gym 版本关节/刚体索引

在 Isaac Gym 中，刚体和关节的索引是按照 URDF 加载顺序硬编码的：

#### 刚体索引 (Rigid Body Indices)

```python
# gym_legged_robot.py 中的索引定义
thigh_indices = [2, 6, 10, 14]    # FL_THIGH, FR_THIGH, HL_THIGH, HR_THIGH
shank_indices = [3, 7, 11, 15]    # FL_SHANK, FR_SHANK, HL_SHANK, HR_SHANK
foot_indices = [4, 8, 12, 16]     # FL_FOOT, FR_FOOT, HL_FOOT, HR_FOOT
front_foot_indices = [4, 8]       # FL_FOOT, FR_FOOT（前腿足端）
hind_foot_indices = [12, 16]      # HL_FOOT, HR_FOOT（后腿足端）
```

#### 完整刚体顺序

| 索引 | 刚体名称 | 含义 |
|-----|---------|------|
| 0 | TORSO | 躯干/基座 |
| 1 | FL_HIP | 前左髋部 |
| 2 | FL_THIGH | 前左大腿 |
| 3 | FL_SHANK | 前左小腿 |
| 4 | FL_FOOT | 前左足端 |
| 5 | FR_HIP | 前右髋部 |
| 6 | FR_THIGH | 前右大腿 |
| 7 | FR_SHANK | 前右小腿 |
| 8 | FR_FOOT | 前右足端 |
| 9 | HL_HIP | 后左髋部 |
| 10 | HL_THIGH | 后左大腿 |
| 11 | HL_SHANK | 后左小腿 |
| 12 | HL_FOOT | 后左足端 |
| 13 | HR_HIP | 后右髋部 |
| 14 | HR_THIGH | 后右大腿 |
| 15 | HR_SHANK | 后右小腿 |
| 16 | HR_FOOT | 后右足端 |

#### 关节索引顺序 (DOF Indices)

Lite3 共有 **12 个关节**：

| 索引 | 关节名称 | 功能 |
|-----|---------|------|
| 0 | FL_HipX_joint | 前左侧摆 |
| 1 | FL_HipY_joint | 前左髋关节 |
| 2 | FL_Knee_joint | 前左膝关节 |
| 3 | FR_HipX_joint | 前右侧摆 |
| 4 | FR_HipY_joint | 前右髋关节 |
| 5 | FR_Knee_joint | 前右膝关节 |
| 6 | HL_HipX_joint | 后左侧摆 |
| 7 | HL_HipY_joint | 后左髋关节 |
| 8 | HL_Knee_joint | 后左膝关节 |
| 9 | HR_HipX_joint | 后右侧摆 |
| 10 | HR_HipY_joint | 后右髋关节 |
| 11 | HR_Knee_joint | 后右膝关节 |

---

### 2.2 Isaac Lab 版本关节配置

Isaac Lab 使用**名称匹配**而非硬编码索引，更加灵活和安全：

#### 关节名称配置 (biped_env_cfg.py)

```python
# 所有关节名称（按顺序）
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
```

#### 连杆名称配置

```python
# 所有连杆名称
link_names = [
    'TORSO', 
    'FL_HIP', 'FR_HIP', 'HL_HIP', 'HR_HIP', 
    'FL_THIGH', 'FR_THIGH', 'HL_THIGH', 'HR_THIGH', 
    'FL_SHANK', 'FR_SHANK', 'HL_SHANK', 'HR_SHANK', 
    'FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT',
]

# 使用正则表达式匹配
rear_foot_link_name = "H[LR]_FOOT"   # 后腿足端：HL_FOOT, HR_FOOT
front_foot_link_name = "F[LR]_FOOT"  # 前腿足端：FL_FOOT, FR_FOOT
```

#### Lab 与 Gym 索引对应关系

| Gym 硬编码索引 | Lab 名称匹配 | 说明 |
|--------------|-------------|------|
| `front_foot_indices = [4, 8]` | `body_names=["F[LR]_FOOT"]` | 前腿足端 |
| `hind_foot_indices = [12, 16]` | `body_names=["H[LR]_FOOT"]` | 后腿足端 |
| `thigh_indices = [2, 6, 10, 14]` | `body_names=[".*THIGH.*"]` | 大腿 |
| `shank_indices = [3, 7, 11, 15]` | `body_names=[".*SHANK.*"]` | 小腿 |

**优势**：Lab 使用正则表达式匹配名称，更灵活且不依赖索引顺序。

---

## 3. 关节限位配置

### 3.1 Isaac Lab 关节限位

在 `deeprobotics.py` 中通过 `ArticulationCfg` 配置：

```python
DEEPROBOTICS_LITE3_CFG = ArticulationCfg(
    ...
    soft_joint_pos_limit_factor=0.99,  # 软限位因子（99% 的硬限位）
    actuators={
        "Hip": DelayedPDActuatorCfg(
            joint_names_expr=[".*_Hip[X,Y]_joint"],
            effort_limit=24.0,      # 力矩限制（N·m）
            velocity_limit=26.2,    # 速度限制（rad/s）
            stiffness=30.0,         # 刚度（N·m/rad）
            damping=1.0,            # 阻尼（N·m·s/rad）
            ...
        ),
        "Knee": DelayedPDActuatorCfg(
            joint_names_expr=[".*_Knee_joint"],
            effort_limit=36.0,      # 膝关节力矩更大
            velocity_limit=17.3,    # 膝关节速度限制
            stiffness=30.0,
            damping=1.0,
            ...
        ),
    },
)
```

### 3.2 URDF 中的关节限位

Lite3 机器人的关节限位（来自 URDF 文件）：

| 关节类型 | 关节 | URDF 下限 (rad) | URDF 上限 (rad) | 力矩限制 (Nm) | 速度限制 (rad/s) |
|---------|-----|----------------|----------------|--------------|----------------|
| HipX | `.*_HipX_joint` | -0.523 | +0.523 | 24 | 26.2 |
| HipY | `.*_HipY_joint` | -2.67 | +0.314 | 24 | 26.2 |
| Knee | `.*_Knee_joint` | +0.524 | +2.792 | 36 | 17.3 |

### 3.3 Policy 控制范围配置

为保护机器人硬件，**Policy 能控制的关节位置范围应小于 URDF 硬限位**。

在 `biped_env_cfg.py` 中通过 `clip` 参数配置：

| 关节类型 | URDF 硬限位 | Policy 控制范围 | 缩减比例 | 说明 |
|---------|-----------|----------------|---------|------|
| HipX | [-0.523, +0.523] | **[-0.45, +0.45]** | ~86% | 侧摆留出安全裕度 |
| HipY | [-2.67, +0.314] | **[-2.3, +0.25]** | ~85% | 髋关节限制范围 |
| Knee | [+0.524, +2.792] | **[+0.6, +2.5]** | ~84% | 膝关节限制范围 |

**配置代码**：

```python
# biped_env_cfg.py 中的 Actions 配置
self.actions.joint_pos.clip = {
    ".*_HipX_joint": (-0.45, 0.45),     # 硬限位 ±0.523，缩小到 ±0.45
    ".*_HipY_joint": (-2.3, 0.25),      # 硬限位 [-2.67,0.314]，缩小范围
    ".*_Knee_joint": (0.6, 2.5),        # 硬限位 [0.524,2.792]，缩小范围
}
```

**动作映射公式**：

```
target_position = clip(action * scale + default_offset, min_limit, max_limit)
```

其中：
- `action`: Policy 输出，范围 [-1, 1]
- `scale`: 动作缩放因子（HipX=0.125，其他=0.25）
- `default_offset`: 默认关节角度（HipX=0, HipY=-0.8, Knee=1.6）
- `clip`: 将结果限制在 Policy 控制范围内

### 3.4 双足站立的关节限位建议

对于双足站立训练，通过 `clip` 配置限制 Policy 的输出范围，确保安全：

#### 前腿（需要抬起并锁定）

```python
# 前腿目标姿态（收起状态）
front_leg_target_positions = {
    "FL_HipX_joint": 0.0,      # 侧摆：保持中立
    "FL_HipY_joint": -1.5,     # 髋关节：向上抬起（在 Policy 范围 [-2.3, 0.25] 内）
    "FL_Knee_joint": 2.6,      # 膝关节：弯曲收起（接近 Policy 上限 2.5，会被 clip）
    "FR_HipX_joint": 0.0,
    "FR_HipY_joint": -1.5,
    "FR_Knee_joint": 2.6,
}
```

> 注意：前腿 Knee 目标值 2.6 超过 Policy clip 上限 2.5，会被自动限制为 2.5
```

#### 后腿（主要行走）

后腿使用 Policy clip 配置的限位范围，同时可以通过奖励函数限制 HipX 偏离：

```python
# 后腿关节 Policy 控制范围
# HipX: [-0.45, +0.45] rad - 侧摆限制
# HipY: [-2.3, +0.25] rad - 髋关节大范围运动
# Knee: [+0.6, +2.5] rad - 膝关节弯曲范围

# 惩罚后腿 HipX 偏离中立位置（可选）
self.rewards.joint_deviation_l1.params["asset_cfg"].joint_names = ["H[LR]_HipX.*"]
```

### 3.5 动作缩放配置

```python
# biped_env_cfg.py 中的动作缩放
self.actions.joint_pos.scale = {
    ".*_HipX_joint": 0.125,  # HipX 缩小动作范围（更稳定）
    "^(?!.*_HipX_joint).*": 0.25  # 其他关节正常范围
}
```

---

## 4. 核心奖励函数详解

### 4.1 `front_legs_height_exp` (权重: +17.5)

**功能**: 鼓励前腿抬高到目标高度

**Gym 实现核心逻辑** (`_reward_handstand_feet_height_exp`):

```python
# 抬腿阈值：高于 0.025m 才算抬离地面
LIFT_THRESHOLD = 0.025

# 判断抬腿状态
left_leg_lifted = front_left_height > LIFT_THRESHOLD
right_leg_lifted = front_right_height > LIFT_THRESHOLD
both_legs_lifted = left_leg_lifted & right_leg_lifted
any_leg_lifted = left_leg_lifted | right_leg_lifted

# 组合奖励系统
height_reward = (
    base_lift_reward +           # 基础抬腿奖励 (0.3)
    single_leg_reward +          # 单腿抬高奖励 (0.4)
    both_legs_reward +           # 双腿协调奖励 (0.5)
    min_lift_reward +            # 最小抬腿奖励 (0.3)
    alternation_reward +         # 交替模式奖励 (0.4)
    target_reward                # 目标高度奖励 (0.6)
)

# 膝盖安全检查：膝盖触地则奖励清零
combined_reward[severe_knee_contact] = 0.0
```

**Lab 实现** (`front_legs_height_exp` in `rewards.py`):

```python
def front_legs_height_exp(env, asset_cfg, target_height, std=0.2, lift_threshold=0.025):
    # 获取前腿足端高度
    foot_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    
    # 抬腿状态判断
    legs_lifted = foot_heights > lift_threshold
    any_leg_lifted = torch.any(legs_lifted, dim=1).float()
    both_legs_lifted = torch.all(legs_lifted, dim=1).float()
    
    # 分阶段奖励
    base_reward = any_leg_lifted * 0.3
    single_leg_reward = (max_lift / (target_height - lift_threshold)) * 0.4
    both_legs_reward = both_legs_lifted * 0.5
    min_lift_reward = (min_lift / (target_height - lift_threshold)) * 0.3
    target_reward = torch.exp(-height_error / (std ** 2)) * 0.6
    
    return total_reward
```

### 4.2 `handstand_orientation_l2` (权重: -0.8)

**功能**: 惩罚姿态偏离目标方向

**数学表达式**:
```
reward = Σ(projected_gravity - target_gravity)²
```

**目标重力方向**:
- 四足站立：`[0, 0, -1]`（重力指向下）
- 双足站立：`[1, 0, 0]`（身体直立，头朝上）

**Lab 实现**:

```python
def handstand_orientation_l2(env, target_gravity, asset_cfg):
    projected_gravity = asset.data.projected_gravity_b
    target_gravity_tensor = torch.tensor(target_gravity, device=env.device)
    return torch.sum(torch.square(projected_gravity - target_gravity_tensor), dim=1)
```

### 4.3 `front_legs_no_contact` (权重: -1.5)

**功能**: 惩罚前腿接触地面

**Gym 实现** (`_reward_handstand_feet_on_air`):

```python
def _reward_handstand_feet_on_air(self):
    # 检查脚部接触
    feet_contact = torch.norm(self.contact_forces[:, feet_indices, :], dim=-1) > 1.0
    # 检查膝盖接触
    knee_contact = torch.norm(self.contact_forces[:, knee_indices, :], dim=-1) > 1.0
    
    # 奖励：所有脚部未接触 AND 所有膝盖未接触
    reward = (~feet_contact).float().prod(dim=1) * (~knee_contact).float().prod(dim=1)
    return reward
```

**Lab 实现**:

```python
def front_legs_no_contact(env, sensor_cfg, threshold=1.0):
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(
        torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), 
        dim=1
    )[0] > threshold
    return torch.any(is_contact, dim=1).float()
```

---

## 5. Isaac Lab 配置详解

### 5.1 环境配置类 (biped_env_cfg.py)

```python
@configclass
class DeeproboticsLite3BipedEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Lite3 双腿站立行走配置"""
    
    base_link_name = "TORSO"
    rear_foot_link_name = "H[LR]_FOOT"   # 后腿足端
    front_foot_link_name = "F[LR]_FOOT"  # 前腿足端
    
    # 前腿目标姿态（收起状态）
    front_leg_target_positions = {
        "FL_HipX_joint": 0.0,
        "FL_HipY_joint": -1.5,
        "FL_Knee_joint": 2.6,
        "FR_HipX_joint": 0.0,
        "FR_HipY_joint": -1.5,
        "FR_Knee_joint": 2.6,
    }
```

### 5.2 完整奖励配置

```python
def _setup_biped_rewards(self):
    """配置双足站立行走的奖励函数
    
    设计思路（按权重大小分层）：
    第一层（核心）：front_legs_height_exp=17.5, tracking_lin_vel=10, tracking_ang_vel=5
    第二层（重要）：front_legs_no_contact=-1.5, biped_feet_air_time=1.5, undesired_contacts=-1.0
    第三层（辅助）：handstand_orientation=-0.8, stand_still=-0.8, ang_vel_xy=-0.3
    第四层（精细）：action_rate=-0.03, torques=-1e-5, dof_acc=-2.5e-7
    """
    
    # ==================== 禁用四足专用奖励 ====================
    self.rewards.feet_air_time.weight = 0.0
    self.rewards.feet_air_time_variance.weight = 0.0
    
    # ==================== 第一层：核心奖励（|scale| > 10）====================
    
    # 前腿抬高奖励
    self.rewards.front_legs_height_exp = RewTerm(
        func=mdp.front_legs_height_exp,
        weight=17.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[self.front_foot_link_name]),
            "target_height": 0.6,
            "std": 0.4,
            "lift_threshold": 0.025,
        },
    )
    
    # 速度跟踪
    self.rewards.track_lin_vel_xy_exp.weight = 10.0
    self.rewards.track_ang_vel_z_exp.weight = 5.0

    # ==================== 第二层：重要奖励（1 < |scale| ≤ 10）====================
    
    # 前腿离地
    self.rewards.front_legs_no_contact = RewTerm(
        func=mdp.front_legs_no_contact,
        weight=-1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.front_foot_link_name]),
            "threshold": 1.0,
        },
    )
    
    # 后腿悬空时间
    self.rewards.biped_feet_air_time = RewTerm(
        func=mdp.biped_feet_air_time,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "threshold": 5.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.rear_foot_link_name]),
        },
    )
    
    # 碰撞惩罚
    self.rewards.undesired_contacts.weight = -1.0
    self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*THIGH.*", ".*SHANK.*"]
    
    # 双足平衡
    self.rewards.biped_balance = RewTerm(
        func=mdp.biped_balance,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[self.rear_foot_link_name]),
        },
    )
    
    # 前腿姿态锁定
    self.rewards.front_legs_fixed_pose = RewTerm(
        func=mdp.front_legs_fixed_pose,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=self.front_leg_joints),
            "target_positions": self.front_leg_target_positions,
        },
    )

    # ==================== 第三层：辅助奖励（|scale| ≤ 1）====================
    
    self.rewards.handstand_orientation = RewTerm(
        func=mdp.handstand_orientation_l2,
        weight=-0.8,
        params={"target_gravity": [1.0, 0.0, 0.0]},
    )
    
    self.rewards.stand_still.weight = -0.8
    self.rewards.ang_vel_xy_l2.weight = -0.3
    self.rewards.lin_vel_z_l2.weight = 0.0

    # ==================== 第四层：精细调节（|scale| < 0.01）====================
    
    self.rewards.action_rate_l2.weight = -0.03
    self.rewards.joint_torques_l2.weight = -1e-5
    self.rewards.joint_acc_l2.weight = -2.5e-7
```

---

## 6. 训练命令与监控

### 6.1 训练命令

```bash
# 启动双足站立训练
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Biped-Deeprobotics-Lite3-v0 \
    --headless \
    --num_envs=4096 \
    --max_iterations=30000
```

### 6.2 TensorBoard 监控

```bash
tensorboard --logdir=logs/rsl_rl
```

**关键指标**:

| 指标 | 期望趋势 | 说明 |
|-----|---------|------|
| `Rewards/front_legs_height_exp` | 上升 | 前腿抬起 |
| `Rewards/handstand_orientation` | 下降 | 姿态稳定 |
| `Rewards/track_lin_vel_xy_exp` | 上升 | 速度跟踪改善 |
| `Episode_Reward` | 上升 | 整体表现改善 |

---

## 7. 常见问题

### Q1: 机器人倒下怎么办？
A: 检查 `handstand_orientation` 权重是否足够大，增加姿态惩罚。同时可以降低速度命令范围。

### Q2: 前腿抬不起来？
A: 增加 `front_legs_height_exp` 权重，降低 `tracking_lin_vel` 权重，让机器人先学站立再学行走。

### Q3: 膝盖着地？
A: 检查 `undesired_contacts` 配置，确保包含 `SHANK` 和 `THIGH` 连杆。

### Q4: 训练不稳定？
A: 降低学习率，增加 `action_rate_l2` 惩罚权重。可以禁用 `randomize_push_robot` 事件。

### Q5: Gym 和 Lab 训练结果差异大？
A: 检查以下配置是否一致：
- 观测空间维度和缩放
- 动作缩放因子
- 仿真时间步和控制频率
- 随机化强度

---

## 附录：Gym 与 Lab 配置对照表

| 配置项 | Gym 配置 | Lab 配置 |
|-------|---------|---------|
| 仿真时间步 | `dt = 0.005` | `decimation * sim_dt` |
| 控制频率 | `decimation = 4` | 由场景配置决定 |
| 观测空间 | 45维（无高度扫描） | 48维（可配置） |
| 动作缩放 | `action_scale = 0.5` | `scale = {".*_HipX": 0.125, ".*": 0.25}` |
| 前腿目标高度 | `target_height = 0.6` | `target_height = 0.6` |
| 机身目标高度 | `base_height_target = 0.95` | `target_height = 0.95` |
| 抬腿阈值 | `LIFT_THRESHOLD = 0.025` | `lift_threshold = 0.025` |
