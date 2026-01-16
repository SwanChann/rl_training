# 四足机器人双腿站立行走训练指南

本文档详细介绍如何基于 Isaac Lab 框架训练 Lite3 四足机器人实现**双腿站立行走**（Bipedal Walking）。

> 参考实现：`gym_legged_robot.py` (Isaac Gym 成功案例)

---

## 1. 奖励函数层次总览

根据 gym 参考实现的 `reward scales` 配置，奖励函数按权重大小分为以下层次：

### 第一层：核心奖励（权重 |scale| > 10）

| 奖励函数 | 权重 | 类型 | 来源 |
|---------|------|------|------|
| `handstand_feet_height_exp` | **+17.5** | 正向奖励 | gym 实现 |
| `tracking_lin_vel` | +10.0 | 正向奖励 | 基础功能 |
| `tracking_ang_vel` | +5.0 | 正向奖励 | 基础功能 |

### 第二层：重要奖励（权重 1 < |scale| ≤ 10）

| 奖励函数 | 权重 | 类型 | 来源 |
|---------|------|------|------|
| `handstand_feet_on_air` | +1.5 | 正向奖励 | gym 实现 |
| `handstand_feet_air_time` | +1.5 | 正向奖励 | gym 实现 |
| `collision` | -1.0 | 惩罚 | 基础安全 |

### 第三层：辅助奖励（权重 |scale| ≤ 1）

| 奖励函数 | 权重 | 类型 | 来源 |
|---------|------|------|------|
| `handstand_orientation_l2` | +0.8 | 姿态控制 | gym 实现 |
| `stand_still` | -0.8 | 惩罚 | 稳定性 |
| `ang_vel_xy` | -0.3 | 惩罚 | 姿态稳定 |
| `action_rate` | -0.03 | 惩罚 | 平滑性 |

### 第四层：精细调节（权重 |scale| < 0.01）

| 奖励函数 | 权重 | 类型 | 来源 |
|---------|------|------|------|
| `torques` | -1e-5 | 惩罚 | 能耗优化 |
| `dof_acc` | -2.5e-7 | 惩罚 | 平滑性 |
| `joint_smoothness` | -2.5e-9 | 惩罚 | 高阶平滑 |

---

## 2. 机器人关节与连杆说明

### 关节名称 (Joints)

Lite3 共有 **12 个关节**，命名格式为 `{Leg}_{Joint}_joint`：

| 腿部代号 | 含义 | 对应关节 |
|---------|------|---------|
| `FL` | 前左腿 | `FL_HipX_joint`, `FL_HipY_joint`, `FL_Knee_joint` |
| `FR` | 前右腿 | `FR_HipX_joint`, `FR_HipY_joint`, `FR_Knee_joint` |
| `HL` | 后左腿 | `HL_HipX_joint`, `HL_HipY_joint`, `HL_Knee_joint` |
| `HR` | 后右腿 | `HR_HipX_joint`, `HR_HipY_joint`, `HR_Knee_joint` |

**关节功能**：
- `HipX`: 侧摆关节（负责左右摆动）
- `HipY`: 髋关节（负责大腿前后摆动）
- `Knee`: 膝关节（负责小腿伸缩）

### 连杆名称 (Links)

| 连杆名称 | 含义 | 刚体索引(gym) |
|---------|------|---------------|
| `TORSO` | 躯干/基座 | 0 |
| `{Leg}_HIP` | 髋部 | 1,5,9,13 |
| `{Leg}_THIGH` | 大腿 | 2,6,10,14 |
| `{Leg}_SHANK` | 小腿 | 3,7,11,15 |
| `{Leg}_FOOT` | 足端 | 4,8,12,16 |

**重要索引（来自 gym 实现）**：
```python
thigh_indices = [2, 6, 10, 14]    # FL_THIGH, FR_THIGH, HL_THIGH, HR_THIGH
shank_indices = [3, 7, 11, 15]    # FL_SHANK, FR_SHANK, HL_SHANK, HR_SHANK
foot_indices = [4, 8, 12, 16]     # FL_FOOT, FR_FOOT, HL_FOOT, HR_FOOT
front_foot_indices = [4, 8]       # FL_FOOT, FR_FOOT（前腿足端）
hind_foot_indices = [12, 16]      # HL_FOOT, HR_FOOT（后腿足端）
```

---

## 3. 核心奖励函数详解

### 3.1 `handstand_feet_height_exp` (权重: 17.5)

**功能**: 鼓励前腿抬高到目标高度

**gym 实现核心逻辑**:

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

**Isaac Lab 实现** (`front_legs_height_exp`):

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

### 3.2 `handstand_feet_on_air` (权重: 1.5)

**功能**: 奖励前腿完全离地

**gym 实现**:
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

**Isaac Lab 实现** (`front_legs_no_contact`):
```python
def front_legs_no_contact(env, sensor_cfg, threshold=1.0):
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    reward = torch.any(is_contact, dim=1).float()
    return reward
```

### 3.3 `handstand_orientation_l2` (权重: 0.8)

**功能**: 惩罚姿态偏离目标方向

**gym 实现**:
```python
def _reward_handstand_orientation_l2(self):
    target_gravity = torch.tensor([1, 0.0, 0.0], device=self.device)  # 目标：机器人站立
    return torch.sum((self.projected_gravity - target_gravity) ** 2, dim=1)
```

**说明**: `projected_gravity` 是机器人基座坐标系下的重力投影向量：
- 四足站立时：`[0, 0, -1]`（重力指向下）
- 双足站立时：目标为 `[1, 0, 0]`（身体直立）

---

## 4. Isaac Lab 配置对照

### 4.1 环境配置 (`biped_env_cfg.py`)

基于 gym 实现的权重配置，Isaac Lab 版本的奖励权重应调整为：

```python
def _setup_biped_rewards(self):
    """配置双足站立行走的奖励函数"""
    
    # ========== 核心奖励（最高优先级）==========
    
    # 前腿抬高奖励 - 对应 gym 的 handstand_feet_height_exp (17.5)
    self.rewards.front_legs_height_exp = RewTerm(
        func=mdp.front_legs_height_exp,
        weight=17.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["F[LR]_FOOT"]),
            "target_height": 0.6,
            "std": 0.4,
            "lift_threshold": 0.025,
        },
    )
    
    # 速度跟踪 - 对应 gym 的 tracking_lin_vel (10.0)
    self.rewards.track_lin_vel_xy_exp.weight = 10.0
    self.rewards.track_ang_vel_z_exp.weight = 5.0
    
    # ========== 重要奖励 ==========
    
    # 前腿离地奖励 - 对应 gym 的 handstand_feet_on_air (1.5)
    self.rewards.front_legs_no_contact = RewTerm(
        func=mdp.front_legs_no_contact,
        weight=-1.5,  # 改为惩罚形式
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["F[LR]_FOOT"]),
            "threshold": 1.0,
        },
    )
    
    # 碰撞惩罚 - 对应 gym 的 collision (-1.0)
    self.rewards.undesired_contacts.weight = -1.0
    
    # ========== 辅助奖励 ==========
    
    # 姿态控制 - 对应 gym 的 handstand_orientation_l2 (0.8)
    self.rewards.handstand_orientation = RewTerm(
        func=mdp.handstand_orientation_l2,
        weight=-0.8,  # 惩罚偏离目标姿态
        params={
            "target_gravity": [1.0, 0.0, 0.0],
        },
    )
    
    # 静止保持 - 对应 gym 的 stand_still (-0.8)
    self.rewards.stand_still.weight = -0.8
    
    # XY角速度惩罚 - 对应 gym 的 ang_vel_xy (-0.3)
    self.rewards.ang_vel_xy_l2.weight = -0.3
    
    # 动作平滑性 - 对应 gym 的 action_rate (-0.03)
    self.rewards.action_rate_l2.weight = -0.03
    
    # ========== 精细调节 ==========
    
    self.rewards.joint_torques_l2.weight = -1e-5
    self.rewards.joint_acc_l2.weight = -2.5e-7
```

### 4.2 关键参数配置

```python
# gym 配置参数
class params:
    handstand_feet_height_exp = {
        "target_height": 0.6,  # 前腿目标高度
        "std": 0.4             # 高斯核标准差
    }
    handstand_orientation_l2 = {
        "target_gravity": [1, 0.0, 0.0]  # 目标重力方向（站立）
    }
    handstand_feet_air_time = {
        "threshold": 5.0  # 悬空时间阈值
    }
```

---

## 5. 训练命令

```bash
# 启动双足站立训练
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task=Biped-Deeprobotics-Lite3-v0 \
    --headless \
    --num_envs=4096 \
    --max_iterations=10000
```

## 6. 训练监控

使用 TensorBoard 监控关键指标：

```bash
tensorboard --logdir=logs/rsl_rl
```

**关键指标**:
- `Rewards/front_legs_height_exp`: 应逐渐上升，表示前腿抬起
- `Rewards/handstand_orientation`: 应逐渐下降，表示姿态稳定
- `Episode_Reward`: 总奖励，应逐渐上升

---

## 7. 常见问题

### Q1: 机器人倒下怎么办？
A: 检查 `handstand_orientation_l2` 权重是否足够大，增加姿态惩罚。

### Q2: 前腿抬不起来？
A: 增加 `front_legs_height_exp` 权重，降低 `tracking_lin_vel` 权重。

### Q3: 膝盖着地？
A: 检查 `undesired_contacts` 配置，确保包含 `SHANK` 和 `THIGH` 连杆。

### Q4: 训练不稳定？
A: 降低学习率，增加 `action_rate` 惩罚权重。
