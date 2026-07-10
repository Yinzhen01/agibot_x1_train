# 当前项目整体运行思路说明

生成时间：2026-06-30  
项目路径：`F:\agibot_x1_train`  
核心任务：`x1_dh_stand`  
机器人模型：X1 12DOF 腿部控制模型  

## 1. 项目一句话概括

当前项目是一个基于 Isaac Gym 的人形机器人强化学习训练工程。它用 PPO 训练 X1 机器人在仿真中完成站立、行走、转向和抗扰动等能力，训练出的策略可以通过 `play.py` 在 Isaac Gym 中回放，也可以导出 JIT/ONNX，再通过 `sim2sim.py` 放到 MuJoCo 里验证。

当前项目的核心思路是：

```text
给机器人速度命令
    ↓
根据步态相位生成左右腿参考摆动姿态
    ↓
策略网络输出 12 个关节动作
    ↓
PD 控制器把动作转成关节力矩
    ↓
Isaac Gym 推进一步仿真
    ↓
根据速度跟踪、姿态稳定、脚步接触、能耗、安全等奖励训练策略
```

## 2. 代码入口与任务注册

任务注册在 `humanoid/envs/__init__.py`：

```python
task_registry.register("x1_dh_stand", X1DHStandEnv, X1DHStandCfg(), X1DHStandCfgPPO())
```

这行把三件事绑定到任务名 `x1_dh_stand`：

| 组件 | 文件 | 作用 |
|---|---|---|
| 环境类 | `humanoid/envs/x1/x1_dh_stand_env.py` | 定义仿真一步怎么走、观测怎么构造、奖励怎么算 |
| 环境配置 | `humanoid/envs/x1/x1_dh_stand_config.py` | 定义机器人资源、动作维度、奖励权重、随机化、命令范围 |
| PPO 配置 | `humanoid/envs/x1/x1_dh_stand_config.py` | 定义网络结构、学习率、迭代次数、runner 类型 |

训练入口：

```bash
python humanoid/scripts/train.py --task=x1_dh_stand --run_name=<run_name> --headless
```

播放入口：

```bash
python humanoid/scripts/play.py --task=x1_dh_stand --load_run=<date_time><run_name>
```

MuJoCo sim2sim：

```bash
python humanoid/scripts/sim2sim.py --task=x1_dh_stand --load_model <exported_policy_dir>
```

## 3. 训练主流程

`humanoid/scripts/train.py` 很薄，主要做三件事：

```python
env, env_cfg = task_registry.make_env(name=args.task, args=args)
ppo_runner, train_cfg, log_dir = task_registry.make_alg_runner(env=env, name=args.task, args=args)
ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=False)
```

也就是：

1. 根据任务名创建 Isaac Gym 环境。
2. 根据 PPO 配置创建 `DHOnPolicyRunner`。
3. 调用 runner 的 `learn()` 开始采样和更新策略。

日志和模型默认保存到：

```text
logs/x1_dh_stand/exported_data/<date_time><run_name>/
```

## 4. 机器人和动作空间

当前项目控制的是 12 个腿部关节：

| 序号 | 关节 |
|---:|---|
| 0 | `left_hip_pitch_joint` |
| 1 | `left_hip_roll_joint` |
| 2 | `left_hip_yaw_joint` |
| 3 | `left_knee_pitch_joint` |
| 4 | `left_ankle_pitch_joint` |
| 5 | `left_ankle_roll_joint` |
| 6 | `right_hip_pitch_joint` |
| 7 | `right_hip_roll_joint` |
| 8 | `right_hip_yaw_joint` |
| 9 | `right_knee_pitch_joint` |
| 10 | `right_ankle_pitch_joint` |
| 11 | `right_ankle_roll_joint` |

默认站姿角：

```python
left_hip_pitch_joint:   0.4
left_hip_roll_joint:    0.05
left_hip_yaw_joint:    -0.31
left_knee_pitch_joint:  0.49
left_ankle_pitch_joint:-0.21
left_ankle_roll_joint:  0.0

right_hip_pitch_joint:  -0.4
right_hip_roll_joint:   -0.05
right_hip_yaw_joint:    0.31
right_knee_pitch_joint: 0.49
right_ankle_pitch_joint:-0.21
right_ankle_roll_joint: 0.0
```

策略网络输出 12 维动作，动作会先裁剪到 `[-clip_actions, clip_actions]`，再通过 PD 控制转成目标关节角偏移：

```text
目标角 = default_dof_pos + action * action_scale
```

当前 `action_scale = 0.5`。

PD 参数：

| 关节类型 | stiffness | damping |
|---|---:|---:|
| hip_pitch | 30 | 3 |
| hip_roll | 40 | 3.0 |
| hip_yaw | 35 | 4 |
| knee_pitch | 100 | 10 |
| ankle_pitch | 35 | 0.5 |
| ankle_roll | 35 | 0.5 |

控制频率由 `decimation = 10` 决定：策略输出一次动作后，底层仿真会用同一个动作推进 10 个物理步。

这里要区分两个时间概念：

```text
physics step：物理仿真的最小步长，当前 sim.dt = 0.001 秒
policy step：策略输出动作的步长，当前 10 个 physics step 才输出一次动作
```

所以当前项目中：

```text
1 policy step = 10 physics steps = 0.01 秒
```

actor 输出的动作不是“下一帧机器人位置”，而是关节控制目标的偏移。环境会把这个动作经过 `action_scale` 和默认关节角转换成目标关节位置，再由 PD 控制器算出关节力矩，最后交给 Isaac Gym 做物理仿真。

直观流程是：

```text
actor 输出 action
    -> action 转成目标关节角
    -> PD 控制器转成 torques
    -> Isaac Gym 根据 torques、质量、摩擦、碰撞等推进物理
    -> 得到新的 obs / reward / done
    -> actor 下一次再根据新 obs 输出动作
```

## 5. 观测设计

当前单帧 actor 观测维度是 `47`：

```text
5 维命令输入
12 维关节位置
12 维关节速度
12 维上一帧动作
3 维 base 角速度
3 维 base 欧拉角
= 47
```

其中 5 维命令输入是：

```text
sin(phase)
cos(phase)
cmd_vel_x
cmd_vel_y
cmd_yaw_vel
```

actor 使用历史堆叠：

```text
frame_stack = 66
num_single_obs = 47
num_observations = 66 * 47 = 3102
```

所以策略看到的不是单帧，而是一段历史窗口。这对行走很重要，因为步态、速度、接触和动作平滑都依赖时间上下文。

critic 使用特权观测，单帧 `73` 维，堆叠 `3` 帧：

```text
num_privileged_obs = 3 * 73 = 219
```

特权观测包含 actor 看不到或不一定可靠的信息，例如：

- base 线速度
- base 角速度
- base 欧拉角
- 参考关节误差 `dof_pos - ref_dof_pos`
- 随机推力/推力矩
- 摩擦系数
- 机体质量扰动
- 站立/接触 mask

这是一种非对称 actor-critic 设计：训练时 critic 可以看更多真值信息，帮助估值更稳定；部署时 actor 只用普通观测。

## 6. 命令与步态相位

训练命令类型配置为：

```python
gait = ["walk_omnidirectional", "stand", "walk_omnidirectional"]
```

命令范围：

| 命令 | 范围 |
|---|---:|
| `lin_vel_x` | `[-0.4, 1.2]` m/s |
| `lin_vel_y` | `[-0.4, 0.4]` m/s |
| `ang_vel_yaw` | `[-0.6, 0.6]` rad/s |

站立判断阈值：

```python
stand_com_threshold = 0.05
```

当速度命令范数小于这个阈值时，环境认为机器人应该站立，并把步态相位归零。

步态周期：

```python
cycle_time = 0.7
```

环境通过 `_get_phase()` 生成连续相位，再通过正弦函数判断哪只脚支撑、哪只脚摆动：

```text
sin(2π phase) >= 0  左脚支撑，右脚摆动
sin(2π phase) <  0  右脚支撑，左脚摆动
|sin| < 0.1          双支撑过渡区
```

## 7. 参考摆腿模板

当前项目没有使用足端 IK 轨迹，而是直接在关节空间定义一个摆腿模板：

```python
final_swing_joint_delta_pos = [
    0.25, 0.05, -0.11, 0.35, -0.16, 0.0,
   -0.25,-0.05,  0.11, 0.35, -0.16, 0.0
]
```

它表示“摆腿峰值时，各腿部关节相对默认站姿的最大偏移量”。它本身不是时间序列，真正让它随时间变化的是 `phase/sin_pos`。

每个时间步都会生成当前参考关节角：

```text
ref_dof_pos = default_dof_pos + phase系数 * final_swing_joint_delta_pos
```

左腿使用正弦波的一半周期，右腿使用另一半周期：

```text
左腿参考偏移 = -sin_pos_l * final_swing_joint_delta_pos[左腿]
右腿参考偏移 =  sin_pos_r * final_swing_joint_delta_pos[右腿]
```

当 `|sin_pos| < 0.1` 时，进入双支撑过渡区，参考偏移清零，让双腿都回到默认站姿附近。

这套设计的直觉是：

```text
final_swing_joint_delta_pos 像关键姿势图
phase/sin_pos 像动画进度条
每一帧根据进度条生成一个参考姿态
```

然后 `ref_joint_pos` 奖励会鼓励真实关节角靠近这个动态参考姿态。

## 8. 环境一步是怎么运行的

一次 `env.step(actions)` 大致流程：

1. 策略输出 12 维动作。
2. 动作被裁剪。
3. 如 `use_ref_actions=True`，会叠加 `ref_action`；当前配置是 `False`，所以默认不叠加。
4. PD 控制器把动作转换为力矩。
5. Isaac Gym 用相同动作推进 `decimation=10` 个物理步。
6. 刷新 root state、dof state、contact force、rigid body state。
7. 更新 base 速度、姿态、脚姿态等中间量。
8. 根据 gait 时间重新采样命令。
9. 施加随机推力或扰动。
10. 检查终止条件。
11. 计算所有启用奖励项。
12. reset 已终止环境。
13. 重新计算 actor/critic 观测。

每次环境 reset 后有 `startup_delay_steps = 10` 个 policy step 的启动等待期。第 0～9 帧命令保持为 0，训练奖励及各奖励项的 episode 累计保持为 0；第 10 帧采样首个运动命令、重置步态相位并开始正常计奖。当前一个 policy step 为 10 ms，因此启动等待约为 100 ms。

终止条件主要包括：

- `base_link` 接触力超过阈值
- episode 超时
- base roll 绝对值大于 `1.5`
- base pitch 绝对值大于 `1.5`

## 9. 训练思路

当前项目的训练目标不是只让机器人“走快”，而是多个目标平衡：

```text
能按命令走
身体不倒
脚不要拖地
脚接触时不要滑
左右脚步态符合相位
关节不要撞限位
动作不要太暴力
低速命令下能站稳
扰动、延迟、摩擦、质量变化下仍然能工作
```

训练设计上有几个关键点：

### 9.1 先给机器人一个步态先验

`final_swing_joint_delta_pos` 提供一个简单的关节空间摆腿模板。它不是硬控制，只是通过 `ref_joint_pos` 奖励引导策略学出“像走路”的腿部节奏。

### 9.2 再用速度奖励让它跟命令

`tracking_lin_vel` 和 `tracking_ang_vel` 让机器人实际速度接近命令速度。这样策略不会只学会原地摆腿，而是要真的向前、侧向或转向移动。

### 9.3 用接触奖励塑造步态

`feet_contact_number`、`feet_air_time`、`feet_clearance`、`foot_slip` 等奖励让机器人形成更合理的支撑/摆动交替，避免拖脚、乱滑或双脚接触节奏混乱。

### 9.4 用稳定性奖励防止摔倒和乱晃

`orientation`、`base_height`、`base_acc`、`vel_mismatch_exp` 让身体保持较平稳的姿态、高度和速度分布。

### 9.5 用能量和限制惩罚约束动作

`torques`、`dof_vel`、`dof_acc`、`action_smoothness`、`dof_pos_limits`、`dof_vel_limits`、`dof_torque_limits` 避免策略用暴力动作、撞限位或高频抖动换取短期奖励。

### 9.6 用 domain randomization 提升泛化

训练时随机化了质量、COM、摩擦、关节阻尼、armature、电机偏移、控制延迟、DOF 延迟、IMU 延迟、推力扰动等。这样策略不会只适应一个“完美仿真世界”，而是更接近 sim2real 需要的鲁棒性。

## 10. PPO 配置

当前 PPO 配置：

| 项目 | 值 |
|---|---:|
| runner | `DHOnPolicyRunner` |
| policy | `ActorCriticDH` |
| algorithm | `DHPPO` |
| actor hidden dims | `[512, 256, 128]` |
| critic hidden dims | `[768, 256, 128]` |
| learning rate | `1e-5` |
| entropy coef | `0.001` |
| gamma | `0.994` |
| lambda | `0.9` |
| num steps per env | `24` |
| num envs | `4096` |
| max iterations | `20000` |
| save interval | `100` |

一次 PPO iteration 大致采样量：

```text
4096 envs * 24 steps = 98304 transitions
```

这是典型 Isaac Gym 大规模并行采样模式：用很多环境并行跑，快速收集经验，然后用 PPO 更新策略。

### 10.1 actor、critic 和 PPO 的关系

当前项目使用 actor-critic 结构，但 actor 和 critic 的职责不一样：

```text
actor：看普通 obs，输出 action，真正控制机器人
critic：看 critic_obs，输出 value，用来估计当前状态未来还能拿多少累计奖励
```

训练时在同一个 policy step 中会同时做两件事：

```python
actions = actor(obs)
value = critic(critic_obs)
```

代码里表现为 `self.alg.act(obs, critic_obs)`，但这不代表 actor 需要 `critic_obs`。`critic_obs` 只是同一步中给 critic 估值用的。部署时通常只保留 actor，控制链路就是：

```text
obs -> actor -> action
```

### 10.2 24 个 policy step 是 rollout，不是滑动窗口

`num_steps_per_env = 24` 表示一次 PPO iteration 中，每个并行环境连续采样 24 个 policy step。它不是滑动窗口，窗口之间不会刻意重叠。

更像这样：

```text
rollout 1: step 0  -> step 23
PPO update
rollout 2: step 24 -> step 47
PPO update
rollout 3: step 48 -> step 71
PPO update
```

不是这样：

```text
window 1: step 0 -> step 23
window 2: step 1 -> step 24
window 3: step 2 -> step 25
```

如果某个环境中途 `done=True`，说明这一局 episode 结束，例如摔倒或超时。该环境会 reset，并在同一个 rollout 里继续采样新 episode 的数据。

这 24 步数据也不是按 `1-24`、`2-24`、`3-24` 这种滑动时间片反复构造训练样本。当前项目会把所有并行环境采到的数据合成一整批：

```text
4096 envs * 24 policy steps = 98304 transitions
```

每一条 transition 都有自己的：

```text
obs
action
reward
done
old value
old action_log_prob
return
advantage
```

计算 `return / advantage` 时会沿时间从后往前用到同一个环境后续步的 reward；但进入 PPO update 时，训练样本仍然是一条条 transition，而不是滑动窗口片段。

### 10.3 reward、value、return、advantage 的含义

PPO 更新 actor 时，不是只看当前这一步 reward 高不高，而是看这个动作“比原本预期好多少”。

几个量的含义是：

```text
reward：环境真实给出的即时奖励
value：critic 对当前状态未来累计奖励的预测
return：用真实 reward 加上 bootstrap 算出来的训练目标
advantage：return - value，表示实际结果比 critic 原先预测好多少
```

直观例子：

```text
critic 原本预测当前状态未来大概值 10 分
actor 做了某个动作后，后续真实奖励推出来大概值 14 分
advantage = 14 - 10 = +4
```

这说明这个动作比预期更好，PPO 会提高 actor 在类似状态下选择这个动作的概率。

反过来：

```text
critic 原本预测 10 分
后续真实结果只有 6 分
advantage = 6 - 10 = -4
```

这说明这个动作比预期更差，PPO 会降低 actor 在类似状态下选择这个动作的概率。

### 10.4 critic 不是只预测 24 步之后的得分

critic 的目标不是“预测 24 步之后那一刻的分数”，而是预测：

```text
从当前状态开始，未来还能拿到多少折扣累计奖励
```

但是训练时不可能每次都等完整 episode 结束，所以当前项目每次真实采样 24 个 policy step，然后用 critic 对第 24 步之后的未来继续估值。

可以近似理解为：

```text
return_t =
  r_t
+ gamma * r_t+1
+ gamma^2 * r_t+2
+ ...
+ gamma^23 * r_t+23
+ gamma^24 * V(s_t+24)
```

其中：

```text
24 步以内：用真实采到的 reward
24 步以后：用 critic 的 V(s_t+24) 补上
```

这叫 bootstrap。它的作用是：每次只采一小段真实未来，也能让 critic 和 actor 学到更长远的效果。

当前项目中：

```text
24 policy steps = 24 * 0.01 秒 = 0.24 秒
```

也就是说，一次 PPO update 真实看到约 0.24 秒的未来，但 value/return 的含义仍然是更长远的累计奖励估计。

如果某个环境在这 24 步中 `done=True`，终止之后就不会继续用后续 value 补未来，因为这一局已经结束。

### 10.5 actor 和 critic 什么时候更新

actor 不是每个 policy step 都更新参数。rollout 采样期间，actor 只是根据当前参数输出动作，critic 只是根据当前参数输出 value，二者参数都不变。

一次 PPO iteration 的节奏是：

```text
1. actor/critic 推理，连续采样 24 个 policy step
2. 用 reward 和 critic value 计算 return / advantage
3. PPO update：用 advantage 更新 actor，用 return 更新 critic
4. 进入下一轮 rollout
```

actor 的更新目标：

```text
advantage > 0：提高这个动作的概率
advantage < 0：降低这个动作的概率
```

critic 的更新目标：

```text
让 critic(critic_obs) 更接近 return
```

PPO 还会用 clip 限制 actor 每次变化不要太大，避免策略因为一次更新跳得过猛导致训练崩掉。

当前项目中，一次 PPO update 也不是只做一次参数更新。配置是：

```python
num_learning_epochs = 2
num_mini_batches = 4
```

所以一次 PPO iteration 采完 24 步 rollout 后，会把 `98304` 条 transition 打乱并分成 `4` 个 mini-batch，然后重复训练 `2` 个 epoch：

```text
2 epochs * 4 mini-batches = 8 次 optimizer.step()
```

也就是说：

```text
24 是每个环境的采样步数
8 是这批数据上的梯度更新次数
```

每次 `optimizer.step()` 都会同时更新 actor、critic 和 state_estimator，因为它们都在同一个 `actor_critic.parameters()` 里。每条 transition 平均会被用 `2` 次，因为 `num_learning_epochs = 2`。

## 11. 奖励计算机制

所有奖励权重配置在 `X1DHStandCfg.rewards.scales` 中。

环境初始化时，`LeggedRobot._prepare_reward_function()` 会：

1. 把权重为 0 的奖励项移除。
2. 根据奖励名寻找对应函数 `_reward_<name>()`。
3. 将非零奖励权重乘以 `dt`。
4. 每一步调用这些奖励函数并累加。

当前配置：

```python
only_positive_rewards = True
```

所以总奖励在累加后会被裁剪到非负：

```text
total_reward = max(total_reward, 0)
```

这可以避免训练早期负奖励过大导致价值函数崩掉，但也意味着部分惩罚项会通过“压低总奖励”而不是直接形成很大的负回报来起作用。

## 12. 当前项目所有启用奖励项

下面列的是当前项目配置中权重非零、会实际参与训练总奖励的奖励项。表中的权重是配置文件中的原始权重，实际计算时还会乘以 `dt`。

### 12.1 参考姿态与默认姿态

| 奖励项 | 权重 | 作用 |
|---|---:|---|
| `ref_joint_pos` | 2.2 | 鼓励当前关节角接近 `compute_ref_state()` 生成的动态摆腿参考。站立命令下目标改为默认站姿。 |
| `default_joint_pos` | 1.0 | 主要约束髋 yaw/roll 和踝 roll 等不要偏离默认姿态太多，防止腿部姿态扭曲。 |
| `stand_still` | 2.5 | 当速度命令接近 0 时，鼓励所有关节靠近默认站姿；非站立命令时该项基本为 0。 |

### 12.2 足端与步态接触

| 奖励项 | 权重 | 作用 |
|---|---:|---|
| `feet_clearance` | 1.0 | 鼓励摆动脚达到目标抬脚高度区间，减少拖脚。目标高度 `0.03`，最大高度 `0.06`。 |
| `feet_contact_number` | 2.0 | 鼓励脚的接触状态符合步态相位：支撑脚接触、摆动脚离地。站立时两脚都应接触。 |
| `feet_air_time` | 1.2 | 鼓励摆动脚有合理离地时间，帮助形成迈步动作。 |
| `foot_slip` | -0.1 | 惩罚脚接触地面时的水平滑动，减少打滑。 |
| `feet_distance` | 0.2 | 约束双脚距离不要过近或过远，防止交叉腿或劈得太开。 |
| `knee_distance` | 0.2 | 约束双膝距离，避免膝盖互相贴近或姿态异常。 |
| `feet_contact_forces` | -0.01 | 惩罚脚底接触力超过 `max_contact_force` 的部分，避免砸地。 |
| `feet_rotation` | 0.3 | 鼓励脚部姿态更平，减少脚 roll/pitch 旋转过大。 |

### 12.3 速度跟踪

| 奖励项 | 权重 | 作用 |
|---|---:|---|
| `tracking_lin_vel` | 1.8 | 鼓励 xy 平面线速度跟随命令。站立命令下使用绝对误差形式，更偏向静止。 |
| `tracking_ang_vel` | 1.1 | 鼓励 yaw 角速度跟随命令。站立命令下同样更偏向静止。 |
| `track_vel_hard` | 0.5 | 额外的硬速度跟踪项，同时考虑线速度误差和角速度误差，并加入线性误差惩罚。 |
| `low_speed` | 0.2 | 根据 x 方向速度是否太慢、太快、方向相反给奖惩；主要防止命令向前时原地摆腿或反向走。 |

### 12.4 身体稳定性

| 奖励项 | 权重 | 作用 |
|---|---:|---|
| `orientation` | 1.0 | 鼓励 base 姿态保持水平，惩罚 roll/pitch 倾斜和重力投影偏差。 |
| `base_height` | 0.2 | 鼓励 base 高度接近 `base_height_target = 0.61`。 |
| `base_acc` | 0.2 | 鼓励 root 速度变化更平滑，减少身体突然加速或抖动。 |
| `vel_mismatch_exp` | 0.5 | 惩罚 z 方向线速度和 roll/pitch 角速度，减少上下窜和身体晃动。 |

### 12.5 能量、平滑与安全

| 奖励项 | 权重 | 作用 |
|---|---:|---|
| `action_smoothness` | -0.002 | 惩罚连续动作的一阶、二阶变化，减少高频抖动。 |
| `torques` | `-8e-9` | 惩罚关节力矩平方和，降低能耗和暴力控制。 |
| `dof_vel` | `-2e-8` | 惩罚关节速度平方和。 |
| `dof_acc` | `-1e-7` | 惩罚关节加速度平方和，让动作更平滑。 |
| `collision` | -1.0 | 惩罚指定身体部位发生非期望碰撞。当前主要关注 `base_link`。 |
| `dof_vel_limits` | -1.0 | 惩罚关节速度接近或超过限制。 |
| `dof_pos_limits` | -10.0 | 惩罚关节位置接近或超过限制，权重较大。 |
| `dof_torque_limits` | -0.1 | 惩罚关节力矩超过软限制。 |

### 12.6 TensorBoard 中怎么看这些奖励

TensorBoard 是训练仪表盘。它不参与训练，只负责读取训练日志并把曲线画出来。

在 TensorBoard 里：

```text
同一张图 = 同一个变量
不同颜色 = 不同 run，也就是不同次训练实验
浅色线 = 原始记录值，波动更明显
深色线 = 平滑后的趋势线，用来看整体方向
横坐标 = PPO learning iteration
纵坐标 = 该变量的数值
```

当前训练中，1 个 iteration 表示每个环境采样 `24` 个 policy step。比如 `NUM_ENVS=3000` 时：

```text
1 iteration ≈ 3000 * 24 = 72000 条 transition
```

TensorBoard 里常见分组如下：

| 分组 | 代表含义 | 重点看什么 |
|---|---|---|
| `Train/*` | 总体训练表现 | `mean_reward` 越高越好，`mean_episode_length` 越长通常越稳 |
| `Loss/*` | PPO 和估计器损失 | 不要求单调下降，但不能长期爆炸或 NaN |
| `Policy/*` | 策略探索状态 | `mean_noise_std` 表示动作随机探索幅度 |
| `Perf/*` | 训练速度 | `total_fps` 越高训练越快，`collection time` 过大说明采样慢 |
| `Observation/*` | 观测均值/方差 | 用来排查观测是否异常、爆炸、长期为 0 或 NaN |
| `Episode/*` | 每个 episode 结束后统计的奖励分项 | 判断机器人具体是哪方面变好或变差 |

`Episode/rew_xxx` 是“每个 episode 内该奖励项的平均贡献”。它已经乘过 reward scale，所以可以直接用来比较趋势。通俗判断：

```text
正奖励项：整体越高越好
负惩罚项：整体越接近 0 越好
曲线偶尔抖动正常，重点看深色趋势线
```

TensorBoard 中各个 `Episode/rew_*` 的含义：

| TensorBoard 名称 | 简单含义 | 希望看到的趋势 |
|---|---|---|
| `Episode/rew_ref_joint_pos` | 关节是否接近参考步态或站立姿态 | 越高越好 |
| `Episode/rew_default_joint_pos` | 髋 yaw/roll、踝 roll 等是否保持自然默认姿态 | 越高越好 |
| `Episode/rew_stand_still` | 零速度命令时是否能安静站住 | 越高越好 |
| `Episode/rew_feet_clearance` | 摆动脚是否抬到合适高度，避免拖脚 | 越高越好 |
| `Episode/rew_feet_contact_number` | 左右脚接触是否符合步态相位 | 越高越好 |
| `Episode/rew_feet_air_time` | 脚是否有合理腾空时间，避免贴地小碎步 | 越高越好 |
| `Episode/rew_foot_slip` | 脚落地时是否打滑 | 负数越接近 0 越好 |
| `Episode/rew_feet_distance` | 两脚距离是否合理，避免交叉或过宽 | 越高越好 |
| `Episode/rew_knee_distance` | 两膝距离是否合理，避免膝盖互相靠太近 | 越高越好 |
| `Episode/rew_feet_contact_forces` | 足底冲击力是否过大 | 负数越接近 0 越好 |
| `Episode/rew_feet_rotation` | 脚掌是否尽量放平 | 越高越好 |
| `Episode/rew_tracking_lin_vel` | 实际前后/左右速度是否跟随命令 | 越高越好 |
| `Episode/rew_tracking_ang_vel` | 实际转向速度是否跟随命令 | 越高越好 |
| `Episode/rew_track_vel_hard` | 更严格的速度和转向综合跟踪 | 越高越好，明显为负说明跟踪差 |
| `Episode/rew_low_speed` | 防止该走时太慢、反向走或速度不合理 | 越高越好，负值说明速度问题明显 |
| `Episode/rew_vel_mismatch_exp` | 抑制身体上下窜和 roll/pitch 晃动 | 越高越好 |
| `Episode/rew_orientation` | 身体是否保持水平，不前后左右歪 | 越高越好 |
| `Episode/rew_base_height` | 机身高度是否接近目标高度 | 越高越好 |
| `Episode/rew_base_acc` | 身体运动是否平顺，少突然加速和抖动 | 越高越好 |
| `Episode/rew_action_smoothness` | 动作是否平滑，是否少高频抖动 | 负数越接近 0 越好 |
| `Episode/rew_torques` | 电机力矩是否小，是否省力 | 负数越接近 0 越好 |
| `Episode/rew_dof_vel` | 关节速度是否过大 | 负数越接近 0 越好 |
| `Episode/rew_dof_acc` | 关节加速度是否过大，动作是否冲 | 负数越接近 0 越好 |
| `Episode/rew_collision` | 是否有非期望身体碰撞 | 负数越接近 0 越好 |
| `Episode/rew_dof_pos_limits` | 关节位置是否接近或超过限位 | 通常应接近 0 |
| `Episode/rew_dof_vel_limits` | 关节速度是否接近或超过限位 | 通常应接近 0 |
| `Episode/rew_dof_torque_limits` | 关节力矩是否接近或超过限位 | 通常应接近 0 |
| `Episode/terrain_level` | 当前课程地形难度 | 逐步升高说明能力在提升 |
| `Episode/max_command_x` | 当前最大前进速度命令范围 | 反映速度课程范围 |

实际看训练时，先盯住这几条就够：

```text
Train/mean_reward
Train/mean_episode_length
Episode/rew_tracking_lin_vel
Episode/rew_ref_joint_pos
Episode/rew_orientation
Episode/rew_foot_slip
Episode/rew_collision
Episode/rew_action_smoothness
Perf/total_fps
```

如果 `mean_reward` 上升、`mean_episode_length` 变长、速度跟踪和姿态奖励提高，同时打滑/碰撞/抖动惩罚接近 0，通常说明训练方向是好的。

## 13. 已定义但当前未启用的奖励函数

环境代码中还定义了以下奖励函数，但当前 `rewards.scales` 没有给它们配置非零权重，所以默认不会进入总奖励：

| 函数 | 未启用原因 | 可能用途 |
|---|---|---|
| `_reward_ankle_torques` | 没有 `ankle_torques` 权重 | 单独惩罚踝关节力矩，避免脚踝过度用力 |
| `_reward_termination` | 没有 `termination` 权重 | 对非 timeout 的终止提供额外惩罚 |
| `_reward_feet_stumble` | 没有 `feet_stumble` 权重 | 惩罚脚撞到垂直障碍或异常绊脚 |

如果要启用，只需要在 `rewards.scales` 中添加对应名称和非零权重，但要注意奖励名必须和函数名去掉 `_reward_` 后一致。

## 14. Domain Randomization 思路

当前项目随机化比较强，主要是为了提高策略鲁棒性：

| 随机化项 | 当前设置 | 意义 |
|---|---|---|
| base mass | `[-3, 3]` | 模拟质量偏差 |
| COM | 开启 | 模拟质心偏差 |
| friction | 开启 | 模拟地面摩擦变化 |
| motor offset | `[-0.035, 0.035]` | 模拟电机零位误差 |
| joint friction/damping/armature | 开启 | 模拟关节物理参数误差 |
| action lag | `[5, 40]` | 模拟控制延迟 |
| dof lag | `[0, 40]` | 模拟关节状态延迟 |
| IMU lag | 开启 | 模拟 IMU 延迟 |
| push robots | 开启 | 模拟外力扰动 |

这类随机化会让训练更难，但训练出来的策略更不容易只适应单一仿真参数。

## 15. Play、导出与 sim2sim

训练完成后，通常流程是：

1. 用 `play.py` 在 Isaac Gym 中加载 checkpoint，看机器人是否能稳定走。
2. 用 `export_policy_dh.py` 导出 JIT 策略。
3. 用 `export_onnx_dh.py` 导出 ONNX 策略。
4. 用 `sim2sim.py` 把导出的策略放到 MuJoCo 模型中验证。

`play.py` 默认会把训练配置改成更适合测试：

- 降低环境数量
- 关闭地形 curriculum
- 关闭大部分随机化
- 加长 episode
- 加载已训练 checkpoint

`sim2sim.py` 会重新构造和训练时一致的 observation：

```text
phase sin/cos
速度命令
关节位置/速度
上一帧动作
base 角速度
base 欧拉角
历史观测堆叠
```

因此，训练配置中的观测维度、动作维度、关节顺序和 `default_joint_angles` 必须和导出/部署端保持一致。

## 16. 当前项目调参时最重要的几个旋钮

| 目标 | 优先看这些参数 |
|---|---|
| 走不起来、只原地摆腿 | `tracking_lin_vel`、`low_speed`、`ref_joint_pos` |
| 抬脚不足、拖地 | `feet_clearance`、`target_feet_height`、`final_swing_joint_delta_pos` 的膝/髋 pitch |
| 脚滑 | `foot_slip`、摩擦随机化、接触力相关参数 |
| 身体晃或容易倒 | `orientation`、`base_height`、`vel_mismatch_exp`、`base_acc` |
| 动作太暴力 | `torques`、`dof_vel`、`dof_acc`、`action_smoothness` |
| 步态太保守、走不快 | 命令范围、`cycle_time`、`final_swing_joint_delta_pos`、速度跟踪权重 |
| 训练不稳定 | 随机化范围、学习率、`only_positive_rewards`、奖励权重比例 |

## 17. 远程云桌面与一键部署

当前项目已经迁入 Gradmotion GUI 云桌面和远端部署工具，入口集中在 `ops/`：

| 能力 | 入口 |
|---|---|
| Gradmotion GUI 云桌面引导 | `ops/gradmotion/bootstrap-gui-desktop.sh` |
| Codex 反向 SSH 隧道 | `ops/gradmotion/start-codex-tunnel.sh` |
| GUI smoke / viewer / train / play / TensorBoard | `ops/gradmotion/gui-desktop-train.sh` |
| 远端一键打包、上传、部署 | `ops/gradmotion/deploy_x1_remote.sh` |
| gm-cli 云任务模板和产物下载 | `ops/gm-cli/` |

常用命令：

```bash
bash ops/gradmotion/start-codex-tunnel.sh
bash ops/gradmotion/gui-desktop-train.sh gui-env
bash ops/gradmotion/gui-desktop-train.sh open-app xclock
bash ops/gradmotion/gui-desktop-train.sh gui-single
NUM_ENVS=16 MAX_ITERATIONS=10 bash ops/gradmotion/gui-desktop-train.sh gui-smoke
NUM_ENVS=4096 MAX_ITERATIONS=3000 RUN_NAME=x1_dh_stand_v1 bash ops/gradmotion/gui-desktop-train.sh train
LOAD_RUN=x1_dh_stand_v1 bash ops/gradmotion/gui-desktop-train.sh play
bash ops/gradmotion/gui-desktop-train.sh tensorboard
```

远端部署：

```bash
bash ops/gradmotion/deploy_x1_remote.sh --package-only
bash ops/gradmotion/deploy_x1_remote.sh --server root@SERVER_IP --conda-env pointfoot_legged_gym --gpu 0
```

详细文档：

```text
docs/ops/x1_remote_training_deployment.md
docs/ops/gradmotion_codex_gui_minimal_repro.md
docs/ops/gradmotion_reverse_ssh_gui_workflow.md
docs/ops/codex_gradmotion_gui_operation_principles.md
docs/ops/gm_cli_task_submission_workflow.md
docs/ops/cloud_task_artifact_layout.md
```

注意：`ops/gm-cli/payloads/*.local.json`、`ops/gm-cli/accounts.local.json`、`cloud_artifacts/` 和 `outputs/` 是本地文件，不应提交。公钥 `ops/gradmotion/codex_gradmotion.pub` 可以随仓库保留；私钥不能入库。

## 18. 总结

当前项目的训练路线可以概括为：

```text
用正弦相位 + 关节偏移模板提供基础步态先验
用速度跟踪奖励驱动机器人按命令移动
用足端接触奖励塑造支撑/摆动节奏
用姿态和高度奖励保证身体稳定
用能量、平滑、限位和碰撞惩罚约束动作质量
用大量 domain randomization 提高泛化能力
最后通过 play / export / sim2sim 验证策略能否迁移到其他仿真或部署链路
```

它的优点是结构清晰、训练目标完整、鲁棒性考虑充分。它的局限是参考步态仍然是关节空间模板，而不是显式足端轨迹或全身运动学规划；如果后续要改善脚端轨迹质量，可以考虑引入足端 minimum-jerk + IK，但需要同步重写参考生成、奖励权重和 sim2sim 对齐逻辑。

