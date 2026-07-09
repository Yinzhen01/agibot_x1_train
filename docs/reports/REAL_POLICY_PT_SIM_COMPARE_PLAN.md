# 真机日志与 PT 策略仿真对比方案

生成时间：2026-07-09  
当前项目：`F:\agibot_x1_train`  
真机控制工程参考：`E:\F1`  
真机日志来源：`E:\F1\test_logs\7.7`  
本地策略来源：`C:\Users\HP\Desktop\policy`  
项目内本地工作副本：`work/real_policy_compare/`（已被 `.gitignore` 忽略，不入库）  

## 1. 目标

使用训练得到的 `.pt` 策略文件，在本地完成两类对比：

1. 离线输入复现：读取真机记录的 policy input，直接喂给本地 `.pt`，检查本地输出与真机记录的 `action_*` 是否一致。
2. 仿真闭环对比：在 MuJoCo 或 Isaac Gym 中加载同一 `.pt`，让策略在仿真中闭环运行，再与真机日志中的观测、动作、关节状态和姿态变化进行对齐。

本方案优先保证“模型输入输出复现”成立，再进入仿真闭环。这样可以把模型版本、观测构造和动力学差异分层排查。

## 2. 已确认的数据语义

根据 `E:\F1\src\module\control_module\src\rl_controller.cc`：

- `tm_obs_input_*.bin` 由 `RLController::LogTmData()` 写出。
- 写出内容是 `observations_`，类型为连续 `float32`。
- 每个策略周期写一帧，每帧长度为 `observations_size * num_hist`。
- 当前 `rl_walk_leg` 配置为：
  - `actions_size = 12`
  - `observations_size = 47`
  - `num_hist = 66`
  - 单帧 policy input 长度为 `47 * 66 = 3102`
- `7.7` 的每个 `.bin` 文件大小为 `12,408,000` 字节，即 `1000 * 3102 * sizeof(float32)`，对应 1000 个策略帧。

单个 47 维 observation 的布局为：

| 区间 | 维度 | 含义 |
| --- | ---: | --- |
| `0:2` | 2 | `phase_sin`, `phase_cos` |
| `2:5` | 3 | 速度命令：`cmd_linear_x * 2`、`cmd_linear_y * 2`、`cmd_angular_z` |
| `5:17` | 12 | `(joint_pos - init_state) * dof_pos_scale` |
| `17:29` | 12 | `joint_vel * dof_vel_scale`，当前 `dof_vel_scale = 0.05` |
| `29:41` | 12 | `last_actions_` |
| `41:44` | 3 | `base_ang_vel * ang_vel_scale` |
| `44:47` | 3 | `base_euler_xyz * quat_scale` |

历史堆叠顺序：

- `propri_history_buffer_` 每次左移一帧。
- 最新单帧 observation 写入 tail。
- 写入 `.bin` 的 `observations_` 顺序为从旧到新排列的 66 帧历史。

`walk_diag_*.csv` 的 `action_<joint>` 字段：

- 来自同一周期 `ComputeActions()` 后的 `actions_`。
- 与对应 `.bin` 中同序号帧的 observation 可以按行对齐。
- `GetJointCmdData()` 中再使用 `action_scale`、关节限位和 LPF/PD 链路生成实际位置或力矩命令。

## 3. 文件与模型版本映射

真机日志：

| 测试序号 | 时间戳 | 输入文件 | 诊断文件 |
| --- | --- | --- | --- |
| 1 | `20260707_163855` | `tm_obs_input_20260707_163855.bin` | `walk_diag_20260707_163855.csv` |
| 2 | `20260707_165431` | `tm_obs_input_20260707_165431.bin` | `walk_diag_20260707_165431.csv` |
| 3 | `20260707_170120` | `tm_obs_input_20260707_170120.bin` | `walk_diag_20260707_170120.csv` |
| 4 | `20260707_170424` | `tm_obs_input_20260707_170424.bin` | `walk_diag_20260707_170424.csv` |

用户给定的初始模型对应关系：

| 测试序号 | 计划使用 PT |
| --- | --- |
| 1、2 | `work/real_policy_compare/policy/origin_v7.3/model_7999.pt` |
| 3、4 | `work/real_policy_compare/policy/origin_v5.8/model_3000.pt` |

执行正式对比时，需要把“用户给定映射”和“离线复现结果”同时记录。若后两次日志无法由 `origin_v5.8` 复现，应先确认当时真机部署的策略文件来源和测试记录，而不是直接进入仿真闭环。

## 4. 阶段一：离线 PT 输入输出复现

目的：

- 验证 `.pt` 是否与真机测试时使用的策略一致。
- 验证本地 PyTorch 重建网络、history 输入顺序和动作输出顺序是否与真机部署一致。

流程：

1. 读取 `.bin` 为 `float32`，reshape 为 `[1000, 3102]`。
2. 用当前仓库的 `ActorCriticDH` 构造 12DOF 网络。
3. 加载指定 `.pt` 中的 `model_state_dict`。
4. 调用 `act_inference(obs)` 得到 `[1000, 12]` 动作。
5. 从 `walk_diag_*.csv` 读取 12 个 `action_<joint>` 字段。
6. 按帧、按关节计算：
   - `mean_abs`
   - `rmse`
   - `max_abs`
   - 最大误差对应的帧和关节
7. 输出 summary 和逐帧差异 CSV。

判定标准：

- 若误差为 `1e-5` 以内，认为 `.pt` 与真机记录动作一致。
- 若误差明显大于 `1e-3`，优先排查模型版本映射、导出文件来源和网络配置。

注意：

- 这一阶段不需要 Isaac Gym、MuJoCo 或 ONNX Runtime。
- 只使用 `.pt`，ONNX 仅作为真机部署来源的背景信息，不作为本地推理依据。

## 5. 阶段二：动作后处理链路对比

目的：

在 PT 输出一致后，继续检查从 `action` 到真机关节命令的转换是否与本地仿真一致。

真机控制链路：

1. `actions_[i]` 为策略输出。
2. `pos_des = actions_[i] * action_scale + init_state[i]`。
3. 对 `pos_des` 做关节限位 clamp。
4. 串联关节：
   - 对 `pos_des` 做低通滤波。
   - 输出位置命令、刚度、阻尼。
5. 并联踝关节：
   - 先计算 `tau_des = kp * (pos_des - joint_pos) + kd * (0 - joint_vel)`。
   - 对 `tau_des` 做低通滤波。
   - 输出力矩命令，位置/刚度/阻尼置零。

需要对齐的关键参数：

- `joint_list`
- `init_state`
- `stiffness`
- `damping`
- `joint_limits`
- `action_scale = 0.5`
- `lpf_conf.wc = 100`
- `lpf_conf.ts = 0.001`
- 并联关节列表：
  - `left_ankle_pitch_joint`
  - `left_ankle_roll_joint`
  - `right_ankle_pitch_joint`
  - `right_ankle_roll_joint`

输出对比：

- `pos_des_raw_*`
- `pos_des_lpf_*`
- `tau_des_raw_*`
- `tau_des_lpf_*`
- `is_parallel_*`

## 6. 阶段三：仿真闭环对比

目的：

在模型输入输出和动作后处理一致后，将同一 `.pt` 放入仿真环境闭环运行，定位真机与仿真的状态演化差异。

推荐先使用 MuJoCo sim2sim，原因：

- 不依赖 Isaac Gym/GPU。
- 更接近真机控制链路中的 1 kHz 仿真步进和 100 Hz 策略频率。
- 便于单实例逐帧导出 CSV。

闭环仿真需要统一：

- 初始姿态和初始关节角。
- 12DOF joint order。
- `default_dof_pos` / `init_state`。
- `action_scale`。
- PD 刚度、阻尼和关节限位。
- 低通滤波逻辑，尤其是踝关节并联力矩控制。
- 控制频率：底层 1000 Hz，策略 decimation 10，即 100 Hz。
- command 序列：优先从真机 CSV 读取每帧 command，而不是手工设定常数。
- observation scale 和 clip。
- history stack 初始化方式。

仿真输出建议包含：

- 每策略帧 obs。
- 每策略帧 action。
- 关节 `pos / vel / effort`。
- `pos_des_raw / pos_des_lpf / tau_des_raw / tau_des_lpf`。
- base euler、gyro、base height。
- command。

对比方式：

1. 先按第 0 帧对齐。
2. 对比 command、phase、action 是否一致。
3. 再对比关节状态和 base 姿态随时间的发散。
4. 若动作一致但状态快速发散，重点检查动力学、PD、延迟、摩擦、质量、关节限位和真机传感器偏置。

## 7. 推荐执行顺序

1. 先固定模型映射，完成四组日志的离线 PT 复现。
2. 若某组日志无法由预期 `.pt` 复现，暂停仿真闭环，先确认真机部署时实际使用的策略。
3. 离线 PT 复现通过后，复现动作后处理链路，确认 `action -> pos/tau command` 一致。
4. 使用真机日志中的 command 序列驱动 MuJoCo 闭环仿真。
5. 输出仿真 CSV，与真机 CSV 生成同名字段对齐报告。
6. 最后再扩展到 Isaac Gym 单环境 replay，用于和训练环境内部状态进行交叉验证。

## 8. 风险与注意事项

- `.bin` 记录的是已经 clip 后的完整 history 输入，不是原始传感器数据。
- CSV 中的 `action_*` 是策略输出，不是最终下发到电机的目标位置或力矩。
- 真机 `phase` 使用 wall-clock 时间，并在小速度命令时重置为 0；仿真中如果用 step count 生成 phase，可能出现相位差。
- 真机 `cmd_angular_z` 在 observation 中没有乘 `obs_scales_.ang_vel`，当前 `ang_vel = 1`，数值等价，但实现细节需记录。
- 踝关节使用并联力矩链路，不能简单按位置控制对比。
- 如果使用 Isaac Gym，需要具备 Isaac Gym/GPU 环境；当前本机环境未验证该依赖。

## 9. 产物规划

建议后续生成以下本地产物，均放入 `work/real_policy_compare/`：

| 产物 | 说明 |
| --- | --- |
| `comparisons/summary.csv` | 离线 PT 与真机 action 总览 |
| `comparisons/*_pt_vs_real_actions.csv` | 每帧、每关节 action 误差 |
| `sim_rollouts/*.csv` | MuJoCo 或 Isaac Gym 闭环仿真日志 |
| `sim_vs_real/*.csv` | 仿真与真机状态对齐结果 |
| `plots/*.png` | 关键关节、base 姿态和 action 误差曲线 |

这些产物属于本地分析数据，不应提交入库。
