# 项目状态

本文档用于长期项目的低成本上下文恢复。只记录会影响后续推进的阶段、目标、关键决策和风险；不要记录每个小改动或完整聊天历史。

## 当前阶段

当前项目处于 X1 12DOF 强化学习训练基线整理和远端训练能力接入阶段。

## 当前目标

- 保持 `x1_dh_stand` 作为当前主训练任务。
- 维护清晰的训练说明、奖励解释、PPO 解释和项目对比报告。
- 支持本地训练、Isaac Gym 回放、导出、sim2sim、Gradmotion GUI 云桌面训练和一键远端部署。
- 避免将 29DOF/F1 分支逻辑误迁移到当前 12DOF 项目。
- 建立真机测试日志与本地 `.pt` 策略推理/仿真的分层对比流程。

## 已完成

- 生成并维护当前项目整体运行说明：`docs/CURRENT_PROJECT_OPERATION_GUIDE.md`。
- 生成当前项目与 `E:\agi_29` 的差异报告：`docs/reports/CURRENT_vs_AGI_29_DIFF_REPORT.md`。
- 生成当前项目与 `E:\agi_origin` 的差异报告：`docs/reports/CURRENT_vs_AGI_ORIGIN_DIFF_REPORT.md`。
- 迁入并适配 Gradmotion GUI 云桌面、反向 SSH、一键部署和 gm-cli 云任务模板。
- 将文字文档统一归入 `docs/`，将 README 媒体素材迁移到 `docs/assets/`。
- 将根 `AGENTS.md` 从通用模板适配为当前项目规则。

## 正在进行

- 按 `AGENTS.md` 规则整理仓库结构、文档路由和忽略规则。
- 继续保持运维脚本默认任务为 `x1_dh_stand`，避免残留 F1/29DOF 默认值。

## 下一步

- 在具备 Isaac Gym/GPU 的环境中运行一次 `x1_dh_stand` 小规模 smoke。
- 如果继续使用 Gradmotion 云任务，填充 `ops/gm-cli/payloads/*.local.json` 中的项目、镜像和资源 ID。
- 如果要迁移 `agi_29` 的 29DOF 奖励或动作逻辑，先逐项核对 DOF 顺序、观测维度、action scale 和 sim2sim 端。
- 按 `docs/reports/REAL_POLICY_PT_SIM_COMPARE_PLAN.md` 先完成真机 `.bin` 输入与 `.pt` 输出的离线一致性验证，再进入 MuJoCo/Isaac Gym 闭环仿真对比。
- 如需提交本次整理，按 `docs/git-workflow.md` 暂存相关路径并使用中文 commit message。

## 关键决策

- `2026-07-01`：当前项目保留 X1 12DOF 路线，`agi_29` 仅作为 29DOF/F1 参考，不直接混入训练配置。
- `2026-07-01`：`docs/assets/` 保留媒体素材，文字文档统一放入 `docs/`。
- `2026-07-01`：`ops/gradmotion/` 和 `ops/gm-cli/` 作为远端训练与云任务能力入口。
- `2026-07-09`：真机日志对比采用分层流程：先用 `.pt` 离线复现真机 policy input/output，再进入动作后处理和仿真闭环对比；ONNX 不作为本地推理依据。
- `2026-07-09`：训练延迟建模按部署链路区分关节类型：hip/knee 保留 action/位置目标延迟，ankle_pitch/ankle_roll 使用扭矩命令延迟 `[5, 8]` timesteps，避免脚踝同时受到 action lag 和 torque lag。
- `2026-07-09`：`x1_dh_stand` 训练初始状态对齐到真机测例 `20260707_163855` 第 0 帧和 MuJoCo 推理接触高度：base_z `0.6101938959661087`，关节初始随机扰动关闭。

## 风险与注意事项

- `docs/assets/play.gif`、`docs/assets/train.gif`、`docs/assets/mujoco.gif` 是较大文件，GitHub 会提示建议使用 Git LFS；当前保留它们作为 README 媒体素材。
- `work/` 是本地缓存目录，应忽略不入库。
- `ops/gm-cli/accounts.local.json`、`ops/gm-cli/payloads/*.local.json`、`cloud_artifacts/`、`outputs/` 都是本地文件，不应提交。
- 当前环境未必具备 Isaac Gym/GPU；无法运行训练 smoke 时需在最终说明中明确。

## 更新规则

- 完成阶段性任务、切换里程碑、发现阻塞或形成关键决策时更新本文档。
- 不要把项目阶段、短期进度或任务流水写入根目录 `AGENTS.md`。
- 如果状态变化只影响当前对话，不影响后续推进，不需要写入本文档。


