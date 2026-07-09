from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from compare_real_policy_inference import load_policy  # noqa: E402
from inverse_replay_real_pose_commands import (  # noqa: E402
    actuator_torque_limits,
    build_qpos_sequence,
    load_real_log,
    optional_float_tag,
)
from replay_real_actions_mujoco import (  # noqa: E402
    DEFAULT_DOF_POS,
    JOINT_LIMIT_LOWER,
    JOINT_LIMIT_UPPER,
    JOINT_NAMES,
    PARALLEL_JOINTS,
    REAL_KD,
    REAL_KP,
    get_joint_addresses,
)


NUM_SINGLE_OBS = 47
FRAME_STACK = 66
ACTION_SCALE = 0.5
CYCLE_TIME = 0.7
OBS_SCALE_LIN_VEL = 2.0
OBS_SCALE_ANG_VEL = 1.0
OBS_SCALE_DOF_POS = 1.0
OBS_SCALE_DOF_VEL = 0.05
OBS_SCALE_QUAT = 1.0
CLIP_OBS = 100.0
CLIP_ACTIONS = 100.0


class DigitalLPFilter:
    def __init__(self, wc: float = 100.0, ts: float = 0.001) -> None:
        self.wc = wc
        self.ts = ts
        self.in_prev = np.zeros(2, dtype=np.float64)
        self.out_prev = np.zeros(2, dtype=np.float64)
        self.out = 0.0
        self.update()

    def update(self) -> None:
        den = 2500.0 * self.ts * self.ts * self.wc * self.wc + 7071.0 * self.ts * self.wc + 10000.0
        self.in1 = 2500.0 * self.ts * self.ts * self.wc * self.wc / den
        self.in2 = 5000.0 * self.ts * self.ts * self.wc * self.wc / den
        self.in3 = 2500.0 * self.ts * self.ts * self.wc * self.wc / den
        self.out1 = -(5000.0 * self.ts * self.ts * self.wc * self.wc - 20000.0) / den
        self.out2 = -(2500.0 * self.ts * self.ts * self.wc * self.wc - 7071.0 * self.ts * self.wc + 10000.0) / den

    def init(self, value: float) -> None:
        self.in_prev[:] = value
        self.out_prev[:] = value
        self.out = value

    def input(self, value: float) -> float:
        self.out = (
            self.in1 * value
            + self.in2 * self.in_prev[0]
            + self.in3 * self.in_prev[1]
            + self.out1 * self.out_prev[0]
            + self.out2 * self.out_prev[1]
        )
        self.in_prev[1] = self.in_prev[0]
        self.in_prev[0] = value
        self.out_prev[1] = self.out_prev[0]
        self.out_prev[0] = self.out
        return self.out


def quat_wxyz_to_euler(q_wxyz: np.ndarray) -> np.ndarray:
    return R.from_quat(q_wxyz[[1, 2, 3, 0]]).as_euler("xyz")


def wrap_euler(euler: np.ndarray) -> np.ndarray:
    out = euler.copy()
    out[out > np.pi] -= 2.0 * np.pi
    out[out < -np.pi] += 2.0 * np.pi
    return out


def initial_phase_offset(csv_path: Path, cycle_time: float) -> float:
    import pandas as pd

    first = pd.read_csv(csv_path, nrows=1)
    if "phase_sin" not in first.columns or "phase_cos" not in first.columns:
        return 0.0
    angle = float(np.arctan2(float(first["phase_sin"].iloc[0]), float(first["phase_cos"].iloc[0])))
    if angle < 0.0:
        angle += 2.0 * np.pi
    return angle / (2.0 * np.pi) * cycle_time


def make_single_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addr: np.ndarray,
    qvel_addr: np.ndarray,
    last_action: np.ndarray,
    command: np.ndarray,
    policy_time: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = data.qpos[qpos_addr].copy()
    dq = data.qvel[qvel_addr].copy()
    euler = wrap_euler(quat_wxyz_to_euler(data.qpos[3:7].copy()))
    gyro = data.qvel[3:6].copy()

    phase = policy_time / CYCLE_TIME
    obs = np.zeros(NUM_SINGLE_OBS, dtype=np.float32)
    obs[0] = np.sin(2.0 * np.pi * phase)
    obs[1] = np.cos(2.0 * np.pi * phase)
    obs[2] = command[0] * OBS_SCALE_LIN_VEL
    obs[3] = command[1] * OBS_SCALE_LIN_VEL
    obs[4] = command[2] * OBS_SCALE_ANG_VEL
    obs[5:17] = (q - DEFAULT_DOF_POS) * OBS_SCALE_DOF_POS
    obs[17:29] = dq * OBS_SCALE_DOF_VEL
    obs[29:41] = last_action
    obs[41:44] = gyro * OBS_SCALE_ANG_VEL
    obs[44:47] = euler * OBS_SCALE_QUAT
    obs = np.clip(obs, -CLIP_OBS, CLIP_OBS)
    return obs, q, dq, gyro, euler


def history_to_policy_input(history: deque[np.ndarray]) -> np.ndarray:
    return np.concatenate(list(history), axis=0).reshape(1, FRAME_STACK * NUM_SINGLE_OBS).astype(np.float32)


def init_real_controller_filters(q: np.ndarray) -> list[DigitalLPFilter]:
    filters = [DigitalLPFilter(wc=100.0, ts=0.001) for _ in JOINT_NAMES]
    for i, joint in enumerate(JOINT_NAMES):
        filters[i].init(0.0 if joint in PARALLEL_JOINTS else float(q[i]))
    return filters


def compute_control_tau(
    q: np.ndarray,
    dq: np.ndarray,
    target_q_raw: np.ndarray,
    filters: list[DigitalLPFilter],
    torque_limits: np.ndarray,
    controller_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    target_q_cmd = target_q_raw.copy()
    if controller_mode == "raw-pd":
        tau = REAL_KP * (target_q_raw - q) - REAL_KD * dq
        return np.clip(tau, torque_limits[:, 0], torque_limits[:, 1]), target_q_cmd

    if controller_mode != "real-lpf":
        raise ValueError(f"unknown controller_mode {controller_mode!r}")

    tau = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    for i, joint in enumerate(JOINT_NAMES):
        if joint in PARALLEL_JOINTS:
            tau_raw = REAL_KP[i] * (target_q_raw[i] - q[i]) - REAL_KD[i] * dq[i]
            tau[i] = filters[i].input(float(tau_raw))
        else:
            target_q_cmd[i] = filters[i].input(float(target_q_raw[i]))
            tau[i] = REAL_KP[i] * (target_q_cmd[i] - q[i]) - REAL_KD[i] * dq[i]
    return np.clip(tau, torque_limits[:, 0], torque_limits[:, 1]), target_q_cmd


def setup_initial_state(
    model: mujoco.MjModel,
    case_id: str,
    log_dir: Path,
    init_velocity: str,
    base_z_mode: str,
    base_z_offset: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    qpos_addr, qvel_addr = get_joint_addresses(model)
    times, real = load_real_log(log_dir / f"walk_diag_{case_id}.csv")
    qpos_seq, qvel_seq, base_zs = build_qpos_sequence(
        model,
        qpos_addr,
        qvel_addr,
        real["q"][:1],
        real["euler"][:1],
        real["gyro"][:1],
        real["dq"][:1],
        base_z_mode,
        initial_base_z=0.62,
        foot_clearance=0.0,
    )
    qvel0 = qvel_seq[0].copy()
    if init_velocity == "zero":
        qvel0[:] = 0.0
    elif init_velocity == "real-frame0":
        qvel0[0:3] = 0.0
    else:
        raise ValueError(f"unknown init_velocity {init_velocity!r}")
    qpos0 = qpos_seq[0].copy()
    qpos0[2] += base_z_offset
    return qpos0, qvel0, float(base_zs[0] + base_z_offset)


def render_frame(renderer: mujoco.Renderer, data: mujoco.MjData, case_id: str, sim_time: float) -> np.ndarray:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [float(data.qpos[0]), float(data.qpos[1]), max(float(data.qpos[2]), 0.45)]
    cam.distance = 2.0
    cam.azimuth = 135.0
    cam.elevation = -16.0
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    text = f"{case_id} closed-loop PT  t={sim_time:.2f}s  z={data.qpos[2]:.3f}"
    cv2.putText(bgr, text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(bgr, text, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
    return bgr


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop MuJoCo rollout from a real initial pose using a PT policy.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "work" / "real_policy_compare")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--xml", type=Path, default=REPO_ROOT / "resources" / "robots" / "x1" / "mjcf" / "xyber_x1_flat.xml")
    parser.add_argument("--case", default="20260707_163855")
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--cmd-x", type=float, default=0.4)
    parser.add_argument("--cmd-y", type=float, default=0.0)
    parser.add_argument("--cmd-yaw", type=float, default=0.0)
    parser.add_argument("--init-velocity", choices=["zero", "real-frame0"], default="zero")
    parser.add_argument("--base-z-mode", choices=["auto-per-frame", "initial"], default="auto-per-frame")
    parser.add_argument("--base-z-offset", type=float, default=0.0)
    parser.add_argument("--phase-source", choices=["zero", "real-frame0"], default="real-frame0")
    parser.add_argument("--controller-mode", choices=["real-lpf", "raw-pd"], default="real-lpf")
    parser.add_argument("--actuator-delay-ms", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--render-every", type=int, default=10)
    args = parser.parse_args()

    log_dir = args.log_dir or (args.root / "real_logs_7_7")
    policy_path = args.policy or (args.root / "policy" / "origin_v7.3" / "model_7999.pt")
    output_dir = args.output_dir or (args.root / "closed_loop_mujoco_policy")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    model.opt.timestep = 0.001
    data = mujoco.MjData(model)
    qpos_addr, qvel_addr = get_joint_addresses(model)
    torque_limits = actuator_torque_limits(model)

    qpos0, qvel0, base_z0 = setup_initial_state(
        model, args.case, log_dir, args.init_velocity, args.base_z_mode, args.base_z_offset
    )
    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    mujoco.mj_forward(model, data)

    policy = load_policy(policy_path)
    command = np.array([args.cmd_x, args.cmd_y, args.cmd_yaw], dtype=np.float64)
    phase_offset = 0.0
    if args.phase_source == "real-frame0":
        phase_offset = initial_phase_offset(log_dir / f"walk_diag_{args.case}.csv", CYCLE_TIME)

    zero_action = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    first_obs, *_ = make_single_obs(model, data, qpos_addr, qvel_addr, zero_action, command, phase_offset)
    history: deque[np.ndarray] = deque([first_obs.copy() for _ in range(FRAME_STACK)], maxlen=FRAME_STACK)
    last_action = zero_action.copy()
    q_init = data.qpos[qpos_addr].copy()
    filters = init_real_controller_filters(q_init)
    target_q_raw = DEFAULT_DOF_POS.copy()
    target_q_cmd = DEFAULT_DOF_POS.copy()

    policy_tag = f"{policy_path.parent.name}_{policy_path.stem}".replace(".", "p").replace(" ", "_")
    tag = (
        f"{args.case}_{policy_tag}_cmdx{args.cmd_x:g}_{optional_float_tag(args.duration_s)}_"
        f"initvel-{args.init_velocity}_phase-{args.phase_source}_ctrl-{args.controller_mode}_"
        f"delay{args.actuator_delay_ms:g}ms_basezoff{args.base_z_offset:g}"
    )
    csv_path = output_dir / f"{tag}.csv"
    mp4_path = output_dir / f"{tag}.mp4"
    preview_path = output_dir / f"{tag}_preview.png"

    renderer = None
    video = None
    if args.render:
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        fps = int(round(1.0 / (model.opt.timestep * args.render_every)))
        video = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (args.width, args.height))
        if not video.isOpened():
            raise RuntimeError(f"failed to open video writer: {mp4_path}")

    fields = [
        "step",
        "time_s",
        "policy_step",
        "root_x",
        "root_y",
        "root_z",
        "roll",
        "pitch",
        "yaw",
        "base_gyro_x",
        "base_gyro_y",
        "base_gyro_z",
    ]
    for joint in JOINT_NAMES:
        fields += [
            f"q_{joint}",
            f"dq_{joint}",
            f"action_{joint}",
            f"target_q_{joint}",
            f"tau_cmd_{joint}",
            f"tau_{joint}",
        ]

    total_steps = int(round(args.duration_s / model.opt.timestep))
    decimation = 10
    actuator_delay_steps = int(round(args.actuator_delay_ms * 1e-3 / model.opt.timestep))
    if actuator_delay_steps < 0:
        raise ValueError("--actuator-delay-ms must be non-negative")
    tau_delay_buffer: deque[np.ndarray] | None = None
    min_root_z = float(data.qpos[2])
    preview_written = False
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in range(total_steps + 1):
            sim_time = step * model.opt.timestep
            row: dict[str, float | int] | None = None
            if step % decimation == 0:
                obs, q, dq, gyro, euler = make_single_obs(
                    model, data, qpos_addr, qvel_addr, last_action, command, sim_time + phase_offset
                )
                if step > 0:
                    history.append(obs.copy())
                policy_input = history_to_policy_input(history)
                with torch.no_grad():
                    action = policy.act_inference(torch.from_numpy(policy_input).float()).cpu().numpy()[0].astype(np.float64)
                action = np.clip(action, -CLIP_ACTIONS, CLIP_ACTIONS)
                target_q_raw = np.clip(action * ACTION_SCALE + DEFAULT_DOF_POS, JOINT_LIMIT_LOWER, JOINT_LIMIT_UPPER)
                last_action = action.copy()

                tau, target_q_cmd = compute_control_tau(
                    q, dq, target_q_raw, filters, torque_limits, args.controller_mode
                )

                row: dict[str, float | int] = {
                    "step": step,
                    "time_s": sim_time,
                    "policy_step": step // decimation,
                    "root_x": float(data.qpos[0]),
                    "root_y": float(data.qpos[1]),
                    "root_z": float(data.qpos[2]),
                    "roll": float(euler[0]),
                    "pitch": float(euler[1]),
                    "yaw": float(euler[2]),
                    "base_gyro_x": float(gyro[0]),
                    "base_gyro_y": float(gyro[1]),
                    "base_gyro_z": float(gyro[2]),
                }
                for i, joint in enumerate(JOINT_NAMES):
                    row[f"q_{joint}"] = float(q[i])
                    row[f"dq_{joint}"] = float(dq[i])
                    row[f"action_{joint}"] = float(action[i])
                    row[f"target_q_{joint}"] = float(target_q_cmd[i])
                    row[f"tau_cmd_{joint}"] = float(tau[i])
            else:
                q = data.qpos[qpos_addr].copy()
                dq = data.qvel[qvel_addr].copy()
                tau, target_q_cmd = compute_control_tau(
                    q, dq, target_q_raw, filters, torque_limits, args.controller_mode
                )

            if actuator_delay_steps > 0:
                if tau_delay_buffer is None:
                    tau_delay_buffer = deque(
                        [tau.copy() for _ in range(actuator_delay_steps + 1)],
                        maxlen=actuator_delay_steps + 1,
                    )
                tau_delay_buffer.append(tau.copy())
                applied_tau = tau_delay_buffer.popleft()
            else:
                applied_tau = tau

            if row is not None:
                for i, joint in enumerate(JOINT_NAMES):
                    row[f"tau_{joint}"] = float(applied_tau[i])
                writer.writerow(row)

            data.ctrl[:] = applied_tau
            if step < total_steps:
                mujoco.mj_step(model, data)
                min_root_z = min(min_root_z, float(data.qpos[2]))

            if args.render and video is not None and renderer is not None and step % args.render_every == 0:
                bgr = render_frame(renderer, data, args.case, sim_time)
                video.write(bgr)
                if not preview_written and sim_time >= min(1.0, args.duration_s):
                    cv2.imwrite(str(preview_path), bgr)
                    preview_written = True

    if video is not None:
        video.release()
    if renderer is not None:
        renderer.close()

    print(f"case={args.case} policy={policy_path}")
    print(f"cmd=({args.cmd_x:g},{args.cmd_y:g},{args.cmd_yaw:g}) duration={args.duration_s:g}s")
    print(
        f"base_z0={base_z0:.9g} init_velocity={args.init_velocity} "
        f"base_z_offset={args.base_z_offset:g} "
        f"phase_offset={phase_offset:.6g} controller_mode={args.controller_mode} "
        f"actuator_delay_ms={args.actuator_delay_ms:g} delay_steps={actuator_delay_steps}"
    )
    print(f"final_root_z={float(data.qpos[2]):.9g} min_root_z={min_root_z:.9g}")
    print(f"csv={csv_path}")
    if args.render:
        print(f"video={mp4_path}")
        print(f"preview={preview_path}")


if __name__ == "__main__":
    main()
