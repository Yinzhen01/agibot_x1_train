from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from replay_real_actions_mujoco import (
    DEFAULT_DOF_POS,
    JOINT_NAMES,
    REAL_KD,
    REAL_KP,
    get_joint_addresses,
    load_rows,
    min_robot_geom_z,
    row_gyro,
    row_quat_wxyz,
    row_vector,
)


def infer_case_id(csv_path: Path) -> str:
    name = csv_path.stem
    prefix = "walk_diag_"
    return name[len(prefix) :] if name.startswith(prefix) else name


def sim_euler(data: mujoco.MjData) -> np.ndarray:
    quat_wxyz = data.qpos[3:7].copy()
    return R.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_euler("xyz")


def run_hold(args: argparse.Namespace) -> dict[str, float | str]:
    rows = load_rows(args.csv)
    first = rows[0]
    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    model.opt.timestep = args.sim_dt
    data = mujoco.MjData(model)
    qpos_addr, qvel_addr = get_joint_addresses(model)

    target_q = DEFAULT_DOF_POS.copy() if args.target == "default" else row_vector(first, "pos")
    data.qpos[0:3] = np.array([0.0, 0.0, args.base_z], dtype=np.float64)
    data.qpos[3:7] = row_quat_wxyz(first, args.orientation_source)
    data.qpos[qpos_addr] = target_q
    data.qvel[0:3] = 0.0
    data.qvel[3:6] = row_gyro(first)
    data.qvel[qvel_addr] = row_vector(first, "vel")
    mujoco.mj_forward(model, data)

    if args.auto_base_z:
        data.qpos[2] += args.foot_clearance - min_robot_geom_z(model, data)
        mujoco.mj_forward(model, data)

    actual_base_z = float(data.qpos[2])
    case_id = infer_case_id(args.csv)
    z_tag = f"{actual_base_z:.3f}".replace(".", "p")
    out_path = args.output_dir / f"{case_id}_hold-initial_{args.orientation_source}_z{z_tag}_dur{args.duration:g}.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fields = ["step", "time_s", "root_z", "roll", "pitch", "yaw", "max_abs_q_err", "max_abs_dq"]
    for joint in JOINT_NAMES:
        fields += [f"q_{joint}", f"target_{joint}", f"err_{joint}", f"dq_{joint}", f"tau_{joint}"]

    total_steps = int(round(args.duration / args.sim_dt))
    log_every_steps = max(1, int(round(args.log_dt / args.sim_dt)))
    tau_limit = np.full(len(JOINT_NAMES), args.tau_limit, dtype=np.float64)
    max_abs_q_err = 0.0
    max_abs_pitch = 0.0
    min_root_z = float(data.qpos[2])
    fall_time = ""

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in range(total_steps + 1):
            q = data.qpos[qpos_addr].copy()
            dq = data.qvel[qvel_addr].copy()
            tau = REAL_KP * (target_q - q) - REAL_KD * dq
            tau = np.clip(tau, -tau_limit, tau_limit)

            euler = sim_euler(data)
            q_err = q - target_q
            max_abs_q_err = max(max_abs_q_err, float(np.max(np.abs(q_err))))
            max_abs_pitch = max(max_abs_pitch, abs(float(euler[1])))
            min_root_z = min(min_root_z, float(data.qpos[2]))
            if fall_time == "" and data.qpos[2] < args.fall_z:
                fall_time = f"{step * args.sim_dt:.6g}"

            if step % log_every_steps == 0:
                row: dict[str, float | int] = {
                    "step": step,
                    "time_s": step * args.sim_dt,
                    "root_z": float(data.qpos[2]),
                    "roll": float(euler[0]),
                    "pitch": float(euler[1]),
                    "yaw": float(euler[2]),
                    "max_abs_q_err": float(np.max(np.abs(q_err))),
                    "max_abs_dq": float(np.max(np.abs(dq))),
                }
                for i, joint in enumerate(JOINT_NAMES):
                    row[f"q_{joint}"] = float(q[i])
                    row[f"target_{joint}"] = float(target_q[i])
                    row[f"err_{joint}"] = float(q_err[i])
                    row[f"dq_{joint}"] = float(dq[i])
                    row[f"tau_{joint}"] = float(tau[i])
                writer.writerow(row)

            if step < total_steps:
                data.ctrl[:] = tau
                mujoco.mj_step(model, data)

    return {
        "case": case_id,
        "csv": str(out_path),
        "init_base_z": actual_base_z,
        "duration": args.duration,
        "min_root_z": min_root_z,
        "fall_time_s": fall_time,
        "max_abs_q_err": max_abs_q_err,
        "max_abs_pitch": max_abs_pitch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hold the real-machine initial pose in MuJoCo using PD.")
    parser.add_argument("--csv", type=Path, default=Path("work/real_policy_compare/real_logs_7_7/walk_diag_20260707_163855.csv"))
    parser.add_argument("--mjcf", type=Path, default=Path("resources/robots/x1/mjcf/xyber_x1_flat.xml"))
    parser.add_argument("--output-dir", type=Path, default=Path("work/real_policy_compare/hold_initial_pose"))
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--sim-dt", type=float, default=0.001)
    parser.add_argument("--log-dt", type=float, default=0.01)
    parser.add_argument("--base-z", type=float, default=0.62)
    parser.add_argument("--auto-base-z", action="store_true", default=True)
    parser.add_argument("--foot-clearance", type=float, default=0.0)
    parser.add_argument("--orientation-source", choices=["imu", "euler", "level"], default="imu")
    parser.add_argument("--tau-limit", type=float, default=500.0)
    parser.add_argument("--fall-z", type=float, default=0.2)
    parser.add_argument("--target", choices=["real-q0", "default"], default="real-q0")
    args = parser.parse_args()

    summary = run_hold(args)
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
