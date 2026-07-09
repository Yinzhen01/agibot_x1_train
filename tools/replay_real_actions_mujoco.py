from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

MJCF_JOINT_NAMES = [name[:-6] if name.endswith("_joint") else name for name in JOINT_NAMES]

PARALLEL_JOINTS = {
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
}

DEFAULT_DOF_POS = np.array(
    [0.4, 0.05, -0.31, 0.49, -0.21, 0.0, -0.4, -0.05, 0.31, 0.49, -0.21, 0.0],
    dtype=np.float64,
)

REAL_KP = np.array(
    [30.0, 40.0, 35.0, 100.0, 35.0, 35.0, 30.0, 40.0, 35.0, 100.0, 35.0, 35.0],
    dtype=np.float64,
)

REAL_KD = np.array(
    [3.0, 3.0, 4.0, 10.0, 1.5, 1.5, 3.0, 3.0, 4.0, 10.0, 1.5, 1.5],
    dtype=np.float64,
)

JOINT_LIMIT_LOWER = np.array(
    [-1.0, -1.5, -1.5, 0.0, -0.41, -0.64, -2.0, -0.2, -1.5, 0.0, -0.41, -0.64],
    dtype=np.float64,
)

JOINT_LIMIT_UPPER = np.array(
    [2.0, 0.2, 1.5, 2.0, 0.35, 0.64, 1.0, 1.5, 1.5, 2.0, 0.35, 0.64],
    dtype=np.float64,
)


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    required = ["timestamp_ns", "imu_quat_w", "imu_quat_x", "imu_quat_y", "imu_quat_z"]
    for joint in JOINT_NAMES:
        required += [f"pos_{joint}", f"vel_{joint}", f"action_{joint}"]
    missing = [name for name in required if name not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return rows


def row_vector(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.array([parse_float(row[f"{prefix}_{joint}"]) for joint in JOINT_NAMES], dtype=np.float64)


def row_euler(row: dict[str, str]) -> np.ndarray:
    return np.array(
        [
            parse_float(row["base_euler_x"]),
            parse_float(row["base_euler_y"]),
            parse_float(row["base_euler_z"]),
        ],
        dtype=np.float64,
    )


def row_gyro(row: dict[str, str]) -> np.ndarray:
    return np.array(
        [
            parse_float(row["base_ang_vel_x"]),
            parse_float(row["base_ang_vel_y"]),
            parse_float(row["base_ang_vel_z"]),
        ],
        dtype=np.float64,
    )


def row_quat_wxyz(row: dict[str, str], source: str) -> np.ndarray:
    if source == "euler":
        euler = row_euler(row)
        xyzw = R.from_euler("xyz", euler).as_quat()
        return xyzw[[3, 0, 1, 2]]
    if source == "level":
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    quat = np.array(
        [
            parse_float(row["imu_quat_w"]),
            parse_float(row["imu_quat_x"]),
            parse_float(row["imu_quat_y"]),
            parse_float(row["imu_quat_z"]),
        ],
        dtype=np.float64,
    )
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm <= 0.0:
        euler = row_euler(row)
        xyzw = R.from_euler("xyz", euler).as_quat()
        return xyzw[[3, 0, 1, 2]]
    return quat / norm


def min_robot_geom_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    zs = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name == "floor":
            continue
        zs.append(float(data.geom_xpos[geom_id, 2]))
    if not zs:
        raise ValueError("No robot geoms found when computing initial base height.")
    return min(zs)


def get_joint_addresses(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    qpos_addr = []
    qvel_addr = []
    for name in MJCF_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MJCF joint not found: {name}")
        qpos_addr.append(model.jnt_qposadr[joint_id])
        qvel_addr.append(model.jnt_dofadr[joint_id])
    return np.asarray(qpos_addr, dtype=np.int32), np.asarray(qvel_addr, dtype=np.int32)


def get_state(data: mujoco.MjData, qpos_addr: np.ndarray, qvel_addr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = data.qpos[qpos_addr].copy()
    dq = data.qvel[qvel_addr].copy()
    quat_wxyz = data.qpos[3:7].copy()
    quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
    euler = R.from_quat(quat_xyzw).as_euler("xyz")
    return q, dq, euler


def finite_or(value: float, fallback: float) -> float:
    return value if math.isfinite(value) else fallback


def control_from_logged_command(row: dict[str, str], q: np.ndarray, dq: np.ndarray) -> np.ndarray:
    tau = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    for i, joint in enumerate(JOINT_NAMES):
        if joint in PARALLEL_JOINTS:
            logged_tau = parse_float(row.get(f"tau_des_lpf_{joint}", "nan"))
            logged_tau = finite_or(logged_tau, parse_float(row.get(f"tau_des_raw_{joint}", "nan")))
            if math.isfinite(logged_tau):
                tau[i] = logged_tau
            else:
                pos_des = parse_float(row.get(f"pos_des_raw_{joint}", "nan"))
                pos_des = finite_or(pos_des, DEFAULT_DOF_POS[i])
                tau[i] = REAL_KP[i] * (pos_des - q[i]) - REAL_KD[i] * dq[i]
        else:
            pos_des = parse_float(row.get(f"pos_des_lpf_{joint}", "nan"))
            pos_des = finite_or(pos_des, parse_float(row.get(f"pos_des_raw_{joint}", "nan")))
            pos_des = finite_or(pos_des, DEFAULT_DOF_POS[i])
            tau[i] = REAL_KP[i] * (pos_des - q[i]) - REAL_KD[i] * dq[i]
    return tau


def control_from_raw_action(row: dict[str, str], q: np.ndarray, dq: np.ndarray) -> np.ndarray:
    actions = row_vector(row, "action")
    pos_des = np.clip(actions * 0.5 + DEFAULT_DOF_POS, JOINT_LIMIT_LOWER, JOINT_LIMIT_UPPER)
    return REAL_KP * (pos_des - q) - REAL_KD * dq


def build_output_fields() -> list[str]:
    fields = [
        "frame",
        "command_frame",
        "time_s",
        "sim_root_x",
        "sim_root_y",
        "sim_root_z",
        "sim_roll",
        "sim_pitch",
        "sim_yaw",
        "real_roll",
        "real_pitch",
        "real_yaw",
        "err_roll",
        "err_pitch",
        "err_yaw",
    ]
    for joint in JOINT_NAMES:
        fields += [
            f"sim_pos_{joint}",
            f"real_pos_{joint}",
            f"err_pos_{joint}",
            f"sim_vel_{joint}",
            f"real_vel_{joint}",
            f"err_vel_{joint}",
            f"applied_tau_{joint}",
        ]
    return fields


def make_output_row(
    frame: int,
    command_frame: int,
    time_s: float,
    data: mujoco.MjData,
    qpos_addr: np.ndarray,
    qvel_addr: np.ndarray,
    real_row: dict[str, str],
    applied_tau: np.ndarray,
) -> dict[str, float | int]:
    sim_q, sim_dq, sim_euler = get_state(data, qpos_addr, qvel_addr)
    real_q = row_vector(real_row, "pos")
    real_dq = row_vector(real_row, "vel")
    real_euler = row_euler(real_row)

    row: dict[str, float | int] = {
        "frame": frame,
        "command_frame": command_frame,
        "time_s": time_s,
        "sim_root_x": float(data.qpos[0]),
        "sim_root_y": float(data.qpos[1]),
        "sim_root_z": float(data.qpos[2]),
        "sim_roll": float(sim_euler[0]),
        "sim_pitch": float(sim_euler[1]),
        "sim_yaw": float(sim_euler[2]),
        "real_roll": float(real_euler[0]),
        "real_pitch": float(real_euler[1]),
        "real_yaw": float(real_euler[2]),
        "err_roll": float(sim_euler[0] - real_euler[0]),
        "err_pitch": float(sim_euler[1] - real_euler[1]),
        "err_yaw": float(sim_euler[2] - real_euler[2]),
    }
    for i, joint in enumerate(JOINT_NAMES):
        row[f"sim_pos_{joint}"] = float(sim_q[i])
        row[f"real_pos_{joint}"] = float(real_q[i])
        row[f"err_pos_{joint}"] = float(sim_q[i] - real_q[i])
        row[f"sim_vel_{joint}"] = float(sim_dq[i])
        row[f"real_vel_{joint}"] = float(real_dq[i])
        row[f"err_vel_{joint}"] = float(sim_dq[i] - real_dq[i])
        row[f"applied_tau_{joint}"] = float(applied_tau[i])
    return row


def infer_case_id(csv_path: Path) -> str:
    name = csv_path.stem
    prefix = "walk_diag_"
    return name[len(prefix) :] if name.startswith(prefix) else name


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay real-machine X1 logged commands in MuJoCo.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("work/real_policy_compare/real_logs_7_7/walk_diag_20260707_163855.csv"),
    )
    parser.add_argument("--mjcf", type=Path, default=Path("resources/robots/x1/mjcf/xyber_x1_flat.xml"))
    parser.add_argument("--output-dir", type=Path, default=Path("work/real_policy_compare/replay_mujoco"))
    parser.add_argument("--frames", type=int, default=1000, help="Number of policy frames to replay, including frame 0.")
    parser.add_argument("--sim-dt", type=float, default=0.001)
    parser.add_argument("--decimation", type=int, default=10)
    parser.add_argument("--base-z", type=float, default=0.62)
    parser.add_argument(
        "--auto-base-z",
        action="store_true",
        help="Adjust base z after setting the CSV joint pose so the lowest robot geom touches the floor.",
    )
    parser.add_argument("--foot-clearance", type=float, default=0.0)
    parser.add_argument("--orientation-source", choices=["imu", "euler", "level"], default="imu")
    parser.add_argument("--mode", choices=["logged-command", "raw-action"], default="logged-command")
    parser.add_argument("--tau-limit", type=float, default=500.0)
    args = parser.parse_args()

    rows = load_rows(args.csv)
    frame_count = min(args.frames, len(rows))
    if frame_count < 2:
        raise ValueError("Need at least 2 frames to replay.")

    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    model.opt.timestep = args.sim_dt
    data = mujoco.MjData(model)
    qpos_addr, qvel_addr = get_joint_addresses(model)

    first = rows[0]
    data.qpos[0:3] = np.array([0.0, 0.0, args.base_z], dtype=np.float64)
    data.qpos[3:7] = row_quat_wxyz(first, args.orientation_source)
    data.qpos[qpos_addr] = row_vector(first, "pos")
    data.qvel[0:3] = 0.0
    data.qvel[3:6] = row_gyro(first)
    data.qvel[qvel_addr] = row_vector(first, "vel")
    mujoco.mj_forward(model, data)
    initial_min_geom_z = min_robot_geom_z(model, data)
    if args.auto_base_z:
        data.qpos[2] += args.foot_clearance - initial_min_geom_z
        mujoco.mj_forward(model, data)
        initial_min_geom_z = min_robot_geom_z(model, data)
    actual_base_z = float(data.qpos[2])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_id = infer_case_id(args.csv)
    base_z_tag = f"z{actual_base_z:.3f}".replace(".", "p")
    if args.auto_base_z:
        base_z_tag += "_auto"
    out_path = args.output_dir / f"{case_id}_{args.mode}_{base_z_tag}_frames{frame_count}_sim_vs_real.csv"

    fields = build_output_fields()
    last_tau = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    max_abs_pos_err = 0.0
    max_abs_pitch_err = 0.0
    min_root_z = float(data.qpos[2])

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(make_output_row(0, -1, 0.0, data, qpos_addr, qvel_addr, rows[0], last_tau))

        for command_frame in range(frame_count - 1):
            for _ in range(args.decimation):
                q, dq, _ = get_state(data, qpos_addr, qvel_addr)
                if args.mode == "logged-command":
                    tau = control_from_logged_command(rows[command_frame], q, dq)
                else:
                    tau = control_from_raw_action(rows[command_frame], q, dq)
                tau = np.clip(tau, -args.tau_limit, args.tau_limit)
                data.ctrl[:] = tau
                mujoco.mj_step(model, data)
                last_tau = tau

            frame = command_frame + 1
            time_s = frame * args.decimation * args.sim_dt
            out_row = make_output_row(frame, command_frame, time_s, data, qpos_addr, qvel_addr, rows[frame], last_tau)
            writer.writerow(out_row)
            min_root_z = min(min_root_z, float(out_row["sim_root_z"]))

            pos_errs = [abs(float(out_row[f"err_pos_{joint}"])) for joint in JOINT_NAMES]
            max_abs_pos_err = max(max_abs_pos_err, max(pos_errs))
            max_abs_pitch_err = max(max_abs_pitch_err, abs(float(out_row["err_pitch"])))

            if frame % 100 == 0:
                print(
                    f"frame={frame} t={time_s:.2f}s z={out_row['sim_root_z']:.3f} "
                    f"pitch_err={out_row['err_pitch']:.4f} max_pos_err={max(pos_errs):.4f}",
                    flush=True,
                )

    print(f"CSV={out_path}")
    print(f"frames={frame_count} mode={args.mode} dt={args.sim_dt:g} decimation={args.decimation}")
    print(
        f"init_base_z={actual_base_z:.6g} auto_base_z={args.auto_base_z} "
        f"orientation_source={args.orientation_source} initial_min_robot_geom_z={initial_min_geom_z:.6g}"
    )
    print(f"min_root_z={min_root_z:.6g} max_abs_pos_err={max_abs_pos_err:.6g} max_abs_pitch_err={max_abs_pitch_err:.6g}")


if __name__ == "__main__":
    main()
