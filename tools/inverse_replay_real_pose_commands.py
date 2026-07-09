from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from replay_real_actions_mujoco import (  # noqa: E402
    JOINT_NAMES,
    MJCF_JOINT_NAMES,
    get_joint_addresses,
    min_robot_geom_z,
)


DEFAULT_CASES = [
    "20260707_163855",
    "20260707_165431",
    "20260707_170120",
    "20260707_170424",
]


def safe_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def optional_float_tag(value: float | None) -> str:
    if value is None:
        return "full"
    return f"dur{safe_tag(value)}"


def load_real_log(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    df = pd.read_csv(path)
    if "timestamp_ns" not in df.columns:
        raise ValueError(f"{path} is missing timestamp_ns")

    t = (df["timestamp_ns"].to_numpy(np.float64) - float(df["timestamp_ns"].iloc[0])) * 1e-9
    if np.any(np.diff(t) <= 0.0):
        raise ValueError(f"{path} has non-increasing timestamps")

    q = np.column_stack([df[f"pos_{joint}"].to_numpy(np.float64) for joint in JOINT_NAMES])
    dq = np.column_stack([df[f"vel_{joint}"].to_numpy(np.float64) for joint in JOINT_NAMES])
    euler = df[["base_euler_x", "base_euler_y", "base_euler_z"]].to_numpy(np.float64)
    euler[:, 2] = np.unwrap(euler[:, 2])

    if {"base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z"}.issubset(df.columns):
        gyro = df[["base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z"]].to_numpy(np.float64)
    else:
        gyro = np.zeros((len(df), 3), dtype=np.float64)

    return t, {"q": q, "dq": dq, "euler": euler, "gyro": gyro}


def resample_real_data(t: np.ndarray, data: dict[str, np.ndarray], dt: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if dt <= 0.0:
        raise ValueError("--dt must be positive")
    out_t = np.arange(0.0, t[-1] + 0.5 * dt, dt, dtype=np.float64)
    out_t = out_t[out_t <= t[-1] + 1e-12]

    out: dict[str, np.ndarray] = {}
    for name, arr in data.items():
        cols = [np.interp(out_t, t, arr[:, i]) for i in range(arr.shape[1])]
        out[name] = np.column_stack(cols).astype(np.float64)
    return out_t, out


def trim_by_duration(t: np.ndarray, data: dict[str, np.ndarray], duration_s: float | None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if duration_s is None:
        return t, data
    if duration_s <= 0.0:
        raise ValueError("--duration-s must be positive")
    keep = t <= duration_s + 1e-12
    if np.count_nonzero(keep) < 2:
        raise ValueError(f"duration {duration_s:g}s leaves fewer than 2 samples")
    return t[keep], {name: arr[keep] for name, arr in data.items()}


def euler_to_quat_wxyz(euler_xyz: np.ndarray) -> np.ndarray:
    xyzw = R.from_euler("xyz", euler_xyz).as_quat()
    return xyzw[[3, 0, 1, 2]]


def find_foot_collision_geoms(model: mujoco.MjModel) -> list[int]:
    geoms: list[int] = []
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
        is_foot_body = body_name in {"left_ankle_roll_link", "right_ankle_roll_link"}
        is_collision_sphere = (
            model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE
            and model.geom_contype[geom_id] != 0
            and abs(float(model.geom_size[geom_id, 0]) - 0.002) < 1e-9
        )
        if is_foot_body and is_collision_sphere:
            geoms.append(geom_id)
    if len(geoms) != 8:
        raise ValueError(f"expected 8 foot collision spheres, found {len(geoms)}: {geoms}")
    return geoms


def min_foot_contact_surface_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    return min(
        float(data.geom_xpos[geom_id, 2] - model.geom_size[geom_id, 0])
        for geom_id in find_foot_collision_geoms(model)
    )


def set_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_addr: np.ndarray,
    qvel_addr: np.ndarray,
    q: np.ndarray,
    euler: np.ndarray,
    gyro: np.ndarray,
    dq: np.ndarray,
    base_z: float,
) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    data.qpos[0:3] = [0.0, 0.0, base_z]
    data.qpos[3:7] = euler_to_quat_wxyz(euler)
    data.qpos[qpos_addr] = q
    data.qvel[0:3] = 0.0
    data.qvel[3:6] = gyro
    data.qvel[qvel_addr] = dq
    mujoco.mj_forward(model, data)


def build_qpos_sequence(
    model: mujoco.MjModel,
    qpos_addr: np.ndarray,
    qvel_addr: np.ndarray,
    q: np.ndarray,
    euler: np.ndarray,
    gyro: np.ndarray,
    dq: np.ndarray,
    base_z_mode: str,
    initial_base_z: float,
    foot_clearance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    qpos = np.zeros((q.shape[0], model.nq), dtype=np.float64)
    qvel_logged = np.zeros((q.shape[0], model.nv), dtype=np.float64)
    base_zs = np.zeros(q.shape[0], dtype=np.float64)

    current_base_z = initial_base_z
    for i in range(q.shape[0]):
        set_pose(model, data, qpos_addr, qvel_addr, q[i], euler[i], gyro[i], dq[i], current_base_z)
        if base_z_mode == "auto-per-frame":
            data.qpos[2] += foot_clearance - min_foot_contact_surface_z(model, data)
            mujoco.mj_forward(model, data)
        elif base_z_mode == "initial":
            if i == 0:
                data.qpos[2] += foot_clearance - min_foot_contact_surface_z(model, data)
                mujoco.mj_forward(model, data)
                current_base_z = float(data.qpos[2])
            else:
                data.qpos[2] = current_base_z
                mujoco.mj_forward(model, data)
        else:
            raise ValueError(f"unknown base_z_mode {base_z_mode!r}")

        qpos[i] = data.qpos
        qvel_logged[i, 0:3] = 0.0
        qvel_logged[i, 3:6] = gyro[i]
        qvel_logged[i, qvel_addr] = dq[i]
        base_zs[i] = float(data.qpos[2])

    return qpos, qvel_logged, base_zs


def qvel_from_qpos_sequence(model: mujoco.MjModel, qpos_seq: np.ndarray, times: np.ndarray) -> np.ndarray:
    qvel = np.zeros((qpos_seq.shape[0], model.nv), dtype=np.float64)
    tmp = np.zeros(model.nv, dtype=np.float64)
    if qpos_seq.shape[0] < 2:
        return qvel

    mujoco.mj_differentiatePos(model, tmp, float(times[1] - times[0]), qpos_seq[0], qpos_seq[1])
    qvel[0] = tmp
    for i in range(1, qpos_seq.shape[0] - 1):
        mujoco.mj_differentiatePos(model, tmp, float(times[i + 1] - times[i - 1]), qpos_seq[i - 1], qpos_seq[i + 1])
        qvel[i] = tmp
    mujoco.mj_differentiatePos(model, tmp, float(times[-1] - times[-2]), qpos_seq[-2], qpos_seq[-1])
    qvel[-1] = tmp
    return qvel


def contact_jacobian_transpose(model: mujoco.MjModel, data: mujoco.MjData, foot_geoms: list[int]) -> np.ndarray:
    blocks = []
    for geom_id in foot_geoms:
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        body_id = int(model.geom_bodyid[geom_id])
        point = data.geom_xpos[geom_id].copy()
        mujoco.mj_jac(model, data, jacp, jacr, point, body_id)
        blocks.append(jacp)
    return np.vstack(blocks).T


def solve_tau_with_fixed_feet(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qvel_addr: np.ndarray,
    foot_geoms: list[int],
    generalized_force: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    jt = contact_jacobian_transpose(model, data, foot_geoms)
    s = np.zeros((model.nv, len(qvel_addr)), dtype=np.float64)
    for i, adr in enumerate(qvel_addr):
        s[int(adr), i] = 1.0

    # generalized_force = J_contact.T * lambda + S * tau.
    a = np.hstack([jt, s])
    x, *_ = np.linalg.lstsq(a, generalized_force, rcond=None)
    residual = a @ x - generalized_force
    lambdas = x[: jt.shape[1]]
    tau = x[jt.shape[1] :]
    rank = int(np.linalg.matrix_rank(a))
    return tau, lambdas, float(np.linalg.norm(residual)), rank


def static_equilibrium_tau(
    model: mujoco.MjModel,
    qpos_addr: np.ndarray,
    qvel_addr: np.ndarray,
    foot_geoms: list[int],
    qpos0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos0
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    return solve_tau_with_fixed_feet(model, data, qvel_addr, foot_geoms, data.qfrc_bias.copy())


def one_step_inverse_commands(
    model: mujoco.MjModel,
    qpos_addr: np.ndarray,
    qvel_addr: np.ndarray,
    foot_geoms: list[int],
    qpos_seq: np.ndarray,
    qvel_logged: np.ndarray,
    times: np.ndarray,
    torque_limits: np.ndarray | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    data = mujoco.MjData(model)
    qvel_avg = np.zeros(model.nv, dtype=np.float64)

    for frame in range(qpos_seq.shape[0] - 1):
        dt = float(times[frame + 1] - times[frame])
        if dt <= 0.0:
            raise ValueError(f"non-positive dt at frame {frame}: {dt}")

        data.qpos[:] = qpos_seq[frame]
        data.qvel[:] = qvel_logged[frame]
        mujoco.mj_forward(model, data)

        mujoco.mj_differentiatePos(model, qvel_avg, dt, qpos_seq[frame], qpos_seq[frame + 1])
        qacc_des = 2.0 * (qvel_avg - qvel_logged[frame]) / dt

        data.qacc[:] = qacc_des
        mujoco.mj_inverse(model, data)
        tau, lambdas, residual, rank = solve_tau_with_fixed_feet(
            model, data, qvel_addr, foot_geoms, data.qfrc_inverse.copy()
        )
        if torque_limits is None:
            tau_clipped = tau.copy()
            clipped = np.zeros_like(tau, dtype=bool)
        else:
            tau_clipped = np.clip(tau, torque_limits[:, 0], torque_limits[:, 1])
            clipped = np.abs(tau_clipped - tau) > 1e-9

        rows.append(
            {
                "frame": frame,
                "target_frame": frame + 1,
                "dt": dt,
                "qvel_avg": qvel_avg.copy(),
                "qacc_des": qacc_des.copy(),
                "tau": tau.copy(),
                "tau_clipped": tau_clipped.copy(),
                "torque_clipped": clipped.copy(),
                "lambda": lambdas.copy(),
                "residual_norm": residual,
                "constraint_rank": rank,
                "qfrc_inverse_norm": float(np.linalg.norm(data.qfrc_inverse)),
            }
        )

    return rows


def select_qvel_for_inverse(
    model: mujoco.MjModel,
    qpos_seq: np.ndarray,
    qvel_logged: np.ndarray,
    times: np.ndarray,
    velocity_source: str,
) -> np.ndarray:
    if velocity_source == "logged":
        return qvel_logged
    if velocity_source == "zero":
        return np.zeros_like(qvel_logged)
    if velocity_source == "finite-difference":
        return qvel_from_qpos_sequence(model, qpos_seq, times)
    raise ValueError(f"unknown velocity_source {velocity_source!r}")


def actuator_torque_limits(model: mujoco.MjModel) -> np.ndarray:
    if model.nu != len(JOINT_NAMES):
        raise ValueError(f"expected {len(JOINT_NAMES)} actuators, got {model.nu}")
    limits = model.actuator_ctrlrange.copy().astype(np.float64)
    for i in range(model.nu):
        if not bool(model.actuator_ctrllimited[i]):
            limits[i] = [-np.inf, np.inf]
    return limits


def write_static_csv(path: Path, case_id: str, t0: float, base_z0: float, tau: np.ndarray, lambdas: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fieldnames = ["case", "time_s", "base_z", "joint", "static_tau", "contact_lambda_norm"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        lambda_norm = float(np.linalg.norm(lambdas))
        for joint, value in zip(JOINT_NAMES, tau):
            writer.writerow(
                {
                    "case": case_id,
                    "time_s": f"{t0:.9g}",
                    "base_z": f"{base_z0:.9g}",
                    "joint": joint,
                    "static_tau": f"{float(value):.9g}",
                    "contact_lambda_norm": f"{lambda_norm:.9g}",
                }
            )


def write_commands_csv(
    path: Path,
    times: np.ndarray,
    base_zs: np.ndarray,
    qpos_addr: np.ndarray,
    qvel_addr: np.ndarray,
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame",
        "time_s",
        "target_frame",
        "target_time_s",
        "dt",
        "base_z",
        "target_base_z",
        "residual_norm",
        "constraint_rank",
        "qfrc_inverse_norm",
    ]
    for joint in JOINT_NAMES:
        fieldnames += [
            f"qvel_avg_{joint}",
            f"qacc_des_{joint}",
            f"tau_cmd_{joint}",
            f"tau_cmd_clipped_{joint}",
            f"tau_cmd_was_clipped_{joint}",
        ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            frame = int(row["frame"])
            target_frame = int(row["target_frame"])
            qvel_avg = row["qvel_avg"]
            qacc_des = row["qacc_des"]
            tau = row["tau"]
            tau_clipped = row["tau_clipped"]
            clipped = row["torque_clipped"]
            assert isinstance(qvel_avg, np.ndarray)
            assert isinstance(qacc_des, np.ndarray)
            assert isinstance(tau, np.ndarray)
            assert isinstance(tau_clipped, np.ndarray)
            assert isinstance(clipped, np.ndarray)

            out = {
                "frame": frame,
                "time_s": f"{float(times[frame]):.9g}",
                "target_frame": target_frame,
                "target_time_s": f"{float(times[target_frame]):.9g}",
                "dt": f"{float(row['dt']):.9g}",
                "base_z": f"{float(base_zs[frame]):.9g}",
                "target_base_z": f"{float(base_zs[target_frame]):.9g}",
                "residual_norm": f"{float(row['residual_norm']):.9g}",
                "constraint_rank": int(row["constraint_rank"]),
                "qfrc_inverse_norm": f"{float(row['qfrc_inverse_norm']):.9g}",
            }
            for i, joint in enumerate(JOINT_NAMES):
                out[f"qvel_avg_{joint}"] = f"{float(qvel_avg[qvel_addr[i]]):.9g}"
                out[f"qacc_des_{joint}"] = f"{float(qacc_des[qvel_addr[i]]):.9g}"
                out[f"tau_cmd_{joint}"] = f"{float(tau[i]):.9g}"
                out[f"tau_cmd_clipped_{joint}"] = f"{float(tau_clipped[i]):.9g}"
                out[f"tau_cmd_was_clipped_{joint}"] = int(bool(clipped[i]))
            writer.writerow(out)


def summarize_commands(case_id: str, dt: float, source_dt: float, rows: list[dict[str, object]], static_tau: np.ndarray) -> dict[str, object]:
    tau = np.stack([row["tau"] for row in rows])
    tau_clipped = np.stack([row["tau_clipped"] for row in rows])
    clipped = np.stack([row["torque_clipped"] for row in rows])
    qacc = np.stack([row["qacc_des"] for row in rows])
    residuals = np.array([float(row["residual_norm"]) for row in rows], dtype=np.float64)
    max_idx = np.unravel_index(np.argmax(np.abs(tau)), tau.shape)
    max_clipped_idx = np.unravel_index(np.argmax(np.abs(tau_clipped)), tau_clipped.shape)
    return {
        "case": case_id,
        "frames": len(rows),
        "dt": dt,
        "source_median_dt": source_dt,
        "static_tau_max_abs": float(np.max(np.abs(static_tau))),
        "tau_cmd_mean_abs": float(np.mean(np.abs(tau))),
        "tau_cmd_max_abs": float(abs(tau[max_idx])),
        "tau_cmd_max_frame": int(max_idx[0]),
        "tau_cmd_max_joint": JOINT_NAMES[int(max_idx[1])],
        "tau_cmd_clipped_mean_abs": float(np.mean(np.abs(tau_clipped))),
        "tau_cmd_clipped_max_abs": float(abs(tau_clipped[max_clipped_idx])),
        "tau_cmd_clipped_max_frame": int(max_clipped_idx[0]),
        "tau_cmd_clipped_max_joint": JOINT_NAMES[int(max_clipped_idx[1])],
        "clipped_value_fraction": float(np.mean(clipped)),
        "clipped_frame_fraction": float(np.mean(np.any(clipped, axis=1))),
        "qacc_des_mean_norm": float(np.mean(np.linalg.norm(qacc, axis=1))),
        "qacc_des_max_norm": float(np.max(np.linalg.norm(qacc, axis=1))),
        "residual_max": float(np.max(residuals)),
        "residual_mean": float(np.mean(residuals)),
    }


def write_summary_csv(path: Path, summaries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not summaries:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def run_case(
    case_id: str,
    log_dir: Path,
    output_dir: Path,
    xml_path: Path,
    dt: float,
    base_z_mode: str,
    foot_clearance: float,
    velocity_source: str,
    duration_s: float | None,
    sample_mode: str,
    clip_torque: bool,
) -> dict[str, object]:
    csv_path = log_dir / f"walk_diag_{case_id}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    t_raw, data_raw = load_real_log(csv_path)
    source_median_dt = float(np.median(np.diff(t_raw)))
    if sample_mode == "raw":
        times, data = t_raw, data_raw
    elif sample_mode == "resample":
        times, data = resample_real_data(t_raw, data_raw, dt)
    else:
        raise ValueError(f"unknown sample_mode {sample_mode!r}")
    times, data = trim_by_duration(times, data, duration_s)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    qpos_addr, qvel_addr = get_joint_addresses(model)
    foot_geoms = find_foot_collision_geoms(model)

    qpos_seq, qvel_logged, base_zs = build_qpos_sequence(
        model,
        qpos_addr,
        qvel_addr,
        data["q"],
        data["euler"],
        data["gyro"],
        data["dq"],
        base_z_mode,
        initial_base_z=0.62,
        foot_clearance=foot_clearance,
    )

    static_tau, static_lambda, static_residual, static_rank = static_equilibrium_tau(
        model, qpos_addr, qvel_addr, foot_geoms, qpos_seq[0]
    )
    qvel_inverse = select_qvel_for_inverse(model, qpos_seq, qvel_logged, times, velocity_source)
    torque_limits = actuator_torque_limits(model) if clip_torque else None
    rows = one_step_inverse_commands(model, qpos_addr, qvel_addr, foot_geoms, qpos_seq, qvel_inverse, times, torque_limits)

    if sample_mode == "raw":
        dt_tag = "rawdt"
    else:
        dt_tag = f"dt{safe_tag(dt)}"
    clip_tag = "clipped" if clip_torque else "unclipped"
    tag = f"{dt_tag}_{base_z_mode}_{velocity_source}_{clip_tag}_{optional_float_tag(duration_s)}"
    static_path = output_dir / f"{case_id}_{tag}_initial_static_tau.csv"
    commands_path = output_dir / f"{case_id}_{tag}_inverse_pose_commands.csv"
    write_static_csv(static_path, case_id, float(times[0]), float(base_zs[0]), static_tau, static_lambda)
    write_commands_csv(commands_path, times, base_zs, qpos_addr, qvel_addr, rows)

    summary_dt = float(np.median(np.diff(times))) if sample_mode == "raw" else dt
    summary = summarize_commands(case_id, summary_dt, source_median_dt, rows, static_tau)
    summary.update(
        {
            "resampled_samples": len(times),
            "sample_mode": sample_mode,
            "actual_median_dt": float(np.median(np.diff(times))),
            "actual_min_dt": float(np.min(np.diff(times))),
            "actual_max_dt": float(np.max(np.diff(times))),
            "base_z0": float(base_zs[0]),
            "base_z_min": float(np.min(base_zs)),
            "base_z_max": float(np.max(base_zs)),
            "velocity_source": velocity_source,
            "clip_torque": clip_torque,
            "duration_s": "" if duration_s is None else duration_s,
            "static_residual": static_residual,
            "static_constraint_rank": static_rank,
            "static_csv": str(static_path),
            "commands_csv": str(commands_path),
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MuJoCo fixed-foot inverse-dynamics torque commands that track real pose samples."
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "work" / "real_policy_compare")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--xml", type=Path, default=REPO_ROOT / "resources" / "robots" / "x1" / "mjcf" / "xyber_x1_flat.xml")
    parser.add_argument("--case", action="append", default=None, help="Case timestamp, e.g. 20260707_163855")
    parser.add_argument("--dt", type=float, default=0.001, help="Replay command timestep after interpolation.")
    parser.add_argument(
        "--sample-mode",
        choices=["raw", "resample"],
        default="resample",
        help="raw uses logged timestamps directly; resample interpolates to --dt.",
    )
    parser.add_argument("--base-z-mode", choices=["auto-per-frame", "initial"], default="auto-per-frame")
    parser.add_argument("--foot-clearance", type=float, default=0.0)
    parser.add_argument(
        "--velocity-source",
        choices=["logged", "finite-difference", "zero"],
        default="finite-difference",
        help="Current qvel used when solving one-step inverse commands.",
    )
    parser.add_argument("--duration-s", type=float, default=None, help="Only generate commands up to this time.")
    parser.add_argument("--no-clip-torque", action="store_true", help="Do not clip output torque by MJCF actuator ctrlrange.")
    args = parser.parse_args()

    log_dir = args.log_dir or (args.root / "real_logs_7_7")
    output_dir = args.output_dir or (args.root / "inverse_pose_commands")
    cases = args.case or DEFAULT_CASES

    summaries = []
    for case_id in cases:
        summary = run_case(
            case_id=case_id,
            log_dir=log_dir,
            output_dir=output_dir,
            xml_path=args.xml,
            dt=args.dt,
            base_z_mode=args.base_z_mode,
            foot_clearance=args.foot_clearance,
            velocity_source=args.velocity_source,
            duration_s=args.duration_s,
            sample_mode=args.sample_mode,
            clip_torque=not args.no_clip_torque,
        )
        summaries.append(summary)
        print(
            f"{case_id}: commands={summary['frames']} dt={summary['dt']:.6g} "
            f"source_median_dt={summary['source_median_dt']:.6g} "
            f"static_tau_max_abs={summary['static_tau_max_abs']:.6g} "
            f"tau_cmd_mean_abs={summary['tau_cmd_mean_abs']:.6g} "
            f"tau_cmd_max_abs={summary['tau_cmd_max_abs']:.6g} "
            f"at frame={summary['tau_cmd_max_frame']} joint={summary['tau_cmd_max_joint']}"
        )

    dt_tag = "rawdt" if args.sample_mode == "raw" else f"dt{safe_tag(args.dt)}"
    clip_tag = "clipped" if not args.no_clip_torque else "unclipped"
    summary_path = output_dir / (
        f"summary_{dt_tag}_{args.base_z_mode}_{args.velocity_source}_{clip_tag}_{optional_float_tag(args.duration_s)}.csv"
    )
    write_summary_csv(summary_path, summaries)
    print(f"summary_csv={summary_path}")


if __name__ == "__main__":
    main()
