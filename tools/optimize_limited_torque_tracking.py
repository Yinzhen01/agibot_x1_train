from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from inverse_replay_real_pose_commands import (  # noqa: E402
    DEFAULT_CASES,
    actuator_torque_limits,
    build_qpos_sequence,
    load_real_log,
    optional_float_tag,
    qvel_from_qpos_sequence,
    resample_real_data,
    safe_tag,
    select_qvel_for_inverse,
    trim_by_duration,
)
from replay_real_actions_mujoco import JOINT_NAMES, get_joint_addresses  # noqa: E402


def qpos_to_euler(qpos: np.ndarray) -> np.ndarray:
    return R.from_quat(qpos[3:7][[1, 2, 3, 0]]).as_euler("xyz")


def wrap_rpy_error(value: np.ndarray) -> np.ndarray:
    out = value.copy()
    out[2] = (out[2] + np.pi) % (2.0 * np.pi) - np.pi
    return out


def rollout_one_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    start_qpos: np.ndarray,
    start_qvel: np.ndarray,
    tau: np.ndarray,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    data.qpos[:] = start_qpos
    data.qvel[:] = start_qvel
    data.ctrl[:] = tau
    mujoco.mj_forward(model, data)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return data.qpos.copy(), data.qvel.copy()


def make_residual(
    model: mujoco.MjModel,
    scratch: mujoco.MjData,
    qpos_addr: np.ndarray,
    qvel_addr: np.ndarray,
    start_qpos: np.ndarray,
    start_qvel: np.ndarray,
    target_qpos: np.ndarray,
    target_qvel: np.ndarray,
    steps: int,
    tau_limits: np.ndarray,
    prev_tau: np.ndarray,
    joint_weight: float,
    joint_vel_weight: float,
    base_rpy_weight: float,
    base_z_weight: float,
    tau_weight: float,
    smooth_weight: float,
):
    limit_scale = np.maximum(np.abs(tau_limits).max(axis=1), 1.0)
    target_joint_q = target_qpos[qpos_addr]
    target_joint_dq = target_qvel[qvel_addr]
    target_rpy = qpos_to_euler(target_qpos)
    target_z = float(target_qpos[2])

    def residual(tau: np.ndarray) -> np.ndarray:
        next_qpos, next_qvel = rollout_one_control(model, scratch, start_qpos, start_qvel, tau, steps)
        parts = []
        if joint_weight > 0.0:
            parts.append(np.sqrt(joint_weight) * (next_qpos[qpos_addr] - target_joint_q))
        if joint_vel_weight > 0.0:
            parts.append(np.sqrt(joint_vel_weight) * (next_qvel[qvel_addr] - target_joint_dq))
        if base_rpy_weight > 0.0:
            parts.append(np.sqrt(base_rpy_weight) * wrap_rpy_error(qpos_to_euler(next_qpos) - target_rpy))
        if base_z_weight > 0.0:
            parts.append(np.sqrt(base_z_weight) * np.array([float(next_qpos[2]) - target_z], dtype=np.float64))
        if tau_weight > 0.0:
            parts.append(np.sqrt(tau_weight) * (tau / limit_scale))
        if smooth_weight > 0.0:
            parts.append(np.sqrt(smooth_weight) * ((tau - prev_tau) / limit_scale))
        return np.concatenate(parts)

    return residual


def run_case(
    case_id: str,
    log_dir: Path,
    output_dir: Path,
    xml_path: Path,
    dt: float,
    sample_mode: str,
    duration_s: float | None,
    base_z_mode: str,
    foot_clearance: float,
    velocity_source: str,
    max_nfev: int,
    joint_weight: float,
    joint_vel_weight: float,
    base_rpy_weight: float,
    base_z_weight: float,
    tau_weight: float,
    smooth_weight: float,
    state_mode: str,
) -> dict[str, object]:
    csv_path = log_dir / f"walk_diag_{case_id}.csv"
    t_raw, data_raw = load_real_log(csv_path)
    source_median_dt = float(np.median(np.diff(t_raw)))
    if sample_mode == "raw":
        times, data = t_raw, data_raw
    elif sample_mode == "resample":
        times, data = resample_real_data(t_raw, data_raw, dt)
    else:
        raise ValueError(f"unknown sample mode {sample_mode!r}")
    times, data = trim_by_duration(times, data, duration_s)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    qpos_addr, qvel_addr = get_joint_addresses(model)
    tau_limits = actuator_torque_limits(model)
    lower = tau_limits[:, 0]
    upper = tau_limits[:, 1]

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

    if velocity_source == "finite-difference":
        qvel_init_seq = qvel_from_qpos_sequence(model, qpos_seq, times)
    else:
        qvel_init_seq = select_qvel_for_inverse(model, qpos_seq, qvel_logged, times, velocity_source)

    sim = mujoco.MjData(model)
    scratch = mujoco.MjData(model)
    sim.qpos[:] = qpos_seq[0]
    sim.qvel[:] = qvel_init_seq[0]
    mujoco.mj_forward(model, sim)

    prev_tau = np.zeros(model.nu, dtype=np.float64)
    rows = []
    joint_errs = []
    joint_vel_errs = []
    base_rpy_errs = []
    root_z_values = []
    clipped_values = []
    eval_counts = []

    for frame in range(len(times) - 1):
        step_dt = float(times[frame + 1] - times[frame])
        steps = max(1, int(round(step_dt / model.opt.timestep)))
        if state_mode == "rollout":
            start_qpos = sim.qpos.copy()
            start_qvel = sim.qvel.copy()
        elif state_mode == "reset-real":
            start_qpos = qpos_seq[frame].copy()
            start_qvel = qvel_init_seq[frame].copy()
        else:
            raise ValueError(f"unknown state_mode {state_mode!r}")
        target_qpos = qpos_seq[frame + 1]
        target_qvel = qvel_init_seq[frame + 1]

        residual = make_residual(
            model,
            scratch,
            qpos_addr,
            qvel_addr,
            start_qpos,
            start_qvel,
            target_qpos,
            target_qvel,
            steps,
            tau_limits,
            prev_tau,
            joint_weight,
            joint_vel_weight,
            base_rpy_weight,
            base_z_weight,
            tau_weight,
            smooth_weight,
        )

        result = least_squares(
            residual,
            np.clip(prev_tau, lower, upper),
            bounds=(lower, upper),
            max_nfev=max_nfev,
            xtol=1e-4,
            ftol=1e-4,
            gtol=1e-4,
            x_scale=np.maximum(np.abs(tau_limits).max(axis=1), 1.0),
        )
        tau = np.clip(result.x, lower, upper)
        next_qpos, next_qvel = rollout_one_control(model, sim, start_qpos, start_qvel, tau, steps)
        if state_mode == "rollout":
            sim.qpos[:] = next_qpos
            sim.qvel[:] = next_qvel
            mujoco.mj_forward(model, sim)

        joint_err = next_qpos[qpos_addr] - target_qpos[qpos_addr]
        joint_vel_err = next_qvel[qvel_addr] - target_qvel[qvel_addr]
        base_rpy_err = wrap_rpy_error(qpos_to_euler(next_qpos) - qpos_to_euler(target_qpos))
        clipped = (np.abs(tau - lower) < 1e-6) | (np.abs(tau - upper) < 1e-6)

        joint_errs.append(joint_err)
        joint_vel_errs.append(joint_vel_err)
        base_rpy_errs.append(base_rpy_err)
        root_z_values.append(float(next_qpos[2]))
        clipped_values.append(clipped)
        eval_counts.append(int(result.nfev))

        row: dict[str, object] = {
            "frame": frame,
            "time_s": f"{float(times[frame]):.9g}",
            "target_frame": frame + 1,
            "target_time_s": f"{float(times[frame + 1]):.9g}",
            "dt": f"{step_dt:.9g}",
            "mujoco_steps": steps,
            "cost": f"{float(result.cost):.9g}",
            "nfev": int(result.nfev),
            "success": int(bool(result.success)),
            "root_z": f"{float(next_qpos[2]):.9g}",
            "target_root_z": f"{float(target_qpos[2]):.9g}",
            "base_roll_err": f"{float(base_rpy_err[0]):.9g}",
            "base_pitch_err": f"{float(base_rpy_err[1]):.9g}",
            "base_yaw_err": f"{float(base_rpy_err[2]):.9g}",
        }
        for i, joint in enumerate(JOINT_NAMES):
            row[f"tau_opt_{joint}"] = f"{float(tau[i]):.9g}"
            row[f"tau_at_limit_{joint}"] = int(bool(clipped[i]))
            row[f"sim_pos_{joint}"] = f"{float(next_qpos[qpos_addr[i]]):.9g}"
            row[f"target_pos_{joint}"] = f"{float(target_qpos[qpos_addr[i]]):.9g}"
            row[f"err_pos_{joint}"] = f"{float(joint_err[i]):.9g}"
            row[f"sim_vel_{joint}"] = f"{float(next_qvel[qvel_addr[i]]):.9g}"
            row[f"target_vel_{joint}"] = f"{float(target_qvel[qvel_addr[i]]):.9g}"
            row[f"err_vel_{joint}"] = f"{float(joint_vel_err[i]):.9g}"
        rows.append(row)
        prev_tau = tau

    if sample_mode == "raw":
        dt_tag = "rawdt"
    else:
        dt_tag = f"dt{safe_tag(dt)}"
    tag = f"{dt_tag}_{base_z_mode}_{velocity_source}_{state_mode}_bounded_opt_{optional_float_tag(duration_s)}"
    out_path = output_dir / f"{case_id}_{tag}_tracking.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    joint_err_arr = np.vstack(joint_errs)
    joint_vel_err_arr = np.vstack(joint_vel_errs)
    base_rpy_err_arr = np.vstack(base_rpy_errs)
    clipped_arr = np.vstack(clipped_values)
    max_idx = np.unravel_index(np.argmax(np.abs(joint_err_arr)), joint_err_arr.shape)

    return {
        "case": case_id,
        "frames": len(rows),
        "sample_mode": sample_mode,
        "state_mode": state_mode,
        "source_median_dt": source_median_dt,
        "actual_median_dt": float(np.median(np.diff(times))),
        "joint_rmse": float(np.sqrt(np.mean(joint_err_arr * joint_err_arr))),
        "joint_mean_abs": float(np.mean(np.abs(joint_err_arr))),
        "joint_max_abs": float(abs(joint_err_arr[max_idx])),
        "joint_max_frame": int(max_idx[0]),
        "joint_max_name": JOINT_NAMES[int(max_idx[1])],
        "final_joint_rmse": float(np.sqrt(np.mean(joint_err_arr[-1] * joint_err_arr[-1]))),
        "joint_vel_rmse": float(np.sqrt(np.mean(joint_vel_err_arr * joint_vel_err_arr))),
        "base_rpy_rmse": float(np.sqrt(np.mean(base_rpy_err_arr * base_rpy_err_arr))),
        "root_z_min": float(np.min(root_z_values)),
        "root_z_final": float(root_z_values[-1]),
        "tau_at_limit_value_fraction": float(np.mean(clipped_arr)),
        "tau_at_limit_frame_fraction": float(np.mean(np.any(clipped_arr, axis=1))),
        "mean_nfev": float(np.mean(eval_counts)),
        "max_nfev": int(np.max(eval_counts)),
        "tracking_csv": str(out_path),
    }


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-step bounded torque optimization for real-pose tracking.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "work" / "real_policy_compare")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--xml", type=Path, default=REPO_ROOT / "resources" / "robots" / "x1" / "mjcf" / "xyber_x1_flat.xml")
    parser.add_argument("--case", action="append", default=None)
    parser.add_argument("--sample-mode", choices=["raw", "resample"], default="raw")
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--base-z-mode", choices=["auto-per-frame", "initial"], default="auto-per-frame")
    parser.add_argument("--foot-clearance", type=float, default=0.0)
    parser.add_argument("--velocity-source", choices=["finite-difference", "logged", "zero"], default="finite-difference")
    parser.add_argument("--max-nfev", type=int, default=25)
    parser.add_argument(
        "--state-mode",
        choices=["rollout", "reset-real"],
        default="reset-real",
        help="rollout accumulates simulated state; reset-real solves each one-step problem from the logged real state.",
    )
    parser.add_argument("--joint-weight", type=float, default=1.0)
    parser.add_argument("--joint-vel-weight", type=float, default=0.01)
    parser.add_argument("--base-rpy-weight", type=float, default=0.05)
    parser.add_argument("--base-z-weight", type=float, default=0.05)
    parser.add_argument("--tau-weight", type=float, default=1e-4)
    parser.add_argument("--smooth-weight", type=float, default=1e-3)
    args = parser.parse_args()

    log_dir = args.log_dir or (args.root / "real_logs_7_7")
    output_dir = args.output_dir or (args.root / "bounded_torque_tracking")
    cases = args.case or DEFAULT_CASES

    summaries = []
    for case_id in cases:
        summary = run_case(
            case_id=case_id,
            log_dir=log_dir,
            output_dir=output_dir,
            xml_path=args.xml,
            dt=args.dt,
            sample_mode=args.sample_mode,
            duration_s=args.duration_s,
            base_z_mode=args.base_z_mode,
            foot_clearance=args.foot_clearance,
            velocity_source=args.velocity_source,
            max_nfev=args.max_nfev,
            joint_weight=args.joint_weight,
            joint_vel_weight=args.joint_vel_weight,
            base_rpy_weight=args.base_rpy_weight,
            base_z_weight=args.base_z_weight,
            tau_weight=args.tau_weight,
            smooth_weight=args.smooth_weight,
            state_mode=args.state_mode,
        )
        summaries.append(summary)
        print(
            f"{case_id}: frames={summary['frames']} "
            f"joint_rmse={summary['joint_rmse']:.6g} "
            f"mean_abs={summary['joint_mean_abs']:.6g} "
            f"max_abs={summary['joint_max_abs']:.6g} "
            f"root_z_final={summary['root_z_final']:.6g} "
            f"limit_frame_frac={summary['tau_at_limit_frame_fraction']:.3f}"
        )

    if args.sample_mode == "raw":
        dt_tag = "rawdt"
    else:
        dt_tag = f"dt{safe_tag(args.dt)}"
    summary_path = output_dir / (
        f"summary_{dt_tag}_{args.base_z_mode}_{args.velocity_source}_{args.state_mode}_bounded_opt_{optional_float_tag(args.duration_s)}.csv"
    )
    write_summary(summary_path, summaries)
    print(f"summary_csv={summary_path}")


if __name__ == "__main__":
    main()
