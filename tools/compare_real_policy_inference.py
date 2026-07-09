from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_actor_critic_class():
    module_path = REPO_ROOT / "humanoid" / "algo" / "ppo" / "actor_critic_dh.py"
    spec = importlib.util.spec_from_file_location("actor_critic_dh", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ActorCriticDH


ActorCriticDH = load_actor_critic_class()


NUM_ACTIONS = 12
NUM_SINGLE_OBS = 47
FRAME_STACK = 66
SHORT_FRAME_STACK = 5
NUM_OBS = NUM_SINGLE_OBS * FRAME_STACK
NUM_PRIVILEGED_OBS = 219

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

DEFAULT_CASES = [
    ("20260707_163855", "origin_v7.3"),
    ("20260707_165431", "origin_v7.3"),
    ("20260707_170120", "origin_v5.8"),
    ("20260707_170424", "origin_v5.8"),
]

POLICY_BY_VERSION = {
    "origin_v7.3": "model_7999.pt",
    "origin_v5.8": "model_3000.pt",
}


def load_policy(checkpoint_path: Path) -> ActorCriticDH:
    num_short_obs = SHORT_FRAME_STACK * NUM_SINGLE_OBS
    kwargs = dict(
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[768, 256, 128],
        state_estimator_hidden_dims=[256, 128, 64],
        in_channels=FRAME_STACK,
        kernel_size=[6, 4],
        filter_size=[32, 16],
        stride_size=[3, 2],
        lh_output_dim=64,
        init_noise_std=1.0,
    )
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        policy = ActorCriticDH(
            num_short_obs,
            NUM_SINGLE_OBS,
            NUM_PRIVILEGED_OBS,
            NUM_ACTIONS,
            **kwargs,
        )
    try:
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        loaded = torch.load(checkpoint_path, map_location="cpu")
    policy.load_state_dict(loaded["model_state_dict"])
    policy.eval()
    return policy


def read_obs_bin(path: Path) -> np.ndarray:
    obs = np.fromfile(path, dtype=np.float32)
    if obs.size % NUM_OBS != 0:
        raise ValueError(f"{path} has {obs.size} float32 values, not divisible by {NUM_OBS}")
    return obs.reshape(-1, NUM_OBS)


def read_real_actions(csv_path: Path) -> np.ndarray:
    action_columns = [f"action_{name}" for name in JOINT_NAMES]
    rows = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = [name for name in action_columns if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{csv_path} is missing action columns: {missing}")
        for row in reader:
            rows.append([float(row[name]) for name in action_columns])
    return np.asarray(rows, dtype=np.float32)


def infer_actions(policy: ActorCriticDH, obs: np.ndarray, batch_size: int) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, obs.shape[0], batch_size):
            batch = torch.from_numpy(obs[start : start + batch_size]).float()
            outputs.append(policy.act_inference(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def write_frame_csv(path: Path, real: np.ndarray, pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame"]
    for joint in JOINT_NAMES:
        fieldnames += [f"real_{joint}", f"pt_{joint}", f"diff_{joint}"]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(real.shape[0]):
            row = {"frame": i}
            for j, joint in enumerate(JOINT_NAMES):
                row[f"real_{joint}"] = f"{real[i, j]:.9g}"
                row[f"pt_{joint}"] = f"{pred[i, j]:.9g}"
                row[f"diff_{joint}"] = f"{pred[i, j] - real[i, j]:.9g}"
            writer.writerow(row)


def summarize_case(case_id: str, version: str, real: np.ndarray, pred: np.ndarray) -> dict:
    diff = pred - real
    abs_diff = np.abs(diff)
    max_flat = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    return {
        "case_id": case_id,
        "version": version,
        "frames": real.shape[0],
        "mean_abs": float(abs_diff.mean()),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs": float(abs_diff[max_flat]),
        "max_frame": int(max_flat[0]),
        "max_joint": JOINT_NAMES[int(max_flat[1])],
    }


def write_summary_csv(path: Path, summaries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id", "version", "frames", "mean_abs", "rmse", "max_abs", "max_frame", "max_joint"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


def parse_case(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("case must be TIMESTAMP=VERSION, e.g. 20260707_163855=origin_v7.3")
    case_id, version = text.split("=", 1)
    if version not in POLICY_BY_VERSION:
        raise argparse.ArgumentTypeError(f"unknown version {version!r}; expected one of {sorted(POLICY_BY_VERSION)}")
    return case_id, version


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare real-machine logged policy outputs with local PT inference.")
    parser.add_argument("--root", type=Path, default=Path("work/real_policy_compare"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--summary-name", default="summary.csv")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--case", action="append", type=parse_case, help="TIMESTAMP=VERSION. Can be repeated.")
    parser.add_argument("--no-frame-csv", action="store_true", help="Only write the summary CSV.")
    args = parser.parse_args()

    log_dir = args.root / "real_logs_7_7"
    policy_root = args.root / "policy"
    out_dir = args.output_dir or (args.root / "comparisons")
    cases = args.case if args.case else DEFAULT_CASES

    policies: dict[str, ActorCriticDH] = {}
    summaries = []

    for case_id, version in cases:
        obs_path = log_dir / f"tm_obs_input_{case_id}.bin"
        csv_path = log_dir / f"walk_diag_{case_id}.csv"
        checkpoint_path = policy_root / version / POLICY_BY_VERSION[version]

        if version not in policies:
            policies[version] = load_policy(checkpoint_path)

        obs = read_obs_bin(obs_path)
        real = read_real_actions(csv_path)
        if obs.shape[0] != real.shape[0]:
            raise ValueError(f"{case_id}: obs frames {obs.shape[0]} != csv rows {real.shape[0]}")

        pred = infer_actions(policies[version], obs, args.batch_size)
        summary = summarize_case(case_id, version, real, pred)
        summaries.append(summary)

        if not args.no_frame_csv:
            write_frame_csv(out_dir / f"{case_id}_{version}_pt_vs_real_actions.csv", real, pred)

        print(
            f"{case_id} {version}: frames={summary['frames']} "
            f"mean_abs={summary['mean_abs']:.8g} rmse={summary['rmse']:.8g} "
            f"max_abs={summary['max_abs']:.8g} "
            f"at frame={summary['max_frame']} joint={summary['max_joint']}"
        )

    summary_path = out_dir / args.summary_name
    write_summary_csv(summary_path, summaries)
    print(f"summary_csv={summary_path}")


if __name__ == "__main__":
    main()
