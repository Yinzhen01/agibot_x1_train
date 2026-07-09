from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

from inverse_replay_real_pose_commands import (  # noqa: E402
    build_qpos_sequence,
    load_real_log,
    optional_float_tag,
    resample_real_data,
    safe_tag,
    trim_by_duration,
)
from replay_real_actions_mujoco import get_joint_addresses  # noqa: E402


def render_frame(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    width: int,
    height: int,
    title: str,
) -> np.ndarray:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.43]
    cam.distance = 1.85
    cam.azimuth = 135.0
    cam.elevation = -15.0

    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    renderer.update_scene(data, camera=cam, scene_option=opt)
    rgb = renderer.render()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(
        bgr,
        title,
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        bgr,
        title,
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return bgr


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a real-machine pose log as a MuJoCo video.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "work" / "real_policy_compare")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--xml", type=Path, default=REPO_ROOT / "resources" / "robots" / "x1" / "mjcf" / "xyber_x1_flat.xml")
    parser.add_argument("--case", default="20260707_163855")
    parser.add_argument("--sample-mode", choices=["raw", "resample"], default="raw")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--base-z-mode", choices=["auto-per-frame", "initial"], default="auto-per-frame")
    parser.add_argument("--foot-clearance", type=float, default=0.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    log_dir = args.log_dir or (args.root / "real_logs_7_7")
    output_dir = args.output_dir or (args.root / "real_log_renders")
    output_dir.mkdir(parents=True, exist_ok=True)

    t_raw, data_raw = load_real_log(log_dir / f"walk_diag_{args.case}.csv")
    if args.sample_mode == "raw":
        times, real = t_raw, data_raw
        dt_tag = "rawdt"
    else:
        times, real = resample_real_data(t_raw, data_raw, args.dt)
        dt_tag = f"dt{safe_tag(args.dt)}"
    times, real = trim_by_duration(times, real, args.duration_s)

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    qpos_addr, qvel_addr = get_joint_addresses(model)
    qpos_seq, qvel_seq, base_zs = build_qpos_sequence(
        model,
        qpos_addr,
        qvel_addr,
        real["q"],
        real["euler"],
        real["gyro"],
        real["dq"],
        args.base_z_mode,
        initial_base_z=0.62,
        foot_clearance=args.foot_clearance,
    )

    tag = f"{args.case}_{dt_tag}_{args.base_z_mode}_{optional_float_tag(args.duration_s)}"
    mp4_path = output_dir / f"{tag}.mp4"
    preview_path = output_dir / f"{tag}_preview.png"

    writer = cv2.VideoWriter(
        str(mp4_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {mp4_path}")

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    frames_written = 0
    preview_written = False
    try:
        for frame in range(0, len(times), max(1, args.frame_stride)):
            data.qpos[:] = qpos_seq[frame]
            data.qvel[:] = qvel_seq[frame]
            mujoco.mj_forward(model, data)
            title = f"{args.case}  t={times[frame]:.2f}s  base_z={base_zs[frame]:.3f}"
            bgr = render_frame(renderer, model, data, args.width, args.height, title)
            writer.write(bgr)
            frames_written += 1
            if not preview_written and times[frame] >= min(1.0, float(times[-1])):
                cv2.imwrite(str(preview_path), bgr)
                preview_written = True
    finally:
        writer.release()
        renderer.close()

    if not preview_written:
        data.qpos[:] = qpos_seq[0]
        data.qvel[:] = qvel_seq[0]
        mujoco.mj_forward(model, data)
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        try:
            cv2.imwrite(str(preview_path), render_frame(renderer, model, data, args.width, args.height, args.case))
        finally:
            renderer.close()

    print(f"video={mp4_path}")
    print(f"preview={preview_path}")
    print(f"frames_written={frames_written} fps={args.fps:g} source_samples={len(times)} frame_stride={args.frame_stride}")


if __name__ == "__main__":
    main()
