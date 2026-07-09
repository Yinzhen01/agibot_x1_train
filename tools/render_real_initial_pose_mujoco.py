from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import mujoco
import numpy as np

from replay_real_actions_mujoco import (
    JOINT_NAMES,
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


def put_label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (255, 255, 255), -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    return out


def render_view(renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData, label: str, azimuth: float, elevation: float, distance: float) -> np.ndarray:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2]) * 0.55]
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    renderer.update_scene(data, camera=cam)
    rgb = renderer.render()
    return put_label(rgb, label)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render MuJoCo initial pose from real-machine CSV frame 0.")
    parser.add_argument("--csv", type=Path, default=Path("work/real_policy_compare/real_logs_7_7/walk_diag_20260707_163855.csv"))
    parser.add_argument("--mjcf", type=Path, default=Path("resources/robots/x1/mjcf/xyber_x1_flat.xml"))
    parser.add_argument("--output-dir", type=Path, default=Path("work/real_policy_compare/replay_mujoco/visuals"))
    parser.add_argument("--base-z", type=float, default=0.62)
    parser.add_argument("--auto-base-z", action="store_true", default=True)
    parser.add_argument("--foot-clearance", type=float, default=0.0)
    parser.add_argument("--orientation-source", choices=["imu", "euler", "level"], default="imu")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    rows = load_rows(args.csv)
    first = rows[0]

    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    data = mujoco.MjData(model)
    qpos_addr, qvel_addr = get_joint_addresses(model)

    data.qpos[0:3] = np.array([0.0, 0.0, args.base_z], dtype=np.float64)
    data.qpos[3:7] = row_quat_wxyz(first, args.orientation_source)
    data.qpos[qpos_addr] = row_vector(first, "pos")
    data.qvel[0:3] = 0.0
    data.qvel[3:6] = row_gyro(first)
    data.qvel[qvel_addr] = row_vector(first, "vel")
    mujoco.mj_forward(model, data)

    if args.auto_base_z:
        data.qpos[2] += args.foot_clearance - min_robot_geom_z(model, data)
        mujoco.mj_forward(model, data)

    actual_base_z = float(data.qpos[2])
    min_z = min_robot_geom_z(model, data)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    views = [
        ("front", 180, -10, 2.5),
        ("right", 90, -10, 2.5),
        ("back", 0, -10, 2.5),
        ("oblique", 135, -18, 2.8),
    ]
    rendered = [
        render_view(renderer, model, data, f"{name} | base_z={actual_base_z:.3f} min_geom_z={min_z:.3f}", az, el, dist)
        for name, az, el, dist in views
    ]

    top = np.concatenate(rendered[:2], axis=1)
    bottom = np.concatenate(rendered[2:], axis=1)
    grid = np.concatenate([top, bottom], axis=0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_id = infer_case_id(args.csv)
    z_tag = f"{actual_base_z:.3f}".replace(".", "p")
    out_path = args.output_dir / f"{case_id}_initial_pose_{args.orientation_source}_z{z_tag}.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    print(f"PNG={out_path}")
    print(f"case={case_id} orientation_source={args.orientation_source} init_base_z={actual_base_z:.9g} min_robot_geom_z={min_z:.9g}")
    print("joint_pos=" + ",".join(f"{name}:{float(first['pos_' + name]):.6g}" for name in JOINT_NAMES))


if __name__ == "__main__":
    main()
