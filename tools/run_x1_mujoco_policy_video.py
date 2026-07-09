import argparse
import csv
import math
import os
from collections import deque
from pathlib import Path

from isaacgym import gymapi  # must be imported before torch through humanoid deps
from isaacgym.torch_utils import *
import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import *
from humanoid.utils import task_registry


def quat_xyzw_to_euler(quat_xyzw):
    return R.from_quat(quat_xyzw).as_euler('xyz')


def build_pd_arrays(env_cfg):
    # X1 order is left 6 joints then right 6 joints. Config stores one leg pattern.
    kps = np.array(list(env_cfg.control.stiffness.values()) * 2, dtype=np.double)
    kds = np.array(list(env_cfg.control.damping.values()) * 2, dtype=np.double)
    default = np.array(list(env_cfg.init_state.default_joint_angles.values()), dtype=np.double)
    return kps, kds, default


def make_obs(data, model, env_cfg, default_dof_pos, action, hist_obs, command, sim_time):
    nq = env_cfg.env.num_actions
    q = data.qpos[-nq:].astype(np.double)
    dq = data.qvel[-nq:].astype(np.double)

    quat_wxyz = data.sensor('body-orientation').data.astype(np.double)
    quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
    rot = R.from_quat(quat_xyzw)
    omega = data.sensor('body-angular-velocity').data.astype(np.double)
    v_world = data.qvel[:3].astype(np.double)
    v_body = rot.apply(v_world, inverse=True).astype(np.double)
    eu_ang = quat_xyzw_to_euler(quat_xyzw)
    eu_ang[eu_ang > math.pi] -= 2 * math.pi

    obs = np.zeros([1, env_cfg.env.num_single_obs], dtype=np.float32)
    if env_cfg.env.num_commands == 5:
        obs[0, 0] = math.sin(2 * math.pi * sim_time / env_cfg.rewards.cycle_time)
        obs[0, 1] = math.cos(2 * math.pi * sim_time / env_cfg.rewards.cycle_time)
        obs[0, 2] = command[0] * env_cfg.normalization.obs_scales.lin_vel
        obs[0, 3] = command[1] * env_cfg.normalization.obs_scales.lin_vel
        obs[0, 4] = command[2] * env_cfg.normalization.obs_scales.ang_vel
    elif env_cfg.env.num_commands == 3:
        obs[0, 0] = command[0] * env_cfg.normalization.obs_scales.lin_vel
        obs[0, 1] = command[1] * env_cfg.normalization.obs_scales.lin_vel
        obs[0, 2] = command[2] * env_cfg.normalization.obs_scales.ang_vel

    c = env_cfg.env.num_commands
    obs[0, c:c+nq] = (q - default_dof_pos) * env_cfg.normalization.obs_scales.dof_pos
    obs[0, c+nq:c+2*nq] = dq * env_cfg.normalization.obs_scales.dof_vel
    obs[0, c+2*nq:c+3*nq] = action
    obs[0, c+3*nq:c+3*nq+3] = omega * env_cfg.normalization.obs_scales.ang_vel
    obs[0, c+3*nq+3:c+3*nq+6] = eu_ang * env_cfg.normalization.obs_scales.quat

    if env_cfg.env.add_stand_bool:
        vel_norm = np.linalg.norm(command)
        obs[0, -1] = vel_norm <= env_cfg.commands.stand_com_threshold

    obs = np.clip(obs, -env_cfg.normalization.clip_observations, env_cfg.normalization.clip_observations)
    hist_obs.append(obs)
    hist_obs.popleft()

    policy_input = np.zeros([1, env_cfg.env.num_observations], dtype=np.float32)
    for i, h in enumerate(hist_obs):
        start = i * env_cfg.env.num_single_obs
        policy_input[0, start:start + env_cfg.env.num_single_obs] = h[0]
    return policy_input, q, dq, v_body, omega, eu_ang, quat_xyzw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', required=True)
    parser.add_argument('--task', default='x1_dh_stand')
    parser.add_argument('--duration', type=float, default=12.0)
    parser.add_argument('--cmd-x', type=float, default=0.5)
    parser.add_argument('--cmd-y', type=float, default=0.0)
    parser.add_argument('--cmd-yaw', type=float, default=0.0)
    parser.add_argument('--init-height', type=float, default=0.62)
    parser.add_argument('--output-dir', default='outputs/policy_videos')
    parser.add_argument('--tag', default='model3600_mujoco_closed_loop')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--render-every', type=int, default=20)
    args = parser.parse_args()

    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    model_path = env_cfg.asset.xml_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    model = mujoco.MjModel.from_xml_path(model_path)
    model.opt.timestep = 0.001
    data = mujoco.MjData(model)

    nq = env_cfg.env.num_actions
    kps, kds, default_dof_pos = build_pd_arrays(env_cfg)
    tau_limit = 500.0 * np.ones(nq, dtype=np.double)

    init_pos = np.array(env_cfg.init_state.pos, dtype=np.double)
    init_pos[2] = args.init_height
    data.qpos[:3] = init_pos
    init_rot = getattr(env_cfg.init_state, 'rot', [0.0, 0.0, 0.0, 1.0])
    data.qpos[3:7] = np.array([init_rot[3], init_rot[0], init_rot[1], init_rot[2]], dtype=np.double)
    data.qpos[-nq:] = default_dof_pos
    mujoco.mj_forward(model, data)

    policy = torch.jit.load(args.policy, map_location='cpu')
    policy.eval()
    command = np.array([args.cmd_x, args.cmd_y, args.cmd_yaw], dtype=np.double)
    hist_obs = deque([np.zeros([1, env_cfg.env.num_single_obs], dtype=np.float32) for _ in range(env_cfg.env.frame_stack)])
    action = np.zeros(nq, dtype=np.double)
    target_q = np.zeros(nq, dtype=np.double)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / f'x1_dh_stand_{args.tag}_cmdx{args.cmd_x:g}_h{args.init_height:g}.mp4'
    gif_path = out_dir / f'x1_dh_stand_{args.tag}_cmdx{args.cmd_x:g}_h{args.init_height:g}.gif'
    csv_path = out_dir / f'x1_dh_stand_{args.tag}_cmdx{args.cmd_x:g}_h{args.init_height:g}_mujoco.csv'

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.8
    cam.azimuth = 135
    cam.elevation = -18
    fps = int(round(1.0 / (model.opt.timestep * args.render_every)))
    video = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (args.width, args.height))
    if not video.isOpened():
        raise RuntimeError(f'Cannot open video writer: {mp4_path}')

    fields = ['step', 'time', 'root_x', 'root_y', 'root_z', 'base_vx', 'base_vy', 'base_vz', 'base_wz', 'roll', 'pitch', 'yaw']
    fields += [f'action_{i}' for i in range(nq)]
    fields += [f'q_{i}' for i in range(nq)]
    fields += [f'dq_{i}' for i in range(nq)]
    fields += [f'tau_{i}' for i in range(nq)]

    gif_frames = []
    min_z = 999.0
    max_abs_vx = 0.0
    total_steps = int(args.duration / model.opt.timestep)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in range(total_steps):
            sim_time = step * model.opt.timestep
            if step % env_cfg.control.decimation == 0:
                policy_input, q, dq, v_body, omega, eu_ang, quat_xyzw = make_obs(
                    data, model, env_cfg, default_dof_pos, action, hist_obs, command, sim_time
                )
                with torch.no_grad():
                    action = policy(torch.from_numpy(policy_input))[0].cpu().numpy().astype(np.double)
                action = np.clip(action, -env_cfg.normalization.clip_actions, env_cfg.normalization.clip_actions)
                target_q = action * env_cfg.control.action_scale
            else:
                q = data.qpos[-nq:].astype(np.double)
                dq = data.qvel[-nq:].astype(np.double)
                quat_wxyz = data.sensor('body-orientation').data.astype(np.double)
                quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
                rot = R.from_quat(quat_xyzw)
                v_body = rot.apply(data.qvel[:3].astype(np.double), inverse=True)
                omega = data.sensor('body-angular-velocity').data.astype(np.double)
                eu_ang = quat_xyzw_to_euler(quat_xyzw)

            tau = (target_q + default_dof_pos - q) * kps - dq * kds
            tau = np.clip(tau, -tau_limit, tau_limit)
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)

            root = data.qpos[:3].copy()
            min_z = min(min_z, float(root[2]))
            max_abs_vx = max(max_abs_vx, abs(float(v_body[0])))

            if step % env_cfg.control.decimation == 0:
                row = {
                    'step': step,
                    'time': sim_time,
                    'root_x': float(root[0]),
                    'root_y': float(root[1]),
                    'root_z': float(root[2]),
                    'base_vx': float(v_body[0]),
                    'base_vy': float(v_body[1]),
                    'base_vz': float(v_body[2]),
                    'base_wz': float(omega[2]),
                    'roll': float(eu_ang[0]),
                    'pitch': float(eu_ang[1]),
                    'yaw': float(eu_ang[2]),
                }
                for i in range(nq):
                    row[f'action_{i}'] = float(action[i])
                    row[f'q_{i}'] = float(q[i])
                    row[f'dq_{i}'] = float(dq[i])
                    row[f'tau_{i}'] = float(tau[i])
                writer.writerow(row)

            if step % args.render_every == 0:
                cam.lookat[:] = [float(root[0]), float(root[1]), max(float(root[2]), 0.55)]
                renderer.update_scene(data, camera=cam)
                rgb = renderer.render()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                video.write(bgr)
                if (step // args.render_every) % 10 == 0:
                    gif_frames.append(rgb.copy())

            if step % 1000 == 0:
                print(f'step={step} t={sim_time:.2f} z={root[2]:.3f} vx={v_body[0]:.3f} pitch={eu_ang[1]:.3f}', flush=True)

    video.release()
    if gif_frames:
        imageio.mimsave(str(gif_path), gif_frames, fps=10)
    print('MP4=' + str(mp4_path))
    print('GIF=' + str(gif_path))
    print('CSV=' + str(csv_path))
    print(f'init_height={args.init_height:g}')
    print(f'min_root_z={min_z:.3f} max_abs_vx={max_abs_vx:.3f} frames={int(total_steps / args.render_every)} fps={fps}')


if __name__ == '__main__':
    main()
