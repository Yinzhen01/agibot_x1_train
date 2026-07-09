import argparse
import csv
import os
import sys
from datetime import datetime

from isaacgym import gymapi
from isaacgym.torch_utils import *
import torch

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import *
from humanoid.utils import get_args, task_registry


def disable_eval_randomization(env_cfg):
    env_cfg.env.num_envs = 1
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.max_init_terrain_level = 0
    env_cfg.env.episode_length_s = 1000
    env_cfg.noise.add_noise = False
    env_cfg.noise.curriculum = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.continuous_push = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_com = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_torque = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_motor_offset = False
    env_cfg.domain_rand.randomize_joint_friction = False
    env_cfg.domain_rand.randomize_joint_damping = False
    env_cfg.domain_rand.randomize_joint_armature = False
    env_cfg.domain_rand.randomize_lag_timesteps = False
    env_cfg.commands.heading_command = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='x1_dh_stand')
    parser.add_argument('--load-run', default='2026-07-01_13-16-27x1_dh_stand_gui_env3000_20260701_1315')
    parser.add_argument('--checkpoint', type=int, default=600)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--command-x', type=float, default=0.5)
    parser.add_argument('--command-y', type=float, default=0.0)
    parser.add_argument('--command-yaw', type=float, default=0.0)
    parser.add_argument('--init-height', type=float, default=0.62)
    parser.add_argument('--output-dir', default=os.path.join(LEGGED_GYM_ROOT_DIR, 'outputs', 'policy_videos'))
    cli = parser.parse_args()

    sys.argv = [
        sys.argv[0], '--task', cli.task, '--load_run', cli.load_run,
        '--checkpoint', str(cli.checkpoint), '--num_envs', '1', '--headless'
    ]
    args = get_args()
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    disable_eval_randomization(env_cfg)
    env_cfg.init_state.pos[2] = cli.init_height
    train_cfg.seed = 123145
    train_cfg.runner.resume = True

    os.makedirs(cli.output_dir, exist_ok=True)
    tag = f"{cli.task}_model{cli.checkpoint}_cmdx{cli.command_x:g}_h{cli.init_height:g}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    csv_path = os.path.join(cli.output_dir, tag + '_rollout.csv')

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    obs = env.get_observations()

    dof_names = list(env.dof_names)
    fieldnames = [
        'step', 'command_x', 'command_y', 'command_yaw',
        'root_x', 'root_y', 'root_z', 'root_qx', 'root_qy', 'root_qz', 'root_qw',
        'base_vel_x', 'base_vel_y', 'base_vel_z', 'base_yaw_vel', 'reward', 'done',
    ]
    fieldnames += [f'action_{name}' for name in dof_names]
    fieldnames += [f'dof_pos_{name}' for name in dof_names]
    fieldnames += [f'dof_vel_{name}' for name in dof_names]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(cli.steps):
            with torch.no_grad():
                actions = policy(obs.detach())
            env.commands[:, 0] = cli.command_x
            env.commands[:, 1] = cli.command_y
            env.commands[:, 2] = cli.command_yaw
            env.commands[:, 3] = 0.0
            obs, critic_obs, rews, dones, infos = env.step(actions.detach())
            root = env.root_states[0]
            row = {
                'step': step,
                'command_x': cli.command_x,
                'command_y': cli.command_y,
                'command_yaw': cli.command_yaw,
                'root_x': float(root[0].item()),
                'root_y': float(root[1].item()),
                'root_z': float(root[2].item()),
                'root_qx': float(root[3].item()),
                'root_qy': float(root[4].item()),
                'root_qz': float(root[5].item()),
                'root_qw': float(root[6].item()),
                'base_vel_x': float(env.base_lin_vel[0, 0].item()),
                'base_vel_y': float(env.base_lin_vel[0, 1].item()),
                'base_vel_z': float(env.base_lin_vel[0, 2].item()),
                'base_yaw_vel': float(env.base_ang_vel[0, 2].item()),
                'reward': float(rews[0].item()),
                'done': int(dones[0].item()),
            }
            for i, name in enumerate(dof_names):
                row[f'action_{name}'] = float(actions[0, i].item())
                row[f'dof_pos_{name}'] = float(env.dof_pos[0, i].item())
                row[f'dof_vel_{name}'] = float(env.dof_vel[0, i].item())
            writer.writerow(row)
            if step % 100 == 0:
                print(f"step={step} reward={row['reward']:.4f} z={row['root_z']:.3f} vx={row['base_vel_x']:.3f} done={row['done']}", flush=True)
    print('CSV=' + csv_path)
    print(f'INIT_HEIGHT={cli.init_height:g}')
    print('DOF_NAMES=' + ','.join(dof_names))


if __name__ == '__main__':
    main()
