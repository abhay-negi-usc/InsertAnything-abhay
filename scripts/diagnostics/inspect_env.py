"""Print an InsertAnything env's state after reset and after a few zero-action steps (port check)."""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="InsertAnything-LHole-III-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=20)
add_launcher_args(parser)
args = parser.parse_args()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import InsertAnything.tasks  # noqa: E402, F401
from InsertAnything.tasks.direct.insertanything import insertanything_math as M  # noqa: E402

torch.set_printoptions(precision=4, sci_mode=False, linewidth=160)


def report(u, tag):
    print(f"--- {tag}")
    print("fingertip_pos      ", u.fingertip_midpoint_pos[:2])
    print("fingertip_quat wxyz", u.fingertip_midpoint_quat[:2])
    print("fixed_pos (hole)   ", u.fixed_pos[:2], " obs frame", u.fixed_pos_obs_frame[:2])
    rel_q, rel_p = M.tf_combine(*M.tf_inverse(u.fingertip_midpoint_quat, u.fingertip_midpoint_pos), u.held_quat, u.held_pos)
    print("held in fingertip p", rel_p[:2], " q", rel_q[:2])
    print("gripper joints     ", u.joint_pos[:2, 7:])
    print("contact_force_local", u.contact_force_local[:2])
    print("ee_linvel_fd       ", u.ee_linvel_fd[:2])
    obs = u._get_observations()["policy"]
    print("policy obs dim", obs.shape, "\n", obs[:2])


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.evaluation_mode = True
    with launch_simulation(cfg, args):
        env = gym.make(args.task, cfg=cfg)
        u = env.unwrapped
        env.reset()
        report(u, "after reset")
        a = torch.zeros((u.num_envs, 6), device=u.device)
        for _ in range(args.steps):
            env.step(a)
        report(u, f"after {args.steps} zero-action steps")
        a[:, 2] = -1.0
        for _ in range(args.steps):
            env.step(a)
        report(u, f"after {args.steps} steps pushing down")
        env.close()


if __name__ == "__main__":
    main()
