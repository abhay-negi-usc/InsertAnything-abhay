"""Evaluate an RL-Games checkpoint in an InsertAnything environment (Isaac Lab 3.0).

Same protocol as the release validation: the first reset is discarded, then every completed
evaluation episode prints the success rate over all environments.

    ./scripts/run.sh python scripts/rl_games/play.py --task InsertAnything-LHole-III-Direct-v0 \
        --num_envs 128 --seed 0 --max_episodes 3 --checkpoint <path.pth>
"""

import argparse
import json
import os
import random
import time

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--max_episodes", type=int, default=None, help="Stop after this many evaluation episodes.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--summary_json", type=str, default=None, help="Write per-episode results to this file.")
parser.add_argument(
    "--set",
    action="append",
    default=[],
    metavar="KEY=VALUE",
    help="Override an env cfg field, dotted, value parsed as JSON (e.g. task.hand_init_pos_range=[0.01,0.01,0.01]).",
)
add_launcher_args(parser)
args_cli = parser.parse_args()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper, register_rl_games_env  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rl_games.common.player import BasePlayer  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

import InsertAnything.tasks  # noqa: E402, F401


def allow_rl_games_checkpoint_globals():
    """torch >= 2.6 loads with weights_only=True. RL-Games checkpoints also pickle numpy scalars
    (epoch, frame, best reward); allowlist exactly those types instead of disabling safe loading."""
    import codecs

    import numpy as np

    core = getattr(np, "_core", None) or np.core
    # Checkpoints pickled under numpy 1 name "numpy.core.multiarray.scalar"; numpy 2 moved it to _core.
    safe = [(core.multiarray.scalar, "numpy.core.multiarray.scalar"), core.multiarray.scalar, np.dtype, codecs.encode]
    safe += [getattr(np.dtypes, n) for n in dir(np.dtypes) if n.endswith("DType")] if hasattr(np, "dtypes") else []
    torch.serialization.add_safe_globals(safe)


allow_rl_games_checkpoint_globals()


def apply_overrides(cfg, overrides):
    for item in overrides:
        key, value = item.split("=", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        obj = cfg
        *path, leaf = key.split(".")
        for name in path:
            obj = obj[name] if isinstance(obj, dict) else getattr(obj, name)
        if isinstance(obj, dict):
            obj[leaf] = value
        else:
            setattr(obj, leaf, value)
        print(f"[INFO] override {key} = {value!r}")


def final_metrics(u):
    """Per-env end-of-episode errors of the held asset against its target (the env's own geometry)."""
    import numpy as np

    from InsertAnything.tasks.direct.insertanything import insertanything_utils as iu

    held_pos, held_quat = iu.get_held_base_pose(u.held_pos, u.held_quat, u.num_envs, u.device)
    tgt_pos, tgt_quat = iu.get_target_held_base_pose(u.fixed_pos, u.fixed_quat, u.cfg_task, u.num_envs, u.device)
    from InsertAnything.tasks.direct.insertanything import insertanything_math as im

    u._diag_offset_in_hole = im.quat_apply(im.quat_conjugate(tgt_quat), held_pos - tgt_pos)
    xy = torch.linalg.vector_norm(tgt_pos[:, 0:2] - held_pos[:, 0:2], dim=1)
    z = held_pos[:, 2] - tgt_pos[:, 2]
    sym = [a * np.pi / 180.0 for a in (u.cfg_task.symmetry_angles_deg or [0.0])]
    _, _, yaw = iu.get_closest_symmetry_transform(held_quat, tgt_quat, sym)
    return xy, z, yaw


def summarize_failures_from(u, xy_z_yaw, peak_force):
    import numpy as np

    xy, z, yaw = xy_z_yaw
    fail = ~u.ep_succeeded.bool()
    h = u.cfg_task.fixed_asset_cfg.height
    stats = {
        "n_fail": int(fail.sum().item()),
        "fail_xy_mm_median": float(xy[fail].median().item() * 1000) if fail.any() else None,
        "fail_z_mm_median": float(z[fail].median().item() * 1000) if fail.any() else None,
        "fail_yaw_deg_median": float(np.degrees(yaw[fail].median().item())) if fail.any() else None,
        "fail_inserted_but_misaligned": int((fail & (z < h * u.cfg_task.success_threshold)).sum().item()),
        "fail_xy_ok_yaw_bad": int((fail & (xy < 0.0025) & (yaw >= u.cfg_task.yaw_success_threshold)).sum().item()),
        "fail_on_surface": int((fail & (z > 0.0) & (xy < 0.005)).sum().item()),
        "peak_force_N_median": float(peak_force.median().item()),
        "fail_offset_in_hole_mm_mean": (u._diag_offset_in_hole[fail].mean(0) * 1000).tolist() if fail.any() else None,
        "fail_offset_in_hole_mm_std": (u._diag_offset_in_hole[fail].std(0) * 1000).tolist() if fail.any() else None,
    }
    print("[FAILURES]", json.dumps(stats), flush=True)
    return stats


def main():
    import numpy as np  # noqa: F401
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)
    apply_overrides(env_cfg, args_cli.set)

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]
    env_cfg.evaluation_mode = True
    resume_path = os.path.abspath(args_cli.checkpoint)
    env_cfg.log_dir = os.path.dirname(os.path.dirname(resume_path))

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RlGamesVecEnvWrapper.from_agent_cfg(env, agent_cfg)
        register_rl_games_env(env)

        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner = Runner()
        runner.load(agent_cfg)
        agent: BasePlayer = runner.create_player()
        agent.restore(resume_path)
        agent.reset()

        dt = env.unwrapped.step_dt
        # Discard the first reset: freshly spawned assets and sensors can leave transient state.
        _ = env.reset()
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["obs"]
        _ = agent.get_batch_size(obs, 1)
        if agent.is_rnn:
            agent.init_rnn()

        unwrapped = env.unwrapped
        results = []
        completed = 0
        peak_force = torch.zeros(unwrapped.num_envs, device=unwrapped.device)
        while True:
            start = time.time()
            with torch.inference_mode():
                obs = agent.obs_to_torch(obs)
                actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
                # Read the episode's end state before the step that resets it.
                pre_xy_z_yaw = final_metrics(unwrapped)
                obs, _, dones, infos = env.step(actions)
                peak_force = torch.maximum(peak_force, torch.linalg.vector_norm(unwrapped.contact_force_local, dim=-1))
                if isinstance(infos, dict) and "eval_printout" in infos:
                    print(infos["eval_printout"], end="", flush=True)
                    succ = unwrapped.ep_succeeded.bool()
                    failures = summarize_failures_from(unwrapped, pre_xy_z_yaw, peak_force)
                    peak_force.zero_()
                    times = unwrapped.ep_success_times[succ].float() * unwrapped.step_dt
                    results.append(
                        {
                            "episode": completed + 1,
                            "successes": int(succ.sum().item()),
                            "num_envs": unwrapped.num_envs,
                            "mean_success_time_s": float(times.mean().item()) if len(times) else None,
                            "failures": failures,
                        }
                    )
                    completed += 1
                if len(dones) > 0 and agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
            if args_cli.max_episodes is not None and completed >= args_cli.max_episodes:
                break
            sleep = dt - (time.time() - start)
            if args_cli.real_time and sleep > 0:
                time.sleep(sleep)

        total = sum(r["successes"] for r in results)
        trials = sum(r["num_envs"] for r in results)
        if trials:
            print(f"\n[RESULT] {args_cli.task}: {total}/{trials} = {100.0 * total / trials:.2f}% over {completed} episodes")
        if args_cli.summary_json:
            os.makedirs(os.path.dirname(os.path.abspath(args_cli.summary_json)), exist_ok=True)
            with open(args_cli.summary_json, "w") as f:
                json.dump(
                    {
                        "task": args_cli.task,
                        "checkpoint": resume_path,
                        "seed": env_cfg.seed,
                        "overrides": args_cli.set,
                        "episodes": results,
                        "success_rate": total / trials if trials else None,
                    },
                    f,
                    indent=2,
                )
        env.close()


if __name__ == "__main__":
    main()
