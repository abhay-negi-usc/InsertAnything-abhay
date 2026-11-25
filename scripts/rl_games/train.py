"""Train an RL-Games agent on an InsertAnything environment (Isaac Lab 3.0).

    ./scripts/run.sh python scripts/rl_games/train.py --task InsertAnything-Usb-NoProngs-Direct-v0 \
        --num_envs 1024 --seed 0 [--max_iterations N] [--checkpoint <resume .pth>] [--set key=json ...]

Checkpoints and TensorBoard logs go to logs/rl_games/<agent config name>/<run name>/ (``--run_name``, else a
timestamp). Env-config overrides use the same ``--set`` syntax as play.py and usb_probe.py, so a policy can be
trained and evaluated under the same controller settings.
"""

import argparse
import json
import os
import random
import time
from datetime import datetime

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed (-1: random).")
parser.add_argument("--max_iterations", type=int, default=None, help="Maximum number of training epochs.")
parser.add_argument("--checkpoint", type=str, default=None, help="Resume from this checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--run_name", type=str, default=None, help="Run folder name (default: a timestamp).")
parser.add_argument(
    "--set",
    action="append",
    default=[],
    metavar="KEY=VALUE",
    help="Override an env cfg field, dotted, value parsed as JSON (e.g. ctrl.payload_gravity_comp=true).",
)
parser.add_argument(
    "--agent_set",
    action="append",
    default=[],
    metavar="KEY=VALUE",
    help="Override an agent cfg field under params.config, e.g. minibatch_size=8192.",
)
add_launcher_args(parser)
args_cli = parser.parse_args()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper, register_rl_games_env  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rl_games.common.algo_observer import IsaacAlgoObserver  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

import InsertAnything.tasks  # noqa: E402, F401


def allow_rl_games_checkpoint_globals():
    """torch >= 2.6 loads with weights_only=True; RL-Games checkpoints pickle numpy scalars (see play.py)."""
    import codecs

    import numpy as np

    core = getattr(np, "_core", None) or np.core
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


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, args_cli.agent)
    apply_overrides(env_cfg, args_cli.set)
    apply_overrides(agent_cfg["params"]["config"], args_cli.agent_set)

    params, config = agent_cfg["params"], agent_cfg["params"]["config"]
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    if args_cli.seed is not None:
        params["seed"] = args_cli.seed
    if args_cli.max_iterations is not None:
        config["max_epochs"] = args_cli.max_iterations
    env_cfg.seed = params["seed"]

    log_root = os.path.abspath(os.path.join("logs", "rl_games", config["name"]))
    run_name = args_cli.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config["train_dir"] = log_root
    config["full_experiment_name"] = run_name
    log_dir = os.path.join(log_root, run_name)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    with open(os.path.join(log_dir, "params", "agent.yaml"), "w") as f:
        yaml.safe_dump(agent_cfg, f)
    with open(os.path.join(log_dir, "params", "run.json"), "w") as f:
        json.dump({"task": args_cli.task, "seed": params["seed"], "num_envs": args_cli.num_envs,
                   "env_overrides": args_cli.set, "agent_overrides": args_cli.agent_set,
                   "checkpoint": args_cli.checkpoint}, f, indent=2)
    print(f"[INFO] Logging experiment in: {log_dir}")

    resume_path = None
    if args_cli.checkpoint is not None:
        resume_path = os.path.abspath(args_cli.checkpoint)
        params["load_checkpoint"] = True
        params["load_path"] = resume_path
        print(f"[INFO] Resuming from: {resume_path}")
    env_cfg.log_dir = log_dir

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RlGamesVecEnvWrapper.from_agent_cfg(env, agent_cfg)
        register_rl_games_env(env)
        config["num_actors"] = env.unwrapped.num_envs

        runner = Runner(IsaacAlgoObserver())
        runner.load(agent_cfg)
        runner.reset()
        run_args = {"train": True, "play": False, "sigma": float(args_cli.sigma) if args_cli.sigma else None}
        if resume_path is not None:
            run_args["checkpoint"] = resume_path
        start = time.time()
        runner.run(run_args)
        print(f"[INFO] Training time: {time.time() - start:.0f} s")
        env.close()


if __name__ == "__main__":
    main()
