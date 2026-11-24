"""Controller and contact probes on an InsertAnything task, no policy needed.

    ./scripts/run.sh python scripts/diagnostics/usb_probe.py --task InsertAnything-Usb-NoProngs-Direct-v0 \
        --mode response --num_envs 32 --out logs/usb/response_franka.json
    ./scripts/run.sh python scripts/diagnostics/usb_probe.py --task ... --mode insert --num_envs 128 --out ...

--mode response: the task-space inertia at the start pose, then a fixed action schedule in free space (up,
    then steps in x, y, z and yaw); the fingertip's trajectory at the physics rate, in the task frame.
--mode insert: a privileged scripted policy (it reads the plug and socket poses) aligns the plug above the
    socket and pushes it in; success, time, peak gripper-plug force and the final plug state.

Domain randomization of the controller (gains, dead zone) is switched off unless --keep_dr; the gains are
then the CtrlCfg defaults (or --set overrides).
"""

import argparse
import json
import os

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--mode", choices=["response", "insert"], required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--out", required=True)
parser.add_argument("--keep_dr", action="store_true")
parser.add_argument("--trace", type=int, default=0, help="Print env 0 every N policy steps (insert mode).")
parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
add_launcher_args(parser)
args = parser.parse_args()

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import InsertAnything.tasks  # noqa: E402, F401
from InsertAnything.tasks.direct.insertanything import insertanything_math as im  # noqa: E402
from InsertAnything.tasks.direct.insertanything import insertanything_utils as iu  # noqa: E402


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
        setattr(obj, leaf, value) if not isinstance(obj, dict) else obj.__setitem__(leaf, value)
        print(f"[INFO] override {key} = {value!r}")


def task_inertia(u):
    J = u.fingertip_midpoint_jacobian
    Minv = torch.inverse(u.arm_mass_matrix)
    return torch.inverse(J @ Minv @ J.transpose(1, 2))


def yaw_of(q):
    return im.get_euler_xyz(q)[2]


# (action 6-vector, policy steps): task-frame pos (x, y, z) then rot (rx, ry, rz)
SCHEDULE = (
    [([0, 0, 0.5, 0, 0, 0], 6), ([0] * 6, 10)]
    + [([0.5, 0, 0, 0, 0, 0], 8), ([0] * 6, 12), ([-0.5, 0, 0, 0, 0, 0], 8), ([0] * 6, 12)]
    + [([0, 0.5, 0, 0, 0, 0], 8), ([0] * 6, 12)]
    + [([0, 0, -0.5, 0, 0, 0], 4), ([0] * 6, 12)]
    + [([0, 0, 0, 0, 0, 0.5], 8), ([0] * 6, 12)]
)


def run_response(env, u):
    log = {"t": [], "pos": [], "yaw": [], "action": []}
    orig = u._apply_action

    def logged_apply():
        orig()
        log["t"].append(float(u.episode_length_buf[0].item()))
        log["pos"].append(u.fingertip_midpoint_pos.mean(0).tolist())
        log["yaw"].append(float(torch.atan2(torch.sin(yaw_of(u.fingertip_midpoint_quat)), torch.cos(yaw_of(u.fingertip_midpoint_quat))).mean().item()))
        log["action"].append(u.actions[0].tolist())

    lam = task_inertia(u)
    info = {
        "lambda_diag_mean": torch.diagonal(lam, dim1=1, dim2=2).mean(0).tolist(),
        "lambda_mean": lam.mean(0).reshape(-1).tolist(),
        "lambda_trans_eig_mean": torch.linalg.eigvalsh(lam[:, :3, :3]).mean(0).tolist(),
        "gains": u.task_prop_gains[0].tolist(),
        "deriv_gains": u.task_deriv_gains[0].tolist(),
        "start_pos": u.fingertip_midpoint_pos.mean(0).tolist(),
    }
    u._apply_action = logged_apply
    for a, n in SCHEDULE:
        act = torch.tensor(a, dtype=torch.float32, device=u.device).repeat(u.num_envs, 1)
        for _ in range(n):
            env.step(act)
    u._apply_action = orig
    info["physics_dt"] = u.physics_dt
    info["log"] = log
    return info


def oracle_action(u, phase):
    """Privileged scripted policy: put the plug origin on the socket's axis, then push it home."""
    held_pos, held_quat = iu.get_held_base_pose(u.held_pos, u.held_quat, u.num_envs, u.device)
    tgt_pos, tgt_quat = iu.get_target_held_base_pose(u.fixed_pos, u.fixed_quat, u.cfg_task, u.num_envs, u.device)
    err = tgt_pos - held_pos
    xy = torch.linalg.vector_norm(err[:, 0:2], dim=1)
    above = 0.012  # plug origin 12 mm above seated = tip ~1.6 mm above the lip
    aligned = (xy < 0.0005) & (phase > 0)
    phase = torch.where(aligned, torch.full_like(phase, 2), phase)
    phase = torch.where(phase == 0, torch.ones_like(phase), phase)
    pos = torch.stack((err[:, 0], err[:, 1], err[:, 2] + above), dim=1)
    # PD on the plug position (the env's EMA on actions and 15 Hz policy rate need the D term).
    vel = u.fingertip_midpoint_linvel
    a_pos = torch.clamp((pos - 0.15 * vel) / 0.02, -0.5, 0.5)
    # Inserting: full push down (the action interface's largest force, Kp * 0.02 m), xy still servoed.
    a_pos[:, 2] = torch.where(phase == 2, -torch.ones_like(a_pos[:, 2]), a_pos[:, 2])
    # Rotation actions turn the fingertip about the task axes, and the plug turns with it.
    dyaw = im.get_euler_xyz(im.quat_mul(tgt_quat, im.quat_conjugate(held_quat)))[2]
    dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))
    a_rot = torch.zeros_like(a_pos)
    a_rot[:, 2] = torch.clamp((dyaw - 0.15 * u.fingertip_midpoint_angvel[:, 2]) / 0.097, -0.5, 0.5)
    return torch.cat((a_pos, a_rot), dim=1), phase


def run_insert(env, u, episodes):
    results = []
    steps = int(u.max_episode_length)
    for ep in range(episodes):
        if ep > 0:
            env.reset()
        phase = torch.zeros(u.num_envs, dtype=torch.long, device=u.device)
        done_t = torch.full((u.num_envs,), -1.0, device=u.device)
        peak_f = torch.zeros(u.num_envs, device=u.device)
        for k in range(steps - 2):
            a, phase = oracle_action(u, phase)
            env.step(a)
            if args.trace and k % args.trace == 0:
                hp, hq = iu.get_held_base_pose(u.held_pos, u.held_quat, u.num_envs, u.device)
                tp, tq = iu.get_target_held_base_pose(u.fixed_pos, u.fixed_quat, u.cfg_task, u.num_envs, u.device)
                e = (tp - hp)[0] * 1000
                tip_in_plug = im.quat_apply(im.quat_conjugate(hq), u.fingertip_midpoint_pos - hp)[0] * 1000
                print(f"[V] k={k} phase={int(phase[0])} err_mm={[round(v, 2) for v in e.tolist()]} "
                      f"a={[round(v, 2) for v in a[0].tolist()]} tip_in_plug_mm={[round(v, 1) for v in tip_in_plug.tolist()]} "
                      f"F={[round(v, 2) for v in u.contact_force_local[0].tolist()]}", flush=True)
            f = torch.linalg.vector_norm(u.contact_force_local, dim=-1)
            peak_f = torch.maximum(peak_f, f)
            succ = u._get_curr_successes(u.cfg_task.success_threshold)
            newly = succ & (done_t < 0)
            done_t[newly] = (k + 1) * u.step_dt
        held_pos, held_quat = iu.get_held_base_pose(u.held_pos, u.held_quat, u.num_envs, u.device)
        tgt_pos, tgt_quat = iu.get_target_held_base_pose(u.fixed_pos, u.fixed_quat, u.cfg_task, u.num_envs, u.device)
        off = im.quat_apply(im.quat_conjugate(tgt_quat), held_pos - tgt_pos) * 1000
        final = u._get_curr_successes(u.cfg_task.success_threshold)
        ever = done_t >= 0
        r = {
            "episode": ep,
            "n": u.num_envs,
            "ever_seated": int(ever.sum()),
            "seated_at_end": int(final.sum()),
            "time_to_seat_s_median": float(done_t[ever].median()) if ever.any() else None,
            "peak_grip_force_N_median": float(peak_f.median()),
            "peak_grip_force_N_p90": float(peak_f.quantile(0.9)),
            "final_depth_mm_median": float(off[:, 2].median()),
            "final_depth_mm_p90": float(off[:, 2].quantile(0.9)),
            "final_xy_mm_median": float(torch.linalg.vector_norm(off[:, 0:2], dim=1).median()),
            "reached_insert_phase": int((phase == 2).sum()),
        }
        print("[INSERT]", json.dumps(r), flush=True)
        results.append(r)
    return results


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    if not args.keep_dr:
        cfg.task.dr_randomize_dynamics = False
    apply_overrides(cfg, args.set)
    cfg.seed = args.seed
    cfg.evaluation_mode = True
    with launch_simulation(cfg, args):
        env = gym.make(args.task, cfg=cfg)
        u = env.unwrapped
        env.reset()
        env.reset()
        if args.mode == "response":
            out = run_response(env, u)
        else:
            out = {"insert": run_insert(env, u, args.episodes)}
        out.update(task=args.task, overrides=args.set, keep_dr=args.keep_dr)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f)
        print("[PROBE] wrote", args.out, flush=True)
        env.close()


if __name__ == "__main__":
    main()
