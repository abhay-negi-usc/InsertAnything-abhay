"""Controller and contact probes on an InsertAnything task, no policy needed.

    ./scripts/run.sh python scripts/diagnostics/usb_probe.py --task InsertAnything-Usb-NoProngs-Direct-v0 \
        --mode response --num_envs 32 --out logs/usb/response_franka.json
    ./scripts/run.sh python scripts/diagnostics/usb_probe.py --task ... --mode insert --num_envs 128 --out ...

--mode response: the task-space inertia at the start pose, then a fixed action schedule in free space (up,
    then steps in x, y, z and yaw); the fingertip's trajectory at the physics rate, in the task frame, with
    the arm's joint positions, velocities and torques, and per step segment the joint-limit margins and
    torque saturation ([RESPONSE] lines).
--mode drift: zero action in free space; the steady drift velocity and the force that explains it.
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
parser.add_argument("--mode", choices=["response", "insert", "drift"], required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--out", required=True)
parser.add_argument("--keep_dr", action="store_true")
parser.add_argument("--trace", type=int, default=0, help="Print env 0 every N policy steps (insert mode).")
parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
parser.add_argument("--keep_fixed_materials", action="store_true",
                    help="Do not let the env rewrite the fixed asset's shape materials (set_friction). A/B test for "
                    "whether that rewrite drops the compliant-contact prong material.")
add_launcher_args(parser)
args = parser.parse_args()

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import InsertAnything.tasks  # noqa: E402, F401
from InsertAnything.tasks.direct.insertanything import insertanything_compat as compat  # noqa: E402
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


def arm_limits(u):
    """(pos_limits (7, 2) rad, effort_limits (7,) N m) of the arm joints, in the env's arm order."""
    data = u._robot.data
    ids = u.arm_joint_ids
    lim = data.soft_joint_pos_limits if hasattr(data, "soft_joint_pos_limits") else data.joint_pos_limits
    pos_lim = compat.t(lim)[0, ids]
    eff_lim = compat.t(data.joint_effort_limits)[0, ids] if hasattr(data, "joint_effort_limits") else None
    return pos_lim, eff_lim


def run_response(env, u):
    log = {"t": [], "pos": [], "yaw": [], "action": [], "q": [], "qd": [], "tau": [], "limit_margin_min": [],
           "tau_cmd_ratio_max": []}
    orig = u._apply_action
    pos_lim, eff_lim = arm_limits(u)

    def logged_apply():
        orig()
        log["t"].append(float(u.episode_length_buf[0].item()))
        log["pos"].append(u.fingertip_midpoint_pos.mean(0).tolist())
        log["yaw"].append(float(torch.atan2(torch.sin(yaw_of(u.fingertip_midpoint_quat)), torch.cos(yaw_of(u.fingertip_midpoint_quat))).mean().item()))
        log["action"].append(u.actions[0].tolist())
        # Arm joints, env 0 [deg, deg/s, N m]; per joint over envs: the smallest distance to a position
        # limit [deg] and the largest |tau| / effort limit (the commanded torque, before PhysX clamps it).
        q = compat.joint_pos(u._robot)[:, u.arm_joint_ids]
        log["q"].append(torch.rad2deg(q[0]).tolist())
        log["qd"].append(torch.rad2deg(compat.joint_vel(u._robot)[0, u.arm_joint_ids]).tolist())
        tau = u.joint_torque[:, 0:7]
        log["tau"].append(tau[0].tolist())
        margin = torch.minimum(q - pos_lim[:, 0], pos_lim[:, 1] - q).min(0).values
        log["limit_margin_min"].append(torch.rad2deg(margin).tolist())
        if eff_lim is not None:
            log["tau_cmd_ratio_max"].append((tau.abs() / eff_lim).max(0).values.tolist())

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
    info["arm_pos_limits_deg"] = torch.rad2deg(pos_lim).tolist()
    info["arm_effort_limits"] = eff_lim.tolist() if eff_lim is not None else None
    info["start_q_deg"] = log["q"][0]
    info["segments"] = segment_summary(log)
    for s in info["segments"]:
        print("[RESPONSE]", json.dumps(s), flush=True)
    info["log"] = log
    return info


def segment_summary(log):
    """Per schedule segment (a step and the hold after it): the fingertip's peak displacement from the
    segment start [mm], the smallest joint-limit margin [deg] and largest |tau|/limit seen, and the joints
    those occur at. A one-sided response (+x moves, -x does not) with a joint at its limit or saturated in
    one direction only is a limit, not inertia."""
    out = []
    per_step = len(log["t"]) // sum(n for _, n in SCHEDULE)  # physics steps per policy step
    for i, (a, n) in enumerate(SCHEDULE):
        if not any(a):
            continue
        hold = SCHEDULE[i + 1][1] if i + 1 < len(SCHEDULE) and not any(SCHEDULE[i + 1][0]) else 0
        k0 = sum(m for _, m in SCHEDULE[:i]) * per_step
        k2 = k0 + (n + hold) * per_step
        p = np.array(log["pos"][k0:k2]) - np.array(log["pos"][k0])
        peak = p[np.abs(p).argmax(0), range(3)] * 1000
        margin = np.array(log["limit_margin_min"][k0:k2]).min(0)
        s = {"action": a, "peak_disp_mm": peak.round(2).tolist(),
             "min_limit_margin_deg": margin.round(2).tolist(),
             "tightest_joint": f"A{int(margin.argmin()) + 1}"}
        if log["tau_cmd_ratio_max"]:
            r = np.array(log["tau_cmd_ratio_max"][k0:k2]).max(0)
            s["max_tau_ratio"] = r.round(3).tolist()
            s["most_loaded_joint"] = f"A{int(r.argmax()) + 1}"
        out.append(s)
    return out


def run_drift(env, u, hold_steps=45):
    """Lift clear of the socket, then hold a zero action: with no position stiffness, any constant force on
    the fingertip (the held part's weight, a bias wrench) shows as a steady drift velocity, task frame."""
    up = torch.tensor([0, 0, 0.5, 0, 0, 0], dtype=torch.float32, device=u.device).repeat(u.num_envs, 1)
    for _ in range(6):
        env.step(up)
    zero = torch.zeros_like(up)
    pos, t = [], []
    for k in range(hold_steps):
        env.step(zero)
        pos.append(u.fingertip_midpoint_pos.mean(0))
        t.append(k * u.step_dt)
    pos, t = torch.stack(pos[15:]), torch.tensor(t[15:], device=u.device)  # after the lift's transient
    tc = t - t.mean()
    vel = ((pos - pos.mean(0)) * tc[:, None]).sum(0) / (tc**2).sum()
    bias = u._payload_gravity_wrench()
    out = {
        "drift_mm_s": (vel * 1000).tolist(),
        "payload_bias_N": bias[0, 0:3].tolist() if bias is not None else None,
        "held_mass_kg": u.cfg_task.held_asset_cfg.mass,
        "deriv_gains": u.task_deriv_gains[0, 0:3].tolist(),
        # Steady drift = F / Kd: the constant task-frame force that explains it.
        "implied_force_N": (vel * u.task_deriv_gains[0, 0:3]).tolist(),
    }
    print("[DRIFT]", json.dumps(out), flush=True)
    return out


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


def force_vs_height(h, f, bin_mm=0.5, top_mm=12.0):
    """Gripper-plug force [N] binned by the plug's height above seated [mm]: per bin the bin's lower edge,
    the sample count and the median / p90 / max force. A rising wall at the prongs' height that the push
    cannot get past shows here as samples piling up at one height."""
    rows = []
    for lo in np.arange(min(float(h.min()), 0.0) if h.numel() else 0.0, top_mm, bin_mm):
        m = (h >= lo) & (h < lo + bin_mm)
        if m.sum() > 0:
            rows.append([round(float(lo), 2), int(m.sum()), round(float(f[m].median()), 3),
                         round(float(f[m].quantile(0.9)), 3), round(float(f[m].max()), 3)])
    return rows


def run_insert(env, u, episodes):
    results = []
    steps = int(u.max_episode_length)
    for ep in range(episodes):
        if ep > 0:
            env.reset()
        phase = torch.zeros(u.num_envs, dtype=torch.long, device=u.device)
        done_t = torch.full((u.num_envs,), -1.0, device=u.device)
        peak_f = torch.zeros(u.num_envs, device=u.device)
        depth_log, force_log = [], []
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
            # Force vs insertion depth, envs pushing in only: depth = the plug's height above seated [mm].
            hp, _ = iu.get_held_base_pose(u.held_pos, u.held_quat, u.num_envs, u.device)
            tp, tq = iu.get_target_held_base_pose(u.fixed_pos, u.fixed_quat, u.cfg_task, u.num_envs, u.device)
            ins = phase == 2
            depth_log.append(im.quat_apply(im.quat_conjugate(tq), hp - tp)[ins, 2] * 1000)
            force_log.append(f[ins])
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
            "force_vs_height_mm": force_vs_height(torch.cat(depth_log), torch.cat(force_log)),
        }
        print("[INSERT]", json.dumps(r), flush=True)
        results.append(r)
    return results


def keep_fixed_materials():
    orig = iu.set_friction

    def set_friction(asset, value, env_ids, device="cpu"):
        if "FixedAsset" in asset.cfg.prim_path:
            print("[INFO] set_friction skipped on", asset.cfg.prim_path, flush=True)
            return
        orig(asset, value, env_ids, device=device)

    iu.set_friction = set_friction


def main():
    if args.keep_fixed_materials:
        keep_fixed_materials()
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
        elif args.mode == "drift":
            out = run_drift(env, u)
        else:
            out = {"insert": run_insert(env, u, args.episodes)}
        out.update(task=args.task, overrides=args.set, keep_dr=args.keep_dr,
                   keep_fixed_materials=args.keep_fixed_materials)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f)
        print("[PROBE] wrote", args.out, flush=True)
        env.close()


if __name__ == "__main__":
    main()
