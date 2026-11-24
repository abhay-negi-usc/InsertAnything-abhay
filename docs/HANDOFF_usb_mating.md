# Handoff: `feature/usb-mating`

Branch goal: develop USB-A mating on the bench's real KUKA iiwa 14 + Robotiq 2F-85 by replicating
InsertAnything's methodology. The plan is `docs/usb_mating_plan.md`. This document records where the
work stands, how to reproduce it on a new machine, and what to do next.

Status date: 2026-11-24.

---

## 1. State in one paragraph

InsertAnything (Isaac Lab 2.2 / Isaac Sim 5.0) is ported to **Isaac Lab 3.0 / Isaac Sim 6.2**. The released
Franka checkpoints reproduce the paper to within ~2 points. The repo also has:

- an iiwa 14 + 2F-85 cell matching cable-assembly's bench (bench-top and vertical-pegboard variants);
- a **USB-A task** with two receptacle models, the shell's spring prongs removed and the prongs as
  compliant contacts;
- a privileged scripted insertion probe and a controller probe;
- two optional controller extensions, payload gravity compensation and task-space inertia shaping.

**No USB policy has been trained yet.** The controller-tuning study (§5) was still running when this was
written; its results are not in git.

---

## 2. Setting up on a new machine

The port runs on the cable-assembly sim's uv venv. That venv provides Isaac Lab 3.0.0, Isaac Sim 6.2,
torch 2.11+cu130 and Python 3.12.

1. **Clone both repos side by side:** `~/abhay_ws/InsertAnything-abhay` on branch `feature/usb-mating`,
   and `~/abhay_ws/cable-assembly`, then build its `sim/.venv` per that repo.
   - Elsewhere, set `ISAACLAB_VENV=<path to that venv>`.
2. **rl_games and gym are not in that venv.** Install them into this repo's `.deps/`, which is git-ignored:
   ```bash
   uv pip install --python ~/abhay_ws/cable-assembly/sim/.venv/bin/python --target .deps --no-deps \
       rl-games==1.6.1 gym==0.23.1 gym-notices tensorboardX setproctitle
   ```
3. **Run everything through `scripts/run.sh`.** It drops the ROS env vars, puts the repo and `.deps` on
   `PYTHONPATH`, and accepts the EULA:
   ```bash
   ./scripts/run.sh python scripts/rl_games/play.py --task ... --checkpoint ...
   ```
4. **Checkpoints are git-ignored.** Download them from Hugging Face into
   `source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints/`. Commands and SHA256s
   are in `checkpoints/README.md`.
5. **Assets that are not in any git repo** (cable-assembly ignores them on purpose):
   - **iiwa + 2F-85:** cable-assembly's `sim/src/cable_assembly_sim/assets/data/` must contain
     `iiwa14_2f85.usda`, `iiwa14_2f85_bench.usda`, `iiwa14/usd/`, `iiwa14/meshes/` and `robotiq_2f85/`.
     They come from cable-assembly's `scripts/build_iiwa14.py`, or copy them from the old PC.
     Override the location with `CABLE_ASSEMBLY_SIM_ASSETS`.
   - **USB CAD** (McMaster 4974T135 cord, 4574N11 adapter): copy
     `cable-assembly/sim/src/cable_assembly_sim/assets/data/usb/` from the old PC. It holds both STEP
     files and cable-assembly's `usb_plug.usd` / `usb_socket.usd` from `build_usb.py`.
   - **This repo's USB assets** (`assets/usb/*.usd`, `usb_prongs.json`) are derived from that CAD and kept
     out of git for the same licensing reason. Rebuild them, about 1 min:
     ```bash
     cd ~/abhay_ws/cable-assembly/sim && ./run.sh --extra cad python \
         ~/abhay_ws/InsertAnything-abhay/scripts/assets/extract_usb_prongs.py \
         src/cable_assembly_sim/assets/data/usb/4574N11_panel_mount_usb_adapter.step \
         ~/abhay_ws/InsertAnything-abhay/source/InsertAnything/InsertAnything/tasks/direct/insertanything/assets/usb/usb_prongs.json
     cd ~/abhay_ws/InsertAnything-abhay && ~/abhay_ws/cable-assembly/sim/.venv/bin/python scripts/assets/build_usb_assets.py
     ```
6. **Smoke test** (about 1 min). Expect `ever_seated: 4`:
   ```bash
   ./scripts/run.sh python scripts/diagnostics/usb_probe.py --task InsertAnything-Usb-NoProngs-Direct-v0 \
       --mode insert --num_envs 4 --out logs/usb/smoke.json
   ```

GPU note: the dev PC has 10 GB. Two Isaac Sim processes at once fail with "no active physics scene", so
run evaluations serially.

---

## 3. What changed (files)

| File | What |
|---|---|
| `insertanything_compat.py` (new) | All simulator I/O for Lab 3.0: `.torch` views, (x,y,z,w)↔(w,x,y,z) at the boundary, `*_index` writes, friction and inertia setters. **Contact force uses `net_normal_forces_w`** (see §4.1). |
| `insertanything_math.py` (new) | Vendored (w,x,y,z) quaternion helpers. Lab 3.0 switched to (x,y,z,w), and the checkpoints observe (w,x,y,z). |
| `insertanything_env.py` | Lab 3.0 port. Also adds: generic arm + gripper joint ordering, a task-frame rotation (`cell_rotation_wxyz`), 2F-85 grasp handling, `held_asset_cfg.grasp_pos`, and the controller hooks `_payload_gravity_wrench` and `_fingertip_external_wrench`. |
| `insertanything_control.py` | OSC unchanged by default. New options: `bias_wrench`, and `ctrl.inertia_shaping` = `nominal` / `nominal_trans` / `nominal_fext` with `ctrl.desired_task_inertia`. |
| `insertanything_utils.py` | Delegates to compat. **Fix:** `get_closest_symmetry_transform` wraps roll and pitch to (−π, π]. Without it, ~1e-9 of roll wrapped to 2π and flipped the sign of the target quaternion. |
| `insertanything_tasks_cfg.py` | `RobotCfg` generalised beyond the Franka: joints, gripper type, fingertip body and offset, Jacobian mode, finger bodies, reset/null-space posture, controller overrides. Task fields: `fixed_asset_pos`, `cell_rotation_wxyz`, `table`. |
| `insertanything_env_cfg.py` | Scene declared via `InteractiveSceneCfg`, `PhysxCfg`, iiwa bench/pegboard scenery, new `CtrlCfg` fields (`inertia_shaping`, `desired_task_inertia`, `payload_gravity_comp`). |
| `insertanything_iiwa_cell_cfg.py` (new) | iiwa 14 + 2F-85 articulation (torque-driven arm), 2F-85 linkage width↔angle, IK seeds for the bench and the pegboard, hexagon/L iiwa tasks. |
| `insertanything_usb_cfg.py` (new) | USB-A tasks: `Usb-{NoProngs,Compliant}[-Iiwa14Bench\|-Iiwa14Pegboard]`. |
| `__init__.py` | Registers the iiwa and USB gym ids. |
| `scripts/rl_games/play.py` | Lab 3.0 player: torch `weights_only` allowlist for numpy, `--max_episodes`, `--summary_json`, `--set key=json` overrides, per-episode failure stats. |
| `scripts/run.sh` (new) | Launcher (§2). |
| `scripts/assets/extract_usb_prongs.py`, `build_usb_assets.py` (new) | USB asset pipeline (§4.3). |
| `scripts/diagnostics/` (new) | `inspect_env.py`, `eval_cell_matrix.sh`, `usb_probe.py` (response + scripted insertion), `compare_response.py`, `fit_apparent_inertia.py`. |
| `docs/usb_mating_plan.md` (new) | The development plan (gaps G1–G8, USB-I..IV ladder, real-robot path, schedule). |

---

## 4. Findings

### 4.1 Two port bugs, both fixed

**1. The contact force observation was dead.** In Lab 3.0, `ContactSensorData.net_forces_w` became
normal + friction. Lab 2.2's version was normal-only, and that is what the policies were trained on.
With a gripped peg, the jaws' friction cancels the socket's reaction, so the "gripper_peg" force read
about 0 N. The fix is `net_normal_forces_w` in `compat.net_contact_forces`.

Released checkpoints, Franka, seed 0, 128 envs × 3 episodes, `logs/eval/matrix_forcefix/`:

| Checkpoint | Before fix | **After fix** | Release (Isaac Sim 5.0) |
|---|---|---|---|
| L-III | 8.07% | **94.79%** | 96.35% |
| Hexagon-III | 90.10% | **94.27%** | 96.61% |

**2. Quaternion sign flip** via Euler wrap-around (the `utils` fix in §3). It sign-flipped the
`fingertip_quat_rel_fixed` observation in about half the envs.

### 4.2 iiwa cell, before the force fix

These are not yet re-run with the force fix; re-run them first.

| Hexagon-III | Franka | iiwa bench | iiwa pegboard |
|---|---|---|---|
| Before force fix | 90.10% | 74.22% | 77.60% |

The L-III iiwa numbers from before the fix are meaningless.

### 4.3 USB assets and the receptacle's prongs

- The CAD tongue is a flat 1.84 mm. It meets the plug's 1.95 mm cavity with 0.04 / 0.07 mm clearance
  (rigid CAD).
- The receptacle shell has **7 spring prongs** (detent tabs), measured from the STEP by
  `extract_usb_prongs.py`. Each prong's reach into the plug's path:

| Prong | Reach into the plug's path |
|---|---|
| top ±3 mm | 0.39 mm |
| top centre | 0.18 mm |
| bottom ±3 mm | 0.61 mm |
| sides | 0.15 mm |

- The user's point: the real tongue gap is over 1 mm because the prongs are compliant.
- **Two receptacles:**
  - `usb_socket_noprongs.usd`: cable-assembly's collider. The pocket is at the USB-A shell's inside
    (prongs cut away), and the tongue is relieved 0.15 mm per face.
  - `usb_socket_compliant.usd`: the same, plus the prongs as 51 convex pieces with a PhysX
    compliant-contact material. Defaults are k = 5000 N/m and c = 10 N·s/m;
    `build_usb_assets.py --prong_stiffness/--prong_damping` changes them.
- **Not yet verified:**
  - that the compliant material survives the env's `set_friction`, which rewrites shape materials;
  - the seating force the compliant prongs produce. Expected order: ~13 N total normal → ~4–5 N
    friction.
- **Plug:** the shell envelope is an SDF; the overmold is a convex-hull collider, which the gripper
  needs. Held at `grasp_pos = (0, 0.456, 25) mm` in the plug frame.
- **Socket yaw:** 180° ± 10°. The hand-down grasp lines the plug up at socket yaw 180°, the same
  convention as the L/hexagon holes.
- Success: seated within 0.55 mm of full depth, xy < 2.5 mm, yaw < 0.05 rad.
- **Insertion-force ceiling.** The action interface pushes at most `Kp × 0.02 m` = 2–6 N for Kp 100–300.
  That may be too little to seat a plug against compliant prongs. If USB-Compliant won't seat with the
  scripted probe, raise `pos_action_threshold` or the gains for this task.

Scripted insertion, Franka, no prongs, 4 envs: **4/4 seated**, 3.1 s. Seated push reads 2.3 N, which is
correct after the force fix.

### 4.4 Controller: why the iiwa differs from the Franka

**1. The action is a force command, not a position target.** The target is re-anchored to the current
pose every step, so `F = Kp·0.02·a − Kd·v` with `Kd = 2√Kp`. Free-space speed and contact force do not
depend on the arm. Only the transient (`Λ/Kd`) and the cross-axis coupling do.

**2. Payload gravity is the big one on the pegboard.** The arm's gravity is off in simulation, but the
held part's isn't. On the Franka, the part's weight (0.08–0.19 N) acts along task −z, the insertion
direction, which the policy saw in training. On the pegboard, the task frame is rotated 90°, so the
weight pushes sideways in task x. With no position stiffness, that is a steady ~4 mm/s drift the policy
never saw.

`ctrl.payload_gravity_comp=true` adds `m(g_ref − g_task)`, which puts the weight back on task −z. In the
response probe, z error vs the Franka dropped from 11.2 to 3.2 mm RMS, and the fitted residual force went
from task x to −z (as on the Franka). It is a no-op on the bench. It is implementable on the real arm,
where the payload is known.

**3. Inertia.** Fitted from trajectories (`fit_apparent_inertia.py`), the effective translational inertia
matches PhysX's Λ translational block, because roll and pitch are held stiffly:

| Arm | Eigenvalues (kg) | x–z coupling (kg) |
|---|---|---|
| Franka | 3.0 / 4.2 / 8.6 | −2.2 |
| iiwa pegboard | 3.3 / 4.6 / 10.6 | **+2.8** |
| iiwa bench | 3.8 / 5.2 / 10.7 | **+3.3** |

Magnitudes are within 10–25% of each other; the coupling has the opposite sign. The Franka's full
6×6 Λ at the USB start pose is in `logs/usb/franka_task_inertia.json`, and is also reproduced in §6.

**4. Inertia shaping, results so far** (response probe only):

| Mode | Result |
|---|---|
| `nominal` (full 6×6) | **worse**: x 33.6 mm RMS. It leaks the orientation-holding torque into force through the Franka's translation–rotation coupling terms. |
| `nominal_trans` (translation block) | Fixes the coupling sign: fitted eigenvalues 3.5 / 4.7 / 6.8, coupling −1.1. But x RMS is 19 mm, because the iiwa's x response is **asymmetric**: a +x step gives +24 mm, a −x step only −7 mm. That is a nonlinearity, not inertia. |
| `nominal_fext` | Untested. |

**Suspect for the asymmetry:** pegboard posture A5 = −151.5° against the ±170° limit, or some other joint
limit. Check this next by logging joint positions in the response probe.

**5. The decision metric is task success, not trajectory RMS.** Trajectory matching was turning into a
rabbit hole.

---

## 5. Controller study in flight when this was written

`scripts/diagnostics/ctrl_study.sh`, run on the old PC, writes to `logs/usb/ctrl2/` (git-ignored). Candidates
on the iiwa pegboard: `baseline`, `grav`, `gravtrans` (grav + `nominal_trans`). For each:

1. scripted USB insertion on both sockets, plus the Franka reference;
2. the hexagon-III policy on the hexagon task (pegboard, all three; bench, baseline and gravtrans);
3. the hexagon-III policy **zero-shot on USB-NoProngs**, Franka vs the three iiwa variants.

First result in: **Franka scripted insertion, no prongs, 128 envs: 107/128 seated**, median 3.13 s, peak
grip-plug force 5.1 N median.

If the rest don't come across, re-run the script (`OUT=... ./scripts/diagnostics/ctrl_study.sh`, about 2 h).
Single runs look like this, with `$L` = the 36 numbers in §6:

```bash
CTRL_grav="--set ctrl.payload_gravity_comp=true"
CTRL_gravtrans="$CTRL_grav --set ctrl.inertia_shaping=\"nominal_trans\" --set ctrl.desired_task_inertia=$L"
./scripts/run.sh python scripts/diagnostics/usb_probe.py --task InsertAnything-Usb-NoProngs-Iiwa14Pegboard-Direct-v0 \
    --mode insert --num_envs 128 --out logs/usb/ctrl2/ins_iiwapeg_NoProngs_grav.json $CTRL_grav
./scripts/run.sh python scripts/rl_games/play.py --task InsertAnything-HexagonHole-III-Iiwa14Pegboard-Direct-v0 \
    --num_envs 128 --seed 0 --max_episodes 2 --checkpoint <hexagon .pth> --summary_json logs/usb/ctrl2/hex_iiwapeg_grav.json $CTRL_grav
```

Pick the variant whose hexagon-on-iiwa and USB numbers come closest to the Franka's. Then make it the
default in `iiwa_robot_cfg()` through `RobotCfg.payload_gravity_comp`, `inertia_shaping` and
`desired_task_inertia`.

---

## 6. Reference data

The Franka's task-space inertia at the USB start pose (6×6, row-major, task frame, kg / kg·m²):

```
[8.01301,0.1524,-2.84546,0.0314,0.89897,0.00384,0.1524,4.34055,-0.21665,-0.47119,0.01503,0.00043,-2.84546,-0.21665,7.81239,0.05739,-0.01116,-2e-05,0.0314,-0.47119,0.05739,0.20658,0.01742,-0.00027,0.89897,0.01503,-0.01116,0.01742,0.22635,-0.00104,0.00384,0.00043,-2e-05,-0.00027,-0.00104,0.0667]
```

Gym ids:
- `InsertAnything-{HexagonHole,LHole}-III[-Iiwa14Bench|-Iiwa14Pegboard]-Direct-v0`
- `InsertAnything-Usb-{NoProngs,Compliant}[-Iiwa14Bench|-Iiwa14Pegboard]-Direct-v0`

The USB tasks use the hexagon agent config, which has the same observation layout.

---

## 7. Next steps, in order

1. **Finish the controller study (§5).** Also re-run hexagon and L on the iiwa bench and pegboard with
   the force fix: the §4.2 numbers predate it.
2. **Debug the iiwa's asymmetric x response.** Log joint positions in `usb_probe.py --mode response` and
   check for joint limits.
3. **Verify the compliant prongs:**
   - that `set_friction` keeps the compliant material;
   - insertion force vs depth with the scripted probe on `Usb-Compliant`;
   - whether the 2–6 N action ceiling (§4.3) can seat the plug.

   Tune k so that peak insertion force falls inside the USB 2.0 spec (≤ 35 N).
4. **Train USB policies**, per `docs/usb_mating_plan.md` §2.2, on the chosen controller and socket model:
   USB-I/II with oversized clearance first, then the CAD clearance with compliant prongs. The training
   script still needs porting to Lab 3.0, as play.py was.
5. **Real-robot path:** plan §§4–6. The real arm runs KUKA Cartesian impedance via FRI, so the sim
   controller and the real one must be matched by system ID (plan M8). Payload gravity compensation and
   external-force feedback both have real-arm counterparts: the tool-load setting, and τ_ext.

Open questions for the user:
- Prong stiffness: measure the real plug's insertion and extraction force if possible.
- Whether the real tongue clearance claim ("over 1 mm") refers to the slot with the prongs deflected.
  The rigid CAD gap is 0.11 mm total.
