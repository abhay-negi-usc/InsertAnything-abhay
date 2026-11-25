# Handoff: `feature/usb-mating`

Branch goal: develop USB-A mating on the bench's real KUKA iiwa 14 + Robotiq 2F-85 by replicating
InsertAnything's methodology. The plan is `docs/usb_mating_plan.md`. This document records where the
work stands, how to reproduce it on a new machine, and what to do next.

Status date: 2026-11-24 (second session: controller study finished, compliant-prong fix).

---

## 1. State in one paragraph

InsertAnything (Isaac Lab 2.2 / Isaac Sim 5.0) is ported to **Isaac Lab 3.0 / Isaac Sim 6.2**. The released
Franka checkpoints reproduce the paper to within ~1 point. The repo also has:

- an iiwa 14 + 2F-85 cell matching cable-assembly's bench (bench-top and vertical-pegboard variants);
- a **USB-A task** with two receptacle models, the shell's spring prongs removed and the prongs as
  compliant contacts;
- a privileged scripted insertion probe and a controller probe (response, drift, force vs depth);
- a Lab 3.0 `train.py`, smoke-tested.

The controller study is done (§5): **payload gravity compensation is now the iiwa default; inertia shaping
is rejected.** The compliant receptacle had a bug that made its prongs rigid (§4.3), now fixed, and the
USB-Compliant task gets a higher push ceiling so it can seat. **The released hexagon-III policy inserts USB
zero-shot at 95.7 % (Franka, no prongs).** No USB policy has been trained yet; everything is ready to start
(§7).

---

## 2. Setting up on a new machine

The port runs on the cable-assembly sim's uv venv. That venv provides Isaac Lab 3.0.0, Isaac Sim 6.2,
torch 2.11+cu130 and Python 3.12.

1. **Clone both repos side by side:** `~/abhay_ws/InsertAnything-abhay` on branch `feature/usb-mating`,
   and `~/abhay_ws/cable-assembly` on branch **`RL-hardware-testing`** (its `main` is an empty initial
   commit). Build its venv per `sim/README.md`:
   ```bash
   cd ~/abhay_ws/cable-assembly/sim && env -u PYTHONPATH -u LD_LIBRARY_PATH uv sync --extra importers --extra isaacsim
   ```
   About 12 GB to download, 25 GB on disk, NVIDIA driver ≥ 580 (CUDA 13). 580.178 works.
   - Elsewhere, set `ISAACLAB_VENV=<path to that venv>`.
2. **rl_games and gym are not in that venv.** Install them into this repo's `.deps/`, which is git-ignored:
   ```bash
   uv pip install --python ~/abhay_ws/cable-assembly/sim/.venv/bin/python --target .deps --no-deps \
       rl-games==1.6.1 gym==0.23.1 gym-notices tensorboardX setproctitle
   ```
3. **Run everything through `scripts/run.sh`.** It drops the ROS env vars, puts the repo and `.deps` on
   `PYTHONPATH`, and accepts the EULA. Lab 3.0 runs headless by default; there is no `--headless` flag.
   ```bash
   ./scripts/run.sh python scripts/rl_games/play.py --task ... --checkpoint ...
   ```
4. **Checkpoints are git-ignored.** Download them from Hugging Face into
   `source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints/`. Commands and SHA256s
   are in `checkpoints/README.md`. The repo is public; with an expired HF token, fetch anonymously:
   `curl -L -o <dir>/<file> https://huggingface.co/tOMRmA/InsertAnything-checkpoints/resolve/main/<file>`.
5. **CAD and robot assets are in this repo** (`source/.../insertanything/assets/`). cable-assembly
   deliberately does not version them, because McMaster and Robotiq CAD are theirs to distribute. Keep that
   in mind if this repo is public.
   - `assets/iiwa_cell/`: `iiwa14_2f85{,_bench}.usda`, `iiwa14/usd` (+ urdf, LICENSE), `robotiq_2f85/`,
     `robotiq_2f85_v3/v3_visuals.usdc`, `tool/tool_meshes.usdc`. These are copied from cable-assembly's
     `sim/src/cable_assembly_sim/assets/data` with the layout kept: the .usda files use relative
     references, all of which resolve inside the repo except Kit's built-in `OmniPBR.mdl`.
     `CABLE_ASSEMBLY_SIM_ASSETS` points at another copy.
   - `assets/usb/cad/`: the McMaster STEPs (4974T135 cord, 4574N11 adapter), and cable-assembly's
     `usb_plug.usd` / `usb_socket.usd` from its `sim/scripts/build_usb.py`, which are the colliders.
   - `assets/usb/`: the built task assets (`usb_plug.usd`, `usb_socket_{noprongs,compliant}.usd`,
     `usb_prongs.json`). Rebuilding from `cad/` reproduces them byte for byte:
     ```bash
     cd ~/abhay_ws/cable-assembly/sim && ./run.sh --extra cad python \
         ~/abhay_ws/InsertAnything-abhay/scripts/assets/extract_usb_prongs.py \
         ~/abhay_ws/InsertAnything-abhay/source/InsertAnything/InsertAnything/tasks/direct/insertanything/assets/usb/cad/4574N11_panel_mount_usb_adapter.step \
         ~/abhay_ws/InsertAnything-abhay/source/InsertAnything/InsertAnything/tasks/direct/insertanything/assets/usb/usb_prongs.json
     cd ~/abhay_ws/InsertAnything-abhay && ~/abhay_ws/cable-assembly/sim/.venv/bin/python scripts/assets/build_usb_assets.py
     ```
     The prong extraction needs gmsh and scipy (cable-assembly's `cad` extra). The USD build needs only
     `pxr`, numpy and scipy. Extra receptacles for a stiffness sweep, without touching the committed ones:
     `build_usb_assets.py --compliant_only --suffix _k1000 --prong_stiffness 1000`. These are untracked;
     don't commit them.
6. **Smoke test** (about 1 min). Expect `ever_seated: 4`:
   ```bash
   ./scripts/run.sh python scripts/diagnostics/usb_probe.py --task InsertAnything-Usb-NoProngs-Direct-v0 \
       --mode insert --num_envs 4 --out logs/usb/smoke.json
   ```

### GPU sizing (12 GB RTX 3060)

- Every Isaac process carries **~4.5 GB fixed** (Kit + PhysX), whatever the env count. Measured peaks:
  hexagon/L eval at 2048 envs 6.4–6.7 GB; USB eval at 2048 envs 7.4–7.8 GB; scripted USB insertion at
  1024 envs 5.6–6.7 GB; USB training at 1024 envs 6.6 GB.
- In steady state the GPU is compute-bound (92–95 % SM). **Scale envs per process, not processes:** 1536 envs
  gave 4x the trials of 384 for 1.65x the wall time. Two large processes do not fit with margin (a 1024-env
  insertion plus a small probe peaked at 10.5–11.1 GB). Small jobs (32-env probes) run fine alongside one
  large one; the old "no active physics scene" failure did not reproduce.
- `scripts/diagnostics/gpu_run.sh <cmd>` logs `[GPU] peak_mem_mib=… wall_s=…` for any run.
- Wall times: hexagon/L eval at 2048 envs × 1 episode 5–10 min; USB eval at 2048 envs 14–27 min; scripted
  USB insertion at 1024 envs 12–20 min. USB is 3–6x slower than hexagon (SDF contact; the receptacle's SDF
  is 512³).
- Waiting on a job in a shell loop: do not use `pgrep -f <pattern>` from a command line that itself contains
  the pattern. It matches itself and waits forever (this cost 40 GPU-minutes once). Chain jobs with `;` in one
  background command instead.

---

## 3. What changed (files)

First session:

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

Second session:

| File | What |
|---|---|
| `insertanything_tasks_cfg.py` | `FixedAssetCfg.keep_usd_materials` (skip all friction writes on that asset); task field `pos_action_threshold` (overrides `CtrlCfg.pos_action_threshold`). |
| `insertanything_env.py` | Honours `keep_usd_materials` at startup and in friction DR. |
| `insertanything_env_cfg.py` | Applies the task's `pos_action_threshold`. |
| `insertanything_usb_cfg.py` | `UsbSocketCompliant.keep_usd_materials = True`; `Usb-Compliant*` push ceiling `pos_action_threshold = [0.02, 0.02, 0.1]` (§4.3). |
| `insertanything_iiwa_cell_cfg.py` | `iiwa_robot_cfg()` sets `payload_gravity_comp=True` (§5). |
| `scripts/rl_games/train.py` | Ported to Lab 3.0 in play.py's style: `--set` env overrides, `--agent_set` agent overrides under `params.config` (dotted, e.g. `central_value_config.minibatch_size=4096`), `--run_name`; writes `params/run.json`. |
| `scripts/diagnostics/usb_probe.py` | `--mode response` logs arm joint q / q̇ / τ and prints per-step `[RESPONSE]` lines (limit margins, τ/effort limit). `--mode drift`: zero action in free space; steady drift and the force behind it. `--mode insert` adds `force_vs_height_mm` (grip force binned by height above seated). `--keep_fixed_materials` (A/B for §4.3). |
| `scripts/diagnostics/ctrl_study.sh` | `NENV_PLAY` / `EP_PLAY` / `NENV_INS`; response probes and the material A/B added; resumable (skips runs whose JSON exists). |
| `scripts/diagnostics/eval_cell_matrix.sh` | `NENV` / `EPISODES`. |
| `scripts/diagnostics/{gpu_run,drift_check,prong_sweep}.sh` (new) | Peak-VRAM wrapper; drift check (§4.4); prong stiffness / push-ceiling sweep (§4.3). |
| `scripts/assets/build_usb_assets.py` | `--compliant_only --suffix` for sweep receptacles. |

---

## 4. Findings

### 4.1 Two port bugs, both fixed

**1. The contact force observation was dead.** In Lab 3.0, `ContactSensorData.net_forces_w` became
normal + friction. Lab 2.2's version was normal-only, and that is what the policies were trained on.
With a gripped peg, the jaws' friction cancels the socket's reaction, so the "gripper_peg" force read
about 0 N. The fix is `net_normal_forces_w` in `compat.net_contact_forces`.

**2. Quaternion sign flip** via Euler wrap-around (the `utils` fix in §3). It sign-flipped the
`fingertip_quat_rel_fixed` observation in about half the envs.

### 4.2 Released checkpoints across cells (force fix in)

Seed 0, 2048 envs × 1 episode = 2048 trials each (`logs/eval/matrix_forcefix_2048/`). Release numbers are
Isaac Sim 5.0, 384 trials.

| Checkpoint | Franka | Release | iiwa bench | iiwa pegboard | pegboard, before force fix |
|---|---|---|---|---|---|
| Hexagon-III | 95.46 % | 96.61 % | 83.15 % | 86.28 % | 77.60 % |
| L-III | 95.95 % | 96.35 % | 74.41 % | 80.22 % | (meaningless) |

The iiwa costs 9–21 points zero-shot. With payload gravity compensation (now default), hexagon-III on the
pegboard is 88.53 % (§5).

### 4.3 USB assets, the receptacle's prongs, and seating

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

- **Two receptacles:**
  - `usb_socket_noprongs.usd`: cable-assembly's collider. The pocket is at the USB-A shell's inside
    (prongs cut away), and the tongue is relieved 0.15 mm per face.
  - `usb_socket_compliant.usd`: the same, plus the prongs as 51 convex pieces with a PhysX
    compliant-contact material. Defaults are k = 5000 N/m and c = 10 N·s/m.
- **Plug:** the shell envelope is an SDF; the overmold is a convex-hull collider, which the gripper
  needs. Held at `grasp_pos = (0, 0.456, 25) mm` in the plug frame. 8.3 g.
- **Socket yaw:** 180° ± 10°. Success: seated within 0.55 mm of full depth, xy < 2.5 mm, yaw < 0.05 rad.

**Bug (fixed): the env's friction write made the prongs rigid.** `compat.set_friction` (PhysX tensor
`set_material_properties`) replaces each shape's material and drops the compliant-contact attributes.
Scripted Franka insertion on Usb-Compliant: **0/1024** with the write, **469/1024** without it (same run,
`--keep_fixed_materials`). Fix: `FixedAssetCfg.keep_usd_materials`, set for the compliant receptacle. The
USD's friction (0.3) equals the cfg value, so nothing is lost, except that friction DR no longer touches the
socket (the plug's friction is still randomized).

**Seating force.** Scripted Franka insertion, force vs height above seated (`force_vs_height_mm`):
- No prongs: the plug slides home at ~0.6 N. The failures (13 %) all jam at the mouth (10–11.5 mm) at the
  push ceiling.
- Compliant, k = 5000: ~1.2 N drag below 7 mm. Pile-ups at 6.9–7.4 mm (first engagement) and 2–3 mm short of
  seated (the 0.61 mm bottom prongs). Both are at the action interface's push ceiling, **Kp × 0.02 m ≈ 2.2 N**.

Sweep (`scripts/diagnostics/prong_sweep.sh`, Franka scripted, 256 envs, `logs/usb/prongs/`):

| Receptacle / ceiling | Seated (ever) | Seated at end | Peak grip force median / p90 |
|---|---|---|---|
| no prongs, default (1024 envs) | 87.0 % | 87.0 % | 5.2 / 6.7 N |
| k = 5000, default ceiling (1024 envs) | 45.8 % | 45.8 % | 2.9 / 5.7 N |
| k = 1000, default ceiling | 88.3 % | 84.4 % | 3.0 / 3.8 N |
| k = 300, default ceiling | 88.3 % | 84.8 % | 3.4 / 3.9 N |
| **k = 5000, z ceiling 5x** | **89.5 %** | **89.5 %** | 5.5 / 8.4 N |

Soft prongs seat as often but ~4 % pop back out (too little retention). **Chosen: keep k = 5000 and raise the
z push ceiling** (`InsertAnythingUsbCompliant.pos_action_threshold = [0.02, 0.02, 0.1]`). Near the seat the
target is clipped by `pos_action_bounds` (5 cm), so the real gain is ~4x. Peak forces stay far inside the USB
2.0 limit of 35 N. Verified with the new defaults and no overrides: see §5.3.

Caveat: the raised ceiling also makes free-space z motion faster. A policy trained on Usb-Compliant sees
that; the hexagon-III checkpoint evaluated zero-shot on Usb-Compliant does not match its training.

### 4.4 Controller: why the iiwa differs from the Franka

**1. The action is a force command, not a position target.** The target is re-anchored to the current
pose every step, so `F = Kp·0.02·a − Kd·v` with `Kd = 2√Kp`. Free-space speed and contact force do not
depend on the arm. Only the transient (`Λ/Kd`) and the cross-axis coupling do. With no position
stiffness, any constant force drifts at F / Kd.

**2. Payload gravity: compensation verified.** `usb_probe.py --mode drift` (zero action, free space), USB
plug, Kd = 20:

| Cell | Drift (mm/s, task x / y / z) | Implied force (N) |
|---|---|---|
| Franka | −0.5 / −0.1 / −3.9 | z −0.077 (= m g = 0.081) |
| iiwa bench | +0.6 / 0.0 / −3.5 | z −0.070 |
| iiwa pegboard | **−3.8** / −0.2 / −0.4 | **x** −0.076 |
| iiwa pegboard + grav comp | +0.4 / −0.1 / −3.6 | z −0.072 |

The hexagon peg (19 g) on the pegboard drifts −9.2 mm/s in x; with compensation, −8.2 mm/s in z. So the
compensation puts the weight exactly where the Franka sees it.

**3. Inertia.** Fitted from trajectories (`fit_apparent_inertia.py`), the effective translational inertia
matches PhysX's Λ translational block:

| Arm | Eigenvalues (kg) | x–z coupling (kg) |
|---|---|---|
| Franka | 3.0 / 4.2 / 8.6 | −2.2 |
| iiwa pegboard | 3.3 / 4.6 / 10.6 | **+2.8** |
| iiwa bench | 3.8 / 5.2 / 10.7 | **+3.3** |

**4. The "asymmetric x response" is not a joint limit or saturation.** With joint logging
(`--mode response`), on every controller every arm joint stays ≥ 12° from its limit (tightest: pegboard A4
and A5, 12–16°), and the commanded torque stays under 1.2 % of the effort limit. Split into symmetric
(step) and common (bias) parts, the ±x steps give a step of 16.7–18.0 mm on the iiwa against 17.8 mm on the
Franka. The rest is motion carried from the previous segment and x–z coupling, not a constant force: the
drift test above shows no residual x force. Not worth chasing further; decide on task success.

**5. The decision metric is task success, not trajectory RMS.**

---

## 5. Controller study (done)

`scripts/diagnostics/ctrl_study.sh` with `NENV_PLAY=2048 EP_PLAY=1 NENV_INS=1024`, results in
`logs/usb/ctrl2/` (git-ignored). iiwa pegboard unless noted. `grav` = `ctrl.payload_gravity_comp=true`;
`gravtrans` = grav + `inertia_shaping="nominal_trans"` with the Franka Λ (§6).

### 5.1 Results

| Test | Franka | baseline | **grav** | gravtrans |
|---|---|---|---|---|
| Scripted USB insertion, no prongs (1024) | 87.0 % | 74.7 % | **83.5 %** | 24.8 % |
| Hexagon-III policy on hexagon (2048) | 95.5 % | 86.3 % | **88.5 %** | 31.4 % |
| Hexagon-III policy on hexagon, iiwa bench (2048) | — | 83.2 % | (no-op) | 19.6 % |
| Hexagon-III policy zero-shot on USB, no prongs (2048) | **95.7 %** | 83.6 % | 82.5 % | 12.7 % |

Scripted USB time to seat: Franka 3.1 s, iiwa 3.9 s.

### 5.2 Decision

**Payload gravity compensation is the iiwa default** (`iiwa_robot_cfg(... payload_gravity_comp=True)`). It
is physically right (§4.4.2), helps or is neutral everywhere, and the real arm has it (the tool-load
setting). `--set ctrl.payload_gravity_comp=false` turns it off.

**Inertia shaping is rejected.** `nominal_trans` matches free-space trajectories better, but in contact it
rescales the response by Λ·Λ_d⁻¹. Plugs stall at the lip with 0.8 mm lateral error (vs 0.13 mm), and every
policy and scripted metric collapses. It stays in the code as an option, off by default.

### 5.3 New defaults verified

Scripted insertion with no `--set` and no flags (`logs/usb/defaults/`, 512 envs):

| Task | Seated (ever) | Seated at end | Time to seat | Peak grip force median / p90 |
|---|---|---|---|---|
| Usb-Compliant (Franka) | 88.7 % | 88.1 % | 3.3 s | 5.5 / 8.5 N |
| Usb-Compliant-Iiwa14Pegboard (grav default) | 77.0 % | **69.7 %** | 4.0 s | 7.4 / 9.8 N |

The Franka matches the sweep (§4.3). On the iiwa, 37 of 512 envs (7 %) seat and then lose the success
condition before the end; the Franka almost never does. Not yet diagnosed. The candidates are the plug
backing out under the stiffer, faster iiwa push, or yaw/xy drifting past the threshold after seating.
Re-run with `--trace` on a failing env, or log seated envs' final xy / yaw / depth.

### 5.4 Zero-shot transfer is not the bottleneck

The hexagon-III policy, never trained on USB, seats the no-prong USB plug **95.7 %** of the time on the
Franka. That beats the scripted oracle (87.0 %), whose failures are lip jams it cannot wiggle out of.
Geometry is not the hard part. The iiwa gap (~12 points) is the dynamics gap of §4.2.

---

## 6. Reference data

The Franka's task-space inertia at the USB start pose (6×6, row-major, task frame, kg / kg·m²):

```
[8.01301,0.1524,-2.84546,0.0314,0.89897,0.00384,0.1524,4.34055,-0.21665,-0.47119,0.01503,0.00043,-2.84546,-0.21665,7.81239,0.05739,-0.01116,-2e-05,0.0314,-0.47119,0.05739,0.20658,0.01742,-0.00027,0.89897,0.01503,-0.01116,0.01742,0.22635,-0.00104,0.00384,0.00043,-2e-05,-0.00027,-0.00104,0.0667]
```

Gym ids:
- `InsertAnything-{HexagonHole,LHole}-III[-Iiwa14Bench|-Iiwa14Pegboard]-Direct-v0`
- `InsertAnything-Usb-{NoProngs,Compliant}[-Iiwa14Bench|-Iiwa14Pegboard]-Direct-v0`

The USB tasks use the hexagon agent config, which has the same observation layout. Its agent `name` is
`InsertAnything` for every task, so runs land in `logs/rl_games/InsertAnything/<run_name>/`: always pass
`--run_name`.

**Training throughput** (`train.py`, Usb-NoProngs, Franka, 1024 envs, solo on the 3060): ~1700 env-steps/s
of simulation, **~970 fps end to end**. 4 epochs (393k frames) in 540 s, 6.6 GB peak. The PPO update is the
gap. `central_value_config.minibatch_size` is still 512 (256 critic minibatches per epoch); raise it with
`--agent_set central_value_config.minibatch_size=4096`. At ~1000 fps, 13M frames (the release budget,
800 epochs × 128 envs × 128 steps) is ~3.7 h; at 1024 envs that is ~100 epochs.

---

## 7. Next steps, in order

1. **Train USB policies** per `docs/usb_mating_plan.md` §2.2. Start from the hexagon-III checkpoint (it is
   already at 95.7 % zero-shot, so fine-tuning should be quick), on the iiwa pegboard with the new defaults:
   ```bash
   ./scripts/run.sh python scripts/rl_games/train.py --task InsertAnything-Usb-NoProngs-Iiwa14Pegboard-Direct-v0 \
       --num_envs 1024 --seed 0 --max_iterations 100 --run_name usb_np_iiwapeg_ft \
       --checkpoint source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints/InsertAnything-HexagonHole-III-Direct-v0.pth \
       --agent_set minibatch_size=4096 --agent_set central_value_config.minibatch_size=4096
   ```
   Then Usb-Compliant (the raised push ceiling changes the action scale in z, so expect a re-adaptation),
   then the USB-I..IV clearance ladder. A from-scratch run on the same budget is the control.
   Not yet checked: whether resuming from the checkpoint restores the optimizer and epoch counter as
   intended (`--checkpoint` sets `load_checkpoint`, as Lab's own trainer does).
2. **Close the iiwa dynamics gap for the released policies** (hexagon 88.5 vs 95.5 %). Inertia shaping is out.
   Candidates: train on the iiwa directly (step 1 does this for USB), or domain-randomize the arm (Λ
   coupling sign, gains) when training, which the plan's DR already partly covers.
3. **Diagnose the iiwa's seat-then-lose cases on Usb-Compliant** (§5.3, 7 % of envs).
4. **Speed up USB simulation** before long training sweeps. The receptacle SDF is 512³ and the plug's 256³.
   Try 256/128 and check scripted seating and the force profile are unchanged. The iiwa articulation uses
   192 position iterations.
5. **Prongs:** measure the real plug's insertion and extraction force if possible, and tune k so that the
   sim's peak insertion force matches. The current 5.5 N median / 8.4 N p90 at k = 5000 is a guess that
   satisfies the spec, not a measurement.
6. **Real-robot path:** plan §§4–6. The real arm runs KUKA Cartesian impedance via FRI, so the sim
   controller and the real one must be matched by system ID (plan M8). Payload gravity compensation and
   external-force feedback both have real-arm counterparts: the tool-load setting, and τ_ext.

Open questions for the user:
- Prong stiffness: measure the real plug's insertion and extraction force if possible.
- Whether the real tongue clearance claim ("over 1 mm") refers to the slot with the prongs deflected.
  The rigid CAD gap is 0.11 mm total.
- `usb_socket_compliant_k{300,1000}.usd` are untracked sweep artifacts in `assets/usb/`; delete them or add a
  gitignore rule.
