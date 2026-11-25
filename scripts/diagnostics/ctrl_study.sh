#!/bin/bash
# iiwa controller study on the USB and hexagon tasks (docs/HANDOFF_usb_mating.md section 5). Serial: one sim at a time.
# L = the Franka task-space inertia at the USB start pose (6x6 row-major, task frame).
# NENV_PLAY (policy evals, 1 episode) and NENV_INS (scripted insertion) size the batch to the GPU: 2048 / 1024
# fit a 12 GB card with margin. The hexagon baselines on the iiwa are in eval_cell_matrix.sh, not here.
cd "$(dirname "$0")/../.."
OUT=${OUT:-logs/usb/ctrl2}; mkdir -p $OUT
NENV_PLAY=${NENV_PLAY:-128}; EP_PLAY=${EP_PLAY:-2}; NENV_INS=${NENV_INS:-128}
L='[8.01301,0.1524,-2.84546,0.0314,0.89897,0.00384,0.1524,4.34055,-0.21665,-0.47119,0.01503,0.00043,-2.84546,-0.21665,7.81239,0.05739,-0.01116,-2e-05,0.0314,-0.47119,0.05739,0.20658,0.01742,-0.00027,0.89897,0.01503,-0.01116,0.01742,0.22635,-0.00104,0.00384,0.00043,-2e-05,-0.00027,-0.00104,0.0667]'
CK=source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints/InsertAnything-HexagonHole-III-Direct-v0.pth
G=./scripts/diagnostics/gpu_run.sh
PROBE="$G ./scripts/run.sh python scripts/diagnostics/usb_probe.py"
PLAY="$G ./scripts/run.sh python scripts/rl_games/play.py --seed 0 --checkpoint $CK"
declare -A CTRL
CTRL[baseline]=""
CTRL[grav]="--set ctrl.payload_gravity_comp=true"
CTRL[gravtrans]="--set ctrl.payload_gravity_comp=true --set ctrl.inertia_shaping='\"nominal_trans\"' --set ctrl.desired_task_inertia=$L"
run() {
  local name=$1; shift
  if [ -s $OUT/$name.json ]; then echo "$name skipped (done)"; return; fi
  "$@" > $OUT/$name.log 2>&1
  echo "$name exit=$? $(grep -hE '\[RESULT\]|\[INSERT\]' $OUT/$name.log | tail -1 | cut -c1-230) $(grep -h '\[GPU\]' $OUT/$name.log)"
}
# 1. free-space response with joint logging (asymmetric x on the iiwa: joint limit or saturation?)
run resp_franka $PROBE --task InsertAnything-Usb-NoProngs-Direct-v0 --mode response --num_envs 32 --out $OUT/resp_franka.json
run resp_iiwabench $PROBE --task InsertAnything-Usb-NoProngs-Iiwa14Bench-Direct-v0 --mode response --num_envs 32 --out $OUT/resp_iiwabench.json
for c in baseline grav gravtrans; do
  run resp_iiwapeg_$c eval $PROBE --task InsertAnything-Usb-NoProngs-Iiwa14Pegboard-Direct-v0 --mode response --num_envs 32 --out $OUT/resp_iiwapeg_$c.json ${CTRL[$c]}
done
# 2. scripted insertion, both sockets
for s in NoProngs Compliant; do
  run ins_franka_$s $PROBE --task InsertAnything-Usb-$s-Direct-v0 --mode insert --num_envs $NENV_INS --out $OUT/ins_franka_$s.json
  for c in baseline grav gravtrans; do
    run ins_iiwapeg_${s}_$c eval $PROBE --task InsertAnything-Usb-$s-Iiwa14Pegboard-Direct-v0 --mode insert --num_envs $NENV_INS --out $OUT/ins_iiwapeg_${s}_$c.json ${CTRL[$c]}
  done
done
# 3. does the env's set_friction drop the prongs' compliant material? (same run, socket materials untouched)
run ins_franka_Compliant_keepmat $PROBE --task InsertAnything-Usb-Compliant-Direct-v0 --mode insert --num_envs $NENV_INS --keep_fixed_materials --out $OUT/ins_franka_Compliant_keepmat.json
# 4. hexagon policy on the hexagon task, with the controller extensions
for c in grav gravtrans; do
  run hex_iiwapeg_$c eval $PLAY --task InsertAnything-HexagonHole-III-Iiwa14Pegboard-Direct-v0 --num_envs $NENV_PLAY --max_episodes $EP_PLAY --summary_json $OUT/hex_iiwapeg_$c.json ${CTRL[$c]}
done
run hex_iiwabench_gravtrans eval $PLAY --task InsertAnything-HexagonHole-III-Iiwa14Bench-Direct-v0 --num_envs $NENV_PLAY --max_episodes $EP_PLAY --summary_json $OUT/hex_iiwabench_gravtrans.json ${CTRL[gravtrans]}
# 5. hexagon policy zero-shot on USB (no prongs)
run usbzs_franka $PLAY --task InsertAnything-Usb-NoProngs-Direct-v0 --num_envs $NENV_PLAY --max_episodes $EP_PLAY --summary_json $OUT/usbzs_franka.json
for c in baseline grav gravtrans; do
  run usbzs_iiwapeg_$c eval $PLAY --task InsertAnything-Usb-NoProngs-Iiwa14Pegboard-Direct-v0 --num_envs $NENV_PLAY --max_episodes $EP_PLAY --summary_json $OUT/usbzs_iiwapeg_$c.json ${CTRL[$c]}
done
echo DONE
