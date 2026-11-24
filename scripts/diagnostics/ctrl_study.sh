#!/bin/bash
# iiwa controller study on the USB and hexagon tasks (docs/HANDOFF_usb_mating.md section 5). Serial: one sim at a time.
# L = the Franka task-space inertia at the USB start pose (6x6 row-major, task frame).
cd "$(dirname "$0")/../.."
OUT=${OUT:-logs/usb/ctrl2}; mkdir -p $OUT
L='[8.01301,0.1524,-2.84546,0.0314,0.89897,0.00384,0.1524,4.34055,-0.21665,-0.47119,0.01503,0.00043,-2.84546,-0.21665,7.81239,0.05739,-0.01116,-2e-05,0.0314,-0.47119,0.05739,0.20658,0.01742,-0.00027,0.89897,0.01503,-0.01116,0.01742,0.22635,-0.00104,0.00384,0.00043,-2e-05,-0.00027,-0.00104,0.0667]'
CK=source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints/InsertAnything-HexagonHole-III-Direct-v0.pth
declare -A CTRL
CTRL[baseline]=""
CTRL[grav]="--set ctrl.payload_gravity_comp=true"
CTRL[gravtrans]="--set ctrl.payload_gravity_comp=true --set ctrl.inertia_shaping='\"nominal_trans\"' --set ctrl.desired_task_inertia=$L"
run() { local name=$1; shift; "$@" > $OUT/$name.log 2>&1; echo "$name exit=$? $(grep -hE '\[RESULT\]|\[INSERT\]' $OUT/$name.log | tail -1 | cut -c1-230)"; }
# scripted insertion, both sockets
for s in NoProngs Compliant; do
  run ins_franka_$s ./scripts/run.sh python scripts/diagnostics/usb_probe.py --task InsertAnything-Usb-$s-Direct-v0 --mode insert --num_envs 128 --out $OUT/ins_franka_$s.json
  for c in baseline grav gravtrans; do
    run ins_iiwapeg_${s}_$c eval ./scripts/run.sh python scripts/diagnostics/usb_probe.py --task InsertAnything-Usb-$s-Iiwa14Pegboard-Direct-v0 --mode insert --num_envs 128 --out $OUT/ins_iiwapeg_${s}_$c.json ${CTRL[$c]}
  done
done
# hexagon policy on the hexagon task
for c in baseline grav gravtrans; do
  run hex_iiwapeg_$c eval ./scripts/run.sh python scripts/rl_games/play.py --task InsertAnything-HexagonHole-III-Iiwa14Pegboard-Direct-v0 --num_envs 128 --seed 0 --max_episodes 2 --checkpoint $CK --summary_json $OUT/hex_iiwapeg_$c.json ${CTRL[$c]}
done
for c in baseline gravtrans; do
  run hex_iiwabench_$c eval ./scripts/run.sh python scripts/rl_games/play.py --task InsertAnything-HexagonHole-III-Iiwa14Bench-Direct-v0 --num_envs 128 --seed 0 --max_episodes 2 --checkpoint $CK --summary_json $OUT/hex_iiwabench_$c.json ${CTRL[$c]}
done
# hexagon policy zero-shot on USB (no prongs)
run usbzs_franka ./scripts/run.sh python scripts/rl_games/play.py --task InsertAnything-Usb-NoProngs-Direct-v0 --num_envs 128 --seed 0 --max_episodes 2 --checkpoint $CK --summary_json $OUT/usbzs_franka.json
for c in baseline grav gravtrans; do
  run usbzs_iiwapeg_$c eval ./scripts/run.sh python scripts/rl_games/play.py --task InsertAnything-Usb-NoProngs-Iiwa14Pegboard-Direct-v0 --num_envs 128 --seed 0 --max_episodes 2 --checkpoint $CK --summary_json $OUT/usbzs_iiwapeg_$c.json ${CTRL[$c]}
done
echo DONE
