#!/bin/bash
# Can the compliant receptacle be seated? Franka scripted insertion vs prong stiffness and push ceiling.
# The sweep sockets come from build_usb_assets.py --compliant_only --suffix _k<K> --prong_stiffness <K>.
cd "$(dirname "$0")/../.."
OUT=${OUT:-logs/usb/prongs}; N=${N:-256}; mkdir -p $OUT
A=$PWD/source/InsertAnything/InsertAnything/tasks/direct/insertanything/assets/usb
P="./scripts/diagnostics/gpu_run.sh ./scripts/run.sh python scripts/diagnostics/usb_probe.py --mode insert --num_envs $N --keep_fixed_materials --task InsertAnything-Usb-Compliant-Direct-v0"
r() { local n=$1; shift; [ -s $OUT/$n.json ] && { echo "$n skipped (done)"; return; }
      "$@" --out $OUT/$n.json > $OUT/$n.log 2>&1; echo "$n exit=$? $(grep -h '\[INSERT\]' $OUT/$n.log | cut -c1-200) $(grep -h '\[GPU\]' $OUT/$n.log)"; }
for k in 1000 300; do
  r k$k $P --set "scene.fixed_asset.spawn.usd_path=\"$A/usb_socket_compliant_k$k.usd\""
done
# z push ceiling 5x (Kp * 0.1 m instead of Kp * 0.02 m), default k = 5000
r k5000_push5x $P --set "ctrl.pos_action_threshold=[0.02,0.02,0.1]"
echo SWEEP_DONE
