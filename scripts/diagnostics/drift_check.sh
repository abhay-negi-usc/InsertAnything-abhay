#!/bin/bash
# Payload-gravity-compensation check: steady free-space drift under zero action, Franka vs iiwa pegboard/bench.
cd "$(dirname "$0")/../.."
OUT=${OUT:-logs/usb/drift}; mkdir -p $OUT
P="./scripts/diagnostics/gpu_run.sh ./scripts/run.sh python scripts/diagnostics/usb_probe.py --mode drift --num_envs 32"
r() { local n=$1; shift; "$@" --out $OUT/$n.json > $OUT/$n.log 2>&1; echo "$n $(grep -h '\[DRIFT\]' $OUT/$n.log)"; }
r franka $P --task InsertAnything-Usb-NoProngs-Direct-v0
r franka_grav $P --task InsertAnything-Usb-NoProngs-Direct-v0 --set ctrl.payload_gravity_comp=true
r iiwabench $P --task InsertAnything-Usb-NoProngs-Iiwa14Bench-Direct-v0
r iiwapeg $P --task InsertAnything-Usb-NoProngs-Iiwa14Pegboard-Direct-v0
r iiwapeg_grav $P --task InsertAnything-Usb-NoProngs-Iiwa14Pegboard-Direct-v0 --set ctrl.payload_gravity_comp=true
r hexpeg $P --task InsertAnything-HexagonHole-III-Iiwa14Pegboard-Direct-v0
r hexpeg_grav $P --task InsertAnything-HexagonHole-III-Iiwa14Pegboard-Direct-v0 --set ctrl.payload_gravity_comp=true
echo DRIFT_DONE
