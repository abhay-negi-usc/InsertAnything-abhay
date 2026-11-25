#!/usr/bin/env bash
# Run a command, sampling GPU memory once a second; append "peak_mem_mib=<n> wall_s=<n>" to stderr.
#   ./scripts/diagnostics/gpu_run.sh ./scripts/run.sh python ...
mem=$(mktemp)
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 > "$mem" &
smi=$!
t0=$(date +%s)
"$@"; rc=$?
kill $smi 2>/dev/null; wait $smi 2>/dev/null
echo "[GPU] peak_mem_mib=$(sort -n "$mem" | tail -1) wall_s=$(( $(date +%s) - t0 )) exit=$rc" >&2
rm -f "$mem"
exit $rc
