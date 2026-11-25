#!/usr/bin/env bash
# Evaluate the released checkpoints on the Franka tasks and on the iiwa 14 cell variants
# (release protocol: 128 envs, seed 0, 3 episodes = 384 trials each; NENV / EPISODES override, e.g.
# NENV=2048 EPISODES=1 on a 12 GB GPU: one batch, ~7 GB, 5x the trials in ~2x the time).
set -uo pipefail
cd "$(dirname "$0")/../.."
CK=source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints
OUT=${OUT:-logs/eval/matrix}
EPISODES=${EPISODES:-3}
NENV=${NENV:-128}
mkdir -p "$OUT"
for shape in HexagonHole-III LHole-III; do
  for cell in "" "-Iiwa14Bench" "-Iiwa14Pegboard"; do
    task="InsertAnything-${shape}${cell}-Direct-v0"
    name="${shape}${cell:-"-Franka"}"
    ./scripts/diagnostics/gpu_run.sh ./scripts/run.sh python scripts/rl_games/play.py --task "$task" --num_envs "$NENV" --seed 0 \
      --max_episodes "$EPISODES" --checkpoint "$CK/InsertAnything-${shape}-Direct-v0.pth" \
      --summary_json "$OUT/$name.json" > "$OUT/$name.log" 2>&1
    echo "$name exit=$? $(grep -hE '\[RESULT\]|\[GPU\]' "$OUT/$name.log" | tail -2 | tr '\n' ' ')"
  done
done
echo MATRIX_DONE
