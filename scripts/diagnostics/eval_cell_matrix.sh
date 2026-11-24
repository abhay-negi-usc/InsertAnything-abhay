#!/usr/bin/env bash
# Evaluate the released checkpoints on the Franka tasks and on the iiwa 14 cell variants
# (release protocol: 128 envs, seed 0, 3 episodes = 384 trials each).
set -uo pipefail
cd "$(dirname "$0")/../.."
CK=source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints
OUT=${OUT:-logs/eval/matrix}
EPISODES=${EPISODES:-3}
mkdir -p "$OUT"
for shape in HexagonHole-III LHole-III; do
  for cell in "" "-Iiwa14Bench" "-Iiwa14Pegboard"; do
    task="InsertAnything-${shape}${cell}-Direct-v0"
    name="${shape}${cell:-"-Franka"}"
    ./scripts/run.sh python scripts/rl_games/play.py --task "$task" --num_envs 128 --seed 0 \
      --max_episodes "$EPISODES" --checkpoint "$CK/InsertAnything-${shape}-Direct-v0.pth" \
      --summary_json "$OUT/$name.json" > "$OUT/$name.log" 2>&1
    echo "$name exit=$? $(grep -h '\[RESULT\]' "$OUT/$name.log" | tail -1)"
  done
done
echo MATRIX_DONE
