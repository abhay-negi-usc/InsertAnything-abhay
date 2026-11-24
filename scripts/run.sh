#!/usr/bin/env bash
# Run a script with the Isaac Lab 3.0 / Isaac Sim 6.2 environment.
#
# This machine has no Isaac Lab 2.2 install; the only Isaac Lab is the cable-assembly sim's uv venv
# (Python 3.12, CUDA 13). InsertAnything is put on PYTHONPATH instead of being installed into that
# venv, and rl_games (absent from it) comes from .deps/ (uv pip install --target .deps rl-games==1.6.1
# gym==0.23.1 gym-notices tensorboardX setproctitle --no-deps). ~/.bashrc's ROS paths are dropped
# because a Python 3.12 / CUDA 13 process must not load them.
#
#   ./scripts/run.sh python scripts/rl_games/play.py --task ... --checkpoint ...
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${ISAACLAB_VENV:-$HOME/abhay_ws/cable-assembly/sim/.venv}"
unset PYTHONPATH LD_LIBRARY_PATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH ROS_PACKAGE_PATH \
    ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION VIRTUAL_ENV
export PYTHONPATH="$REPO/source/InsertAnything:$REPO/.deps"
export OMNI_KIT_ACCEPT_EULA=YES
cd "$REPO"
cmd="$1"; shift
if [ "$cmd" = "python" ]; then cmd="$VENV/bin/python"; fi
exec "$cmd" "$@"
