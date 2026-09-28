#!/usr/bin/env bash
set -euo pipefail
PI05_20HZ_SLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$PI05_20HZ_SLOW_DIR/../fr3_wuji/env.sh"
export PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cuda
exec taskset -c "$PI05_CPUSET" nice -n 10 "$PI05_ROOT/.venv/bin/python" -B \
  -m experiments.weight_motion_eval.oneshot.policy_server \
  --checkpoint "$PI05_ROOT/checkpoints/tomato_lora_0918_20hz/19999" --port 8001 "$@"
