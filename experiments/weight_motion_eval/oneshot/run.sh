#!/usr/bin/env bash
set -euo pipefail
PI05_ONESHOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$PI05_ONESHOT_DIR/../../../deploy/fr3_wuji/env.sh"
# The deployment gives gateway/splitter/devices their own CPU groups. Keep the
# host controller, observation readers and camera launchers on the remainder.
PI05_ONESHOT_HOST_CPUSET="${PI05_ONESHOT_HOST_CPUSET:-0-1,20-23}"
exec taskset -c "$PI05_ONESHOT_HOST_CPUSET" "$PI05_ROOT/.venv/bin/python" -m experiments.weight_motion_eval.oneshot.deploy "$@"
