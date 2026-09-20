#!/usr/bin/env bash
set -euo pipefail
PI05_SLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$PI05_SLOW_DIR/../fr3_wuji/env.sh"
# Preserve the original slow entry's CPU placement. The old variable remains
# accepted for existing deployments; PI05_SLOW_HOST_CPUSET takes precedence.
PI05_SLOW_HOST_CPUSET="${PI05_SLOW_HOST_CPUSET:-${PI05_ONESHOT_HOST_CPUSET:-0-1,20-23}}"
exec taskset -c "$PI05_SLOW_HOST_CPUSET" "$PI05_ROOT/.venv/bin/python" -m deploy.fr3_wuji_slow.deploy "$@"
