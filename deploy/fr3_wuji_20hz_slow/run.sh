#!/usr/bin/env bash
set -euo pipefail
PI05_20HZ_SLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$PI05_20HZ_SLOW_DIR/../fr3_wuji/env.sh"
PI05_20HZ_SLOW_HOST_CPUSET="${PI05_20HZ_SLOW_HOST_CPUSET:-${PI05_SLOW_HOST_CPUSET:-${PI05_ONESHOT_HOST_CPUSET:-0-1,20-23}}}"
exec taskset -c "$PI05_20HZ_SLOW_HOST_CPUSET" "$PI05_ROOT/.venv/bin/python" -B \
  -m deploy.fr3_wuji_20hz_slow.deploy "$@"
