#!/usr/bin/env bash
set -euo pipefail
PI05_MEDIUM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$PI05_MEDIUM_DIR/../fr3_wuji/env.sh"
PI05_MEDIUM_HOST_CPUSET="${PI05_MEDIUM_HOST_CPUSET:-0-1,20-23}"
exec taskset -c "$PI05_MEDIUM_HOST_CPUSET" "$PI05_ROOT/.venv/bin/python" -m deploy.fr3_wuji_medium.deploy "$@"
