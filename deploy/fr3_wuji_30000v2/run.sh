#!/usr/bin/env bash
set -euo pipefail
PI05_V2_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$PI05_V2_DIR/../fr3_wuji/run.sh" "$PI05_V2_DIR/serve.py" "$@"
