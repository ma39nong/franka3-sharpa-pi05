#!/usr/bin/env bash
set -euo pipefail
PI05_25000_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$PI05_25000_DIR/../fr3_wuji/run.sh" "$PI05_25000_DIR/serve.py" "$@"
