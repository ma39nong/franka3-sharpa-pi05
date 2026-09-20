#!/usr/bin/env bash
set -euo pipefail
PI05_25000_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$PI05_25000_DIR/../fr3_wuji_medium/run.sh" \
  --model 25000 \
  --checkpoint "$PI05_25000_DIR/../../checkpoints/25000" \
  --uri ws://127.0.0.1:8004 \
  "$@"
