#!/usr/bin/env bash
set -euo pipefail
PI05_SINGLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$PI05_SINGLE_DIR/../fr3_wuji_medium/run.sh" \
  --model 64-full-15hz-a \
  --checkpoint "$PI05_SINGLE_DIR/../../checkpoints/25000-single" \
  --uri ws://127.0.0.1:8005 \
  "$@"
