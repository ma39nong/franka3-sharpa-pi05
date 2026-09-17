#!/usr/bin/env bash
set -euo pipefail
PI05_ADAPTER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PI05_ROOT="$(cd -- "$PI05_ADAPTER_DIR/../.." && pwd)"
exec bash "$PI05_ADAPTER_DIR/run.sh" \
  --checkpoint "$PI05_ROOT/checkpoints/30000v2" \
  --full-finetune \
  --port 8002 \
  "$@"
