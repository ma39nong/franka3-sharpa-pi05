#!/usr/bin/env bash
# First argument is a NEW recording directory; remaining args match runtime_delta.sh.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/deploy/fr3_sharpa_runtime/env.sh"
[[ $# -ge 1 ]] || { echo 'Usage: record_runtime_delta.sh RUN_DIR [runtime options]' >&2; exit 2; }
RECORD_DIR="$1"
shift
mkdir -p "$RECORD_DIR"
RECORD_DIR="$(cd -- "$RECORD_DIR" && pwd)"
cd "$ROOT"
sharpa_python -m deploy.fr3_sharpa_runtime.record_console \
  --log "$RECORD_DIR/runtime_console.log" -- \
  bash "$ROOT/deploy/fr3_sharpa_full/runtime_delta.sh" "$@" \
  --execution-trace-dir "$RECORD_DIR/execution_traces"
