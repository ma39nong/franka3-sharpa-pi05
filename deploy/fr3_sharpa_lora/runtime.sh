#!/usr/bin/env bash
# Compatibility alias; all model selection and launch logic lives in the shared runtime.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$ROOT/deploy/fr3_sharpa_runtime/runtime.sh" --model lora "$@"
