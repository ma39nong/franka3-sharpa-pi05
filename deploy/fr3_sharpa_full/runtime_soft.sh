#!/usr/bin/env bash
# Serial Soft Inpainting defaults; motion still requires the existing explicit flags.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$ROOT/deploy/fr3_sharpa_runtime/runtime.sh" \
  --inference-preset paper-rolling --inpainting soft --compression off --broker-mode serial "$@"
