#!/usr/bin/env bash
# Opt-in Soft Inpainting server; reuse the existing model and environment selection.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$ROOT/deploy/fr3_sharpa_runtime/serve.sh" --enable-inference-experiments "$@"
