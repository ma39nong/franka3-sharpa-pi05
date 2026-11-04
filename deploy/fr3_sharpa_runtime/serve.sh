#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source "$ROOT/deploy/fr3_sharpa_runtime/model_env.sh" "$@"
set -- "${SHARPA_MODEL_ARGS[@]}"
if [[ "$SHARPA_MODEL_PROFILE" == flash-absolute ]]; then
  exec bash "$ROOT/deploy/fr3_sharpa_runtime/serve_flash.sh" "$@"
fi
sharpa_python -m deploy.fr3_sharpa_runtime.serve "$@"
