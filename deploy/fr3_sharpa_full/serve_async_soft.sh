#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source "$ROOT/deploy/fr3_sharpa_runtime/model_env.sh" "$@"
set -- "${SHARPA_MODEL_ARGS[@]}"
sharpa_python -m deploy.fr3_sharpa_runtime.async_soft_serve "$@"
