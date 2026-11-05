#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/deploy/fr3_sharpa_runtime/model_env.sh" --model flash-absolute "$@"
set -- "${SHARPA_MODEL_ARGS[@]}"
: "${SHARPA_FLASH_REPO:?Set training FLASH source directory}"
: "${SHARPA_FLASH_PYTHON:?Set independent FLASH Python interpreter}"
: "${SHARPA_FLASH_CHECKPOINT:?Set complete FLASH checkpoint step}"
[[ "$SHARPA_FLASH_PYTHON" == /* && "$SHARPA_FLASH_REPO" == /* && -x "$SHARPA_FLASH_PYTHON" && -f "$SHARPA_FLASH_REPO/src/openpi/models/pi0_flash.py" ]] || {
  echo 'Missing FLASH interpreter/source; do not substitute the old deployment environment' >&2; exit 1;
}
# ROOT makes deploy.* visible but never ROOT/src (the other openpi version).
export PYTHONPATH="$SHARPA_FLASH_REPO/src:$ROOT"
export PYTHONNOUSERSITE=1
cd "$SHARPA_FLASH_REPO"
exec "$SHARPA_FLASH_PYTHON" -m deploy.fr3_sharpa_flash.serve "$@"
