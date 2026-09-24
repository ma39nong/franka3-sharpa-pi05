#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$ROOT"

export JAX_PLATFORMS=cuda
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

"$ROOT/.venv/bin/python" "$ROOT/scripts/prepare_fr3_wuji_20hz.py"
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/train.py" pi05_fr3_wuji_20hz \
  --exp-name tomato_lora_0918_20hz "$@"
