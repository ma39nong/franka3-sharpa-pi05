#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$ROOT"

# Prefer this project's uv environment; an existing compatible environment can
# be selected explicitly without changing its installed OpenPI package.
TRAIN_PYTHON="${SHARPA_TRAIN_PYTHON:-$ROOT/.venv/bin/python}"
[[ -x "$TRAIN_PYTHON" ]] || {
  echo "Training Python missing: $TRAIN_PYTHON. Run GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen." >&2
  exit 1
}
export PYTHONPATH="$ROOT/src:$ROOT/packages/openpi-client/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

CONFIG=pi05_fr3_sharpa_tier_a_b1_20hz
if [[ "${1:-}" == "--compute-norm-stats" ]]; then
  shift
  exec "$TRAIN_PYTHON" "$ROOT/scripts/compute_norm_stats.py" --config-name "$CONFIG" "$@"
fi

exec "$TRAIN_PYTHON" "$ROOT/scripts/train.py" "$CONFIG" \
  --exp-name sharpa_camfix_a70_b130_20hz_30k "$@"
