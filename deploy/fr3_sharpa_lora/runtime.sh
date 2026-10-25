#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
# Fail before the session manager can touch hardware when the model is absent
# or a full/partial checkpoint has accidentally been selected for LoRA.
previous=""
for argument in "$@"; do
  if [[ "$argument" == --execute || "$argument" == --mode=execute || ( "$previous" == --mode && "$argument" == execute ) ]]; then
    interpreter="${SHARPA_PYTHON:-$LORA_ROOT/.venv/bin/python}"
    "$interpreter" -m deploy.fr3_sharpa_runtime.serve --check-only > /dev/null
    break
  fi
  previous="$argument"
done
exec bash "$LORA_ROOT/deploy/fr3_sharpa_runtime/runtime.sh" "$@"
