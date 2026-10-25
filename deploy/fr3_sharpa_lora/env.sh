#!/usr/bin/env bash
# Source in a child shell; generic full-model settings remain untouched.
LORA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$LORA_ROOT/deploy/fr3_sharpa_runtime/env.sh"
LORA_ENV="${SHARPA_LORA_ENV_FILE:-$LORA_ROOT/.deployment/lora.env}"
if [[ -f "$LORA_ENV" ]]; then
  source "$LORA_ENV"
elif [[ -n "${SHARPA_LORA_ENV_FILE:-}" ]]; then
  echo "Missing explicitly selected LoRA configuration: $LORA_ENV" >&2
  return 1
fi
export SHARPA_MODEL_PROFILE=lora
