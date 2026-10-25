#!/usr/bin/env bash
# Shared launch-time selection, before Python imports its model configuration.
# Source with "$@"; launchers then forward only SHARPA_MODEL_ARGS.
SHARPA_MODEL_ARGS=()
selected_model=""
list_models=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      [[ $# -ge 2 && "$2" != --* ]] || { echo '--model requires full or lora' >&2; return 2; }
      candidate="$2"; shift 2 ;;
    --model=*) candidate="${1#--model=}"; shift ;;
    --list-models) list_models=true; shift; continue ;;
    *) SHARPA_MODEL_ARGS+=("$1"); shift; continue ;;
  esac
  [[ "$candidate" == full || "$candidate" == lora ]] || {
    echo "Unknown model '$candidate'; select full or lora" >&2; return 2;
  }
  [[ -z "$selected_model" || "$selected_model" == "$candidate" ]] || {
    echo 'Conflicting --model selections' >&2; return 2;
  }
  selected_model="$candidate"
done
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
export SHARPA_MODEL_PROFILE="${selected_model:-${SHARPA_MODEL_PROFILE:-full}}"
case "$SHARPA_MODEL_PROFILE" in
  full) ;;
  lora)
    model_env="${SHARPA_LORA_ENV_FILE:-$SHARPA_ROOT/.deployment/lora.env}"
    if [[ -f "$model_env" ]]; then
      source "$model_env"
    elif [[ -n "${SHARPA_LORA_ENV_FILE:-}" ]]; then
      echo "Missing explicitly selected LoRA configuration: $model_env" >&2; return 1
    fi
    # The model flag has precedence over any contents of the defaults file.
    export SHARPA_MODEL_PROFILE=lora ;;
  *) echo 'SHARPA_MODEL_PROFILE must be full or lora' >&2; return 2 ;;
esac
if $list_models; then
  sharpa_python -m deploy.fr3_sharpa_runtime.model_profiles
fi
