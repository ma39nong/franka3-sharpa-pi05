#!/usr/bin/env bash
# Shared launch-time selection, before Python imports its model configuration.
# Source with "$@"; launchers then forward only SHARPA_MODEL_ARGS.
SHARPA_MODEL_ARGS=()
selected_model=""
list_models=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      [[ $# -ge 2 && "$2" != --* ]] || { echo '--model requires full, lora, full-delta or flash-absolute' >&2; return 2; }
      candidate="$2"; shift 2 ;;
    --model=*) candidate="${1#--model=}"; shift ;;
    --list-models) list_models=true; shift; continue ;;
    *) SHARPA_MODEL_ARGS+=("$1"); shift; continue ;;
  esac
  [[ "$candidate" == full || "$candidate" == lora || "$candidate" == full-delta || "$candidate" == flash-absolute ]] || {
    echo "Unknown model '$candidate'; select full, lora, full-delta or flash-absolute" >&2; return 2;
  }
  [[ -z "$selected_model" || "$selected_model" == "$candidate" ]] || {
    echo 'Conflicting --model selections' >&2; return 2;
  }
  selected_model="$candidate"
done
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
export SHARPA_MODEL_PROFILE="${selected_model:-${SHARPA_MODEL_PROFILE:-full}}"
case "$SHARPA_MODEL_PROFILE" in
  flash-absolute)
    model_env="${SHARPA_FLASH_ENV_FILE:-$SHARPA_ROOT/.deployment/flash.env}"
    # GUI/CLI choices take precedence over stale defaults from an earlier run.
    declare -A flash_caller_settings=()
    for flash_setting_key in SHARPA_FLASH_CHECKPOINT SHARPA_FLASH_NORM_STATS SHARPA_FLASH_PORT SHARPA_FLASH_MAX_PLAN_AGE_MS SHARPA_FLASH_REPO SHARPA_FLASH_PYTHON SHARPA_FLASH_TOKENIZER SHARPA_FLASH_TRAINING_CONFIG; do
      if [[ -v "$flash_setting_key" ]]; then
        flash_caller_settings["$flash_setting_key"]="${!flash_setting_key}"
      fi
    done
    if [[ -f "$model_env" ]]; then
      source "$model_env"
    elif [[ -n "${SHARPA_FLASH_ENV_FILE:-}" ]]; then
      echo "Missing explicitly selected FLASH configuration: $model_env" >&2; return 1
    fi
    for flash_setting_key in "${!flash_caller_settings[@]}"; do
      export "$flash_setting_key=${flash_caller_settings[$flash_setting_key]}"
    done
    unset flash_caller_settings flash_setting_key
    export SHARPA_MODEL_PROFILE=flash-absolute ;;
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
  full-delta)
    model_env="${SHARPA_FULL_DELTA_ENV_FILE:-$SHARPA_ROOT/.deployment/full_delta.env}"
    if [[ -f "$model_env" ]]; then
      source "$model_env"
    elif [[ -n "${SHARPA_FULL_DELTA_ENV_FILE:-}" ]]; then
      echo "Missing explicitly selected full-delta configuration: $model_env" >&2; return 1
    fi
    export SHARPA_MODEL_PROFILE=full-delta ;;
  *) echo 'SHARPA_MODEL_PROFILE must be full, lora, full-delta or flash-absolute' >&2; return 2 ;;
esac
if $list_models; then
  sharpa_python -m deploy.fr3_sharpa_runtime.model_profiles
fi
