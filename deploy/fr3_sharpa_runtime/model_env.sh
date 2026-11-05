#!/usr/bin/env bash
# Shared launch-time selection, before Python imports its model configuration.
# Source with "$@"; launchers then forward only SHARPA_MODEL_ARGS.
SHARPA_MODEL_ARGS=()
selected_model=""
list_models=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      [[ $# -ge 2 && "$2" != --* ]] || { echo '--model requires full, lora, full-delta, mamba or mamba-delta' >&2; return 2; }
      candidate="$2"; shift 2 ;;
    --model=*) candidate="${1#--model=}"; shift ;;
    --list-models) list_models=true; shift; continue ;;
    *) SHARPA_MODEL_ARGS+=("$1"); shift; continue ;;
  esac
  [[ "$candidate" == full || "$candidate" == lora || "$candidate" == full-delta || "$candidate" == mamba || "$candidate" == mamba-delta ]] || {
    echo "Unknown model '$candidate'; select full, lora, full-delta, mamba or mamba-delta" >&2; return 2;
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
  full-delta)
    model_env="${SHARPA_FULL_DELTA_ENV_FILE:-$SHARPA_ROOT/.deployment/full_delta.env}"
    if [[ -f "$model_env" ]]; then
      source "$model_env"
    elif [[ -n "${SHARPA_FULL_DELTA_ENV_FILE:-}" ]]; then
      echo "Missing explicitly selected full-delta configuration: $model_env" >&2; return 1
    fi
    export SHARPA_MODEL_PROFILE=full-delta ;;
  mamba|mamba-delta)
    mamba_selection="$SHARPA_MODEL_PROFILE"
    if [[ "$mamba_selection" == mamba ]]; then
      model_env="${SHARPA_MAMBA_ENV_FILE:-$SHARPA_ROOT/.deployment/mamba.env}"
      explicit_env="${SHARPA_MAMBA_ENV_FILE:-}"
    else
      model_env="${SHARPA_MAMBA_DELTA_ENV_FILE:-$SHARPA_ROOT/.deployment/mamba_delta.env}"
      explicit_env="${SHARPA_MAMBA_DELTA_ENV_FILE:-}"
    fi
    # Save caller-provided Mamba settings before applying this profile's defaults.
    mamba_prefix=SHARPA_MAMBA
    [[ "$mamba_selection" != mamba-delta ]] || mamba_prefix=SHARPA_MAMBA_DELTA
    declare -A mamba_overrides=()
    for suffix in CHECKPOINT NORM_STATS PORT EXECUTION_TRACE_DIR; do
      mamba_key="${mamba_prefix}_${suffix}"
      if [[ -v "$mamba_key" ]]; then mamba_overrides["$mamba_key"]="${!mamba_key}"; fi
    done
    if [[ -f "$model_env" ]]; then
      source "$model_env"
      for mamba_key in "${!mamba_overrides[@]}"; do
        export "$mamba_key=${mamba_overrides[$mamba_key]}"
      done
    elif [[ -n "$explicit_env" ]]; then
      echo "Missing explicitly selected Mamba configuration: $model_env" >&2; return 1
    fi
    export SHARPA_MODEL_PROFILE="$mamba_selection" ;;
  *) echo 'SHARPA_MODEL_PROFILE must be full, lora, full-delta, mamba or mamba-delta' >&2; return 2 ;;
esac
if $list_models; then
  sharpa_python -m deploy.fr3_sharpa_runtime.model_profiles
fi
