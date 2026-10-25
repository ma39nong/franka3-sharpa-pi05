#!/usr/bin/env bash
# Compatibility for shells that sourced the LoRA environment directly.
source "$(dirname -- "${BASH_SOURCE[0]}")/../fr3_sharpa_runtime/model_env.sh" --model lora
