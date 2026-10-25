#!/usr/bin/env bash
# Select full-finetune defaults while sharing the workstation environment.
source "$(dirname -- "${BASH_SOURCE[0]}")/../fr3_sharpa_runtime/model_env.sh" --model full
