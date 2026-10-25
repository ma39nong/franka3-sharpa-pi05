#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
exec bash "$LORA_ROOT/deploy/fr3_sharpa_runtime/serve.sh" "$@"
