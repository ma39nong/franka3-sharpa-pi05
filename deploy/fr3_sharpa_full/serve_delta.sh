#!/usr/bin/env bash
# Full-finetune entry; execution and model loading remain shared.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$ROOT/deploy/fr3_sharpa_runtime/serve.sh" --model full-delta "$@"
