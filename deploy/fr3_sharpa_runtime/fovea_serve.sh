#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source "$ROOT/deploy/fr3_sharpa_runtime/env.sh"
sharpa_python -m deploy.fr3_sharpa_runtime.fovea_serve "$@"
