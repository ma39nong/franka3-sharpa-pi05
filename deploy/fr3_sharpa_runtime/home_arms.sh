#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
sharpa_python -m deploy.fr3_sharpa_runtime.home_arms "$@"
