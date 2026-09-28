#!/usr/bin/env bash
set -euo pipefail
HOME_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$HOME_DIR/run.sh" --execute --supervised-trial "$@"
