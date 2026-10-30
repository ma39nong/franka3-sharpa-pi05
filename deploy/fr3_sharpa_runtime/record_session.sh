#!/usr/bin/env bash
# Compatibility entry; implementation is in fr3_sharpa_control.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$ROOT/deploy/fr3_sharpa_control/record_session.sh" "$@"
