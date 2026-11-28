#!/usr/bin/env bash
# Default: synthetic payload through actual ROS and MCAP. --hardware: tactile only.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../fr3_sharpa_runtime/env.sh"
export PYTHONPATH="$SHARPA_ROOT/.deployment/mcap-python:$SHARPA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec pixi run --locked --manifest-path "$SHARPA_GELLO_REPO/portable_deps/litchi_hardware/pixi.toml" -e ros2 \
  python -m deploy.fr3_sharpa_control.validation.validate_tactile_recording "$@"
