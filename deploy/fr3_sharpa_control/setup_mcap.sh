#!/usr/bin/env bash
# Install only into this checkout; leave the shared hardware environment untouched.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/deploy/fr3_sharpa_runtime/env.sh"
PYTHON="$SHARPA_GELLO_REPO/portable_deps/litchi_hardware/.pixi/envs/ros2/bin/python"
uv pip install --python "$PYTHON" --target "$ROOT/.deployment/mcap-python" -r "$ROOT/deploy/fr3_sharpa_control/mcap-requirements.txt"
echo "MCAP dependencies installed in $ROOT/.deployment/mcap-python"

c++ -O3 -std=c++11 -shared -fPIC "$ROOT/deploy/fr3_sharpa_control/rvl_encode.cpp" -o "$ROOT/.deployment/mcap-rvl.so"
