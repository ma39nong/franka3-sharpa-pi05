#!/usr/bin/env bash
# Build message types only. Does not start ROS nodes or connect to hardware.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/deploy/fr3_sharpa_runtime/env.sh"
GELLO="${SHARPA_GELLO_REPO:-/home/user/gello_upper_body_teleop}"
BUILD_ROOT="$ROOT/.deployment/sharpa_ros_messages"
SOURCE="$GELLO/ros_ws/src/teleop_interfaces"
CAMERA_SOURCE="$GELLO/ros_ws/src/orbbec_camera_msgs"
[[ -f "$SOURCE/package.xml" ]] || { echo "Missing message sources: $SOURCE" >&2; exit 1; }
[[ -f "$CAMERA_SOURCE/package.xml" ]] || { echo "Missing camera message sources: $CAMERA_SOURCE" >&2; exit 1; }
# The hardware workspace intentionally ignores its vendor message package.
# Stage message definitions only, leaving its COLCON_IGNORE and source untouched.
"${SHARPA_PYTHON:-$ROOT/.venv/bin/python}" - "$CAMERA_SOURCE" "$BUILD_ROOT/src/orbbec_camera_msgs" <<'PY'
from pathlib import Path
import shutil
import sys
source, target = map(Path, sys.argv[1:])
shutil.copytree(source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns('COLCON_IGNORE'))
PY
CAMERA_SOURCE="$BUILD_ROOT/src/orbbec_camera_msgs"
pixi run --locked --manifest-path "$GELLO/portable_deps/litchi_hardware/pixi.toml" -e ros2 \
  bash -c '
    set -eo pipefail
    colcon --log-base "$2/log" build --base-paths "$1" "$3" \
      --packages-select teleop_interfaces orbbec_camera_msgs --build-base "$2/build" --install-base "$2/install" \
      --cmake-args -DPython3_EXECUTABLE="$(command -v python)" -DPYTHON_EXECUTABLE="$(command -v python)"
    source "$2/install/setup.bash"
    python -c "from teleop_interfaces.msg import ArmCommand; ArmCommand(); print(\"ArmCommand import: OK\")"
    python -c "from orbbec_camera_msgs.srv import GetDeviceInfo; GetDeviceInfo.Request(); print(\"Camera identity service import: OK\")"
  ' -- "$SOURCE" "$BUILD_ROOT" "$CAMERA_SOURCE"
echo "ROS message overlay ready: $BUILD_ROOT/install"
