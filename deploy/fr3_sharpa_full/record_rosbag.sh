#!/usr/bin/env bash
# Record existing ROS topics only; never start hardware or publish commands.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/deploy/fr3_sharpa_runtime/env.sh"
CHECK_ONLY=false
if [[ "${1:-}" == --check ]]; then
  CHECK_ONLY=true
  shift
fi
if [[ $# -ne 1 || "${1:-}" == --help ]]; then
  echo 'Usage: bash deploy/fr3_sharpa_full/record_rosbag.sh [--check] RUN_NAME_OR_ABSOLUTE_DIR'
  exit 0
fi
case "$1" in
  /*) RECORD_DIR="$1" ;;
  *) RECORD_DIR="${SHARPA_RECORDINGS_DIR:-$HOME/桌面/sharpa_deploy_records}/$1" ;;
esac
OVERLAY="${SHARPA_ROS_MESSAGES_OVERLAY:-$SHARPA_ROOT/.deployment/sharpa_ros_messages/install}"
[[ -f "$OVERLAY/setup.bash" ]] || { echo "Missing ROS messages: $OVERLAY/setup.bash" >&2; exit 1; }
[[ ! -e "$RECORD_DIR/rosbag" ]] || { echo "Recording already exists: $RECORD_DIR/rosbag; use a new run name." >&2; exit 1; }
DOMAIN="$(awk -F= '$1 == "TELEOP_ROS_DOMAIN_ID" {print $2; exit}' "$SHARPA_GELLO_REPO/docker/.env")"
[[ "$DOMAIN" =~ ^[0-9]+$ ]] || { echo 'Invalid TELEOP_ROS_DOMAIN_ID in hardware docker/.env' >&2; exit 1; }
export ROS_DOMAIN_ID="$DOMAIN" ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$SHARPA_GELLO_REPO/config/cyclonedds.xml"
echo "Recording directory: $RECORD_DIR/rosbag"
if $CHECK_ONLY; then
  echo "Configuration OK; ROS domain $ROS_DOMAIN_ID. No recorder or hardware started."
  exit 0
fi
mkdir -p "$RECORD_DIR"
exec pixi run --locked \
  --manifest-path "$SHARPA_GELLO_REPO/portable_deps/litchi_hardware/pixi.toml" \
  -e ros2 bash -c '
    set -e
    source "$1"
    exec ros2 bag record -o "$2" --max-bag-size 2147483648 \
      /cam0/color/image_raw /cam1/color/image_raw /cam2/color/image_raw \
      /left/franka/joint_states /right/franka/joint_states \
      /sharpa/left/joint_states /sharpa/right/joint_states \
      /teleop/arm_commands
  ' -- "$OVERLAY/setup.bash" "$RECORD_DIR/rosbag"
