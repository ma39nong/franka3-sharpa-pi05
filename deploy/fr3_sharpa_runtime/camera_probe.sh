#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
: "${SHARPA_GELLO_REPO:?Run workstation setup first}"
DOMAIN="$(awk -F= '$1 == "TELEOP_ROS_DOMAIN_ID" {print $2; exit}' "$SHARPA_GELLO_REPO/docker/.env")"
[[ "$DOMAIN" =~ ^[0-9]+$ ]] || { echo 'Invalid ROS domain' >&2; exit 1; }
export ROS_DOMAIN_ID="$DOMAIN" ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$SHARPA_GELLO_REPO/config/cyclonedds.xml"
OVERLAY="${SHARPA_ROS_MESSAGES_OVERLAY:-$SHARPA_ROOT/.deployment/sharpa_ros_messages/install}"
exec pixi run --locked --manifest-path "$SHARPA_GELLO_REPO/portable_deps/litchi_hardware/pixi.toml" -e ros2 \
  bash -c 'source "$1"; export PYTHONPATH="$2:$2/src${PYTHONPATH:+:$PYTHONPATH}"; shift 2; exec python -m deploy.fr3_sharpa_runtime.camera_probe "$@"' \
  -- "$OVERLAY/setup.bash" "$SHARPA_ROOT" "$@"
