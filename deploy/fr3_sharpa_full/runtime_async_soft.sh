#!/usr/bin/env bash
# Independent fourth entry; the serial/RTC/serial-Soft launchers are untouched.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source "$ROOT/deploy/fr3_sharpa_runtime/model_env.sh" "$@"
set -- "${SHARPA_MODEL_ARGS[@]}"
for argument in "$@"; do
  if [[ "$argument" == --check || "$argument" == --help || "$argument" == -h ]]; then
    sharpa_python -m deploy.fr3_sharpa_runtime.async_soft_run "$@"
  fi
done
# Parse all options without hardware before entering a managed hardware session.
check_args=()
execute=false
hardware=false
for argument in "$@"; do
  case "$argument" in
    --execute) execute=true ;;
    --enable-hardware) hardware=true ;;
    *) check_args+=("$argument") ;;
  esac
done
if ! $execute; then
  echo 'Select --check or --execute --enable-hardware' >&2
  exit 2
fi
# The real parser still enforces --enable-hardware; reject omission before cleanup ownership.
if ! $hardware; then
  echo '--execute requires --enable-hardware' >&2
  exit 2
fi
interpreter="${SHARPA_PYTHON:-$SHARPA_ROOT/.venv/bin/python}"
"$interpreter" -m deploy.fr3_sharpa_runtime.async_soft_run --check "${check_args[@]}" >/dev/null
"$interpreter" -m deploy.fr3_sharpa_runtime.async_soft_serve --check-only >/dev/null
GELLO="${SHARPA_GELLO_REPO:-/home/user/gello_upper_body_teleop}"
DOMAIN="$(awk -F= '$1 == "TELEOP_ROS_DOMAIN_ID" {print $2; exit}' "$GELLO/docker/.env")"
[[ "$DOMAIN" =~ ^[0-9]+$ ]] || { echo 'Missing TELEOP_ROS_DOMAIN_ID' >&2; exit 1; }
export ROS_DOMAIN_ID="$DOMAIN" ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$GELLO/config/cyclonedds.xml"
ROS_MESSAGES_OVERLAY="${SHARPA_ROS_MESSAGES_OVERLAY:-$ROOT/.deployment/sharpa_ros_messages/install}"
if [[ ! -f "$ROS_MESSAGES_OVERLAY/setup.bash" && -z "${SHARPA_ROS_MESSAGES_OVERLAY:-}" ]]; then
  bash "$ROOT/deploy/fr3_sharpa_runtime/build_messages.sh"
fi
[[ -f "$ROS_MESSAGES_OVERLAY/setup.bash" ]] || { echo 'Missing ROS messages overlay' >&2; exit 1; }
sharpa_python -m deploy.fr3_sharpa_runtime.async_soft_session --ros-overlay "$ROS_MESSAGES_OVERLAY" "$@"
