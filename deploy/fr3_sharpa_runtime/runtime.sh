#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/deploy/fr3_sharpa_runtime/model_env.sh" "$@"
set -- "${SHARPA_MODEL_ARGS[@]}"
# Mamba admission is disk-only and happens before ROS overlays/hardware sessions.
if [[ "$SHARPA_MODEL_PROFILE" == mamba || "$SHARPA_MODEL_PROFILE" == mamba-delta ]]; then
  mamba_help=false
  mamba_rtc_flags=()
  mamba_broker=""
  previous=""
  for argument in "$@"; do
    case "$argument" in
      --help|-h) mamba_help=true ;;
      --train-rtc) mamba_rtc_flags+=(--enable-train-rtc) ;;
      --train-paper-rtc) mamba_rtc_flags+=(--enable-train-paper-rtc) ;;
      --paper-rtc) echo 'Mamba rejects soft Paper RTC; use --train-paper-rtc' >&2; exit 2 ;;
      --broker-mode=*) mamba_broker="${argument#--broker-mode=}" ;;
    esac
    if [[ "$previous" == --broker-mode ]]; then mamba_broker="$argument"; fi
    previous="$argument"
  done
  if ! $mamba_help; then
    if [[ "$mamba_broker" == rtc && ${#mamba_rtc_flags[@]} -eq 0 ]]; then
      echo 'Mamba RTC requires --train-rtc or --train-paper-rtc' >&2; exit 2
    fi
    interpreter="${SHARPA_PYTHON:-$SHARPA_ROOT/.venv/bin/python}"
    "$interpreter" -m deploy.fr3_sharpa_runtime.serve --check-only "${mamba_rtc_flags[@]}" > /dev/null
  fi
fi
# Explicit alternate-model validation must happen before any hardware session is entered.
if [[ "$SHARPA_MODEL_PROFILE" != full && "$SHARPA_MODEL_PROFILE" != mamba && "$SHARPA_MODEL_PROFILE" != mamba-delta ]]; then
  previous=""
  for argument in "$@"; do
    if [[ "$argument" == --execute || "$argument" == --mode=execute || ( "$previous" == --mode && "$argument" == execute ) ]]; then
      interpreter="${SHARPA_PYTHON:-$SHARPA_ROOT/.venv/bin/python}"
      "$interpreter" -m deploy.fr3_sharpa_runtime.serve --check-only > /dev/null
      break
    fi
    previous="$argument"
  done
fi
# Offline inspection must work without the robot workstation or ROS overlay.
previous=""
for argument in "$@"; do
  if [[ "$argument" == --check || "$argument" == --help || "$argument" == -h || "$argument" == --mode=check || ( "$previous" == --mode && "$argument" == check ) ]]; then
    cd "$ROOT"
    sharpa_python -m deploy.fr3_sharpa_runtime.run "$@"
    exit $?
  fi
  previous="$argument"
done
GELLO="${SHARPA_GELLO_REPO:-/home/user/gello_upper_body_teleop}"
DOMAIN="$(awk -F= '$1 == "TELEOP_ROS_DOMAIN_ID" {print $2; exit}' "$GELLO/docker/.env")"
[[ "$DOMAIN" =~ ^[0-9]+$ ]] || { echo "Missing TELEOP_ROS_DOMAIN_ID in docker/.env" >&2; exit 1; }
export ROS_DOMAIN_ID="$DOMAIN" ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$GELLO/config/cyclonedds.xml"
# Persist the Pixi-compatible message build across reboots; /tmp is ephemeral.
ROS_MESSAGES_OVERLAY="${SHARPA_ROS_MESSAGES_OVERLAY:-$ROOT/.deployment/sharpa_ros_messages/install}"
if [[ ! -f "$ROS_MESSAGES_OVERLAY/setup.bash" && -z "${SHARPA_ROS_MESSAGES_OVERLAY:-}" ]]; then
  bash "$ROOT/deploy/fr3_sharpa_runtime/build_messages.sh"
fi
[[ -f "$ROS_MESSAGES_OVERLAY/setup.bash" ]] || {
  echo "Missing teleop_interfaces overlay: $ROS_MESSAGES_OVERLAY (check SHARPA_ROS_MESSAGES_OVERLAY)" >&2
  exit 1
}
cd "$ROOT"
MODULE=deploy.fr3_sharpa_runtime.run
previous=""
for argument in "$@"; do
  if [[ "$argument" == --execute || "$argument" == --mode=execute || ( "$previous" == --mode && "$argument" == execute ) ]]; then
    MODULE=deploy.fr3_sharpa_runtime.hardware_session
  fi
  previous="$argument"
done
sharpa_python -m "$MODULE" \
  --ros-overlay "$ROS_MESSAGES_OVERLAY" "$@"
