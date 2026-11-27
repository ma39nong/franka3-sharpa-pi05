#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
: "${SHARPA_GELLO_REPO:?Run setup_local.py first}"
: "${SHARPA_GATEWAY_CONFIG:?Run setup_local.py first}"
# Select recording depth only for this deployment; never rewrite the shared camera profile.
if [[ "${SHARPA_RECORD_HEAD_DEPTH:-0}" == 1 ]]; then
  SHARPA_CAMERA_CAM0_PARAMS="$("${SHARPA_PYTHON:-$SHARPA_ROOT/.venv/bin/python}" -m deploy.fr3_sharpa_control.recording_options --depth-profile "${SHARPA_CAMERA_CAM0_PARAMS:?Missing head profile}")"
  export SHARPA_CAMERA_CAM0_PARAMS
fi
COMPOSE=(docker compose --profile openpi-hardware --project-directory "$SHARPA_GELLO_REPO/docker"
  --env-file "$SHARPA_GELLO_REPO/docker/.env"
  -f "$SHARPA_GELLO_REPO/docker/compose.yaml"
  -f "$SHARPA_ROOT/deploy/fr3_sharpa_runtime/compose.hardware.yaml")
if [[ -n "${SHARPA_CAMERA_CAM1_PARAMS:-}" ]]; then
  [[ "$SHARPA_CAMERA_CAM1_PARAMS" == /* && -f "$SHARPA_CAMERA_CAM1_PARAMS" ]] || {
    echo 'SHARPA_CAMERA_CAM1_PARAMS must be an existing absolute file path' >&2
    exit 1
  }
  COMPOSE+=(-f "$SHARPA_ROOT/deploy/fr3_sharpa_runtime/compose.camera-profile.yaml")
fi
if [[ -n "${SHARPA_CAMERA_CAM0_PARAMS:-}" ]]; then
  [[ "$SHARPA_CAMERA_CAM0_PARAMS" == /* && -f "$SHARPA_CAMERA_CAM0_PARAMS" ]] || {
    echo 'SHARPA_CAMERA_CAM0_PARAMS must be an existing absolute file path' >&2
    exit 1
  }
  COMPOSE+=(-f "$SHARPA_ROOT/deploy/fr3_sharpa_runtime/compose.camera-head.yaml")
fi
case "${1:-check}" in
  check) "${COMPOSE[@]}" config --quiet ;;
  camera-check) sharpa_python -m deploy.fr3_sharpa_runtime.camera_startup --inspect ;;
  up)
    shift
    sharpa_python -m deploy.fr3_sharpa_runtime.camera_startup "$@"
    ;;
  stop) sharpa_python -m deploy.fr3_sharpa_runtime.hardware_session stop ;;
  *) echo 'Usage: hardware.sh [check|camera-check|up [--exclusive] [--defer-camera-check]|stop]' >&2; exit 2 ;;
esac
