#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
: "${SHARPA_GELLO_REPO:?Run setup_local.py first}"
: "${SHARPA_GATEWAY_CONFIG:?Run setup_local.py first}"
COMPOSE=(docker compose --profile openpi-hardware --project-directory "$SHARPA_GELLO_REPO/docker"
  --env-file "$SHARPA_GELLO_REPO/docker/.env"
  -f "$SHARPA_GELLO_REPO/docker/compose.yaml"
  -f "$SHARPA_ROOT/deploy/fr3_sharpa_runtime/compose.hardware.yaml")
case "${1:-check}" in
  check) "${COMPOSE[@]}" config --quiet ;;
  camera-check) sharpa_python -m deploy.fr3_sharpa_runtime.camera_startup --inspect ;;
  up)
    shift
    sharpa_python -m deploy.fr3_sharpa_runtime.camera_startup "$@"
    ;;
  stop) sharpa_python -m deploy.fr3_sharpa_runtime.hardware_session stop ;;
  *) echo 'Usage: hardware.sh [check|camera-check|up [--exclusive]|stop]' >&2; exit 2 ;;
esac
