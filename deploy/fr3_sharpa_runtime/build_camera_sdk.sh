#!/usr/bin/env bash
# Build a separate camera image; do not change shared teleop installations.
set -euo pipefail
runtime_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
sdk_archive="${1:?Usage: build_camera_sdk.sh /path/to/official-SDK-v2.10.6-linux_x86_64.tar.gz}"
sdk_archive="$(realpath -- "$sdk_archive")"
expected_sha=23ec9f7023881cc9061bf0b4ba65735fb16c10e168fd5d62b1548c847e7ddb46
actual_sha="$(sha256sum -- "$sdk_archive")"
[[ "${actual_sha%% *}" == "$expected_sha" ]] || { echo 'SDK archive SHA-256 mismatch' >&2; exit 1; }
context="$(mktemp -d)"
trap 'rm -rf -- "$context"' EXIT
mkdir "$context/sdk"
tar -xzf "$sdk_archive" --strip-components=1 -C "$context/sdk"
cp "$runtime_dir/camera_sdk_entrypoint.sh" "$context/"
docker build --network=none -f "$runtime_dir/camera-sdk.Dockerfile" \
  -t franka-sharpa-cameras:orbbec-sdk-2.10.6 "$context"
