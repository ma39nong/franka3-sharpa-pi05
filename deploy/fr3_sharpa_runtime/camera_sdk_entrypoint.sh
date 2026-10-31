#!/usr/bin/env bash
set -e

# Apply the SDK override AFTER the existing entrypoint has sourced ROS overlays.
# This keeps the patched host driver and its timestamp logger, while ensuring
# that both host and vendor drivers resolve the same SDK and matching plugins.
exec /entrypoint.sh bash -e -c '
  export LD_LIBRARY_PATH="/opt/openpi-orbbec-sdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  python3 - <<\PY
import ctypes
sdk = ctypes.CDLL("libOrbbecSDK.so.2")
version = tuple(getattr(sdk, "ob_get_" + part + "_version")() for part in ("major", "minor", "patch"))
if version != (2, 10, 6):
    raise RuntimeError(f"Unexpected camera SDK: {version}; expected 2.10.6")
print("[camera-sdk] Orbbec SDK 2.10.6; existing ROS driver retained", flush=True)
PY
  exec "$@"
' camera-sdk "$@"
