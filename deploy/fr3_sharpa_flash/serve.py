"""Check, smoke or serve FLASH; imports the model only after metadata checks."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from deploy.fr3_sharpa_runtime.flash_contract import RESAMPLER
from deploy.fr3_sharpa_runtime.flash_contract import inspect_flash


def warmup_observation():
    anchor = 2_000_000_000
    grid = [anchor - round((7 - i) * 1e9 / 30) for i in range(8)]
    return {
        "observation/state": np.zeros(58, np.float32),
        "observation/history": np.zeros((8, 58), np.float32),
        **{
            key: np.zeros((224, 224, 3), np.uint8)
            for key in ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image")
        },
        "prompt": "Synthetic warmup only; never execute.",
        "flash_context": {
            "protocol_version": 1,
            "request_id": "synthetic-warmup",
            "session_id": "warmup-only",
            "observation_anchor_source_ns": anchor,
            "history_grid_source_ns": grid,
            "history_source_ns": [[stamp] * 4 for stamp in grid],
            "captured_monotonic_ns": time.monotonic_ns(),
            "history_resampler_version": RESAMPLER,
        },
    }


def serve(policy, metadata, port):
    from websockets.sync.server import serve as websocket_serve

    from . import codec

    def handler(ws):
        ws.send(codec.packb(metadata))
        for message in ws:
            try:
                result = policy.infer(codec.unpackb(message))
                ws.send(codec.packb(result))
            except Exception as error:
                ws.send(f"FLASH inference rejected: {type(error).__name__}: {error}")

    with websocket_serve(handler, "127.0.0.1", port, compression=None, max_size=None) as server:
        server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check-only", action="store_true")
    group.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    checkpoint = os.environ.get("SHARPA_FLASH_CHECKPOINT")
    repo = os.environ.get("SHARPA_FLASH_REPO")
    if not checkpoint or not repo:
        raise ValueError("Set SHARPA_FLASH_CHECKPOINT and SHARPA_FLASH_REPO")
    metadata = inspect_flash(checkpoint, norm_stats=os.environ.get("SHARPA_FLASH_NORM_STATS"), backend=repo)
    if args.check_only:
        print(json.dumps(metadata, indent=2))
        return
    # Check exact import origin before loading any model. The deployment src/
    # must never be added to this interpreter's PYTHONPATH.
    import openpi

    if not Path(openpi.__file__).resolve().is_relative_to(Path(repo).resolve() / "src"):
        raise ValueError("Wrong openpi import origin for FLASH")
    from .policy import FlashPolicy

    policy = FlashPolicy(checkpoint, metadata)
    start = time.monotonic()
    policy.infer(warmup_observation())
    metadata.update(warmed_up=True, compilation_warmup_s=time.monotonic() - start)
    print(json.dumps(metadata), flush=True)
    if not args.smoke_only:
        serve(policy, metadata, int(os.environ.get("SHARPA_FLASH_PORT", "18768")))


if __name__ == "__main__":
    main()
