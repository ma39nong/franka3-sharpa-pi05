"""Standalone async-Soft model service; does not alter the three existing modes."""

import argparse
import json
from pathlib import Path
import time
import uuid

import numpy as np

from . import config
from .async_soft_client import default_port
from .async_soft_stream import CAPABILITY
from .async_soft_stream import CONTEXT_KEY
from .async_soft_stream import digest
from .model_loader import build_config
from .model_loader import checkpoint_contract
from .timing import timed_server_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=default_port())
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or config.MODEL_PROFILE.name not in ("full", "full-delta"):
        parser.error("Async Soft requires full/full-delta and a valid port")
    metadata = checkpoint_contract()
    metadata.update(
        async_soft_version=CAPABILITY,
        async_soft_warmed_up=False,
        server_instance_id=uuid.uuid4().hex,
        async_soft_port=args.port,
        async_soft_algorithm="fixed_50_30_26_plus_4_to_20_phase_soft",
        async_soft_execution_targets=20,
    )

    def report():
        print(json.dumps(metadata, indent=2), flush=True)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(metadata, indent=2) + "\n")

    if args.check_only:
        report()
        return
    import jax

    from openpi import transforms
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.shared import normalize

    from .async_soft_policy import AsyncSoftPolicy

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("GPU required for async Soft model smoke/serving")
    training = build_config()
    config.validate_training_contract(training)
    stats = normalize.load(config.NORM_STATS.parent)
    data = training.data.create(training.assets_dirs, training.model)
    loaded = policy_config.create_trained_policy(training, config.CHECKPOINT, norm_stats=stats)
    served = AsyncSoftPolicy(
        loaded,
        metadata,
        normalizer=transforms.Normalize(stats, use_quantiles=data.use_quantile_norm),
        delta=config.MODEL_PROFILE.extra_delta_transform,
    )
    observation = {"observation/state": np.zeros(58, np.float32), "prompt": config.DEFAULT_PROMPT}
    for key in ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image"):
        observation[key] = np.zeros((224, 224, 3), np.uint8)
    from .protocol import as_finite_action_chunk

    result = served.infer(observation)
    actions = as_finite_action_chunk(result["actions"])
    now = time.monotonic()
    # Long synthetic warmup window allows JIT compilation; no worker/device connected.
    context = {
        "version": CAPABILITY,
        "token": "warmup",
        "source_number": 1,
        "source_sha256": "0" * 64,
        "observation_monotonic_s": now,
        "snapshot_monotonic_s": now,
        "node_hz": 0.01,
        "reference_start": 0,
        "handover_monotonic_s": now + 10000,
        "reference_sha256": digest(actions[26:30]),
        "previous_actions": actions[26:30],
    }
    guided = served.infer({**observation, CONTEXT_KEY: context})
    as_finite_action_chunk(guided["actions"])
    metadata.update(
        warmed_up=True,
        async_soft_warmed_up=True,
        smoke_action_shape=[50, 58],
        async_soft_warmup=guided["async_soft_diagnostics"],
    )
    report()
    print("Async Soft Warmup OK: 50x58, 50->30->26+4->20, phase-aligned Soft, no RTC guidance", flush=True)
    if not args.smoke_only:
        timed_server_class(WebsocketPolicyServer)(
            served, host="127.0.0.1", port=args.port, metadata=metadata
        ).serve_forever()


if __name__ == "__main__":
    main()
