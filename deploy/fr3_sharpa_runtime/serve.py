"""Serve/check/smoke-test the selected native 58D absolute checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import config
from .timing import TimedPolicy, timed_server_class


def main() -> None:
    parser = argparse.ArgumentParser(epilog="Shell launcher options: --model {full,lora}, --list-models")
    parser.add_argument("--port", type=int, default=config.POLICY_PORT)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-only", action="store_true", help="Check metadata/statistics without loading weights")
    modes.add_argument("--smoke-only", action="store_true", help="Load GPU model, infer dummy observations and exit; no hardware")
    parser.add_argument("--report", type=Path, help="Save checkpoint/smoke validation as JSON")
    args = parser.parse_args()
    from .model_loader import checkpoint_contract, build_config

    metadata = checkpoint_contract()
    if args.check_only:
        print(json.dumps(metadata, indent=2))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(metadata, indent=2) + "\n")
        return
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.shared import normalize
    import jax

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("GPU required for model loading/smoke test")
    training = build_config()
    config.validate_training_contract(training)
    stats = normalize.load(config.NORM_STATS.parent)
    print(f"Loading {config.MODEL_PROFILE.name} 58D absolute model", flush=True)
    policy = policy_config.create_trained_policy(
        training, config.CHECKPOINT, norm_stats=stats, default_prompt=config.DEFAULT_PROMPT)
    observation = {"observation/state": np.zeros(58, dtype=np.float32), "prompt": config.DEFAULT_PROMPT}
    for key in ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image"):
        observation[key] = np.zeros((224, 224, 3), dtype=np.uint8)
    result = policy.infer(observation)
    from .protocol import as_finite_action_chunk
    actions = as_finite_action_chunk(result["actions"])
    metadata.update(warmed_up=True, smoke_action_shape=list(actions.shape), smoke_finite=True)
    print(f"Warmup OK: actions=(50, 58), {config.MODEL_PROFILE.name}, absolute, 30 Hz", flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(metadata, indent=2) + "\n")
    if args.smoke_only:
        return
    timed_server_class(WebsocketPolicyServer)(TimedPolicy(policy), host="127.0.0.1", port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
