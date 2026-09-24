"""Serve a native 54D checkpoint trained on the 20 Hz 0918 trajectory grid."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PROMPT = (
    "Pick up a tomato truss with the right hand, then pick a cherry tomato with the left hand "
    "and place it in the left basket."
)


def checkpoint_contract(checkpoint):
    checkpoint = Path(checkpoint).expanduser().resolve()
    manifest = checkpoint / "params/manifest.ocdbt"
    stats = checkpoint / "assets/fr3_wuji/0918_20hz/norm_stats.json"
    if not manifest.is_file() or not stats.is_file():
        raise ValueError("Checkpoint must contain params and its fr3_wuji/0918_20hz normalization assets")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return {
        "config": "pi05_fr3_wuji_20hz",
        "action_dim": 54,
        "action_horizon": 50,
        "action_order": ["left_arm_7", "left_hand_20", "right_arm_7", "right_hand_20"],
        "action_representation": "absolute_joint_positions_after_arm_delta_transform",
        "source_hz": 20,
        "trajectory_seconds": 2.5,
        "hardware_output": False,
        "checkpoint": str(checkpoint),
        "checkpoint_manifest_sha256": digest(manifest),
        "normalization_sha256": digest(stats),
    }, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8006)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    metadata, stats_file = checkpoint_contract(args.checkpoint)
    if args.check_only:
        print(metadata)
        return

    import jax

    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import checkpoints
    from openpi.training import config

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("CUDA GPU required")
    checkpoint = Path(metadata["checkpoint"])
    norm_stats = checkpoints.load_norm_stats(checkpoint / "assets", "fr3_wuji/0918_20hz")
    policy = policy_config.create_trained_policy(
        config.get_config("pi05_fr3_wuji_20hz"), checkpoint, norm_stats=norm_stats, default_prompt=args.prompt
    )
    observation = {"observation/state": np.zeros(54, np.float32), "prompt": args.prompt}
    observation.update(
        {
            "observation/image": np.zeros((400, 640, 3), np.uint8),
            "observation/left_wrist_image": np.zeros((480, 640, 3), np.uint8),
            "observation/right_wrist_image": np.zeros((480, 640, 3), np.uint8),
        }
    )
    result = policy.infer(observation)
    print(f"Warmup OK: actions={result['actions'].shape}, source_hz=20", flush=True)
    policy.metadata.update(metadata, warmed_up=True)
    if args.smoke_only:
        return
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
