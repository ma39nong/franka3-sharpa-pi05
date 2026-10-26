"""Serve/check/smoke-test the selected 58D checkpoint, serving absolute joint targets."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import numpy as np

from . import config
from .timing import TimedPolicy, timed_server_class


def normalize_transform(stats):
    from openpi.transforms import Normalize
    return Normalize(stats, use_quantiles=True)


def main() -> None:
    parser = argparse.ArgumentParser(epilog="Shell launcher options: --model {full,lora,full-delta}, --list-models")
    parser.add_argument("--port", type=int, default=config.POLICY_PORT)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-only", action="store_true", help="Check metadata/statistics without loading weights")
    modes.add_argument("--smoke-only", action="store_true", help="Load GPU model, infer dummy observations and exit; no hardware")
    parser.add_argument("--report", type=Path, help="Save checkpoint/smoke validation as JSON")
    algorithm = parser.add_mutually_exclusive_group()
    algorithm.add_argument("--enable-rtc", action="store_true", help="Warm and enable Sharpa 58D JAX soft-prefix guidance")
    algorithm.add_argument("--enable-inference-experiments", action="store_true")
    parser.add_argument("--covariance-asset", type=Path,
                        help="Optional normalized 50x58 action covariance; never changes noise sampling")
    args = parser.parse_args()
    if args.enable_rtc and config.MODEL_PROFILE.finetune_mode != "full":
        parser.error("RTC currently supports full/full-delta checkpoints only")
    if args.covariance_asset and not args.enable_inference_experiments:
        parser.error("--covariance-asset requires --enable-inference-experiments")
    if args.enable_inference_experiments and config.MODEL_PROFILE.name not in ("full", "full-delta"):
        parser.error("Inference experiments support full and full-delta only")
    from .model_loader import checkpoint_contract, build_config

    metadata = checkpoint_contract()
    if args.enable_rtc:
        from openpi_client import rtc
        metadata.update(rtc_supported=True, rtc_warmed_up=False, rtc_action_dim=58,
                        rtc_protocol_version=rtc.PROTOCOL_VERSION, rtc_algorithm="soft_prefix_guidance")
    matrix = covariance_info = None
    if args.enable_inference_experiments:
        from .rolling import CAPABILITY
        metadata.update(inference_capabilities=[CAPABILITY], server_instance_id=uuid.uuid4().hex,
                        inpainting_modes=["off", "soft"] + (["correlation"] if args.covariance_asset else []))
        if args.covariance_asset:
            from .correlation import load_correction
            matrix, covariance_info = load_correction(args.covariance_asset, metadata)
            metadata["covariance_asset"] = covariance_info
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
    print(f"Loading {config.MODEL_PROFILE.name} 58D model ({config.MODEL_PROFILE.training_action_supervision} supervision; absolute output)", flush=True)
    from .rtc_stream import canonical_prefix
    policy = policy_config.create_trained_policy(
        training, config.CHECKPOINT, norm_stats=stats, default_prompt=config.DEFAULT_PROMPT,
        rtc_action_repack=canonical_prefix if args.enable_rtc else None)
    observation = {"observation/state": np.zeros(58, dtype=np.float32), "prompt": config.DEFAULT_PROMPT}
    for key in ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image"):
        observation[key] = np.zeros((224, 224, 3), dtype=np.uint8)
    result = policy.infer(observation)
    from .protocol import as_finite_action_chunk
    actions = as_finite_action_chunk(result["actions"])
    if args.enable_rtc:
        import hashlib
        context = rtc.validate_context({
            "protocol_version": rtc.PROTOCOL_VERSION, "token": "sharpa-rtc-warmup",
            "source_number": 1, "source_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
            "previous_actions": actions, "inference_delay_steps": 5,
            "prefix_attention_horizon": 25, "prefix_attention_schedule": "exp", "max_guidance_weight": 5.0})
        for _ in range(2):
            guided = policy.infer(observation, rtc_context=context)
            as_finite_action_chunk(guided["actions"])
            rtc.validate_receipt(guided.get("rtc"), context)
        metadata.update(rtc_warmed_up=True, rtc_smoke_action_shape=[50, 58],
                        rtc_warm_infer_ms=guided["policy_timing"]["infer_ms"])
        print(f"RTC guided Warmup OK: 58D, {metadata['rtc_warm_infer_ms']:.1f} ms warmed model sampling", flush=True)
    metadata.update(warmed_up=True, smoke_action_shape=list(actions.shape), smoke_finite=True)
    print(f"Warmup OK: actions=(50, 58), {config.MODEL_PROFILE.name}, absolute, 30 Hz", flush=True)
    served = TimedPolicy(policy)
    if args.enable_inference_experiments:
        from .model_service import ExperimentalPolicy
        from .rolling import CAPABILITY, CONTEXT_KEY, RollingOptions
        served = ExperimentalPolicy(served, metadata,
            normalizer=normalize_transform(stats), delta=config.MODEL_PROFILE.extra_delta_transform,
            matrix=matrix, covariance_info=covariance_info)
        # Warm both referenced sampler variants before motion-capable requests.
        from .rolling import model_identity
        import time
        options = RollingOptions(inpainting="correlation" if matrix is not None else "soft")
        warm = dict(observation)
        warm[CONTEXT_KEY] = {"version": CAPABILITY, "run_id": "warmup", "chunk_index": 1,
            "options": options.as_dict(), "alignment": "trajectory_phase_not_wall_clock",
            "reference": {"absolute_actions": actions[:4].tolist(), "run_id": "warmup", "chunk_index": 0,
                "source_indices": [26, 30], "consumed_model_steps": 26,
                "completed_monotonic_ns": time.monotonic_ns(), "model_identity": model_identity(metadata),
                "server_instance_id": metadata["server_instance_id"]}}
        warmed = served.infer(warm)
        as_finite_action_chunk(warmed["actions"])
        if matrix is not None:
            warm[CONTEXT_KEY]["options"]["inpainting"] = "soft"
            warm[CONTEXT_KEY]["reference"]["completed_monotonic_ns"] = time.monotonic_ns()
            as_finite_action_chunk(served.infer(warm)["actions"])
        metadata["inpainting_warmed_up"] = True
        print("Inference experiments ready: paper-rolling-v1 (independent noise)", flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(metadata, indent=2) + "\n")
    if args.smoke_only:
        return
    timed_server_class(WebsocketPolicyServer)(served, host="127.0.0.1", port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
