"""Serve/check/smoke-test the selected 58D checkpoint, serving absolute joint targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import config
from .timing import TimedPolicy, timed_server_class


def main() -> None:
    parser = argparse.ArgumentParser(epilog="Shell launcher options: --model {full,lora,full-delta}, --list-models")
    parser.add_argument("--port", type=int, default=config.POLICY_PORT)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-only", action="store_true", help="Check metadata/statistics without loading weights")
    modes.add_argument("--smoke-only", action="store_true", help="Load GPU model, infer dummy observations and exit; no hardware")
    parser.add_argument("--report", type=Path, help="Save checkpoint/smoke validation as JSON")
    parser.add_argument("--enable-rtc", action="store_true", help="Warm and enable Sharpa 58D JAX soft-prefix guidance")
    parser.add_argument("--enable-paper-rtc", action="store_true", help="Enable Paper RTC with time-indexed short references")
    args = parser.parse_args()
    if args.enable_paper_rtc:
        args.enable_rtc = True
    if args.enable_rtc and config.MODEL_PROFILE.finetune_mode != "full":
        parser.error("RTC currently supports full/full-delta checkpoints only")
    from .model_loader import checkpoint_contract, build_config

    metadata = checkpoint_contract()
    if args.enable_paper_rtc and not metadata["paper_algorithm"]:
        parser.error("Paper RTC requires a complete matching Paper delta checkpoint")
    if args.enable_rtc and metadata["paper_algorithm"] and not args.enable_paper_rtc:
        parser.error("Paper checkpoints require --enable-paper-rtc; ordinary RTC does not support their time-indexed statistics")
    if args.enable_rtc:
        from openpi_client import rtc
        metadata.update(rtc_supported=True, rtc_warmed_up=False, rtc_action_dim=58,
                        rtc_protocol_version=rtc.PROTOCOL_VERSION, rtc_algorithm="soft_prefix_guidance")
        if args.enable_paper_rtc:
            from .rtc_stream import PAPER_RTC_ADAPTER
            metadata.update(rtc_input_adapter=PAPER_RTC_ADAPTER, rtc_short_prefix_warmed_up=False)
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
    policy_options = {}
    if args.enable_paper_rtc:
        from .paper_rtc import PaperRTCPolicy
        policy_options["policy_class"] = PaperRTCPolicy
    policy = policy_config.create_trained_policy(
        training, config.CHECKPOINT, norm_stats=stats, default_prompt=config.DEFAULT_PROMPT,
        rtc_action_repack=canonical_prefix if args.enable_rtc else None, **policy_options)
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
        if args.enable_paper_rtc:
            for length in (35, 1):
                short_context = rtc.validate_context({**context.to_dict(),
                    "token": f"paper-rtc-warmup-{length}", "previous_actions": actions[:length],
                    "inference_delay_steps": min(5, length), "prefix_attention_horizon": min(25, length)})
                short_result = policy.infer(observation, rtc_context=short_context)
                as_finite_action_chunk(short_result["actions"])
                rtc.validate_receipt(short_result.get("rtc"), short_context)
            metadata.update(rtc_short_prefix_warmed_up=True, rtc_smoke_prefix_lengths=[50, 35, 1])
            print("Paper RTC short-prefix Warmup OK: K=50/35/1 -> actions=(50, 58)", flush=True)
        metadata.update(rtc_warmed_up=True, rtc_smoke_action_shape=[50, 58],
                        rtc_warm_infer_ms=guided["policy_timing"]["infer_ms"])
        print(f"RTC guided Warmup OK: 58D, {metadata['rtc_warm_infer_ms']:.1f} ms warmed model sampling", flush=True)
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
