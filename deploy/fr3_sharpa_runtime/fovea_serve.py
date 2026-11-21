"""Separate Fovea server entry; existing locator/RTC server defaults stay unchanged."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--enable-async", action="store_true")
    args = parser.parse_args()
    from .fovea_contract import checkpoint_contract

    metadata = checkpoint_contract(args.checkpoint)
    if args.check_only:
        print(json.dumps(metadata, indent=2))
        return
    from .fovea_policy import load_policy
    policy = load_policy(args.checkpoint, enable_async=args.enable_async)
    # Compile with dummy read-only inputs before accepting any executable query.
    import numpy as np

    from openpi.subtasks import fovea

    obs = {
        "observation/state": np.zeros(58, np.float32),
        "observation/image": np.zeros((400, 640, 3), np.uint8),
        "observation/left_wrist_image": np.zeros((480, 640, 3), np.uint8),
        "observation/right_wrist_image": np.zeros((480, 640, 3), np.uint8),
        "prompt": "Right hand grasps and lifts the truss to the center",
        "_fovea": {
            "session_id": "warmup",
            "query_index": 0,
            "executed_steps": 0,
            "current_box": list(fovea.INITIAL_BOX),
        },
    }
    result = policy.infer(obs)
    if args.enable_async:
        from openpi_client import rtc
        from .rtc_stream import RTCTimeline
        async_obs = {**obs, "_fovea": {k: v for k, v in obs["_fovea"].items() if k != "executed_steps"}}
        async_obs["_fovea"].update(execution_mode="fovea-observation-clock-v1",
            nominal_observation_steps=15, source_monotonic_s=0.0, used_target_monotonic_s=None)
        timeline = RTCTimeline(result["actions"], 0.0, 1.0)
        for phase in (15.0, 40.0):
            context = timeline.snapshot(phase / 30, delay_steps=5, schedule="exp", weight=5.0)
            policy.infer(async_obs, rtc_context=context)
            timeline.pending = None
        metadata.update(fovea_async_protocol="fovea-observation-clock-v1",
                        rtc_supported=True, rtc_warmed_up=True,
                        rtc_protocol_version=rtc.PROTOCOL_VERSION, rtc_action_dim=58,
                        rtc_algorithm="soft_prefix_guidance")
    metadata.update(warmed_up=True, smoke_action_shape=list(result["actions"].shape))
    print(json.dumps(metadata, indent=2), flush=True)
    if not args.smoke_only:
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer

        try:
            WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()
        except KeyboardInterrupt:
            print("Fovea推理服务已停止", flush=True)


if __name__ == "__main__":
    main()
