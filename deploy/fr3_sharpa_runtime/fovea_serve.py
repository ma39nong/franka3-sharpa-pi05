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
    args = parser.parse_args()
    from .fovea_contract import checkpoint_contract

    metadata = checkpoint_contract(args.checkpoint)
    if args.check_only:
        print(json.dumps(metadata, indent=2))
        return
    from .fovea_policy import load_policy
    policy = load_policy(args.checkpoint)
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
