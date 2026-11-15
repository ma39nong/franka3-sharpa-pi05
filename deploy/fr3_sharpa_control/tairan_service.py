"""JSONL service for the pinned Tairan Joint+1s handoff. No ROS or hardware."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--pretrained-dir", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.repository))
    import numpy as np
    from tools.verify_tairan_deployment import verify
    from tools import tomato_phase_model as model

    inventory = a.repository / "configs/tairan_deployment.json"
    receipt = verify(inventory, run_dir=a.checkpoint.parent, encoder_dir=a.pretrained_dir, repo=a.repository)
    saved, head = model.load_checkpoint(a.checkpoint, device="cuda")
    encoder, provenance = model.load_encoder(a.pretrained_dir, "cuda", saved["contract"]["seed"])
    if provenance != saved["pretrained"]:
        raise ValueError("Encoder identity mismatch")
    version = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    protocol = "tomato-stage-prediction-v1"

    def ready():
        return dict(
            protocol=protocol,
            status="ready",
            model_version=version,
            provider="tairan",
            future_horizon_s=1.0,
            verified_artifacts=receipt["verified_artifacts"],
        )

    print(json.dumps(ready()), flush=True)
    for line in sys.stdin:
        req = {}
        try:
            req = json.loads(line)
            if req.get("operation") == "health":
                reply = ready()
            else:
                started = time.time()
                timestamp = req["timestamp"]
                if (
                    req["protocol"] != protocol
                    or not isinstance(req["observation_id"], str)
                    or not req["observation_id"]
                    or not isinstance(req["session_id"], str)
                    or not req["session_id"]
                    or not np.isfinite(timestamp)
                    or not 0 <= started - timestamp <= 2
                ):
                    raise ValueError("Invalid request identity/time")
                with np.load(req["observation_npz"], allow_pickle=False) as data:
                    names = data["joint_names"].tolist()
                    q = np.asarray(data["joints"], np.float32)
                    rgb = np.asarray(data["rgb"])
                if (
                    len(names) != 58
                    or len(set(names)) != 58
                    or set(names) != set(saved["joint_names"])
                    or q.shape != (16, 58)
                ):
                    raise ValueError("58 named measured joints required")
                q = q[:, [names.index(n) for n in saved["joint_names"]]]
                feature = model.fuse_features(encoder, rgb[None], q[None])
                logits = model.predict_heads(head, feature, saved["mean"], saved["std"])
                reply = dict(
                    protocol=protocol,
                    provider="tairan",
                    model_version=version,
                    observation_id=req["observation_id"],
                    session_id=req["session_id"],
                    input_timestamp=timestamp,
                    labels=["pickup", "harvest", "return", "adjust"],
                    future_horizon_s=1.0,
                    prediction_mode="joint",
                    valid=True,
                    error=None,
                )
                for h in ("current", "future"):
                    z = logits[h + "_logits"][0].astype(float)
                    prob = np.exp(z - z.max())
                    prob /= prob.sum()
                    reply[h + "_logits"] = z.tolist()
                    reply[h + "_probabilities"] = prob.tolist()
                reply["result_timestamp"] = time.time()
                reply["latency_ms"] = (reply["result_timestamp"] - started) * 1000
        except Exception as error:
            reply = dict(
                protocol=protocol,
                provider="tairan",
                valid=False,
                error=str(error),
                observation_id=req.get("observation_id"),
                session_id=req.get("session_id"),
            )
        print(json.dumps(reply, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
