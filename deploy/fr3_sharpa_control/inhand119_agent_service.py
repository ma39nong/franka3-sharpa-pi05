"""Four-view V-JEPA 2 inference for the read-only 119-episode observer."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from deploy.fr3_sharpa_runtime.agent_history import JOINT_NAMES

PROTOCOL = "tomato-stage-prediction-v1"


def model_inputs(path, joint_names, cropper, anchor_ns, session):
    with np.load(path, allow_pickle=False) as data:
        rgb, state = data["rgb"], data["joints"]
        native, stamps = data["native_cam0"], data["cam0_stamp_ns"]
        names = tuple(data["joint_names"].tolist())
    if (rgb.shape != (3, 16, 256, 256, 3) or rgb.dtype != np.uint8
            or native.ndim != 4 or native.shape[0] != 16 or native.shape[-1] != 3
            or native.dtype != np.uint8 or stamps.shape != (16,) or stamps.dtype.kind != "i"
            or stamps[0] <= 0 or np.any(np.diff(stamps) <= 0) or stamps[-1] > anchor_ns
            or names != JOINT_NAMES or state.shape != (16, 58) or not np.isfinite(state).all()):
        raise ValueError("119 条四路观测历史格式无效")
    local = []
    fallback = 0
    cropper.reset()
    for image, joints, stamp in zip(native, state, stamps, strict=True):
        observation = cropper.apply({
            "observation/image": image,
            "observation/state": joints,
            "_crop_context": {"source_monotonic_ns": int(stamp), "session": session},
        })
        local.append(cv2.resize(observation["observation/interaction_image"], (256, 256),
                                interpolation=cv2.INTER_AREA))
        fallback += not cropper.last["local_roi_valid"]
    views = np.concatenate((rgb, np.stack(local)[None]), axis=0)
    return views[None], state[None, :, [names.index(name) for name in joint_names]], fallback


def probabilities(logits):
    exp = np.exp(logits.astype(np.float64) - np.max(logits))
    return (exp / exp.sum()).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("repository", "checkpoint", "pretrained-dir", "locator", "norm-stats"):
        parser.add_argument("--" + option, required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    sys.path.insert(0, str(args.repository))
    import torch
    from tools import tomato_phase_model as model
    from tools import tomato_dataset as ds
    from deploy.fr3_sharpa_runtime.crop_vendor.repair import RgbLocator
    from deploy.fr3_sharpa_runtime.interaction_crop import InteractionCropper, locator_identity
    from deploy.fr3_sharpa_runtime.tomato_model import IN_HAND_STATS_SHA256

    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    manifest = args.checkpoint.with_name("manifest.json")
    saved, head = model.load_checkpoint(
        args.checkpoint, ds.digest(manifest), args.device, json.loads(manifest.read_text())["split_sha256"],
    )
    joint_names = saved["joint_names"]
    if (tuple(saved["contract"]["cameras"]) != ds.FOUR_CAMERAS
            or saved["prediction_mode"] != "joint" or saved["future_horizon_s"] != 1.0
            or len(joint_names) != len(JOINT_NAMES) or set(joint_names) != set(JOINT_NAMES)):
        raise ValueError("权重必须匹配 119 条四路、58 关节、当前与 +1 秒预测")
    encoder, provenance = model.load_encoder(args.pretrained_dir, args.device, saved["contract"]["seed"])
    if provenance != saved["pretrained"]:
        raise ValueError("编码器与训练权重的来源不匹配")
    norm = args.norm_stats.read_bytes()
    if hashlib.sha256(norm).hexdigest() != IN_HAND_STATS_SHA256:
        raise ValueError("119 条 ROI 归一化与已部署版本不匹配")
    stats = json.loads(norm)["norm_stats"]["state"]
    locator, locator_sha = locator_identity(args.locator)
    cropper = InteractionCropper(
        RgbLocator(locator, device=args.device),
        np.asarray(stats["q99"]) - np.asarray(stats["q01"]),
        roi_profile="in_hand119_right_truss_v1",
    )
    cropper.locator_sha256 = locator_sha
    identity = dict(protocol=PROTOCOL, model_version=ds.digest(args.checkpoint),
                    prediction_mode="joint", labels=saved["label_order"],
                    future_horizon_s=1.0, control_eligible=False)
    print(json.dumps(dict(identity, status="ready")), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        start = time.perf_counter()
        packet = dict(identity, observation_id=request["observation_id"], session_id=request["session_id"],
                      input_timestamp=request["timestamp"])
        try:
            if request["protocol"] != PROTOCOL:
                raise ValueError("未知观察协议")
            views, state, fallback = model_inputs(
                request["observation_npz"], joint_names, cropper, int(request["observation_id"]),
                request["session_id"],
            )
            features = model.fuse_features(encoder, views, state)
            logits = model.predict_heads(head, features, saved["mean"], saved["std"])
            for name in ("current", "future"):
                scores = logits[name + "_logits"][0]
                packet[name + "_probabilities"] = probabilities(scores)
            packet.update(valid=True, roi_fallback_frames=fallback)
        except (ValueError, OSError, KeyError) as error:
            packet.update(valid=False, error=str(error))
        packet.update(result_timestamp=time.time(), latency_ms=(time.perf_counter() - start) * 1000)
        print(json.dumps(packet, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
