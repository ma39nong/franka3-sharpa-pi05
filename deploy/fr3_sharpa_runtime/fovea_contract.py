"""Lightweight Fovea checkpoint identity, shared by UI and inference."""
import hashlib
import json
from pathlib import Path

from openpi.subtasks import fovea


def checkpoint_contract(checkpoint):
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    record = json.loads((checkpoint / "assets/fovea_contract.json").read_text())
    contract = record["contract"]
    expected = {
        "version": fovea.VERSION,
        "image_keys": list(fovea.IMAGE_KEYS),
        "source_size": [640, 400],
        "action_dim": 58,
        "action_horizon": 50,
        "action_hz": 30,
        "execute_steps": fovea.EXECUTE_STEPS,
        "box_loss_weight": 0.1,
        "location_start": 256000,
        "bins": 1024,
        "initial_box": list(fovea.INITIAL_BOX),
        "memory_enabled": False,
        "crop_expansion": 1.5,
        "coordinate_order": "ymin_xmin_ymax_xmax",
        "source_coordinates": "normalized_xyxy",
        "token_coordinates": "padded_full224",
        "target_priors": {"left_basket": list(fovea.LEFT_BASKET), "right_plate": list(fovea.RIGHT_PLATE)},
    }
    if (
        contract != expected
        or not record.get("training_checkpoint")
        or record["action_semantics"] != "delta_arm_joint_target_absolute_hand"
    ):
        raise ValueError("Matching trained Fovea v1 checkpoint contract required")
    asset = record["norm_asset_id"]
    if not isinstance(asset, str) or Path(asset).name != asset or asset in (".", ".."):
        raise ValueError("Invalid checkpoint normalization asset")
    stats = checkpoint / "assets" / asset / "norm_stats.json"
    if hashlib.sha256(stats.read_bytes()).hexdigest() != record["norm_stats_sha256"]:
        raise ValueError("Checkpoint normalization identity changed")
    if not (checkpoint / "params/_METADATA").is_file():
        raise ValueError("Complete Orbax parameter checkpoint required")
    identity = hashlib.sha256(b"".join(
        hashlib.sha256(path.read_bytes()).digest() for path in (
            checkpoint / "params/manifest.ocdbt", checkpoint / "params/_METADATA",
            checkpoint / "assets/fovea_contract.json", stats,
        )
    )).hexdigest()
    return {
        "fovea_identity": identity,
        "foveation": contract,
        "checkpoint": str(checkpoint),
        "norm_stats": str(stats),
        "labels_manifest_sha256": record["labels_manifest_sha256"],
        "action_representation": "absolute_joint_position",
        "rtc_supported": False,
        "memory_enabled": False,
    }
