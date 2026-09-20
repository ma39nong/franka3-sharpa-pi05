"""The 25000 checkpoint has physical-order state and arm-first actions."""

import json
from pathlib import Path

from deploy.fr3_wuji_30000 import serve as base


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "checkpoints/25000"
DEFAULT_URI = "ws://127.0.0.1:8004"


def checked_contract(checkpoint, *, asset_id, config_name):
    metadata, stats_file = base.checkpoint_contract(checkpoint)
    if stats_file.parent.name != asset_id:
        raise ValueError(f"Checkpoint requires its {asset_id} normalization statistics")
    tree = json.loads((Path(metadata["checkpoint"]) / "params/_METADATA").read_text())["tree_metadata"]
    if any("lora" in key.lower() for key in tree):
        raise ValueError("Checkpoint requires full-finetune parameters without LoRA branches")
    metadata.update(
        config=config_name,
        adapter_revision=3,
        model_state_order=["left_arm_7", "left_hand_20", "right_arm_7", "right_hand_20", "padding_10"],
        model_finetune_mode="full",
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        source_hz=15,
    )
    return metadata, stats_file


def checkpoint_contract(checkpoint=DEFAULT_CHECKPOINT):
    return checked_contract(
        checkpoint, asset_id="tomato_AB_15hz", config_name="pi05_fr3_wuji_25000_full_64to54"
    )
