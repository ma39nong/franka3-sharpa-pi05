"""Checkpoint identity for the 30000v2 full-finetune deployment."""

from pathlib import Path

from deploy.fr3_wuji_30000 import serve as base


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "checkpoints/30000v2"
DEFAULT_URI = "ws://127.0.0.1:8003"
BASE_CHECKPOINT_CONTRACT = base.checkpoint_contract


def checkpoint_contract(checkpoint=DEFAULT_CHECKPOINT):
    metadata, stats_file = BASE_CHECKPOINT_CONTRACT(checkpoint)
    metadata.update(
        config="pi05_fr3_wuji_30000v2_full_64to54",
        model_finetune_mode="full",
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
    )
    return metadata, stats_file
