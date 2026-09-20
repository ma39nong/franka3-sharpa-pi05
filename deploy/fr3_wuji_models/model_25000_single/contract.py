"""The tomato_A_15hz full-finetune checkpoint identity."""

from pathlib import Path

from deploy.fr3_wuji_models.model_25000.contract import checked_contract

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = ROOT / "checkpoints/25000-single"
DEFAULT_URI = "ws://127.0.0.1:8005"


def checkpoint_contract(checkpoint=DEFAULT_CHECKPOINT):
    return checked_contract(
        checkpoint,
        asset_id="tomato_A_15hz",
        config_name="pi05_fr3_wuji_25000_single_full_64to54",
    )
