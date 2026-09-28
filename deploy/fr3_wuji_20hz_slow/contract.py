"""Bind the 20 Hz slow entry to a checkpoint-owned 20 Hz normalization."""

from pathlib import Path

from deploy.fr3_wuji_models.model_19999.serve import checkpoint_contract as base_contract

EXPECTED_STATS = Path("fr3_wuji/0918_20hz/norm_stats.json")


def checkpoint_contract(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    metadata, stats = base_contract(checkpoint)
    relative_stats = stats.relative_to(checkpoint / "assets")
    if relative_stats != EXPECTED_STATS:
        raise ValueError(
            "The 20 Hz slow entry requires checkpoint normalization at "
            f"assets/{EXPECTED_STATS}; got assets/{relative_stats}"
        )
    return metadata, stats
