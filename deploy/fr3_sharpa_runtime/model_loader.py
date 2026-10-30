"""Shared model-loading boundary; native 58D full and LoRA profiles share the runtime.

The inference adapter accepts already-canonical live positions. Converter
116D storage is reduced/reordered during training, not applied a second time
at deployment. This is an inference config, not a recovered training manifest.
"""
from __future__ import annotations

import hashlib
import json

from . import config
from .model_profiles import validate_parameter_layout


def checkpoint_contract():
    checkpoint = config.require_checkpoint()
    manifest = checkpoint / "params/manifest.ocdbt"
    tree_path = checkpoint / "params/_METADATA"
    if not manifest.is_file() or not tree_path.is_file():
        raise ValueError("Missing complete Orbax params metadata/manifest")
    entries = json.loads(tree_path.read_text())["tree_metadata"]
    validate_parameter_layout(entries, config.MODEL_PROFILE)
    projection_shapes = {}
    for entry in entries.values():
        names = [part["key"] for part in entry["key_metadata"]]
        if "action_in_proj" in names and "kernel" in names:
            projection_shapes["action_in_proj"] = entry["value_metadata"]["write_shape"]
        if "action_out_proj" in names and "bias" in names:
            projection_shapes["action_out_bias"] = entry["value_metadata"]["write_shape"]
    # write_shape can be a shard shape: expert width is not inferred from it.
    if (projection_shapes.get("action_in_proj", [None])[0] != config.ACTION_DIM
            or projection_shapes.get("action_out_bias") != [config.ACTION_DIM]):
        raise ValueError(f"Expected native 58D model projections, got {projection_shapes}")
    from .paper_contract import inspect_paper, validate_stats
    algorithm = inspect_paper(config.NORM_STATS, entries)
    if algorithm and config.MODEL_PROFILE.name != "full-delta":
        raise ValueError("Paper delta权重必须使用full-delta模型配置，不能作为absolute/LoRA加载")
    stats = json.loads(config.NORM_STATS.read_text())["norm_stats"]
    validate_stats(stats, paper_algorithm=algorithm)
    return {
        "paper_algorithm": algorithm,
        "config_name": config.CONFIG_NAME,
        "checkpoint": str(checkpoint), "norm_stats": str(config.NORM_STATS),
        "checkpoint_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "normalization_sha256": hashlib.sha256(config.NORM_STATS.read_bytes()).hexdigest(),
        "model_profile": config.MODEL_PROFILE.name,
        "model_finetune_mode": config.MODEL_PROFILE.finetune_mode,
        "training_action_supervision": config.MODEL_PROFILE.training_action_supervision,
        "delta_reference": "chunk_start_measured_joint_positions" if config.MODEL_PROFILE.extra_delta_transform else None,
        "delta_dimensions": list(range(7)) + list(range(29, 36)) if config.MODEL_PROFILE.extra_delta_transform else [],
        "paligemma_variant": config.MODEL_PROFILE.paligemma_variant,
        "action_expert_variant": config.MODEL_PROFILE.action_expert_variant,
        "lora_recipe": ({"vlm_rank": 16, "vlm_alpha": 16, "expert_rank": 32, "expert_alpha": 32,
                         "rank_source": "configured recipe; global shapes checked at model load"}
                        if config.MODEL_PROFILE.name == "lora" else None), "action_dim": config.ACTION_DIM,
        "action_horizon": config.ACTION_HORIZON, "action_hz": config.ACTION_HZ,
        "action_representation": "absolute_joint_position",
        "action_order": config.ACTION_ORDER,
        "camera_mapping": config.CAMERA_MAPPING,
        "params_leaves": len(entries), "warmed_up": False,
    }


def build_config():
    from openpi.models.pi0_config import Pi0Config
    from openpi.training.config import AssetsConfig, LeRobotFr3SharpaDataConfig, TrainConfig

    # Reuse the canonical 58D runtime input adapter. The training converter's
    # action layout (LA, RA, LH, RH) was already reordered before normalization.
    # For delta profiles the existing output transform restores only the arms
    # against this observation's measured state, after action unnormalization.
    model = Pi0Config(pi05=True, dtype="bfloat16", action_dim=58,
                      action_horizon=50, max_token_len=256,
                      paligemma_variant=config.MODEL_PROFILE.paligemma_variant,
                      action_expert_variant=config.MODEL_PROFILE.action_expert_variant)
    metadata = checkpoint_contract()
    data_class = LeRobotFr3SharpaDataConfig
    if metadata["paper_algorithm"]:
        from .paper_model import PaperPi0Config, PaperSharpaData
        from dataclasses import fields
        algorithm = metadata["paper_algorithm"]
        model = PaperPi0Config(**{field.name: getattr(model, field.name) for field in fields(Pi0Config)},
            paper_asset_dir=str(config.NORM_STATS.parent), paper_mode=algorithm["mode"],
            paper_beta=algorithm["beta"], paper_std_floor=algorithm["std_floor"],
            paper_asset_sha256=algorithm["bundle_sha256"])
        data_class = PaperSharpaData
    data = data_class(
        repo_id=config.NORM_STATS.parent.name,
        assets=AssetsConfig(assets_dir=str(config.NORM_STATS.parent.parent),
                            asset_id=config.NORM_STATS.parent.name),
        extra_delta_transform=config.MODEL_PROFILE.extra_delta_transform, action_hz=30.0)
    return TrainConfig(name=config.CONFIG_NAME, model=model, data=data,
                       freeze_filter=model.get_freeze_filter())
