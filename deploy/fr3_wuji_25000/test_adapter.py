"""CPU-only 25000 identity, ordering and timebase checks."""

import json

import numpy as np
import pytest

from deploy.fr3_wuji_25000 import serve
from deploy.fr3_wuji_25000.contract import DEFAULT_CHECKPOINT, DEFAULT_URI, checkpoint_contract
from deploy.fr3_wuji_30000 import serve as old_service
from deploy.fr3_wuji_medium.deploy import parse_args
from deploy.fr3_wuji_medium.planner import build_plan
from deploy.fr3_wuji_medium.profile import planning_config, slow_motion_config


def test_checkpoint_identity_is_full_64d_with_separate_state_and_action_orders():
    metadata, stats_file = checkpoint_contract()
    assert metadata["checkpoint"] == str(DEFAULT_CHECKPOINT.resolve())
    assert metadata["model_action_dim"] == 64
    assert metadata["action_dim"] == 54
    assert metadata["model_finetune_mode"] == "full"
    assert metadata["model_state_order"][:4] == [
        "left_arm_7", "left_hand_20", "right_arm_7", "right_hand_20"
    ]
    assert metadata["model_action_order"][:4] == [
        "left_arm_7", "right_arm_7", "left_hand_20", "right_hand_20"
    ]
    assert metadata["source_hz"] == 15
    assert stats_file.parent.name == "tomato_AB_15hz"
    stats = json.loads(stats_file.read_text())["norm_stats"]
    state = np.asarray(stats["state"]["mean"])
    actions = np.asarray(stats["actions"]["mean"])
    action_to_state = np.r_[0:7, 14:34, 7:14, 34:54]
    assert np.mean(np.abs(state[:54] - actions[action_to_state])) < .05
    assert np.mean(np.abs(state[:54] - actions[:54])) > .3
    assert not any("lora" in key.lower() for key in
                   json.loads((DEFAULT_CHECKPOINT / "params/_METADATA").read_text())["tree_metadata"])
    old, _ = old_service.checkpoint_contract(DEFAULT_CHECKPOINT)
    assert old["model_state_order"] != metadata["model_state_order"]
    assert old_service.ModelOrderedState is not serve.PhysicalOrderedState


def test_physical_state_is_only_padded_while_absolute_actions_are_reordered():
    from openpi import transforms
    from openpi.training import checkpoints, config

    original = config.get_config("pi05_fr3_wuji")
    model = old_service.full_finetune_model(original.model)
    adapted = serve.AdaptData(original.data).create(original.assets_dirs, model)
    physical = np.arange(54, dtype=np.float32) / 100
    observation = {"observation/state": physical, "prompt": old_service.PROMPT}
    for name in ("image", "left_wrist_image", "right_wrist_image"):
        observation["observation/" + name] = np.zeros((224, 224, 3), np.uint8)
    data = transforms.compose(adapted.data_transforms.inputs)(observation)
    np.testing.assert_array_equal(data["state"], np.pad(physical, (0, 10)))
    stats_file = checkpoint_contract()[1]
    stats = checkpoints.load_norm_stats(
        DEFAULT_CHECKPOINT / "assets", stats_file.parent.name
    )
    normalized = transforms.Normalize(stats, use_quantiles=adapted.use_quantile_norm)(data)
    tokens = transforms.compose(adapted.model_transforms.inputs)(normalized)
    assert tokens["state"].shape == (64,)
    assert np.isfinite(tokens["state"]).all()
    model_actions = np.tile(np.r_[physical[:7], physical[27:34], physical[7:27],
                                  physical[34:54], np.zeros(10)], (50, 1))
    model_actions[:, 54:] = 99
    result = transforms.compose(adapted.data_transforms.outputs)(
        {"state": np.full(64, 1000.), "actions": model_actions}
    )
    np.testing.assert_array_equal(result["actions"], np.tile(physical, (50, 1)))
    assert original.data.extra_delta_transform
    assert original.model.action_dim == 54


def test_invalid_state_and_actions_are_rejected():
    for state in (np.zeros(64), np.full(54, np.nan)):
        with pytest.raises(ValueError):
            serve.PhysicalOrderedState()({"state": state})
    for actions in (np.zeros((50, 54)), np.zeros((1, 64)), np.full((50, 64), np.nan)):
        with pytest.raises(ValueError):
            old_service.PhysicalAbsoluteActions()({"actions": actions})


def test_medium_profile_uses_15hz_without_changing_existing_30hz_profiles():
    config = planning_config("25000")
    assert config["source_hz"] == 15
    assert config["arm_raw_initial_delta_rad"] == 1.5
    assert slow_motion_config(config)["source_hz"] == 15
    assert all(planning_config(model)["source_hz"] == 30
               for model in ("19999", "30000", "30000v2"))
    with pytest.raises(ValueError, match="15 Hz"):
        from deploy.fr3_wuji_medium.planner import validate_config
        validate_config(dict(config, source_hz=30))

    start = np.zeros(54)
    raw = np.tile(start, (50, 1))
    raw[:, 0] = np.linspace(0, .001, 50)
    limits = {"lower": np.full(54, -2.), "upper": np.full(54, 2.),
              "speed": np.full(54, 2.)}
    names = [str(index) for index in range(54)]
    slow_knot_rate = build_plan(raw, start, dict(config, hand_control="slider", execution_steps=20),
                                limits.copy(), names)
    normal_knot_rate = build_plan(raw, start,
                                  dict(planning_config("30000"), hand_control="slider", execution_steps=20),
                                  limits.copy(), names)
    assert slow_knot_rate.report["source_hz"] == 15
    assert normal_knot_rate.report["source_hz"] == 30
    assert slow_knot_rate.playback.duration == pytest.approx(2 * normal_knot_rate.playback.duration)


def test_parallel_medium_entry_defaults():
    args = parse_args(["--model", "25000"])
    assert args.checkpoint == DEFAULT_CHECKPOINT
    assert args.uri == DEFAULT_URI
    assert args.minimum_time_scale == .625
    assert args.slow_after_seconds == 25.
