from pathlib import Path

import numpy as np
import pytest

from deploy.fr3_wuji_20hz_slow.contract import checkpoint_contract
from deploy.fr3_wuji_20hz_slow.deploy import CONFIG
from deploy.fr3_wuji_20hz_slow.deploy import DEFAULT_CHECKPOINT
from deploy.fr3_wuji_20hz_slow.deploy import parse_args
from deploy.fr3_wuji_slow.planner import build_plan
from deploy.fr3_wuji_slow.planner import read_config


def inputs(config):
    actions = np.zeros((50, 54))
    actions[:, 0] = np.arange(50) * 0.001
    start = actions[0].copy()
    limits = {
        "lower": np.full(54, -3.0),
        "upper": np.full(54, 3.0),
        "speed": np.full(54, 10.0),
        "sources": {},
    }
    names = [f"joint_{index}" for index in range(54)]
    return actions, start, config, limits, names


def test_profile_is_20hz_and_has_20hz_checkpoint_default():
    config = read_config(CONFIG)
    args = parse_args([])
    assert config["source_hz"] == 20
    assert args.checkpoint == DEFAULT_CHECKPOINT
    assert "tomato_lora_0918_20hz/19999" in str(args.checkpoint)


def test_20hz_timing_and_diagnostics_use_profile_rate():
    config = read_config(CONFIG)
    plan = build_plan(*inputs(config))
    assert plan.report["source_hz"] == 20
    assert plan.report["groups"]["left_arm"]["raw_velocity_max_rad_s"] == pytest.approx(0.02)
    assert plan.playback.duration == pytest.approx((49 / 20) * 2.5, rel=1e-7)


def test_30hz_profile_timing_is_unchanged():
    config = read_config(Path(__file__).parents[1] / "fr3_wuji_slow/config.yaml")
    plan = build_plan(*inputs(config))
    assert plan.report["source_hz"] == 30
    assert plan.report["groups"]["left_arm"]["raw_velocity_max_rad_s"] == pytest.approx(0.03)
    assert plan.playback.duration / plan.playback.scale == pytest.approx(49 / 30)
    assert plan.playback.scale >= 2.5


def test_checkpoint_contract_accepts_only_20hz_normalization():
    metadata, stats = checkpoint_contract(DEFAULT_CHECKPOINT)
    assert metadata["action_horizon"] == 50
    assert metadata["action_dim"] == 54
    assert stats.relative_to(DEFAULT_CHECKPOINT / "assets").as_posix() == "fr3_wuji/0918_20hz/norm_stats.json"

    with pytest.raises(ValueError, match="20 Hz slow entry"):
        checkpoint_contract(Path(__file__).parents[2] / "checkpoints/19999_269/19999")


@pytest.mark.parametrize("source_hz", [15, 25, 60, None])
def test_shared_slow_planner_rejects_unprofiled_rates(source_hz):
    config = read_config(CONFIG)
    config["source_hz"] = source_hz
    with pytest.raises(ValueError, match="explicit 20 Hz or 30 Hz"):
        build_plan(*inputs(config))
