from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime import config
from deploy.fr3_sharpa_runtime.action_adapter import SharpaActionAdapter
from deploy.fr3_sharpa_runtime.protocol import DT_NS, FPS


def test_deployment_timeline_and_records_are_30hz():
    assert FPS == config.ACTION_HZ == 30
    assert DT_NS == 33_333_333
    chunk = SharpaActionAdapter().adapt_chunk(np.zeros((50, 58)))
    assert chunk.fps == 30
    assert chunk.frames[-1].timestamp_ns - chunk.frames[0].timestamp_ns == 49 * DT_NS
    with pytest.raises(ValueError, match="30Hz"):
        replace(chunk, fps=20)


def test_policy_handshake_rejects_mismatched_frequency(monkeypatch):
    monkeypatch.setattr(config, "CHECKPOINT", Path("/test/30hz"))
    monkeypatch.setattr(config, "NORM_STATS", Path("/test/30hz/norm_stats.json"))
    metadata = {"config_name": config.CONFIG_NAME, "checkpoint": str(config.CHECKPOINT),
                "norm_stats": str(config.NORM_STATS), "action_dim": 58,
                "action_horizon": 50, "action_hz": 30, "model_finetune_mode": "full",
                "paligemma_variant": "gemma_2b", "action_expert_variant": "gemma_300m",
                "action_representation": "absolute_joint_position", "warmed_up": True,
                "action_order": config.ACTION_ORDER, "camera_mapping": config.CAMERA_MAPPING}
    config.validate_policy_metadata(metadata)
    metadata["action_hz"] = 20
    with pytest.raises(ValueError, match="action_hz"):
        config.validate_policy_metadata(metadata)


def test_training_frequency_must_match_deployment():
    training = SimpleNamespace(data=SimpleNamespace(action_hz=30, extra_delta_transform=False),
                               model=SimpleNamespace(action_dim=58, action_horizon=50,
                                     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"))
    config.validate_training_contract(training)
    training.data.action_hz = 20
    with pytest.raises(ValueError, match="Training action_hz"):
        config.validate_training_contract(training)


def test_no_silent_checkpoint_or_statistics_fallback(monkeypatch):
    monkeypatch.setattr(config, "CHECKPOINT", None)
    with pytest.raises(ValueError, match="SHARPA_CHECKPOINT"):
        config.require_checkpoint()
    monkeypatch.setattr(config, "CHECKPOINT", Path("/test/30hz"))
    monkeypatch.setattr(config, "NORM_STATS", None)
    with pytest.raises(ValueError, match="SHARPA_NORM_STATS"):
        config.require_checkpoint()


def test_checkpoint_owned_stats_selected_only_when_unambiguous(monkeypatch, tmp_path):
    monkeypatch.delenv("SHARPA_NORM_STATS", raising=False)
    monkeypatch.setattr(config, "CHECKPOINT", tmp_path)
    stats = tmp_path / "assets" / "training_dataset" / "norm_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text("{}")
    assert config._norm_stats_path() == stats
    second = tmp_path / "assets" / "other_dataset" / "norm_stats.json"
    second.parent.mkdir()
    second.write_text("{}")
    assert config._norm_stats_path() is None
    monkeypatch.setenv("SHARPA_NORM_STATS", str(second))
    assert config._norm_stats_path() == second
