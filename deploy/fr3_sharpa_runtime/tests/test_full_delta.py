"""Delta is an inference recipe; the shared executor still receives absolutes."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from deploy.fr3_sharpa_full.setup_delta import write_defaults
from deploy.fr3_sharpa_runtime import config
from deploy.fr3_sharpa_runtime.model_loader import build_config, checkpoint_contract
from deploy.fr3_sharpa_runtime.model_profiles import get_model_profile, validate_parameter_layout
from openpi import transforms
from openpi.shared import normalize

ROOT = Path(__file__).resolve().parents[3]
ARMS = np.r_[0:7, 29:36]
HANDS = np.r_[7:29, 36:58]


@pytest.fixture
def delta_checkpoint(tmp_path):
    root = tmp_path / "4000"
    params = root / "params"
    params.mkdir(parents=True)
    entries = {}
    for name, kind, shape in (("action_in_proj", "kernel", [58, 128]),
                              ("action_out_proj", "bias", [58])):
        entries[name] = {"key_metadata": [{"key": name}, {"key": kind}],
                         "value_metadata": {"write_shape": shape}}
    (params / "_METADATA").write_text(json.dumps({"tree_metadata": entries}))
    (params / "manifest.ocdbt").write_bytes(b"synthetic metadata, not weights")
    stats = root / "assets/delta_stats/norm_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text(json.dumps({"norm_stats": {group: {
        "mean": [0.1] * 58, "std": [0.5] * 58, "q01": [-2.] * 58, "q99": [3.] * 58,
    } for group in ("state", "actions")}}))
    return root, stats


def select_delta(monkeypatch, fixture):
    checkpoint, stats = fixture
    profile = get_model_profile("full-delta")
    for name, value in {"MODEL_PROFILE": profile, "CONFIG_NAME": profile.config_name,
                        "CHECKPOINT": checkpoint, "NORM_STATS": stats}.items():
        monkeypatch.setattr(config, name, value)


def environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith("SHARPA_")}
    env.update(SHARPA_PYTHON=sys.executable, PYTHONPATH=f"{ROOT}:{ROOT / 'src'}", JAX_PLATFORMS="cpu")
    return env


def test_delta_subtracts_chunk_start_arms_and_restores_after_unnormalization(monkeypatch, delta_checkpoint):
    select_delta(monkeypatch, delta_checkpoint)
    training = build_config()
    config.validate_training_contract(training)
    data = training.data.create(training.assets_dirs, training.model)
    assert [type(t).__name__ for t in data.data_transforms.inputs] == ["Fr3SharpaInputs", "DeltaActions"]
    assert [type(t).__name__ for t in data.data_transforms.outputs] == ["AbsoluteActions", "Fr3SharpaOutputs"]
    state = np.linspace(-0.7, 0.6, 58, dtype=np.float32)
    targets = np.broadcast_to(np.linspace(0.1, 0.4, 58, dtype=np.float32), (50, 58)).copy()
    targets += np.arange(50, dtype=np.float32)[:, None] * 0.001
    obs = {"observation/state": state.copy(), "actions": targets.copy()}
    for key in ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image"):
        obs[key] = np.zeros((8, 8, 3), dtype=np.uint8)
    encoded = transforms.compose(data.data_transforms.inputs)(obs)
    np.testing.assert_allclose(encoded["actions"][:, ARMS], targets[:, ARMS] - state[ARMS])
    np.testing.assert_allclose(encoded["actions"][:, HANDS], targets[:, HANDS])
    stats = normalize.load(delta_checkpoint[1].parent)
    encoded = transforms.Normalize(stats, use_quantiles=True)(encoded)
    restored = transforms.compose([transforms.Unnormalize(stats, use_quantiles=True),
                                    *data.data_transforms.outputs])(encoded)
    np.testing.assert_allclose(restored["actions"], targets, atol=2e-6)
    np.testing.assert_array_equal(obs["observation/state"], state)


def test_delta_contract_announces_absolute_output_and_rejects_missing_semantics(monkeypatch, delta_checkpoint):
    select_delta(monkeypatch, delta_checkpoint)
    metadata = checkpoint_contract()
    assert metadata["model_profile"] == "full-delta" and metadata["model_finetune_mode"] == "full"
    assert metadata["action_representation"] == "absolute_joint_position"
    assert metadata["delta_dimensions"] == ARMS.tolist()
    assert metadata["delta_reference"] == "chunk_start_measured_joint_positions"
    metadata["warmed_up"] = True
    config.validate_policy_metadata(metadata)
    metadata["training_action_supervision"] = "absolute"
    with pytest.raises(ValueError, match="training_action_supervision"):
        config.validate_policy_metadata(metadata)


def test_delta_requires_delta_transform_and_full_parameters(monkeypatch, delta_checkpoint):
    from dataclasses import replace
    select_delta(monkeypatch, delta_checkpoint)
    training = build_config()
    with pytest.raises(ValueError, match="arm delta"):
        config.validate_training_contract(replace(training, data=replace(training.data, extra_delta_transform=False)))
    with pytest.raises(ValueError, match="LoRA parameter branches"):
        validate_parameter_layout({"lora_a": {}}, get_model_profile("full-delta"))


def test_delta_never_falls_back_to_absolute_or_lora_weights():
    env = environment()
    env.update(SHARPA_MODEL_PROFILE="full-delta", SHARPA_CHECKPOINT="/absolute/checkpoint",
               SHARPA_NORM_STATS="/absolute/stats", SHARPA_LORA_CHECKPOINT="/lora/checkpoint",
               SHARPA_EXECUTION_TRACE_DIR="/absolute/trace")
    code = """from deploy.fr3_sharpa_runtime import config as c
assert c.CHECKPOINT is None and c.NORM_STATS is None
assert c.POLICY_PORT == 18767
assert 'fr3_sharpa_full_delta' in str(c.EXECUTION_TRACE_DIR)
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, check=True)


@pytest.mark.parametrize("home", [[], ["--home"]])
def test_delta_entry_uses_own_defaults_and_shares_current_home_modes(delta_checkpoint, tmp_path, home):
    defaults = write_defaults(delta_checkpoint[0], output=tmp_path / "delta.env")
    env = environment()
    env.update(SHARPA_FULL_DELTA_ENV_FILE=str(defaults), SHARPA_CHECKPOINT="/absolute/checkpoint")
    result = subprocess.run(["bash", "deploy/fr3_sharpa_full/runtime_delta.sh", "--check", *home],
                            cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    report = json.loads(result.stdout)
    assert report["model_profile"] == "full-delta" and report["policy_url"].endswith(":18767")
    assert report["checkpoint"] == str(delta_checkpoint[0])
    assert report["training_action_supervision"] == "delta_arm_joint_target_absolute_hand"
    assert report["action_representation"] == "absolute_joint_position"
    assert report["initialization"] == ("home" if home else "current")
    assert report["execute_steps"] == 15 and report["hardware_output"] is False


def test_delta_check_only_selects_full_parameter_layout(delta_checkpoint, tmp_path):
    defaults = write_defaults(delta_checkpoint[0], output=tmp_path / "delta.env")
    env = environment()
    env["SHARPA_FULL_DELTA_ENV_FILE"] = str(defaults)
    result = subprocess.run(["bash", "deploy/fr3_sharpa_full/serve_delta.sh", "--check-only"],
                            cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    metadata = json.loads(result.stdout)
    assert metadata["model_finetune_mode"] == "full" and metadata["warmed_up"] is False
    with pytest.raises(FileExistsError):
        write_defaults(delta_checkpoint[0], output=defaults)


def test_missing_delta_model_fails_before_hardware(tmp_path):
    defaults = tmp_path / "empty.env"
    defaults.write_text("")
    env = environment()
    env.update(SHARPA_FULL_DELTA_ENV_FILE=str(defaults),
               SHARPA_FULL_DELTA_CHECKPOINT=str(tmp_path / "absent"),
               SHARPA_FULL_DELTA_NORM_STATS=str(tmp_path / "absent_stats.json"))
    result = subprocess.run(["bash", "deploy/fr3_sharpa_full/runtime_delta.sh", "--execute", "--enable-hardware"],
                            cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode != 0 and "Missing complete Orbax" in result.stderr


def test_rolling_reference_matches_actual_training_delta_and_quantile_pipeline(monkeypatch, delta_checkpoint):
    from deploy.fr3_sharpa_runtime.model_service import encode_reference
    select_delta(monkeypatch, delta_checkpoint)
    training = build_config()
    data = training.data.create(training.assets_dirs, training.model)
    stats = normalize.load(delta_checkpoint[1].parent)
    normalizer = transforms.Normalize(stats, use_quantiles=data.use_quantile_norm)
    state = np.linspace(-0.8, 0.8, 58, dtype=np.float32)
    targets = np.broadcast_to(np.linspace(-0.2, 0.3, 58, dtype=np.float32), (4, 58)).copy()
    obs = {"observation/state": state.copy(), "actions": targets.copy()}
    for key in ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image"):
        obs[key] = np.zeros((8, 8, 3), dtype=np.uint8)
    expected = normalizer(transforms.compose(data.data_transforms.inputs)(obs))["actions"]
    encoded = encode_reference(targets, state, delta=True, normalizer=normalizer)
    np.testing.assert_allclose(encoded, expected, atol=1e-7)


def test_covariance_tool_uses_selected_delta_stats_and_refuses_overwrite(monkeypatch, delta_checkpoint, tmp_path):
    from deploy.fr3_sharpa_runtime import correlation
    select_delta(monkeypatch, delta_checkpoint)
    states = np.stack([np.full(58, n * 0.01, dtype=np.float32) for n in range(3)])
    targets = np.stack([np.full((50, 58), n * 0.02, dtype=np.float32) for n in range(3)])
    source, output = tmp_path / "windows.npz", tmp_path / "covariance.npz"
    np.savez(source, absolute_actions=targets, states=states)
    monkeypatch.setattr(sys, "argv", ["correlation", "--input", str(source), "--output", str(output)])
    correlation.main()
    with np.load(output, allow_pickle=False) as asset:
        info = json.loads(asset["metadata"].item())
        assert info["sample_count"] == 3
        assert info["space"] == "normalized_training_actions"
        assert asset["covariance"].shape == (2900, 2900)
        # Arms subtract the new state; hands retain the absolute targets.
        assert asset["covariance"][7, 7] == pytest.approx(4 * asset["covariance"][0, 0], rel=1e-4)
    with pytest.raises(FileExistsError):
        correlation.main()
