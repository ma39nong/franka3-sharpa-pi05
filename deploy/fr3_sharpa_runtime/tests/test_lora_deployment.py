"""LoRA adapter selection and isolated launchers; no robot or GPU required."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime import config
from deploy.fr3_sharpa_runtime.model_loader import build_config, checkpoint_contract
from deploy.fr3_sharpa_runtime.model_profiles import get_model_profile, validate_parameter_layout
from deploy.fr3_sharpa_lora.setup_local import write_defaults

ROOT = Path(__file__).resolve().parents[3]


def adapter_entries():
    paths = []
    for suffix in ("", "_1"):
        for site in ("q_einsum", "kv_einsum", "attn_vec_einsum"):
            for factor in ("lora_a", "lora_b"):
                paths.append(("layers", "attn", site + suffix, factor, "value"))
        for site in ("gating_einsum", "linear"):
            for factor in ("lora_a", "lora_b"):
                paths.append(("layers", "mlp" + suffix, site + "_" + factor, "value"))
    return {str(path): {"key_metadata": [{"key": p} for p in ("params", "PaliGemma", "llm", *path)],
                       "value_metadata": {"write_shape": [1]}} for path in paths}


@pytest.fixture
def lora_checkpoint(tmp_path):
    params = tmp_path / "checkpoint/params"
    params.mkdir(parents=True)
    entries = adapter_entries()
    for name, shape, kind in (("action_in_proj", [58, 128], "kernel"), ("action_out_proj", [58], "bias")):
        entries[name] = {"key_metadata": [{"key": name}, {"key": kind}],
                         "value_metadata": {"write_shape": shape}}
    (params / "_METADATA").write_text(json.dumps({"tree_metadata": entries}))
    (params / "manifest.ocdbt").write_bytes(b"synthetic metadata, not actual weights")
    stats = params.parent / "assets/new_lora_absolute_30hz/norm_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text(json.dumps({"norm_stats": {group: {
        "mean": [0.] * 58, "std": [1.] * 58, "q01": [-1.] * 58, "q99": [1.] * 58,
    } for group in ("state", "actions")}}))
    return params.parent, stats


def select_lora(monkeypatch, fixture):
    checkpoint, stats = fixture
    profile = get_model_profile("lora")
    for name, value in {"MODEL_PROFILE": profile, "CONFIG_NAME": profile.config_name,
                        "CHECKPOINT": checkpoint, "NORM_STATS": stats}.items():
        monkeypatch.setattr(config, name, value)


def test_lora_layout_requires_both_complete_adapter_groups():
    entries = adapter_entries()
    validate_parameter_layout(entries, get_model_profile("lora"))
    with pytest.raises(ValueError, match="LoRA parameter branches"):
        validate_parameter_layout(entries, get_model_profile("full"))
    entries.pop(next(iter(entries)))
    with pytest.raises(ValueError, match="complete VLM"):
        validate_parameter_layout(entries, get_model_profile("lora"))
    with pytest.raises(ValueError, match="complete VLM"):
        validate_parameter_layout({}, get_model_profile("lora"))


def test_lora_model_contract_and_absolute_transform(monkeypatch, lora_checkpoint):
    select_lora(monkeypatch, lora_checkpoint)
    metadata = checkpoint_contract()
    assert metadata["model_finetune_mode"] == "lora"
    assert metadata["paligemma_variant"] == "gemma_2b_lora"
    assert metadata["action_expert_variant"] == "gemma_300m_lora"
    assert metadata["lora_recipe"]["vlm_rank"] == 16
    assert metadata["lora_recipe"]["expert_rank"] == 32
    training = build_config()
    config.validate_training_contract(training)
    assert training.model.max_token_len == 256
    assert training.model.action_dim == 58 and training.model.action_horizon == 50
    data = training.data.create(training.assets_dirs, training.model)
    assert [type(t).__name__ for t in data.data_transforms.inputs] == ["Fr3SharpaInputs"]
    assert [type(t).__name__ for t in data.data_transforms.outputs] == ["Fr3SharpaOutputs"]
    targets = np.full((50, 58), 0.25, dtype=np.float32)
    np.testing.assert_array_equal(data.data_transforms.outputs[0]({"actions": targets, "state": np.ones(58)})["actions"], targets)
    metadata["warmed_up"] = True
    config.validate_policy_metadata(metadata)
    metadata["config_name"] = get_model_profile("full").config_name
    with pytest.raises(ValueError, match="config_name"):
        config.validate_policy_metadata(metadata)


def clean_environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith("SHARPA_")}
    env.update(SHARPA_PYTHON=sys.executable, PYTHONPATH=f"{ROOT}:{ROOT / 'src'}", JAX_PLATFORMS="cpu")
    return env


def test_lora_profile_never_reuses_full_weights_or_stats():
    env = clean_environment()
    env.update(SHARPA_MODEL_PROFILE="lora", SHARPA_CHECKPOINT="/full/checkpoint",
               SHARPA_NORM_STATS="/full/stats.json", SHARPA_EXECUTION_TRACE_DIR="/full/trace")
    code = """import json
from deploy.fr3_sharpa_runtime import config as c
assert c.CHECKPOINT is None and c.NORM_STATS is None
assert c.POLICY_PORT == 18766 and 'fr3_sharpa_lora' in str(c.EXECUTION_TRACE_DIR)
try: c.require_checkpoint()
except ValueError as e: assert 'SHARPA_LORA_CHECKPOINT' in str(e)
else: raise AssertionError('missing LoRA checkpoint must fail')
print('PASS')
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_full_profile_remains_the_default():
    env = clean_environment()
    env.update(SHARPA_CHECKPOINT="/full/checkpoint", SHARPA_NORM_STATS="/full/norm_stats.json")
    code = """from deploy.fr3_sharpa_runtime import config as c
assert c.MODEL_PROFILE.name == 'full' and c.POLICY_PORT == 18765
assert str(c.CHECKPOINT) == '/full/checkpoint'
assert str(c.NORM_STATS) == '/full/norm_stats.json'
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_lora_wrapper_reads_its_own_defaults_and_preserves_home_prefix(lora_checkpoint, tmp_path):
    checkpoint, _ = lora_checkpoint
    output = write_defaults(checkpoint, output=tmp_path / "lora.env", port=18767)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        write_defaults(checkpoint, output=output)
    assert output.read_bytes() == before
    env = clean_environment()
    env["SHARPA_LORA_ENV_FILE"] = str(output)
    result = subprocess.run(["bash", "deploy/fr3_sharpa_lora/runtime.sh", "--check", "--speed", "slow"],
                             cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["checkpoint"] == str(checkpoint)
    assert report["policy_url"] == "ws://127.0.0.1:18767"
    assert report["model_profile"] == "lora"
    assert report["execute_steps"] == 15 and report["initialization"] == "home"
    assert report["hand_initialization"] == "fixed_named_hand_home"
    assert report["hardware_output"] is False


def test_lora_execute_with_absent_files_stops_before_hardware(tmp_path):
    # Never inherit a real workstation LoRA defaults file in an execute test.
    defaults = tmp_path / "empty.env"
    defaults.write_text("")
    env = clean_environment()
    env.update(SHARPA_LORA_ENV_FILE=str(defaults),
               SHARPA_LORA_CHECKPOINT=str(tmp_path / "absent-checkpoint"),
               SHARPA_LORA_NORM_STATS=str(tmp_path / "absent-stats.json"))
    result = subprocess.run(["bash", "deploy/fr3_sharpa_lora/runtime.sh", "--execute", "--enable-hardware"],
                             cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "Missing complete Orbax" in result.stderr


def test_model_profile_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown SHARPA_MODEL_PROFILE"):
        get_model_profile("typo")


@pytest.mark.parametrize("model", ["full", "lora"])
def test_unified_runtime_selects_model_without_changing_execution(model, lora_checkpoint, tmp_path):
    checkpoint, _ = lora_checkpoint
    defaults = write_defaults(checkpoint, output=tmp_path / "lora.env")
    env = clean_environment()
    env.update(SHARPA_LORA_ENV_FILE=str(defaults), SHARPA_CHECKPOINT="/full/checkpoint")
    result = subprocess.run(["bash", "deploy/fr3_sharpa_runtime/runtime.sh", "--check", "--speed", "medium",
                             "--num-chunks", "2", "--model", model],
                             cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["model_profile"] == model
    assert report["policy_url"].endswith(":18765" if model == "full" else ":18766")
    assert report["checkpoint"] == ("/full/checkpoint" if model == "full" else str(checkpoint))
    assert report["execute_steps"] == 15 and report["initialization"] == "home"
    assert report["num_chunks"] == 2 and report["execute_speed_scale"] == 0.5
    assert report["hardware_output"] is False


def test_unified_full_overrides_inherited_lora_profile():
    env = clean_environment()
    env.update(SHARPA_MODEL_PROFILE="lora", SHARPA_CHECKPOINT="/full/checkpoint",
               SHARPA_LORA_ENV_FILE="/nonexistent/should-not-be-read")
    result = subprocess.run(["bash", "deploy/fr3_sharpa_runtime/runtime.sh", "--model=full", "--check"],
                             cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["model_profile"] == "full"


@pytest.mark.parametrize("script", ["serve.sh", "runtime.sh"])
def test_shared_model_registry_is_available_without_weights(script):
    result = subprocess.run(["bash", f"deploy/fr3_sharpa_runtime/{script}", "--list-models"],
                             cwd=ROOT, env=clean_environment(), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    registry = {model["name"]: model for model in json.loads(result.stdout)}
    assert set(registry) == {"full", "lora"}
    assert registry["full"]["default_port"] == 18765
    assert registry["lora"]["checkpoint_env"] == "SHARPA_LORA_CHECKPOINT"
    assert registry["lora"]["training_action_supervision"] == "absolute"


@pytest.mark.parametrize("args", [["--model"], ["--model", "unknown"], ["--model", "full", "--model", "lora"]])
def test_invalid_model_selection_fails_before_loading_or_hardware(args):
    result = subprocess.run(["bash", "deploy/fr3_sharpa_runtime/runtime.sh", *args, "--check"],
                             cwd=ROOT, env=clean_environment(), text=True, capture_output=True)
    assert result.returncode == 2
    assert "model" in result.stderr.lower()


def test_legacy_lora_alias_does_not_allow_model_override():
    result = subprocess.run(["bash", "deploy/fr3_sharpa_lora/runtime.sh", "--model", "full", "--check"],
                             cwd=ROOT, env=clean_environment(), text=True, capture_output=True)
    assert result.returncode == 2
    assert "Conflicting" in result.stderr


def test_unified_lora_execute_keeps_pre_hardware_checkpoint_validation(tmp_path):
    defaults = tmp_path / "empty.env"
    defaults.write_text("")
    env = clean_environment()
    env.update(SHARPA_LORA_ENV_FILE=str(defaults),
               SHARPA_LORA_CHECKPOINT=str(tmp_path / "absent-checkpoint"),
               SHARPA_LORA_NORM_STATS=str(tmp_path / "absent-stats.json"))
    result = subprocess.run(["bash", "deploy/fr3_sharpa_runtime/runtime.sh", "--model=lora",
                             "--execute", "--enable-hardware"], cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "Missing complete Orbax" in result.stderr


@pytest.mark.parametrize("model", ["full", "lora"])
def test_unified_model_service_loads_the_matching_metadata_only(model, lora_checkpoint, tmp_path):
    checkpoint, stats = lora_checkpoint
    if model == "full":
        path = checkpoint / "params/_METADATA"
        data = json.loads(path.read_text())
        data["tree_metadata"] = {name: entry for name, entry in data["tree_metadata"].items() if "lora" not in name}
        path.write_text(json.dumps(data))
    defaults = write_defaults(checkpoint, output=tmp_path / "lora.env") if model == "lora" else None
    env = clean_environment()
    env.update(SHARPA_CHECKPOINT=str(checkpoint), SHARPA_NORM_STATS=str(stats))
    if defaults is not None:
        env["SHARPA_LORA_ENV_FILE"] = str(defaults)
    result = subprocess.run(["bash", "deploy/fr3_sharpa_runtime/serve.sh", "--model", model, "--check-only"],
                             cwd=ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["model_finetune_mode"] == model
    assert report["checkpoint"] == str(checkpoint)
    assert report["warmed_up"] is False
