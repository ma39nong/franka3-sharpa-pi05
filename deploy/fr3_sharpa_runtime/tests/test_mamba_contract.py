# ruff: noqa: PT011
"""Production identity and tamper gates, tested independently of robot hardware."""

import copy
import json
import os
import subprocess
import sys

import pytest

from deploy.fr3_sharpa_runtime.mamba_contract import inspect_mamba
from deploy.fr3_sharpa_runtime.mamba_contract import reconstruct_model
from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import checkpoint
from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import select


@pytest.mark.parametrize("paper", [False, True])
@pytest.mark.parametrize("delay", [0, 3])
def test_identity_reconstruction_and_no_manual_rtc(tmp_path, monkeypatch, paper, delay):
    root, assets, record = checkpoint(tmp_path, paper=paper, delay=delay)
    monkeypatch.setenv("SHARPA_TRAIN_RTC_MAX_DELAY", "49")
    metadata = inspect_mamba(root, assets / "norm_stats.json")
    model = reconstruct_model(metadata, assets / "norm_stats.json")
    assert model.rtc_max_delay == delay
    assert bool(metadata["train_rtc"]) == bool(delay)
    assert model.normalization == ("paper" if paper else "ordinary_per_dimension")
    if paper:
        assert model.paper_asset_dir == str(assets)
        assert model.paper_asset_sha256 == record["model"]["paper_asset_sha256"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("architecture", "transformer"),
        ("stage", "paper_delta_fm"),
        ("pi05", False),
        ("diagnostic_only", True),
        ("paligemma_variant", "gemma_2b_lora"),
        ("action_dim", 64),
        ("rtc_max_delay", True),
        ("rtc_max_delay", 50),
        ("normalization", "global_quantile"),
        ("extra_field", "bad"),
        ("mamba_depth", 0),
        ("max_token_len", 8),
    ],
)
def test_wrong_model_fields_fail_closed(tmp_path, field, value):
    root, assets, record = checkpoint(tmp_path)
    record["model"][field] = value
    (assets / "mamba_architecture.json").write_text(json.dumps(record))
    with pytest.raises(ValueError):
        inspect_mamba(root, assets / "norm_stats.json")


@pytest.mark.parametrize(
    "change",
    [
        "norm",
        "upstream",
        "schema",
        "mixer",
        "extra_asset",
        "missing_manifest",
        "extra_tree",
        "lora",
        "missing_block",
        "old_expert",
    ],
)
def test_asset_and_tree_tamper(tmp_path, change):
    root, assets, record = checkpoint(tmp_path)
    path = assets / "mamba_architecture.json"
    if change == "norm":
        with (assets / "norm_stats.json").open("a") as stream:
            stream.write(" ")
    elif change == "extra_asset":
        second = root / "assets/other"
        second.mkdir()
        (second / path.name).write_text(path.read_text())
    elif change == "missing_manifest":
        path.unlink()
    elif change in ("upstream", "schema", "mixer"):
        if change == "upstream":
            record["upstream"] = {}
        if change == "schema":
            record["schema_version"] = 2
        if change == "mixer":
            record["model"]["mamba"]["chunk_size"] = float("nan")
        path.write_text(json.dumps(record))
    else:
        tree_path = root / "params/_METADATA"
        tree = json.loads(tree_path.read_text())
        entries = tree["tree_metadata"]
        example = copy.deepcopy(next(iter(entries.values())))
        if change == "extra_tree":
            example["key_metadata"] = [{"key": "action_in_proj"}, {"key": "kernel"}]
            entries["old_transformer_projection"] = example
        if change == "lora":
            entries["lora_a"] = example
        if change == "missing_block":
            key = next(k for k in entries if "block_2" in k)
            del entries[key]
        if change == "old_expert":
            example["key_metadata"] = [{"key": n} for n in ("PaliGemma", "llm", "layers", "mlp_1", "kernel")]
            entries["old_gemma_expert"] = example
        tree_path.write_text(json.dumps(tree))
    with pytest.raises(ValueError):
        inspect_mamba(root, assets / "norm_stats.json")


@pytest.mark.parametrize("name", ["delta_paper.json", "delta_paper.npz", "norm_metadata.json"])
def test_paper_asset_tampering(tmp_path, name):
    root, assets, _ = checkpoint(tmp_path, paper=True)
    with (assets / name).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="SHA256"):
        inspect_mamba(root, assets / "norm_stats.json")


def test_check_only_imports_no_jax_and_shell_profile_isolation(tmp_path):
    root, assets, _ = checkpoint(tmp_path, delay=3)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SHARPA_")}
    env.update(
        SHARPA_PYTHON=sys.executable,
        SHARPA_CHECKPOINT="/wrong/old/full",
        SHARPA_MAMBA_CHECKPOINT=str(root),
        SHARPA_MAMBA_NORM_STATS=str(assets / "norm_stats.json"),
    )
    result = subprocess.run(
        ["bash", "deploy/fr3_sharpa_runtime/serve.sh", "--model", "mamba", "--check-only", "--enable-train-rtc"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["model_profile"] == "mamba"
    code = "from deploy.fr3_sharpa_runtime.model_loader import checkpoint_contract; checkpoint_contract(); import sys; assert 'jax' not in sys.modules; assert 'rclpy' not in sys.modules"
    env["SHARPA_MODEL_PROFILE"] = "mamba"
    result = subprocess.run([sys.executable, "-c", code], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("paper", [False, True])
def test_handshake_cache_invalidates_manifest_and_stats(tmp_path, monkeypatch, paper):
    from deploy.fr3_sharpa_runtime import config
    from deploy.fr3_sharpa_runtime import model_loader

    root, assets, record = checkpoint(tmp_path, paper=paper, delay=3)
    select(monkeypatch, root, assets, "mamba-delta" if paper else "mamba")
    metadata = model_loader.checkpoint_contract()
    metadata["warmed_up"] = True
    config.validate_policy_metadata(metadata)
    record["model"]["rtc_max_delay"] = 2
    (assets / "mamba_architecture.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="mamba_manifest_sha256|train_rtc"):
        config.validate_policy_metadata(metadata)
    metadata = model_loader.checkpoint_contract()
    metadata["warmed_up"] = True
    config.validate_policy_metadata(metadata)
    (assets / "mamba_architecture.json").unlink()
    with pytest.raises(ValueError):
        config.validate_policy_metadata(metadata)


def test_no_shard_shape_inference_and_external_stats_rejected(tmp_path):
    root, assets, _ = checkpoint(tmp_path)
    tree_path = root / "params/_METADATA"
    tree = json.loads(tree_path.read_text())
    for entry in tree["tree_metadata"].values():
        if "write_shape" in entry["value_metadata"]:
            entry["value_metadata"]["write_shape"] = [1]
    tree_path.write_text(json.dumps(tree))
    inspect_mamba(root, assets / "norm_stats.json")  # Full global shape check happens on restore.
    outside = tmp_path / "norm_stats.json"
    outside.write_bytes((assets / "norm_stats.json").read_bytes())
    with pytest.raises(ValueError, match="checkpoint-owned"):
        inspect_mamba(root, outside)


def test_manifest_removal_cannot_disguise_mamba_as_old_transformer(tmp_path, monkeypatch):
    from deploy.fr3_sharpa_control.plan import inspect_checkpoint
    from deploy.fr3_sharpa_runtime.model_loader import checkpoint_contract

    root, assets, _ = checkpoint(tmp_path)
    (assets / "mamba_architecture.json").unlink()
    select(monkeypatch, root, assets, "full")
    with pytest.raises(ValueError, match="Mamba parameter tree"):
        checkpoint_contract()
    with pytest.raises(ValueError, match="Mamba parameter tree"):
        inspect_checkpoint(root, semantics="absolute")


def test_explicit_mamba_environment_wins_over_profile_defaults(tmp_path):
    root, assets, _ = checkpoint(tmp_path)
    defaults = tmp_path / "mamba.env"
    defaults.write_text(
        "export SHARPA_MODEL_PROFILE=full\nexport SHARPA_MAMBA_CHECKPOINT=/wrong\nexport SHARPA_MAMBA_PORT=12345\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("SHARPA_")}
    env.update(
        SHARPA_PYTHON=sys.executable,
        SHARPA_MAMBA_CHECKPOINT=str(root),
        SHARPA_MAMBA_PORT="18768",
        SHARPA_MAMBA_ENV_FILE=str(defaults),
    )
    result = subprocess.run(
        ["bash", "deploy/fr3_sharpa_runtime/runtime.sh", "--model=mamba", "--check"],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["policy_url"] == "ws://127.0.0.1:18768"


def test_cached_gate_detects_new_ambiguous_stats_directory(tmp_path, monkeypatch):
    from deploy.fr3_sharpa_runtime import config
    from deploy.fr3_sharpa_runtime import model_loader

    root, assets, _ = checkpoint(tmp_path)
    select(monkeypatch, root, assets, "mamba")
    metadata = model_loader.checkpoint_contract()
    metadata["warmed_up"] = True
    config.validate_policy_metadata(metadata)
    duplicate = root / "assets/duplicate"
    duplicate.mkdir()
    (duplicate / "norm_stats.json").write_bytes((assets / "norm_stats.json").read_bytes())
    with pytest.raises(ValueError, match="uniquely owned"):
        config.validate_policy_metadata(metadata)
