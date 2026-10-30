"""Paper assets, time indexing, distribution and mode boundaries; no devices."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from deploy.fr3_sharpa_control.plan import inspect_checkpoint, DeploymentSettings
from deploy.fr3_sharpa_runtime.paper_contract import inspect_paper, validate_stats
from deploy.fr3_sharpa_runtime.paper_model import PaperPi0, PaperPi0Config, PaperSharpaData, reference_normalizer
from openpi.shared import normalize
from openpi.shared import delta_paper as paper
from openpi import transforms
from openpi.models.pi0 import Pi0
from openpi.training.config import AssetsConfig


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def checkpoint(tmp_path):
    root = tmp_path / "7999"
    params = root / "params"
    params.mkdir(parents=True)
    entries = {name: {"key_metadata": [{"key": name}, {"key": kind}], "value_metadata": {"write_shape": shape}}
        for name, kind, shape in (("action_in_proj", "kernel", [58, 128]), ("action_out_proj", "bias", [58]))}
    (params / "_METADATA").write_text(json.dumps({"tree_metadata": entries}))
    (params / "manifest.ocdbt").write_bytes(b"test only")
    assets = root / "assets/paper"
    assets.mkdir(parents=True)
    stats = {g: {"mean": np.zeros(shape).tolist(), "std": np.ones(shape).tolist(),
        "q01": np.full(shape, -2.).tolist(), "q99": np.full(shape, 2.).tolist()}
        for g, shape in (("state", (58,)), ("actions", (50, 58)))}
    (assets / "norm_stats.json").write_text(json.dumps({"norm_stats": stats}))
    np.savez(assets / paper.BUNDLE, arm_mean=np.repeat(np.arange(50)[:, None], 14, axis=1),
        arm_std=np.full((50, 14), .5), noise_cholesky=np.zeros((0, 0)))
    contract = {"action_dim": 58, "action_horizon": 50, "action_hz": 30,
        "vector_transforms": {"delta_paper": {"version": paper.VERSION, "mode": "timestamp_zscore_arms", "beta": 0., "std_floor": 1e-4}, "vectors": [{"type": "openpi.transforms.DeltaActions", "fields": {
            "mask": [i < 7 or 29 <= i < 36 for i in range(58)]}}]}}
    metadata = {"version": paper.VERSION, "mode": "timestamp_zscore_arms", "beta": 0.,
        "std_floor": 1e-4, "shape": [50, 58], "flatten_order": "time-major", "windows": "complete_only",
        "training_windows": 2, "norm_stats_sha256": digest(assets / "norm_stats.json"),
        "bundle_sha256": digest(assets / paper.BUNDLE), "source_contract": contract}
    (assets / paper.MANIFEST).write_text(json.dumps(metadata))
    (assets / "norm_metadata.json").write_text(json.dumps({"norm_stats_sha256": metadata["norm_stats_sha256"], "contract": contract}))
    return root, assets, entries


def model_for(assets):
    meta = json.loads((assets / paper.MANIFEST).read_text())
    return PaperPi0Config(pi05=True, action_dim=58, action_horizon=50, max_token_len=256,
        paper_asset_dir=str(assets), paper_mode=meta["mode"], paper_beta=meta["beta"],
        paper_asset_sha256=meta["bundle_sha256"])


def test_auto_detect_and_semantic_conflict(checkpoint):
    root, assets, entries = checkpoint
    found = inspect_checkpoint(root)
    assert found["profile"] == "full-delta" and found["semantics"] == "delta"
    assert found["normalization"] == "timestamp_zscore_arms"
    assert found["supported_modes"] == ["serial", "rtc", "soft-serial", "soft-async"]
    validate_stats(json.loads((assets / "norm_stats.json").read_text())["norm_stats"], paper_algorithm=found["paper_algorithm"])
    with pytest.raises(ValueError, match="冲突"):
        inspect_checkpoint(root, semantics="absolute")
    # No name-based fallback when an algorithm asset is missing.
    (assets / paper.BUNDLE).unlink()
    with pytest.raises(ValueError, match="缺少"):
        inspect_checkpoint(root, semantics="delta")


def test_paper_state_never_silently_loads_as_normal_model(checkpoint):
    root, assets, entries = checkpoint
    entries["paper_noise_cholesky"] = {"value_metadata": {"write_shape": [2900, 2900]}}
    with pytest.raises(ValueError, match="噪声参数"):
        inspect_paper(assets / "norm_stats.json", entries)
    (assets / paper.MANIFEST).unlink()
    with pytest.raises(ValueError, match="缺少"):
        inspect_paper(assets / "norm_stats.json", entries)


def test_asset_fingerprint_includes_algorithm_and_detects_corruption(checkpoint):
    root, assets, _ = checkpoint
    first = inspect_checkpoint(root)["identity"]
    meta = json.loads((assets / paper.MANIFEST).read_text())
    meta["mode"] = "global_zscore_arms"
    (assets / paper.MANIFEST).write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="冲突"):
        inspect_checkpoint(root)
    meta["source_contract"]["vector_transforms"]["delta_paper"]["mode"] = meta["mode"]
    (assets / paper.MANIFEST).write_text(json.dumps(meta))
    audit = json.loads((assets / "norm_metadata.json").read_text())
    audit["contract"] = meta["source_contract"]
    (assets / "norm_metadata.json").write_text(json.dumps(audit))
    assert inspect_checkpoint(root)["identity"] != first
    with (assets / paper.BUNDLE).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="digest"):
        inspect_checkpoint(root)


def test_reference_and_output_match_training_time_indices(checkpoint):
    _, assets, _ = checkpoint
    model = model_for(assets)
    stats = normalize.load(assets)
    data = PaperSharpaData(repo_id="paper", assets=AssetsConfig(assets_dir=str(assets.parent), asset_id="paper"),
        extra_delta_transform=True).create(Path("."), model)
    state = np.linspace(-.3, .3, 58).astype(np.float32)
    absolute = np.broadcast_to(state, (50, 58)).copy()
    absolute[:, paper.ARMS] += np.arange(50)[:, None] + 1.
    sample = {"state": state, "actions": absolute.copy()}
    relative = transforms.DeltaActions([i in paper.ARMS for i in range(58)])(sample)
    encoded = data.model_transforms.inputs[0](transforms.Normalize(stats, use_quantiles=True)(relative))
    np.testing.assert_allclose(encoded["actions"][:, paper.ARMS], 2., atol=3e-5)
    restored = replace(data.model_transforms.outputs[-1], inverse=True)(encoded)
    restored = transforms.Unnormalize(stats, use_quantiles=True)(restored)
    restored = transforms.AbsoluteActions([i in paper.ARMS for i in range(58)])(restored)
    np.testing.assert_allclose(restored["actions"], absolute, atol=1e-5)
    normalize_reference = reference_normalizer(stats, model)
    # Old reference26..29 must be expressed in NEW positions0..3.
    reference = absolute[26:30].copy()
    reference[:, paper.ARMS] -= state[paper.ARMS]
    result = normalize_reference({"actions": reference})["actions"]
    np.testing.assert_allclose(result[:, paper.ARMS], 54., atol=3e-5)
    np.testing.assert_allclose(result[:, np.r_[7:29, 36:58]],
        (reference[:, np.r_[7:29, 36:58]] + 2) / (4 + 1e-6) * 2 - 1, atol=1e-6)


def test_correlated_sampling_and_explicit_noise_are_not_applied_twice(monkeypatch):
    chol = np.array([[1., 0.], [.5, np.sqrt(.75)]], dtype=np.float32)
    fake = SimpleNamespace(paper_beta=.5, action_horizon=1, action_dim=2,
        paper_noise_cholesky=SimpleNamespace(value=chol))
    key = jax.random.key(42)
    got = PaperPi0.noise(fake, key, (1, 1, 2))
    np.testing.assert_allclose(got, np.asarray(jax.random.normal(key, (1, 1, 2))) @ chol.T, atol=1e-6)
    fake.noise = lambda rng, shape: PaperPi0.noise(fake, rng, shape)
    monkeypatch.setattr(Pi0, "sample_actions", lambda self, rng, obs, **kw: kw)
    # A lightweight subclass instance exercises super() without constructing Gemma.
    class Lightweight(PaperPi0):
        def __init__(self):
            self.paper_beta = .5
            self.action_horizon, self.action_dim = 1, 2
            self.paper_noise_cholesky = SimpleNamespace(value=chol)
    instance = Lightweight()
    obs = SimpleNamespace(state=np.zeros((1, 2)))
    explicit = np.ones((1, 1, 2))
    out = instance.sample_actions(key, obs, noise=explicit, rtc_max_guidance_weight=5.)
    assert out["noise"] is explicit and out["rtc_max_guidance_weight"] == 5.
    np.testing.assert_allclose(instance.sample_actions(key, obs)["noise"], got)


def test_old_checkout_is_rejected_for_paper(checkpoint, tmp_path):
    root, _, _ = checkpoint
    with pytest.raises(ValueError, match="尚未同步Paper"):
        DeploymentSettings(str(tmp_path), str(root)).plan()
