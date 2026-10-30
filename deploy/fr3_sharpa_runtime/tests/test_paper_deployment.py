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
    assert found["supported_modes"] == ["serial", "paper-rtc", "soft-serial", "soft-async"]
    validate_stats(json.loads((assets / "norm_stats.json").read_text())["norm_stats"], paper_algorithm=found["paper_algorithm"])
    with pytest.raises(ValueError, match="冲突"):
        inspect_checkpoint(root, semantics="absolute")
    # No name-based fallback when an algorithm asset is missing.
    (assets / paper.BUNDLE).unlink()
    with pytest.raises(ValueError, match="缺少"):
        inspect_checkpoint(root, semantics="delta")


def test_paper_handshake_cache_reuses_validation_and_rejects_changed_bundle(checkpoint, monkeypatch):
    from deploy.fr3_sharpa_runtime import config, model_loader
    from deploy.fr3_sharpa_runtime.model_profiles import get_model_profile
    from deploy.fr3_sharpa_runtime.runtime import SharpaRuntime
    root, assets, _ = checkpoint
    profile = get_model_profile("full-delta")
    monkeypatch.setattr(config, "CHECKPOINT", root)
    monkeypatch.setattr(config, "NORM_STATS", assets / "norm_stats.json")
    monkeypatch.setattr(config, "MODEL_PROFILE", profile)
    monkeypatch.setattr(config, "CONFIG_NAME", profile.config_name)
    monkeypatch.setattr(config, "_validation_cache", None)
    original = model_loader.checkpoint_contract
    calls = []
    def verify():
        calls.append(True)
        return original()
    monkeypatch.setattr(model_loader, "checkpoint_contract", verify)
    metadata = original()
    metadata["warmed_up"] = True
    SharpaRuntime()  # Only local preflight; no socket/ROS/model/SDK connection.
    for _ in range(3):
        config.validate_policy_metadata(metadata)
    assert len(calls) == 1
    # In-place modification changes the stat identity and forces full validation.
    bundle = assets / paper.BUNDLE
    bundle.write_bytes(bundle.read_bytes() + b"changed")
    for _ in range(2):
        with pytest.raises(ValueError, match="digest mismatch"):
            config.validate_policy_metadata(metadata)
    assert len(calls) == 3  # Failed verification is never cached.


@pytest.mark.parametrize("asset", ["manifest.ocdbt", "_METADATA", "norm_stats.json",
                                  "delta_paper.json", "delta_paper.npz", "norm_metadata.json"])
def test_all_contract_files_invalidate_cached_identity(tmp_path, monkeypatch, asset):
    from deploy.fr3_sharpa_runtime import config, model_loader
    params, assets = tmp_path / "params", tmp_path / "assets"
    params.mkdir()
    assets.mkdir()
    for name in ("manifest.ocdbt", "_METADATA"):
        (params / name).write_bytes(b"before")
    for name in ("norm_stats.json", "delta_paper.json", "delta_paper.npz", "norm_metadata.json"):
        (assets / name).write_bytes(b"before")
    monkeypatch.setattr(config, "CHECKPOINT", tmp_path)
    monkeypatch.setattr(config, "NORM_STATS", assets / "norm_stats.json")
    monkeypatch.setattr(config, "_validation_cache", None)
    calls = []
    def verify():
        calls.append(True)
        return {"checkpoint_manifest_sha256": "a", "normalization_sha256": "b", "paper_algorithm": None}
    monkeypatch.setattr(model_loader, "checkpoint_contract", verify)
    config.prepare_policy_validation()
    config.prepare_policy_validation()
    assert len(calls) == 1
    path = (params if asset in ("manifest.ocdbt", "_METADATA") else assets) / asset
    path.write_bytes(b"after-longer")
    config.prepare_policy_validation()
    assert len(calls) == 2
    (params / "_METADATA").unlink()
    with pytest.raises(ValueError, match="removed"):
        config.prepare_policy_validation()


def test_expired_observation_diagnostics_preserve_each_attempt_without_images(tmp_path, capsys):
    from deploy.fr3_sharpa_runtime.run import log_expired_inference
    observation = {"_captured_monotonic_ns": 1_000_000_000,
                   "_timing": {"observation_encoding_ms": 12.0},
                   "observation/image": np.zeros((224, 224, 3))}
    timing = {"infer_chunk_entered_monotonic_ns": 1_020_000_000,
              "metadata_validation_ms": 0.1, "server": {"model_sampling_ms": 60.0}}
    for attempt in (1, 2):
        log_expired_inference(observation, timing, attempt=attempt, age_ns=250_000_000, directory=tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "inference_timing.jsonl").read_text().splitlines()]
    assert [row["attempt"] for row in rows] == [1, 2]
    assert rows[0]["limit_ms"] == 200 and rows[0]["age_ms"] == 250
    assert rows[0]["age_at_inference_start_ms"] == 20
    assert rows[0]["policy_client"] == timing
    assert "observation/image" not in rows[0]
    assert "observation-expired" in capsys.readouterr().err


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


@pytest.mark.parametrize("length", [1, 4, 15, 35, 49, 50])
def test_paper_rtc_short_reference_uses_new_state_and_time_axis(checkpoint, monkeypatch, length):
    from deploy.fr3_sharpa_runtime.paper_rtc import PaperRTCPolicy
    from deploy.fr3_sharpa_runtime.paper_model import normalization_bridge
    from deploy.fr3_sharpa_runtime.rtc_stream import canonical_prefix
    from openpi.policies.fr3_sharpa_policy import Fr3SharpaInputs, Fr3SharpaOutputs
    from openpi.models.model import ModelType
    from openpi.shared import nnx_utils
    from openpi_client import rtc

    _, assets, _ = checkpoint
    monkeypatch.setattr(nnx_utils, "module_jit", lambda f: f)
    stats = normalize.load(assets)
    bridge = normalization_bridge(model_for(assets), stats)
    mask = transforms.make_bool_mask(7, -22, 7, -22)
    sampled = {}

    class Model:
        action_dim, action_horizon = 58, 50

        def sample_actions(self, rng, observation, **kwargs):
            sampled.update(kwargs)
            # Exercise the actual Policy input/output path without a GPU model.
            return kwargs["rtc_actions"]

    policy = PaperRTCPolicy(Model(), rtc_action_repack=canonical_prefix,
        transforms=[Fr3SharpaInputs(ModelType.PI05), transforms.DeltaActions(mask),
                    transforms.Normalize(stats, use_quantiles=True), bridge],
        output_transforms=[replace(bridge, inverse=True), transforms.Unnormalize(stats, use_quantiles=True),
                           transforms.AbsoluteActions(mask), Fr3SharpaOutputs()])
    old = np.repeat((np.arange(50, dtype=np.float32) * .01 + .3)[:, None], 58, axis=1)
    reference = old[50-length:].copy()
    context = rtc.validate_context({"protocol_version": rtc.PROTOCOL_VERSION,
        "token": "paper-test", "source_number": 1,
        "source_sha256": hashlib.sha256(old.tobytes()).hexdigest(),
        "previous_actions": reference, "inference_delay_steps": min(5, length),
        "prefix_attention_horizon": min(25, length), "prefix_attention_schedule": "exp",
        "max_guidance_weight": 5.})
    for state in (np.linspace(-.3, .3, 58, dtype=np.float32), np.linspace(.1, .5, 58, dtype=np.float32)):
        observation = {"observation/state": state}
        for key in ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image"):
            observation[key] = np.zeros((8, 8, 3), dtype=np.uint8)
        result = policy.infer(observation, rtc_context=context)
        assert result["actions"].shape == (50, 58)
        np.testing.assert_allclose(result["actions"][:length], reference, atol=1e-5)
        target = np.asarray(sampled["rtc_actions"])[0]
        expected_arms = (reference[:, paper.ARMS] - state[paper.ARMS] - np.arange(length)[:, None]) / .5
        np.testing.assert_allclose(target[:length, paper.ARMS], expected_arms, atol=2e-5)
        np.testing.assert_array_equal(target[length:], 0.)
        np.testing.assert_array_equal(np.asarray(sampled["rtc_weights"])[0, length:], 0.)
        rtc.validate_receipt(result["rtc"], context)
    np.testing.assert_array_equal(context.previous_actions, reference)
    np.testing.assert_array_equal(old[50-length:], reference)


def test_paper_rtc_recipe_and_mode_boundaries(checkpoint):
    from deploy.fr3_sharpa_control.plan import repository_info
    root, _, _ = checkpoint
    repo = Path(__file__).resolve().parents[3]
    settings = DeploymentSettings(str(repo), str(root), mode="paper-rtc", speed="fast")
    plan = settings.plan()
    assert "paper-rtc" in repository_info(repo)["modes"]
    assert "--enable-paper-rtc" in plan["service"]
    assert "--paper-rtc" in plan["execute"] and "--paper-rtc" in plan["check"]
    assert plan["execute"][plan["execute"].index("--broker-mode") + 1] == "rtc"
    assert plan["execute"][plan["execute"].index("--num-chunks") + 1] == "0"
    with pytest.raises(ValueError, match="Paper RTC"):
        replace(settings, mode="rtc").plan()
    with pytest.raises(ValueError, match="RTC需fast"):
        replace(settings, speed="slow").plan()


def test_paper_rtc_rejects_unadapted_or_unwarmed_service():
    from deploy.fr3_sharpa_runtime.rtc_stream import require_paper_capabilities, PAPER_RTC_ADAPTER
    from openpi_client import rtc
    metadata = {"rtc_supported": True, "rtc_warmed_up": True, "rtc_action_dim": 58,
        "rtc_protocol_version": rtc.PROTOCOL_VERSION, "rtc_algorithm": "soft_prefix_guidance",
        "paper_algorithm": {"mode": "timestamp_zscore_arms"}, "rtc_input_adapter": PAPER_RTC_ADAPTER,
        "rtc_short_prefix_warmed_up": True}
    require_paper_capabilities(metadata)
    for key in ("paper_algorithm", "rtc_input_adapter", "rtc_short_prefix_warmed_up", "rtc_warmed_up"):
        with pytest.raises(ValueError, match="RTC"):
            require_paper_capabilities({k: v for k, v in metadata.items() if k != key})


def test_paper_rtc_capability_guard_precedes_sdk_connection(monkeypatch, tmp_path):
    from deploy.fr3_sharpa_runtime import run, motion_worker
    from deploy.fr3_sharpa_runtime.rtc_stream import require_paper_capabilities
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "setup.bash").write_text("")
    calls = []

    class LegacyRuntime:
        def __init__(self, url, *, paper_rtc=False):
            calls.append(paper_rtc)

        def inspect_metadata(self, *, require_rtc=False):
            assert require_rtc
            require_paper_capabilities({})

    monkeypatch.setattr(run, "SharpaRuntime", LegacyRuntime)
    monkeypatch.setattr(motion_worker, "MotionWorker", lambda *a, **kw: pytest.fail("SDK must not start"))
    args = SimpleNamespace(ros_overlay=overlay, sharpa_backend="sdk_direct", initialization="current",
                           broker_mode="rtc", paper_rtc=True, policy_url="test")
    with pytest.raises(ValueError, match="RTC"):
        run.execute_policy_prepare(args)
    assert calls == [True]
