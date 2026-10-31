"""Train RTC evidence, UI/queue boundaries and normalized prefix conditioning; no devices."""

# ruff: noqa: PLC0415 -- heavyweight model imports stay inside offline model tests.

from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np
from openpi_client import rtc
import pytest

from deploy.fr3_sharpa_control.plan import DeploymentSettings
from deploy.fr3_sharpa_control.plan import inspect_checkpoint
from deploy.fr3_sharpa_control.plan import plan_environment
from deploy.fr3_sharpa_runtime.rtc_stream import RTCTimeline
from deploy.fr3_sharpa_runtime.rtc_stream import require_capabilities
from deploy.fr3_sharpa_runtime.train_rtc_contract import inspect_train_rtc
from deploy.fr3_sharpa_runtime.train_rtc_contract import require_train_capabilities


@pytest.fixture
def trained_checkpoint(tmp_path):
    root = tmp_path / "2000"
    params = root / "params"
    params.mkdir(parents=True)
    entries = {
        name: {"key_metadata": [{"key": name}, {"key": kind}], "value_metadata": {"write_shape": shape}}
        for name, kind, shape in [("action_in_proj", "kernel", [58, 128]), ("action_out_proj", "bias", [58])]
    }
    (params / "_METADATA").write_text(json.dumps({"tree_metadata": entries}))
    (params / "manifest.ocdbt").write_bytes(b"fixture")
    stats = root / "assets/train/norm_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text(
        json.dumps(
            {
                "norm_stats": {
                    k: {"mean": [0.0] * 58, "std": [1.0] * 58, "q01": [-2.0] * 58, "q99": [3.0] * 58}
                    for k in ("state", "actions")
                }
            }
        )
    )
    model = {
        "pi05": True,
        "action_dim": 58,
        "action_horizon": 50,
        "max_token_len": 256,
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "rtc_max_delay": 10,
    }
    record = {
        "config": {
            "fields": {
                "model": {"fields": model},
                "data": {"fields": {"action_hz": 30.0, "extra_delta_transform": True}},
            }
        },
        "inputs": {"norm_stats_sha256": hashlib.sha256(stats.read_bytes()).hexdigest()},
    }
    path = root / "train_rtc_record.json"
    path.write_text(json.dumps(record))
    return root, stats, path, record


def test_evidence_and_plan(trained_checkpoint):
    root, stats, path, _ = trained_checkpoint
    info = inspect_checkpoint(root)
    assert info["supported_modes"] == ["serial", "train-rtc"]
    assert info["profile"] == "full-delta"
    repo = str(Path(__file__).resolve().parents[3])
    plan = DeploymentSettings(repo, str(root), mode="train-rtc", speed="fast").plan()
    assert "--enable-train-rtc" in plan["service"]
    assert "--train-rtc" in plan["execute"]
    assert plan["execute"][plan["execute"].index("--rtc-initial-delay-steps") + 1] == "10"
    assert plan_environment(plan)["SHARPA_TRAIN_RTC_RECORD"] == str(path)
    with pytest.raises(ValueError, match="不支持所选推理模式"):
        DeploymentSettings(repo, str(root), mode="rtc", speed="fast").plan()
    with pytest.raises(ValueError, match="冲突"):
        inspect_checkpoint(root, semantics="absolute")
    stats.write_text(stats.read_text() + " ")
    with pytest.raises(ValueError, match="hash"):
        inspect_train_rtc(stats, root)


@pytest.mark.parametrize("delay", [-1, 50, True, 2.5])
def test_invalid_record_delay(trained_checkpoint, delay):
    root, stats, path, record = trained_checkpoint
    record["config"]["fields"]["model"]["fields"]["rtc_max_delay"] = delay
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="integer"):
        inspect_train_rtc(stats, root)


def test_algorithm_capabilities_are_separate():
    meta = {
        "rtc_algorithm": "trained_prefix_conditioning",
        "rtc_supported": True,
        "rtc_warmed_up": True,
        "rtc_protocol_version": 1,
        "rtc_action_dim": 58,
        "train_rtc": {"max_delay": 3},
        "rtc_warmed_prefix_lengths": [0, 1, 2, 3],
    }
    require_train_capabilities(meta)
    with pytest.raises(ValueError, match="soft-prefix"):
        require_capabilities(meta)
    with pytest.raises(ValueError, match="all supported"):
        require_train_capabilities({**meta, "rtc_warmed_prefix_lengths": [0, 3]})


def test_queue_stops_at_committed_prefix_deadline():
    actions = np.zeros((50, 58), np.float32)
    timeline = RTCTimeline(actions, 0, 1)
    context = timeline.snapshot(0, delay_steps=3, schedule="exp", weight=5, hard_prefix=True)
    assert context["prefix_attention_horizon"] == 3
    with pytest.raises(TimeoutError):
        timeline.install(actions, rtc.make_receipt(context), 4 / 30)
    timeline = RTCTimeline(actions, 0, 1)
    context = timeline.snapshot(0, delay_steps=3, schedule="exp", weight=5, hard_prefix=True)
    assert timeline.install(actions, rtc.make_receipt(context), 3 / 30)["skipped"] == 3


def test_policy_reanchors_delta_and_uses_hard_sampler(monkeypatch):
    from deploy.fr3_sharpa_runtime.rtc_stream import canonical_prefix
    from deploy.fr3_sharpa_runtime.train_rtc import TrainRTCPolicy
    from openpi import transforms
    from openpi.models.model import ModelType
    from openpi.policies.fr3_sharpa_policy import Fr3SharpaInputs
    from openpi.policies.fr3_sharpa_policy import Fr3SharpaOutputs
    from openpi.shared import nnx_utils
    from openpi.shared.normalize import NormStats

    monkeypatch.setattr(nnx_utils, "module_jit", lambda fn, **kw: fn)
    recorded = {}

    class Model:
        action_dim = 58
        action_horizon = 50
        rtc_max_delay = 10

        def sample_actions(self, rng, obs, **kwargs):
            recorded.update(kwargs)
            return kwargs["prefix_actions"]

    mask = transforms.make_bool_mask(7, -22, 7, -22)
    stats = {
        k: NormStats(mean=np.zeros(58), std=np.ones(58), q01=np.full(58, -2.0), q99=np.full(58, 3.0))
        for k in ("state", "actions")
    }
    policy = TrainRTCPolicy(
        Model(),
        rtc_action_repack=canonical_prefix,
        transforms=[
            Fr3SharpaInputs(ModelType.PI05),
            transforms.DeltaActions(mask),
            transforms.Normalize(stats, use_quantiles=True),
        ],
        output_transforms=[
            transforms.Unnormalize(stats, use_quantiles=True),
            transforms.AbsoluteActions(mask),
            Fr3SharpaOutputs(),
        ],
    )
    old = np.full((50, 58), 0.3, np.float32)
    context = RTCTimeline(old, 0, 1).snapshot(0, delay_steps=5, schedule="exp", weight=5, hard_prefix=True)
    for state in (np.full(58, 0.1, np.float32), np.full(58, -0.2, np.float32)):
        obs = {
            "observation/state": state,
            **{
                k: np.zeros((4, 4, 3), np.uint8)
                for k in ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image")
            },
        }
        result = policy.infer(obs, rtc_context=context)
        assert recorded["prefix_length"] == 5
        assert "rtc_weights" not in recorded
        expected = old.copy()
        expected[:, np.asarray(mask)] -= state[np.asarray(mask)]
        np.testing.assert_allclose(recorded["prefix_actions"][0], (expected + 2) / 5 * 2 - 1, atol=1e-6)
        np.testing.assert_allclose(result["actions"][:5], old[:5], atol=1e-6)
        rtc.validate_receipt(result["rtc"], context)
    bad = {**context, "inference_delay_steps": 11, "prefix_attention_horizon": 11}
    with pytest.raises(ValueError, match="training max"):
        policy.infer(obs, rtc_context=bad)


def test_real_model_checkpoint_compatibility_and_static_jit(monkeypatch):
    from flax import nnx
    import jax

    from deploy.fr3_sharpa_runtime.train_rtc_model import TrainRTCPi0Config
    from openpi.models import gemma
    from openpi.models import siglip
    from openpi.models.pi0_config import Pi0Config
    from openpi.shared import nnx_utils

    original = siglip.Module
    monkeypatch.setattr(siglip, "Module", lambda **kw: original(**{**kw, "variant": "mu/224"}))
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 32)
    base_config = Pi0Config(
        pi05=True,
        dtype="float32",
        action_dim=2,
        action_horizon=4,
        max_token_len=2,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    base = base_config.create(jax.random.key(0))
    config = TrainRTCPi0Config(**{f.name: getattr(base_config, f.name) for f in fields(Pi0Config)}, rtc_max_delay=3)
    model = config.load(nnx.state(base).to_pure_dict())
    assert jax.tree.map(lambda x: x.shape, nnx.state(base).to_pure_dict()) == jax.tree.map(
        lambda x: x.shape, nnx.state(model).to_pure_dict()
    )
    obs, act = config.fake_obs(1), config.fake_act(1)
    sample = nnx_utils.module_jit(model.sample_actions, static_argnames=("prefix_length",))
    baseline = nnx_utils.module_jit(base.sample_actions)(jax.random.key(2), obs, num_steps=2)
    np.testing.assert_array_equal(sample(jax.random.key(2), obs, num_steps=2), baseline)
    for d in (1, 3):
        result = sample(jax.random.key(2), obs, num_steps=2, prefix_actions=act, prefix_length=d)
        np.testing.assert_array_equal(result[:, :d], act[:, :d])
        assert np.isfinite(result).all()


def test_loader_selects_trained_model_and_check_only_needs_no_gpu(trained_checkpoint, monkeypatch, capsys):
    import sys

    from deploy.fr3_sharpa_runtime import config
    from deploy.fr3_sharpa_runtime import model_loader
    from deploy.fr3_sharpa_runtime import serve
    from deploy.fr3_sharpa_runtime.model_profiles import get_model_profile
    from deploy.fr3_sharpa_runtime.train_rtc_model import TrainRTCPi0Config

    root, stats, _, _ = trained_checkpoint
    profile = get_model_profile("full-delta")
    monkeypatch.setattr(config, "CHECKPOINT", root)
    monkeypatch.setattr(config, "NORM_STATS", stats)
    monkeypatch.setattr(config, "MODEL_PROFILE", profile)
    monkeypatch.setattr(config, "CONFIG_NAME", profile.config_name)
    training = model_loader.build_config()
    assert isinstance(training.model, TrainRTCPi0Config)
    assert training.model.rtc_max_delay == 10
    assert training.data.extra_delta_transform
    monkeypatch.setattr(sys, "argv", ["serve", "--enable-train-rtc", "--check-only"])
    serve.main()
    report = json.loads(capsys.readouterr().out)
    assert report["rtc_algorithm"] == "trained_prefix_conditioning"
    assert report["rtc_warmed_up"] is False
    monkeypatch.setattr(sys, "argv", ["serve", "--enable-rtc", "--check-only"])
    with pytest.raises(SystemExit):
        serve.main()
