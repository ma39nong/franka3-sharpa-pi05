"""Combined training semantics: evidence, new time axis, sampler and service boundaries."""

from dataclasses import fields
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from deploy.fr3_sharpa_control.plan import DeploymentSettings
from deploy.fr3_sharpa_control.plan import inspect_checkpoint
from deploy.fr3_sharpa_control.plan import repository_info
from deploy.fr3_sharpa_runtime.paper_model import PaperPi0Config
from deploy.fr3_sharpa_runtime.paper_model import normalization_bridge
from deploy.fr3_sharpa_runtime.tests import test_paper_deployment as paper_fixtures
from deploy.fr3_sharpa_runtime.tests.test_paper_deployment import model_for
from deploy.fr3_sharpa_runtime.train_paper_rtc_model import TrainPaperRTCPi0Config
from deploy.fr3_sharpa_runtime.train_rtc_contract import require_train_paper_capabilities
from openpi.shared import delta_paper as paper


@pytest.fixture
def paper_checkpoint(tmp_path):
    return paper_fixtures.checkpoint.__wrapped__(tmp_path)


@pytest.fixture
def trained_paper(paper_checkpoint):
    root, assets, _ = paper_checkpoint
    algorithm = inspect_checkpoint(root)["paper_algorithm"]
    model = dict(
        pi05=True,
        action_dim=58,
        action_horizon=50,
        max_token_len=256,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        rtc_max_delay=10,
        paper_mode=algorithm["mode"],
        paper_beta=algorithm["beta"],
        paper_std_floor=algorithm["std_floor"],
        paper_asset_sha256=algorithm["bundle_sha256"],
    )
    record = {
        "config": {
            "fields": {"model": {"fields": model}, "data": {"fields": {"action_hz": 30, "extra_delta_transform": True}}}
        },
        "inputs": {
            "norm_stats_sha256": hashlib.sha256((assets / "norm_stats.json").read_bytes()).hexdigest(),
            "paper_bundle_sha256": algorithm["bundle_sha256"],
        },
    }
    path = root / "train_rtc_record.json"
    path.write_text(json.dumps(record))
    return root, assets, path, record


def test_recognition_plan_and_loader(trained_paper, monkeypatch, capsys):
    import sys

    from deploy.fr3_sharpa_runtime import config
    from deploy.fr3_sharpa_runtime import model_loader
    from deploy.fr3_sharpa_runtime import serve
    from deploy.fr3_sharpa_runtime.model_profiles import get_model_profile

    root, assets, _, _ = trained_paper
    info = inspect_checkpoint(root)
    assert info["supported_modes"] == ["serial", "train-paper-rtc"]
    repo = Path(__file__).resolve().parents[3]
    assert "train-paper-rtc" in repository_info(repo)["modes"]
    settings = DeploymentSettings(str(repo), str(root), mode="train-paper-rtc", speed="fast")
    plan = settings.plan()
    assert "--enable-train-paper-rtc" in plan["service"]
    assert "--train-paper-rtc" in plan["execute"]
    assert plan["execute"][plan["execute"].index("--rtc-initial-delay-steps") + 1] == "10"
    for mode in ("rtc", "paper-rtc", "train-rtc"):
        with pytest.raises(ValueError, match="不支持所选"):
            replace(settings, mode=mode).plan()
    profile = get_model_profile("full-delta")
    for key, value in dict(
        CHECKPOINT=root, NORM_STATS=assets / "norm_stats.json", MODEL_PROFILE=profile, CONFIG_NAME=profile.config_name
    ).items():
        monkeypatch.setattr(config, key, value)
    training = model_loader.build_config()
    assert isinstance(training.model, TrainPaperRTCPi0Config)
    assert training.model.rtc_max_delay == 10
    assert training.model.paper_asset_sha256 == info["paper_algorithm"]["bundle_sha256"]
    monkeypatch.setattr(sys, "argv", ["serve", "--enable-train-paper-rtc", "--check-only"])
    serve.main()
    assert json.loads(capsys.readouterr().out)["rtc_algorithm"] == "trained_prefix_conditioning"
    from deploy.fr3_sharpa_runtime import run

    base_args = ["runtime", "--broker-mode", "rtc", "--speed", "fast", "--num-chunks", "0", "--check"]
    monkeypatch.setattr(sys, "argv", [*base_args, "--train-paper-rtc"])
    run.main()
    checked = json.loads(capsys.readouterr().out)
    assert checked["train_rtc"] and checked["paper_rtc"]
    monkeypatch.setattr(sys, "argv", [*base_args, "--train-rtc"])
    with pytest.raises(SystemExit):
        run.main()
    for flag in ("--enable-train-rtc", "--enable-paper-rtc", "--enable-rtc"):
        monkeypatch.setattr(sys, "argv", ["serve", flag, "--check-only"])
        with pytest.raises(SystemExit):
            serve.main()


@pytest.mark.parametrize(
    "field,value",
    [
        ("paper_beta", 0.6),
        ("paper_mode", "global_zscore_arms"),
        ("paper_asset_sha256", "wrong"),
        ("paper_std_floor", 0.01),
    ],
)
def test_training_paper_contract_must_match_assets(trained_paper, field, value):
    root, _, path, record = trained_paper
    record["config"]["fields"]["model"]["fields"][field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="mismatch"):
        inspect_checkpoint(root)


def test_known_records_match_exact_manifest_and_metadata(tmp_path, monkeypatch):
    from deploy.fr3_sharpa_runtime import train_rtc_contract as contract

    root = tmp_path / "renamed-checkpoint"
    (root / "params").mkdir(parents=True)
    (root / "params/manifest.ocdbt").write_bytes(b"original")
    (root / "params/_METADATA").write_bytes(b"metadata")
    records = tmp_path / "training_records"
    records.mkdir()
    record = {
        "checkpoint_manifest_sha256": hashlib.sha256(b"original").hexdigest(),
        "checkpoint_metadata_sha256": hashlib.sha256(b"metadata").hexdigest(),
    }
    path = records / "known.json"
    path.write_text(json.dumps(record))
    monkeypatch.delenv("SHARPA_TRAIN_RTC_RECORD", raising=False)
    monkeypatch.setattr(contract, "__file__", str(tmp_path / "contract.py"))
    assert contract.resolve_training_record(root) == path
    (root / "params/manifest.ocdbt").write_bytes(b"another weight with same stats")
    assert contract.resolve_training_record(root) is None
    (root / "params/manifest.ocdbt").write_bytes(b"original")
    (root / "params/_METADATA").write_bytes(b"changed")
    with pytest.raises(ValueError, match="metadata hash"):
        contract.resolve_training_record(root)


@pytest.mark.parametrize("length", [1, 10, 35, 50])
def test_short_prefix_normalization_and_absolute_roundtrip(paper_checkpoint, monkeypatch, length):
    from openpi_client import rtc

    from deploy.fr3_sharpa_runtime.rtc_stream import canonical_prefix
    from deploy.fr3_sharpa_runtime.train_paper_rtc import TrainPaperRTCPolicy
    from openpi import transforms
    from openpi.models.model import ModelType
    from openpi.policies.fr3_sharpa_policy import Fr3SharpaInputs
    from openpi.policies.fr3_sharpa_policy import Fr3SharpaOutputs
    from openpi.shared import nnx_utils
    from openpi.shared import normalize

    _, assets, _ = paper_checkpoint
    monkeypatch.setattr(nnx_utils, "module_jit", lambda fn, **kw: fn)
    stats = normalize.load(assets)
    bridge = normalization_bridge(model_for(assets), stats)
    mask = transforms.make_bool_mask(7, -22, 7, -22)
    captured = {}

    class Model:
        action_dim, action_horizon, rtc_max_delay = 58, 50, 10

        def sample_actions(self, rng, obs, **kwargs):
            captured.update(kwargs)
            return kwargs["prefix_actions"]

    policy = TrainPaperRTCPolicy(
        Model(),
        rtc_action_repack=canonical_prefix,
        transforms=[
            Fr3SharpaInputs(ModelType.PI05),
            transforms.DeltaActions(mask),
            transforms.Normalize(stats, use_quantiles=True),
            bridge,
        ],
        output_transforms=[
            replace(bridge, inverse=True),
            transforms.Unnormalize(stats, use_quantiles=True),
            transforms.AbsoluteActions(mask),
            Fr3SharpaOutputs(),
        ],
    )
    reference = np.full((length, 58), 0.3, np.float32)
    delay = min(length, 10)
    context = dict(
        protocol_version=1,
        token="combined",
        source_number=1,
        source_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
        previous_actions=reference,
        inference_delay_steps=delay,
        prefix_attention_horizon=delay,
        prefix_attention_schedule="exp",
        max_guidance_weight=5.0,
    )
    for state in (np.full(58, 0.1, np.float32), np.full(58, -0.2, np.float32)):
        obs = {
            "observation/state": state,
            **{
                k: np.zeros((8, 8, 3), np.uint8)
                for k in ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image")
            },
        }
        result = policy.infer(obs, rtc_context=context)
        assert captured["prefix_length"] == delay
        assert "rtc_weights" not in captured
        target = np.asarray(captured["prefix_actions"])[0]
        expected = (reference[:, paper.ARMS] - state[paper.ARMS] - np.arange(length)[:, None]) / 0.5
        np.testing.assert_allclose(target[:length, paper.ARMS], expected, atol=2e-5)
        np.testing.assert_array_equal(target[length:], 0)
        np.testing.assert_allclose(result["actions"][:delay], reference[:delay], atol=1e-5)
        rtc.validate_receipt(result["rtc"], rtc.validate_context(context))


def test_combined_capabilities_reject_soft_paper_and_plain_train():
    from deploy.fr3_sharpa_runtime.rtc_stream import PAPER_RTC_ADAPTER
    from deploy.fr3_sharpa_runtime.runtime import SharpaRuntime

    meta = dict(
        rtc_algorithm="trained_prefix_conditioning",
        rtc_supported=True,
        rtc_warmed_up=True,
        rtc_protocol_version=1,
        rtc_action_dim=58,
        train_rtc={"max_delay": 10},
        rtc_warmed_prefix_lengths=list(range(11)),
        paper_algorithm={"beta": 0.5},
        rtc_input_adapter=PAPER_RTC_ADAPTER,
        rtc_short_prefix_warmed_up=True,
    )
    require_train_paper_capabilities(meta)
    for key in ("paper_algorithm", "rtc_short_prefix_warmed_up", "rtc_input_adapter"):
        with pytest.raises(ValueError):
            require_train_paper_capabilities({k: v for k, v in meta.items() if k != key})
    with pytest.raises(ValueError):
        require_train_paper_capabilities({**meta, "rtc_algorithm": "soft_prefix_guidance"})
    # No socket or hardware: exercise just the dispatch guard.
    runtime = object.__new__(SharpaRuntime)
    runtime.train_rtc = True
    runtime.paper_rtc = False
    with pytest.raises(ValueError, match="Train Paper RTC"):
        runtime._require_rtc(meta)


def test_real_combined_sampler_matches_paper_and_fixes_prefix(monkeypatch):
    from flax import nnx

    from openpi.models import gemma
    from openpi.models import siglip
    from openpi.shared import nnx_utils

    original = siglip.Module
    monkeypatch.setattr(siglip, "Module", lambda **kw: original(**{**kw, "variant": "mu/224"}))
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 32)
    chol = np.eye(8, dtype=np.float32)
    chol[1, 0] = 0.5
    chol[1, 1] = np.sqrt(0.75)
    monkeypatch.setattr(paper, "load_bundle", lambda *a, **kw: ({}, {"noise_cholesky": chol}))
    base_config = PaperPi0Config(
        pi05=True,
        dtype="float32",
        action_dim=2,
        action_horizon=4,
        max_token_len=2,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        paper_beta=0.5,
        paper_asset_sha256="fixture",
    )
    base = base_config.create(jax.random.key(0))
    config = TrainPaperRTCPi0Config(
        **{f.name: getattr(base_config, f.name) for f in fields(PaperPi0Config)}, rtc_max_delay=3
    )
    model = config.load(nnx.state(base).to_pure_dict())
    assert model.rtc_max_delay == 3
    sample = nnx_utils.module_jit(model.sample_actions, static_argnames=("prefix_length",))
    obs, act = config.fake_obs(1), config.fake_act(1)
    key = jax.random.key(7)
    baseline = nnx_utils.module_jit(base.sample_actions)(key, obs, num_steps=2)
    np.testing.assert_allclose(sample(key, obs, num_steps=2), baseline, atol=1e-6)
    np.testing.assert_allclose(
        model.noise(key, (1, 4, 2)),
        np.asarray(jax.random.normal(key, (1, 4, 2))).reshape(1, 8).dot(chol.T).reshape(1, 4, 2),
    )
    for delay in (1, 3):
        result = sample(key, obs, num_steps=2, prefix_actions=act, prefix_length=delay)
        np.testing.assert_array_equal(result[:, :delay], act[:, :delay])
        assert np.isfinite(result).all()
