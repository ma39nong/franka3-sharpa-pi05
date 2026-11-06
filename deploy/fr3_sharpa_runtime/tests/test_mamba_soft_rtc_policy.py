"""Synthetic sampler invariants; real trained weights are intentionally deferred."""

# ruff: noqa: SLF001, PT011
from dataclasses import replace
from pathlib import Path
import subprocess
import types

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import checkpoint
from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import record_evidence
from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import tiny_config
from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import tiny_network
from deploy.fr3_sharpa_runtime.tests.test_mamba_rtc import _paper_config
from openpi.models.pi0_mamba import MambaPi0
from openpi.shared import nnx_utils


@pytest.fixture
def original_sampler():
    root = Path(__file__).resolve().parents[3]
    source = subprocess.check_output(
        ["git", "show", "35071da7c3b0c5e996c936ec4775ef31488c4200:src/openpi/models/pi0_mamba.py"], cwd=root, text=True
    )
    module = types.ModuleType("mamba_before_soft_rtc")
    exec(compile(source, "pinned_pre_soft_rtc_model", "exec"), module.__dict__)
    return module.MambaPi0.sample_actions


@pytest.mark.parametrize("paper", [False, True])
def test_serial_exactly_matches_prior_sampler_and_does_not_call_vjp(tmp_path, monkeypatch, original_sampler, paper):
    import openpi.models.pi0_mamba as module

    with tiny_network():
        cfg = _paper_config(checkpoint(tmp_path, paper=True)[1]) if paper else tiny_config()
        model = cfg.create(jax.random.key(42))
        before = jax.tree.map(lambda v: (v.shape, str(v.dtype)), nnx.state(model).to_pure_dict())
        obs = cfg.fake_obs()
        noise = jax.random.normal(jax.random.key(2), (1, 50, 58))
        expected = original_sampler(model, jax.random.key(3), obs, noise=noise, num_steps=2)
        monkeypatch.setattr(module, "_rtc_guided_velocity", lambda *a, **k: pytest.fail("Serial must not build VJP"))
        actual = nnx_utils.module_jit(model.sample_actions)(jax.random.key(3), obs, noise=noise, num_steps=2)
        np.testing.assert_array_equal(actual, expected)
        after = jax.tree.map(lambda v: (v.shape, str(v.dtype)), nnx.state(model).to_pure_dict())
        assert before == after


@pytest.mark.parametrize("batch", [1, 2])
def test_soft_batch_zero_weights_no_cache_and_query_condition_once(monkeypatch, batch):
    with tiny_network():
        cfg = tiny_config()
        model = cfg.create(jax.random.key(42))
        parameters = nnx.state(model).to_pure_dict()
        obs = cfg.fake_obs(batch)
        noise = jax.random.normal(jax.random.key(2), (batch, 50, 58)) * 0.1
        targets = jax.random.normal(jax.random.key(3), noise.shape) * 0.1
        weights = jnp.zeros((batch, 50, 1)).at[:, :5].set(0.1)
        original = MambaPi0.encode_condition
        calls = []
        monkeypatch.setattr(MambaPi0, "encode_condition", lambda self, obs: (calls.append(1), original(self, obs))[1])
        with jax.disable_jit():
            guided = model.sample_actions(
                jax.random.key(5), obs, noise=noise, num_steps=2, rtc_actions=targets, rtc_weights=weights
            )
        assert len(calls) == 1
        sample = nnx_utils.module_jit(model.sample_actions)
        ordinary = sample(jax.random.key(5), obs, noise=noise, num_steps=2)
        zero = sample(jax.random.key(5), obs, noise=noise, num_steps=2, rtc_actions=targets, rtc_weights=weights * 0)
        np.testing.assert_allclose(zero, ordinary, rtol=1e-6, atol=1e-6)
        a = sample(jax.random.key(5), obs, noise=noise, num_steps=2, rtc_actions=targets, rtc_weights=weights)
        sample(jax.random.key(6), obs, noise=noise * 2, num_steps=2, rtc_actions=-targets, rtc_weights=weights)
        np.testing.assert_array_equal(
            a, sample(jax.random.key(5), obs, noise=noise, num_steps=2, rtc_actions=targets, rtc_weights=weights)
        )
        # Eager vs XLA-fused VJP differs by <=2.2e-5 on FP32 CPU;
        # repeat compiled queries below remain exactly equal.
        np.testing.assert_allclose(guided, a, rtol=1e-5, atol=3e-5)
        for i in range(batch):
            one = jax.tree.map(lambda x, index=i: x[index : index + 1], obs)
            single = sample(
                jax.random.key(5),
                one,
                noise=noise[i : i + 1],
                num_steps=2,
                rtc_actions=targets[i : i + 1],
                rtc_weights=weights[i : i + 1],
            )
            # GPU batch-size-dependent reduction order is amplified by VJP.
            # Bound both global relative error and every absolute error.
            difference = np.asarray(single - a[i : i + 1])
            relative = float(np.linalg.norm(difference) / max(np.linalg.norm(np.asarray(single)), 1e-8))
            maximum = float(np.max(np.abs(difference)))
            assert relative < 1e-5
            assert maximum < 1e-4
            record_evidence(f"soft_rtc_batch_{batch}_{i}", {"relative_l2": relative, "max_abs": maximum})
        assert not np.array_equal(a[:, :5], targets[:, :5])  # No physical/normalized hard copy.
        after = nnx.state(model).to_pure_dict()
        assert jax.tree.structure(parameters) == jax.tree.structure(after)
        for before, now in zip(jax.tree.leaves(parameters), jax.tree.leaves(after), strict=True):
            assert before.shape == now.shape
            assert before.dtype == now.dtype
            np.testing.assert_array_equal(before, now)


@pytest.mark.parametrize(
    "change", ["shape", "weights_shape", "nan", "negative", "large", "cap", "nan_cap", "hard", "trained"]
)
def test_invalid_soft_sampler_inputs_fail_before_inference(change):
    with tiny_network():
        cfg = tiny_config(rtc_max_delay=3 if change == "trained" else 0)
        model = cfg.create(jax.random.key(42))
        actions = jnp.zeros((1, 50, 58))
        kwargs = {"rtc_actions": actions, "rtc_weights": jnp.ones((1, 50, 1)), "rtc_max_guidance_weight": 5.0}
        if change == "shape":
            kwargs["rtc_actions"] = actions[:, :49]
        if change == "weights_shape":
            kwargs["rtc_weights"] = jnp.ones((1, 50))
        if change == "nan":
            kwargs["rtc_actions"] = actions.at[0, 0, 0].set(jnp.nan)
        if change == "negative":
            kwargs["rtc_weights"] *= -1
        if change == "large":
            kwargs["rtc_weights"] *= 2
        if change == "cap":
            kwargs["rtc_max_guidance_weight"] = -1.0
        if change == "nan_cap":
            kwargs["rtc_max_guidance_weight"] = float("nan")
        if change == "hard":
            kwargs["prefix_actions"] = actions
        with pytest.raises(ValueError):
            model.sample_actions(jax.random.key(1), cfg.fake_obs(), **kwargs)


@pytest.mark.parametrize("paper", [False, True])
@pytest.mark.parametrize("length", [1, 35, 50])
def test_soft_policy_encodes_physical_prefix_on_new_state_and_time(tmp_path, paper, length):
    from openpi_client import rtc

    from deploy.fr3_sharpa_runtime.mamba_model import MambaPaperSoftRTCPolicy
    from deploy.fr3_sharpa_runtime.mamba_model import MambaSoftRTCPolicy
    from deploy.fr3_sharpa_runtime.paper_model import normalization_bridge
    from deploy.fr3_sharpa_runtime.rtc_stream import canonical_prefix
    from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import observation
    from deploy.fr3_sharpa_runtime.tests.test_mamba_rtc import Tokens
    from openpi import transforms
    from openpi.models import model as model_lib
    from openpi.policies.fr3_sharpa_policy import Fr3SharpaInputs
    from openpi.policies.fr3_sharpa_policy import Fr3SharpaOutputs
    from openpi.shared import normalize
    from openpi.shared.delta_paper import ARMS

    _, assets, _ = checkpoint(tmp_path, paper=paper)
    stats = normalize.load(assets)
    mask = transforms.make_bool_mask(7, -22, 7, -22)
    with tiny_network():
        cfg = replace(_paper_config(assets), rtc_max_delay=0) if paper else tiny_config()
        model = cfg.create(jax.random.key(42))
        bridge = normalization_bridge(cfg, stats) if paper else lambda x: x
        inverse = replace(bridge, inverse=True) if paper else lambda x: x
        policy_class = MambaPaperSoftRTCPolicy if paper else MambaSoftRTCPolicy
        inputs = [
            Fr3SharpaInputs(model_lib.ModelType.PI05),
            *([transforms.DeltaActions(mask)] if paper else []),
            transforms.Normalize(stats, use_quantiles=paper),
            bridge,
            Tokens(),
        ]
        outputs = [
            inverse,
            transforms.Unnormalize(stats, use_quantiles=paper),
            *([transforms.AbsoluteActions(mask)] if paper else []),
            Fr3SharpaOutputs(),
        ]
        policy = policy_class(
            model,
            transforms=inputs,
            output_transforms=outputs,
            rtc_action_repack=canonical_prefix,
            sample_kwargs={"num_steps": 2},
        )
        reference = np.random.default_rng(5).normal(size=(length, 58)).astype(np.float32)
        context = rtc.validate_context(
            {
                "protocol_version": 1,
                "token": "soft-new-anchor",
                "source_number": 1,
                "source_sha256": "a" * 64,
                "previous_actions": reference,
                "inference_delay_steps": min(1, length),
                "prefix_attention_horizon": min(25, length),
                "prefix_attention_schedule": "exp",
                "max_guidance_weight": 5.0,
            }
        )
        captured = []
        actual_sample = policy._sample_actions

        def sample(rng, obs, **kwargs):
            captured.append(kwargs)
            return jnp.zeros((1, 50, 58))

        policy._sample_actions = sample
        for anchor in (-0.75, 0.9):
            obs = {**observation(), "observation/state": np.full(58, anchor, np.float32)}
            result = policy.infer(obs, rtc_context=context)
            padded = np.zeros((50, 58), np.float32)
            padded[:length] = reference
            if paper:
                padded[:, ARMS] -= anchor
            expected = bridge(
                transforms.Normalize({"actions": stats["actions"]}, use_quantiles=paper)({"actions": padded})
            )["actions"]
            np.testing.assert_allclose(captured[-1]["rtc_actions"][0, :length], expected[:length], rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(captured[-1]["rtc_actions"][0, length:], 0.0)
            rtc.validate_receipt(result["rtc"], context)
            assert not np.array_equal(result["actions"][:length], reference)
        np.testing.assert_array_equal(context.previous_actions, reference)
        # Execute the actual JIT/VJP graph as well, with the existing transforms.
        if length == 1:
            policy._sample_actions = actual_sample
            result = policy.infer(observation(), noise=np.zeros((50, 58), np.float32), rtc_context=context)
            assert np.isfinite(result["actions"]).all()
            assert result["actions"].shape == (50, 58)


def test_soft_policy_rejects_trained_and_mixed_requests_before_jit():
    from openpi_client import rtc

    from deploy.fr3_sharpa_runtime.mamba_model import MambaSoftRTCPolicy

    with tiny_network():
        model = tiny_config(rtc_max_delay=3).create(jax.random.key(42))
        with pytest.raises(ValueError, match="untrained RTC"):
            MambaSoftRTCPolicy(model)
        model = tiny_config().create(jax.random.key(42))
        with pytest.raises(ValueError, match="validated request"):
            MambaSoftRTCPolicy(model, sample_kwargs={"prefix_length": 0})
        policy = MambaSoftRTCPolicy(model, rtc_action_repack=lambda a: a)
        obs = jax.tree.map(lambda v: np.asarray(v)[0], model.config.fake_obs().to_dict())
        policy._sample_actions = lambda *a, **k: pytest.fail("invalid request must fail before sampler")
        for cap in (0.0, -1.0, 101.0, float("nan")):
            context = {
                "protocol_version": 1,
                "token": "bad-cap",
                "source_number": 1,
                "source_sha256": "a" * 64,
                "previous_actions": np.zeros((50, 58), np.float32),
                "inference_delay_steps": 1,
                "prefix_attention_horizon": 5,
                "prefix_attention_schedule": "exp",
                "max_guidance_weight": cap,
            }
            with pytest.raises(rtc.RTCProtocolError):
                policy.infer(obs, rtc_context=context)
        policy._sample_actions = lambda *a, **k: jnp.full((1, 50, 58), jnp.nan)
        with pytest.raises(ValueError, match="NaN or Inf"):
            policy.infer(obs)
