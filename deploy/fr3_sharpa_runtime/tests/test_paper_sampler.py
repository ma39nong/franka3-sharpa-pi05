"""CPU JAX tests of the actual Pi0 Euler loop and parameter-free inpainting."""
# ruff: noqa: PLC0415, E402
# Optional JAX tests skip before framework imports; model imports remain local.

import numpy as np
import pytest

jax = pytest.importorskip("jax")
from flax import nnx
import jax.numpy as jnp

from openpi.models.inpainting import correct_inpainting


def test_threshold_uses_updated_time_and_releases_final_steps():
    x = jnp.zeros((1, 50, 58))
    reference = jnp.ones((1, 4, 58)) * 0.2
    noise = jnp.ones_like(x) * 0.8
    corrected, counts = jax.jit(correct_inpainting)(x, reference, noise, jnp.float32(0.8))
    np.testing.assert_allclose(corrected[:, :4], 0.68, atol=1e-6)
    assert np.asarray(counts).tolist() == [1, 0]
    np.testing.assert_array_equal(corrected[:, 4:], 0)
    for threshold in [0.3, 0.2999999, 0.2, 0.0]:
        released, counts = correct_inpainting(x, reference, noise, jnp.float32(threshold))
        np.testing.assert_array_equal(released, x)
        assert not np.asarray(counts).any()


def test_identity_correction_is_soft_and_unstable_propagation_is_audited():
    x = jnp.zeros((1, 50, 58))
    ref = jnp.ones((1, 4, 58)) * 0.1
    noise = jnp.zeros_like(x)
    soft, _ = correct_inpainting(x, ref, noise, 0.5)
    identity, counts = correct_inpainting(x, ref, noise, 0.5, correction_matrix=jnp.zeros((46 * 58, 4 * 58)))
    np.testing.assert_array_equal(identity, soft)
    assert np.asarray(counts).tolist() == [1, 0]
    guarded, counts = correct_inpainting(x, ref, noise, 0.5, correction_matrix=jnp.ones((46 * 58, 4 * 58)))
    np.testing.assert_array_equal(guarded, soft)
    assert np.asarray(counts).tolist() == [1, 1]


class DummyLLM(nnx.Module):
    def __call__(self, inputs, **kwargs):
        return ((None, inputs[1]), {})


def tiny_sampler(monkeypatch):
    from openpi.models import model
    from openpi.models import pi0

    monkeypatch.setattr(model, "preprocess_observation", lambda rng, observation, **kw: observation)

    class Sampler(nnx.Module):
        sample_actions = pi0.Pi0.sample_actions

        def __init__(self):
            self.action_dim, self.action_horizon = 58, 50
            self.PaliGemma = nnx.Dict(llm=DummyLLM())

        def embed_prefix(self, obs):
            return jnp.zeros((1, 1, 58)), jnp.ones((1, 1), bool), jnp.zeros(1, bool)

        def embed_suffix(self, observation, x, time):
            return x, jnp.ones((1, 50), bool), jnp.zeros(50, bool), None

        def action_out_proj(self, x):
            return x * 0.2 + 0.1

    observation = model.Observation(images={}, image_masks={}, state=jnp.zeros((1, 58)))
    return Sampler(), observation


def test_actual_sampler_default_matches_original_euler_and_jitted_reference_path(monkeypatch):
    from openpi.shared.nnx_utils import module_jit

    sampler, obs = tiny_sampler(monkeypatch)
    sample = module_jit(sampler.sample_actions)
    noise = jax.random.normal(jax.random.key(8), (1, 50, 58))
    baseline = sample(jax.random.key(4), obs, noise=noise)
    expected = noise
    for _ in range(10):
        expected = expected - 0.1 * (expected * 0.2 + 0.1)
    np.testing.assert_allclose(baseline, expected, atol=1e-6)
    reference = jnp.ones((1, 4, 58)) * 0.25
    guided, counts = sample(jax.random.key(4), obs, noise=noise, initial_actions=reference)
    assert guided.shape == baseline.shape == (1, 50, 58)
    assert np.asarray(counts).tolist() == [6, 0]
    assert not np.allclose(guided[:, :4], baseline[:, :4])
    assert not np.allclose(guided[:, :4], reference)  # Final steps are free, never a pasted prefix.
    np.testing.assert_allclose(guided[:, 4:], baseline[:, 4:])
    np.testing.assert_array_equal(sample(jax.random.key(4), obs, noise=noise), baseline)


def test_actual_sampler_rejects_wrong_reference_shape(monkeypatch):
    sampler, obs = tiny_sampler(monkeypatch)
    with pytest.raises(ValueError, match="four reference"):
        sampler.sample_actions(jax.random.key(0), obs, initial_actions=jnp.zeros((1, 3, 58)))


def test_real_policy_and_timing_wrapper_return_diagnostics_without_changing_output_shape(monkeypatch):
    from deploy.fr3_sharpa_runtime.timing import TimedPolicy
    from openpi.policies.policy import Policy

    sampler, _ = tiny_sampler(monkeypatch)
    policy = Policy(
        sampler,
        transforms=[lambda obs: {"state": obs["observation/state"], "image": {}, "image_mask": {}}],
        output_transforms=[lambda out: {"actions": np.asarray(out["actions"], np.float32)}],
    )
    timed = TimedPolicy(policy)
    observation = {"observation/state": np.zeros(58, np.float32)}
    baseline = timed.infer(observation)
    guided = timed.infer(observation, initial_actions=np.ones((4, 58), np.float32) * 0.2)
    assert baseline["actions"].shape == guided["actions"].shape == (50, 58)
    assert guided["inpainting_diagnostics"]["applied_steps"] == 6
    assert guided["diagnostic_timing"]["model_sampling_ms"] >= 0
    assert "inpainting_diagnostics" not in baseline
