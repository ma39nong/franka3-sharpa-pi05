"""Independent reverse-flow/VJP checks, using synthetic action fields only."""

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import record_evidence
from openpi.models.mamba3.action_expert import MambaActionExpert
from openpi.models.mamba3.config import Mamba3Config
from openpi.models.pi0 import _rtc_guided_velocity


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("time", [0.0, 1e-6, 0.35, 1.0])
def test_linear_vjp_matches_analytic_reverse_time_direction(batch, time):
    rng = np.random.default_rng(9)
    x = jnp.asarray(rng.normal(size=(batch, 50, 58)), jnp.float32)
    target = jnp.asarray(rng.normal(size=x.shape), jnp.float32)
    weights = jnp.linspace(1.0, 0.0, 50)[None, :, None] * jnp.ones((batch, 1, 1))
    # Cross-dimension coupling catches accidental J @ e in place of J.T @ e.
    matrix = jnp.eye(58) * 0.2 + jnp.diag(jnp.full(57, 0.03), 1)
    bias = jnp.arange(batch, dtype=jnp.float32)[:, None, None] * 0.1

    def velocity(a):
        return jnp.matmul(a, matrix.T, precision=jax.lax.Precision.HIGHEST) + bias

    v = velocity(x)
    clean = x - time * v
    error = (target - clean) * weights
    correction = jnp.matmul(error, jnp.eye(58) - time * matrix, precision=jax.lax.Precision.HIGHEST)
    cap = 5.0
    coefficient = cap if time in (0.0, 1.0) else min(cap, (time**2 + (1 - time) ** 2) / (time * (1 - time)))
    expected = v - coefficient * correction
    actual = _rtc_guided_velocity(velocity, x, jnp.asarray(time), target, weights, cap)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    for i in range(batch):

        def single_velocity(a, sample_bias=bias[i : i + 1]):
            return jnp.matmul(a, matrix.T, precision=jax.lax.Precision.HIGHEST) + sample_bias

        single = _rtc_guided_velocity(single_velocity, x[i : i + 1], time, target[i : i + 1], weights[i : i + 1], cap)
        np.testing.assert_allclose(single, actual[i : i + 1], rtol=1e-6, atol=1e-6)


def test_controlled_prefix_error_decreases_without_hard_copy():
    x = jnp.zeros((1, 50, 58))
    target = jnp.ones_like(x)
    weights = jnp.zeros((1, 50, 1)).at[:, :5].set(0.1)

    def velocity(a):
        return a * 0.2

    plain = x - 0.1 * velocity(x)
    guided = x - 0.1 * _rtc_guided_velocity(velocity, x, 0.5, target, weights, 5.0)
    assert float(jnp.mean((guided[:, :5] - target[:, :5]) ** 2)) < float(jnp.mean((plain[:, :5] - target[:, :5]) ** 2))
    assert not np.array_equal(guided[:, :5], target[:, :5])
    np.testing.assert_array_equal(guided[:, 5:], plain[:, 5:])
    zero = _rtc_guided_velocity(velocity, x, 0.5, target, weights * 0, 5.0)
    np.testing.assert_array_equal(zero, velocity(x))


def test_small_mamba_input_gradient_adjoint_and_finite_difference():
    cfg = Mamba3Config(hidden_size=16, state_size=8, head_dim=8, dtype="float32")
    expert = MambaActionExpert(cfg, 58, 1, 32, 32, rngs=nnx.Rngs(42))
    x = jax.random.normal(jax.random.key(1), (2, 50, 58)) * 0.1
    tangent = jax.random.normal(jax.random.key(2), x.shape)
    cotangent = jax.random.normal(jax.random.key(3), x.shape)
    condition = expert.prepare_condition(jax.random.normal(jax.random.key(4), (2, 16)))
    scalar_time = 0.4
    time = jnp.full((2,), scalar_time)

    def clean(a):
        return a - scalar_time * expert(a, time, condition)

    _, jvp = jax.jvp(clean, (x,), (tangent,))
    _, pullback = jax.vjp(clean, x)
    vjp = pullback(cotangent)[0]
    np.testing.assert_allclose(jnp.vdot(jvp, cotangent), jnp.vdot(tangent, vjp), rtol=1e-5, atol=1e-4)
    eps = 1e-3
    finite = (clean(x + eps * tangent) - clean(x - eps * tangent)) / (2 * eps)
    relative = float(jnp.linalg.norm(jvp - finite) / jnp.linalg.norm(jvp))
    assert relative < 0.003
    target = x * 0.2
    weights = jnp.ones((2, 50, 1)) * 0.1
    actual = _rtc_guided_velocity(lambda a: expert(a, time, condition), x, scalar_time, target, weights, 5.0)
    for i in range(2):
        single_condition = jax.tree.map(lambda v, index=i: v[index : index + 1], condition)

        def one_velocity(a, sample_time=time[i : i + 1], sample_condition=single_condition):
            return expert(a, sample_time, sample_condition)

        single = _rtc_guided_velocity(
            one_velocity,
            x[i : i + 1],
            scalar_time,
            target[i : i + 1],
            weights[i : i + 1],
            5.0,
        )
        np.testing.assert_allclose(actual[i : i + 1], single, rtol=1e-5, atol=1e-5)
    assert np.isfinite(actual).all()
    record_evidence(
        "soft_rtc_mamba_input_gradient", {"finite_difference_relative_l2": relative, "batch": 2, "shape": [2, 50, 58]}
    )
