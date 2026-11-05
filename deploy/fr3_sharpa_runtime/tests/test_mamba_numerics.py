"""Independent recurrent and Torch references for the extracted JAX SSD/mixer."""
# ruff: noqa: PLC0415, N803, N806, N812

from dataclasses import replace

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import record_evidence
from openpi.models.mamba3 import core
from openpi.models.mamba3.config import Mamba3Config
from openpi.models.mamba3.mixer import Mamba3Mixer


def ssd_scan(x, q, k, adt, scale, skip, d):
    """Independent state-space recurrence, not the production triangular SSD."""
    state = jnp.zeros((x.shape[0], x.shape[2], x.shape[3], k.shape[-1]), jnp.float32)

    def step(carry, inputs):
        xt, qt, kt, at, st, diag = inputs
        decayed = jnp.exp(at)[..., None, None] * carry
        yt = jnp.einsum("bhn,bhpn->bhp", qt, decayed, precision=jax.lax.Precision.HIGHEST) + (diag + d)[..., None] * xt
        return decayed + st[..., None, None] * xt[..., :, None] * kt[..., None, :], yt

    _, result = jax.lax.scan(step, state, tuple(jnp.swapaxes(v, 0, 1) for v in (x, q, k, adt, scale, skip)))
    return jnp.swapaxes(result, 0, 1)


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("length", [1, 50, 65])
def test_ssd_independent_forward_grad_jit_and_batch(batch, length):
    keys = jax.random.split(jax.random.key(42), 7)
    x = jax.random.normal(keys[0], (batch, length, 2, 3)) * 0.3
    q = jax.random.normal(keys[1], (batch, length, 2, 4)) * 0.3
    k = jax.random.normal(keys[2], q.shape) * 0.3
    adt = -jax.random.uniform(keys[3], (batch, length, 2)) * 0.2 - 0.01
    scale = jax.random.uniform(keys[4], adt.shape) * 0.1
    skip = jax.random.normal(keys[5], adt.shape) * 0.1
    d = jax.random.uniform(keys[6], (2,))
    args = (x, q, k, adt, scale, skip, d)

    def actual(*xs):
        return core.ssd_forward(*xs, chunk_size=64)[0]

    expected = ssd_scan(*args)
    np.testing.assert_allclose(actual(*args), expected, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(jax.jit(actual)(*args), expected, rtol=1e-4, atol=1e-5)
    for i in range(batch):
        single = (*tuple(v[i : i + 1] for v in args[:-1]), d)
        np.testing.assert_allclose(actual(*single), expected[i : i + 1], rtol=1e-4, atol=1e-5)
    g_actual = jax.grad(lambda *xs: (actual(*xs) ** 2).sum(), argnums=tuple(range(7)))(*args)
    g_reference = jax.grad(lambda *xs: (ssd_scan(*xs) ** 2).sum(), argnums=tuple(range(7)))(*args)
    for a, b in zip(g_actual, g_reference, strict=True):
        assert np.isfinite(a).all()
        assert np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-8) <= 1e-3
    # Internal multi-chunk padding does not alter H=50 outputs.
    padded = (*tuple(jnp.pad(v, ((0, 0), (0, 128 - length), *(((0, 0),) * (v.ndim - 2)))) for v in args[:-1]), d)
    np.testing.assert_allclose(actual(*padded)[:, :length], expected, rtol=1e-4, atol=1e-5)


def torch_mixer_reference(values, hidden, cfg):
    """Explicit exponential-trapezoidal recurrence in a separate autodiff system."""
    import torch
    import torch.nn.functional as F

    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32

    def linear(x, name):
        return x.to(dtype) @ values[name + "/kernel"].to(dtype)

    bsz, length, _ = hidden.shape
    h, p, n = cfg.num_heads, cfg.head_dim, cfg.state_size
    sizes = (
        cfg.intermediate_size,
        cfg.intermediate_size,
        n * cfg.ngroups,
        n * cfg.ngroups,
        h,
        h,
        h,
        cfg.num_rope_angles,
    )
    z, x, b, c, raw_dt, raw_a, trap, angle = linear(hidden, "in_proj").split(sizes, dim=-1)

    def norm_bc(v, name):
        v = v.float().reshape(bsz, length, cfg.ngroups, n)
        v = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + cfg.bcnorm_eps) * values[name + "_norm/weight"]
        return v.repeat_interleave(h // cfg.ngroups, dim=2) + values[name + "_bias"].squeeze(1)

    b, c = norm_bc(b, "B"), norm_bc(c, "C")
    dt, trap = F.softplus(raw_dt.float() + values["dt_bias"]), torch.sigmoid(trap.float())
    a = -torch.where(raw_a.float() >= 0, 1 + raw_a.float().clamp(min=0), 1 / (1 - raw_a.float().clamp(max=0)))
    a = torch.minimum(a, torch.tensor(-cfg.A_floor))
    theta = torch.cumsum(torch.tanh(angle.float())[:, :, None] * torch.pi * dt[..., None], dim=1) % (2 * torch.pi)

    def rotate(v):
        pairs = v.reshape(bsz, length, h, n // 2, 2)
        angles = F.pad(theta, (0, n // 2 - cfg.num_rope_angles))
        x0, x1 = pairs[..., 0], pairs[..., 1]
        return torch.stack(
            [x0 * angles.cos() - x1 * angles.sin(), x0 * angles.sin() + x1 * angles.cos()], -1
        ).reshape_as(v)

    q, k = rotate(c), rotate(b)
    x = x.float().reshape(bsz, length, h, p)
    z = z.float().reshape_as(x)
    state = torch.zeros((bsz, h, p, n))
    prev_kv = torch.zeros_like(state)
    outputs = []
    for t in range(length):
        kv = x[:, t, :, :, None] * k[:, t, :, None, :]
        # Current exponential-trapezoidal update: includes previous endpoint.
        state = (
            torch.exp(a[:, t] * dt[:, t])[..., None, None]
            * (state + (dt[:, t] * (1 - trap[:, t]))[..., None, None] * prev_kv)
            + (dt[:, t] * trap[:, t])[..., None, None] * kv
        )
        outputs.append(torch.einsum("bhn,bhpn->bhp", q[:, t], state) + values["D"][None, :, None] * x[:, t])
        prev_kv = kv
    y = torch.stack(outputs, 1) * F.silu(z)
    return linear(y.reshape(bsz, length, -1), "out_proj").float()


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_full_mixer_torch_forward_and_parameter_gradients(dtype):
    from flax import traverse_util
    import torch

    cfg = Mamba3Config(hidden_size=16, state_size=8, head_dim=8, dtype=dtype)
    mixer = Mamba3Mixer(cfg, rngs=nnx.Rngs(4))
    hidden = jax.random.normal(jax.random.key(3), (3, 50, 16)) * 0.2
    values = {
        k: v
        for k, v in traverse_util.flatten_dict(nnx.state(mixer, nnx.Param).to_pure_dict(), sep="/").items()
        if v is not None
    }
    tv = {k: torch.tensor(np.asarray(v), requires_grad=True) for k, v in values.items()}
    tx = torch.tensor(np.asarray(hidden), requires_grad=True)
    reference = torch_mixer_reference(tv, tx, cfg)
    actual = mixer(hidden).astype(jnp.float32)
    tolerance = 1e-4 if dtype == "float32" else 3e-2
    rel = np.linalg.norm(np.asarray(actual) - reference.detach().numpy()) / max(
        np.linalg.norm(reference.detach().numpy()), 1e-8
    )
    record_evidence(
        f"mixer_{dtype}",
        {
            "forward_relative_l2": float(rel),
            "forward_max_abs": float(np.max(np.abs(np.asarray(actual) - reference.detach().numpy()))),
            "forward_limit": tolerance,
        },
    )
    assert rel <= tolerance
    graph, params = nnx.split(mixer)

    def objective(parameters, inputs):
        return (nnx.merge(graph, parameters)(inputs).astype(jnp.float32) ** 2).mean()

    pg, xg = jax.grad(objective, argnums=(0, 1))(params, hidden)
    reference.square().mean().backward()
    pg = traverse_util.flatten_dict(pg.to_pure_dict(), sep="/")
    tolerance = 1e-3 if dtype == "float32" else 5e-2
    gradient_errors = {}
    for key in values:
        a, b = np.asarray(pg[key]), tv[key].grad.numpy()
        rel = np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-8)
        gradient_errors[key] = {"relative_l2": float(rel), "max_abs": float(np.max(np.abs(a - b)))}
        assert rel <= tolerance, (key, rel)
    record_evidence(f"mixer_{dtype}_gradients", {"parameters": gradient_errors, "relative_limit": tolerance})
    a, b = np.asarray(xg), tx.grad.numpy()
    assert np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-8) <= tolerance
    np.testing.assert_allclose(
        jax.jit(lambda ps, x: nnx.merge(graph, ps)(x))(params, hidden),
        actual,
        rtol=1e-4 if dtype == "float32" else 0.03,
        atol=1e-4,
    )
    # No cross-batch state or dependence on previous calls.
    repeated = mixer(hidden)
    np.testing.assert_array_equal(repeated, actual)
    single = mixer(hidden[1:2])
    np.testing.assert_allclose(single, actual[1:2], rtol=tolerance, atol=1e-5)
    assert not any("state" in path or "cache" in path for path in values)


def test_config_rejects_invalid_heads():
    with pytest.raises(ValueError, match="divide"):
        replace(Mamba3Config(), head_dim=63)


def test_main_action_expert_h50_d58_jit_backward():
    """Production 4x512 SISO expert; synthetic condition, full VLM not involved."""
    from openpi.models.mamba3.action_expert import MambaActionExpert

    cfg = Mamba3Config()
    expert = MambaActionExpert(cfg, 58, 4, 2048, 2048, rngs=nnx.Rngs(42))
    graph, params = nnx.split(expert)
    x = jax.random.normal(jax.random.key(15), (1, 50, 58)) * 0.1
    z = jax.random.normal(jax.random.key(16), (1, 512))

    def loss(ps):
        m = nnx.merge(graph, ps)
        prediction = m(x, jnp.array([0.3]), m.prepare_condition(z))
        return jnp.mean((prediction - 0.2) ** 2)

    value, grads = jax.jit(jax.value_and_grad(loss))(params)
    assert np.isfinite(value)
    assert all(np.isfinite(v).all() for v in jax.tree.leaves(grads))
    assert sum(float(jnp.sum(v.astype(jnp.float32) ** 2)) for v in jax.tree.leaves(grads)) > 0
