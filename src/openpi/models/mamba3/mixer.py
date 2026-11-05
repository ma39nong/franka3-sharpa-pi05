# Copyright 2026 Cosmo Santoni.
# Licensed under the Apache License, Version 2.0; see third_party/mamba3/LICENSE-jax.
"""Adapted from mamba3-jax modeling.py, ffa55848fc513ca91aa330bda4c33c86dab74827.

Modified: NNX 0.10.2, SISO only, explicit compute dtype, FP32 sensitive math,
no persistent carry or inference/packed-sequence API. Projection order unchanged.
"""

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models.mamba3 import core
from openpi.models.mamba3.config import Mamba3Config


class RMSNorm(nnx.Module):
    def __init__(self, width, eps=1e-6):
        self.weight = nnx.Param(jnp.ones(width, jnp.float32))
        self.eps = eps

    def __call__(self, x):
        value = x.astype(jnp.float32)
        value *= jax.lax.rsqrt(jnp.mean(value**2, axis=-1, keepdims=True) + self.eps)
        return (value * self.weight.value).astype(x.dtype)


class Mamba3Mixer(nnx.Module):
    def __init__(self, cfg: Mamba3Config, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.in_proj = nnx.Linear(
            cfg.hidden_size,
            cfg.d_in_proj,
            use_bias=False,
            dtype=cfg.dtype,
            rngs=rngs,
            precision=jax.lax.Precision.HIGHEST,
        )
        initial_dt = jnp.exp(
            jax.random.uniform(rngs.params(), (cfg.num_heads,))
            * (jnp.log(cfg.time_step_max) - jnp.log(cfg.time_step_min))
            + jnp.log(cfg.time_step_min)
        )
        initial_dt = jnp.maximum(initial_dt, cfg.time_step_floor)
        self.dt_bias = nnx.Param(initial_dt + jnp.log(-jnp.expm1(-initial_dt)))
        self.B_bias = nnx.Param(jnp.ones((cfg.num_heads, 1, cfg.state_size), jnp.float32))
        self.C_bias = nnx.Param(jnp.ones((cfg.num_heads, 1, cfg.state_size), jnp.float32))
        self.B_norm = RMSNorm(cfg.state_size, cfg.bcnorm_eps)
        self.C_norm = RMSNorm(cfg.state_size, cfg.bcnorm_eps)
        self.D = nnx.Param(jnp.ones(cfg.num_heads, jnp.float32))
        self.out_proj = nnx.Linear(
            cfg.intermediate_size,
            cfg.hidden_size,
            use_bias=False,
            dtype=cfg.dtype,
            rngs=rngs,
            precision=jax.lax.Precision.HIGHEST,
        )

    def streams(self, hidden):
        """Expose the numerical boundary for independent reference/gradient checks."""
        cfg = self.cfg
        batch, length, _ = hidden.shape
        inner, heads, state = cfg.intermediate_size, cfg.num_heads, cfg.state_size
        sizes = (inner, inner, state * cfg.ngroups, state * cfg.ngroups, heads, heads, heads)
        boundaries = [sum(sizes[: i + 1]) for i in range(len(sizes))]
        z, x, b, c, raw_dt, raw_a, raw_trap, angles = jnp.split(self.in_proj(hidden), boundaries, axis=-1)
        # FP32 before normalization/rotation, exponentials, dt and cumulative angles.
        b = self.B_norm(b.astype(jnp.float32).reshape(batch, length, cfg.ngroups, state))
        c = self.C_norm(c.astype(jnp.float32).reshape(batch, length, cfg.ngroups, state))
        b = jnp.repeat(b, heads // cfg.ngroups, axis=2) + self.B_bias.value.squeeze(1)
        c = jnp.repeat(c, heads // cfg.ngroups, axis=2) + self.C_bias.value.squeeze(1)
        dt = jax.nn.softplus(raw_dt.astype(jnp.float32) + self.dt_bias.value)
        a = jnp.minimum(-core.heavy_tail_activation(raw_a.astype(jnp.float32)), -cfg.A_floor)
        trap = jax.nn.sigmoid(raw_trap.astype(jnp.float32))
        angles = jnp.broadcast_to(
            angles[:, :, None, :].astype(jnp.float32), (batch, length, heads, cfg.num_rope_angles)
        )
        angles = core.cumulative_angles(angles, dt)
        q = core.apply_rope_pairwise(c, jnp.cos(angles), jnp.sin(angles))
        k = core.apply_rope_pairwise(b, jnp.cos(angles), jnp.sin(angles))
        return (
            z.reshape(batch, length, heads, cfg.head_dim),
            x.reshape(batch, length, heads, cfg.head_dim).astype(jnp.float32),
            q,
            k,
            a * dt,
            dt,
            trap,
        )

    def __call__(self, hidden):
        z, x, q, k, adt, dt, trap = self.streams(hidden)
        gamma = dt * trap
        next_dt = jnp.concatenate([dt[:, 1:], jnp.zeros_like(dt[:, :1])], axis=1)
        next_trap = jnp.concatenate([trap[:, 1:], jnp.zeros_like(trap[:, :1])], axis=1)
        y, _ = core.ssd_forward(
            x,
            q,
            k,
            adt,
            gamma + next_dt * (1 - next_trap),
            gamma * jnp.sum(q * k, axis=-1),
            self.D.value,
            self.cfg.chunk_size,
        )
        gated = (y * jax.nn.silu(z.astype(jnp.float32))).reshape(hidden.shape[0], hidden.shape[1], -1)
        return self.out_proj(gated)
