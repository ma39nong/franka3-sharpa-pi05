# Adapted from MambaAct 97414cfffc82225c6be34abf486f06c7eb11d22d.
# Apache-2.0; see third_party/mamba3/LICENSE-MambaAct.
"""Native NNX gated FiLM action blocks, with official zero initialization."""

import math

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models.mamba3.mixer import Mamba3Mixer
from openpi.models.mamba3.mixer import RMSNorm
from openpi.models.pi0 import posemb_sincos


class ActionBlock(nnx.Module):
    def __init__(self, cfg, mlp_dim, film_dim, *, rngs):
        width = cfg.hidden_size
        self.cond_proj = nnx.Linear(
            width,
            width,
            kernel_init=nnx.initializers.zeros,
            dtype=cfg.dtype,
            rngs=rngs,
            precision=jax.lax.Precision.HIGHEST,
        )
        self.mamba_norm = RMSNorm(width)
        self.mamba = Mamba3Mixer(cfg, rngs=rngs)
        self.film_norm = RMSNorm(width)
        self.film_in = nnx.Linear(width, film_dim, dtype=cfg.dtype, rngs=rngs, precision=jax.lax.Precision.HIGHEST)
        self.film_out = nnx.Linear(
            film_dim,
            3 * width,
            kernel_init=nnx.initializers.zeros,
            dtype=cfg.dtype,
            rngs=rngs,
            precision=jax.lax.Precision.HIGHEST,
        )
        self.film_out.bias.value = jnp.concatenate([jnp.zeros(2 * width), jnp.full(width, math.log(0.2 / 0.8))])
        self.ffn_norm = RMSNorm(width)
        self.ffn_in = nnx.Linear(width, mlp_dim, dtype=cfg.dtype, rngs=rngs, precision=jax.lax.Precision.HIGHEST)
        self.ffn_out = nnx.Linear(mlp_dim, width, dtype=cfg.dtype, rngs=rngs, precision=jax.lax.Precision.HIGHEST)

    def condition(self, z):
        scale, shift, gate = jnp.split(self.film_out(jax.nn.silu(self.film_in(self.film_norm(z)))), 3, axis=-1)
        return scale, shift, jax.nn.sigmoid(gate)

    def __call__(self, h, time, film):
        projected_time = self.cond_proj(jax.nn.silu(time))
        h = h + (projected_time[:, None] if projected_time.ndim == 2 else projected_time)
        scale, shift, gate = film
        u = self.mamba(self.mamba_norm(h))
        h = h + gate[:, None] * ((1 + scale[:, None]) * u + shift[:, None])
        return h + self.ffn_out(jax.nn.gelu(self.ffn_in(self.ffn_norm(h)), approximate=True))


class MambaActionExpert(nnx.Module):
    def __init__(self, cfg, action_dim, depth, mlp_dim, film_dim, *, rngs):
        width = cfg.hidden_size
        self.width = width
        self.action_in_proj = nnx.Linear(
            action_dim, width, dtype=cfg.dtype, rngs=rngs, precision=jax.lax.Precision.HIGHEST
        )
        self.time_mlp_in = nnx.Linear(width, width, dtype=cfg.dtype, rngs=rngs, precision=jax.lax.Precision.HIGHEST)
        self.time_mlp_out = nnx.Linear(width, width, dtype=cfg.dtype, rngs=rngs, precision=jax.lax.Precision.HIGHEST)
        self.blocks = nnx.Dict({f"block_{i}": ActionBlock(cfg, mlp_dim, film_dim, rngs=rngs) for i in range(depth)})
        self.final_norm = RMSNorm(width)
        self.action_out_proj = nnx.Linear(
            width, action_dim, dtype=cfg.dtype, rngs=rngs, precision=jax.lax.Precision.HIGHEST
        )

    def prepare_condition(self, z):
        return tuple(block.condition(z) for block in self.blocks.values())

    def __call__(self, noisy_actions, time, condition):
        if time.shape not in (noisy_actions.shape[:1], noisy_actions.shape[:2]):
            raise ValueError("Flow time must have shape (batch,) or (batch, action_horizon)")
        time = posemb_sincos(time.reshape(-1), self.width, min_period=4e-3, max_period=4.0).reshape(
            *time.shape, self.width
        )
        time = jax.nn.silu(self.time_mlp_out(jax.nn.silu(self.time_mlp_in(time))))
        h = self.action_in_proj(noisy_actions)
        for block, film in zip(self.blocks.values(), condition, strict=True):
            h = block(h, time, film)
        return self.action_out_proj(self.final_norm(h)).astype(jnp.float32)
