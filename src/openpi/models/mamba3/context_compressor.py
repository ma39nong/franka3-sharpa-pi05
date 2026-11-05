# Adapted from MambaAct pi0_mamba_pytorch.py at 97414cfffc82225c6be34abf486f06c7eb11d22d.
# Apache-2.0; see third_party/mamba3/LICENSE-MambaAct.
"""Masked segment pooling and global queries, ported to Flax NNX 0.10.2.

True means valid. All three camera slots are pooled in fixed order, followed by
all language/state-text slots. Empty rows yield zero condition (never NaN).
"""

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models.mamba3.mixer import RMSNorm


def segment_mean_pool(tokens, valid, count):
    length = tokens.shape[1]
    # Same integer segment boundaries as official implementation, vectorized.
    starts = jnp.arange(count) * length // count
    ends = (jnp.arange(count) + 1) * length // count
    positions = jnp.arange(length)
    weights = ((positions[None, :] >= starts[:, None]) & (positions[None, :] < ends[:, None]))[None] & valid[:, None, :]
    sums = jnp.einsum(
        "bsl,bld->bsd", weights.astype(jnp.float32), tokens.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST
    )
    result = sums / jnp.maximum(weights.sum(-1, keepdims=True), 1)
    return result.astype(tokens.dtype), weights.any(-1)


class ContextCompressor(nnx.Module):
    def __init__(
        self,
        prefix_width,
        width,
        mlp_dim,
        heads,
        *,
        vision_tokens=44,
        language_tokens=16,
        global_tokens=4,
        dtype="bfloat16",
        rngs,
    ):
        self.vision_tokens, self.language_tokens, self.global_tokens = vision_tokens, language_tokens, global_tokens
        self.heads = heads
        self.context_proj = nnx.Linear(prefix_width, width, dtype=dtype, rngs=rngs, precision=jax.lax.Precision.HIGHEST)
        self.vision_type_embed = nnx.Param(jnp.zeros((1, 1, width)))
        self.language_type_embed = nnx.Param(jnp.zeros((1, 1, width)))
        self.global_type_embed = nnx.Param(jnp.zeros((1, 1, width)))
        self.global_queries = nnx.Param(jax.random.normal(rngs.params(), (global_tokens, width)) * 0.02)
        # Torch MultiheadAttention's Q/K/V and out projection, with FP32 softmax.
        self.query = nnx.Linear(width, width, dtype=dtype, rngs=rngs, precision=jax.lax.Precision.HIGHEST)
        self.key = nnx.Linear(width, width, dtype=dtype, rngs=rngs, precision=jax.lax.Precision.HIGHEST)
        self.value = nnx.Linear(width, width, dtype=dtype, rngs=rngs, precision=jax.lax.Precision.HIGHEST)
        self.out = nnx.Linear(width, width, dtype=dtype, rngs=rngs, precision=jax.lax.Precision.HIGHEST)
        self.norm = RMSNorm(width)
        self.mlp_in = nnx.Linear(width, mlp_dim, dtype=dtype, rngs=rngs, precision=jax.lax.Precision.HIGHEST)
        self.mlp_out = nnx.Linear(mlp_dim, width, dtype=dtype, rngs=rngs, precision=jax.lax.Precision.HIGHEST)

    def __call__(self, hidden, mask, vision_length):
        if not 0 < vision_length <= hidden.shape[1]:
            raise ValueError("Compressor requires explicit visual/language layout")
        projected = self.context_proj(jnp.where(mask[..., None], hidden, 0))
        projected = jnp.where(mask[..., None], projected, 0)
        vision, vm = segment_mean_pool(projected[:, :vision_length], mask[:, :vision_length], self.vision_tokens)
        language, lm = segment_mean_pool(projected[:, vision_length:], mask[:, vision_length:], self.language_tokens)
        batch, _, width = projected.shape
        q = self.query(jnp.broadcast_to(self.global_queries.value, (batch, self.global_tokens, width)))
        q = q.reshape(batch, self.global_tokens, self.heads, -1)
        k = self.key(projected).reshape(batch, projected.shape[1], self.heads, -1)
        v = self.value(projected).reshape(batch, projected.shape[1], self.heads, -1)
        logits = jnp.einsum(
            "bqhd,bkhd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST
        ) / jnp.sqrt(width // self.heads)
        logits = jnp.where(mask[:, None, None, :], logits, jnp.finfo(jnp.float32).min)
        weights = jax.nn.softmax(logits, axis=-1) * mask[:, None, None, :]
        attended = jnp.einsum(
            "bhqk,bkhd->bqhd", weights, v.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST
        ).reshape(batch, self.global_tokens, width)
        global_ctx = self.out(attended)
        gm = jnp.broadcast_to(mask.any(-1)[:, None], (batch, self.global_tokens))
        tokens = jnp.concatenate(
            [
                vision + self.vision_type_embed.value,
                language + self.language_type_embed.value,
                global_ctx + self.global_type_embed.value,
            ],
            axis=1,
        )
        valid = jnp.concatenate([vm, lm, gm], axis=1)
        tokens += self.mlp_out(jax.nn.gelu(self.mlp_in(self.norm(tokens)), approximate=True))
        return (jnp.where(valid[..., None], tokens, 0).sum(1) / jnp.maximum(valid.sum(1, keepdims=True), 1)).astype(
            projected.dtype
        )
