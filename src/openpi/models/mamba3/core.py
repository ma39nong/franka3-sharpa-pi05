# Copyright 2026 Cosmo Santoni.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SISO SSD adapted from mamba3-jax ffa55848fc513ca91aa330bda4c33c86dab74827.

Modified: extract only stateless SISO operations; retain upstream FP32 SSD math.
No LM, MIMO, decode cache, packed sequences or newer NNX APIs are needed.
"""

# Keep upstream mathematical variable names and callable signature.
# ruff: noqa: N803, N806, E741, FBT001, FBT002, RET504

import jax
import jax.numpy as jnp


def heavy_tail_activation(x: jnp.ndarray) -> jnp.ndarray:
    """Positive heavy-tail activation for data-dependent A: 1 + x for x >= 0, else 1 / (1 - x)."""
    neg = jnp.clip(x, max=0.0)
    pos = jnp.clip(x, min=0.0)
    return pos + jnp.reciprocal(1.0 - neg)


def cumulative_angles(
    angle_proj: jnp.ndarray,
    dt: jnp.ndarray,
    init_state: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Cumulative rotary angle cumsum(tanh(angle) * pi * dt) over the sequence, mod 2*pi.

    Packed sequences are intentionally unsupported in this extraction.
    """
    vals = jnp.tanh(angle_proj) * jnp.pi * dt[..., None]
    cumulative = jnp.cumsum(vals, axis=1)
    ang = cumulative
    if init_state is not None:
        ang = ang + init_state[:, None]
    return jnp.mod(ang, 2.0 * jnp.pi)


def apply_rope_pairwise(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """Rotate the leading 2*S channels of x as (..., N/2, 2) pairs; identity on the rest."""
    n = x.shape[-1]
    s = cos.shape[-1]
    pairs = x.reshape(*x.shape[:-1], n // 2, 2)
    x0, x1 = pairs[..., 0], pairs[..., 1]
    if s < n // 2:
        pad = n // 2 - s
        cos = jnp.concatenate([cos, jnp.ones((*cos.shape[:-1], pad), cos.dtype)], axis=-1)
        sin = jnp.concatenate([sin, jnp.zeros((*sin.shape[:-1], pad), sin.dtype)], axis=-1)
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    return jnp.stack([o0, o1], axis=-1).reshape(x.shape)


def _pad_seq_dim(x: jnp.ndarray, pad_size: int) -> jnp.ndarray:
    """Pad zeros at the end of the sequence dimension (axis=1)."""
    if pad_size == 0:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[1] = (0, pad_size)
    return jnp.pad(x, pad_width, mode="constant", constant_values=0.0)


def segsum(x: jnp.ndarray) -> jnp.ndarray:
    """Stable segment sum calculation. Input: (..., T) -> Output: (..., T, T)."""
    T = x.shape[-1]
    x_cumsum = jnp.cumsum(x, axis=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = jnp.tril(jnp.ones((T, T), dtype=bool), k=0)
    x_segsum = jnp.where(mask, x_segsum, -jnp.inf)
    return x_segsum


def _ssd_single_chunk(x, q, k, adt, scale, skip, D, initial_states, return_final_states):
    """Single-chunk fast path (seq_len <= chunk_size), skipping the inter-chunk machinery."""
    num_heads = x.shape[2]
    dtype = x.dtype
    k_scaled = k * scale[..., None]
    D_residual = D.reshape(1, 1, num_heads, 1) * x
    skip_residual = skip[..., None] * x

    A_t = jnp.transpose(adt, (0, 2, 1))
    A_cumsum = jnp.cumsum(A_t, axis=-1)
    l = A_cumsum.shape[-1]
    diff = A_cumsum[..., :, None] - A_cumsum[..., None, :]
    strict_lower = jnp.tril(jnp.ones((l, l), dtype=bool), k=-1)
    L_mat = jnp.where(strict_lower, jnp.exp(jnp.minimum(diff, 0.0)), 0.0).astype(dtype)
    y = (
        jnp.einsum("blhn,bshn,bhls,bshp->blhp", q, k_scaled, L_mat, x, precision=jax.lax.Precision.HIGHEST)
        + D_residual
        + skip_residual
    )

    if initial_states is not None:
        init = initial_states[:, 0]
        y = y + jnp.einsum(
            "blhn,bhpn,bhl->blhp", q, init, jnp.exp(A_cumsum).astype(dtype), precision=jax.lax.Precision.HIGHEST
        )

    final_state = None
    if return_final_states:
        decay = jnp.exp(A_cumsum[..., -1:] - A_cumsum)
        final_state = jnp.einsum("blhn,bhl,blhp->bhpn", k_scaled, decay, x, precision=jax.lax.Precision.HIGHEST)
        if initial_states is not None:
            final_state = final_state + init * jnp.exp(A_cumsum[..., -1])[..., None, None]
    return y, final_state


def ssd_forward(
    x: jnp.ndarray,
    q: jnp.ndarray,
    k: jnp.ndarray,
    adt: jnp.ndarray,
    scale: jnp.ndarray,
    skip: jnp.ndarray,
    D: jnp.ndarray,
    chunk_size: int,
    initial_states: jnp.ndarray | None = None,
    return_final_states: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """Exponential-trapezoidal SSD forward pass with chunked computation.

    Args:
        x: Value tensor (batch_size, seq_len, num_heads, head_dim)
        q: Rotated, biased C (batch_size, seq_len, num_heads, state_size)
        k: Rotated, biased B (batch_size, seq_len, num_heads, state_size)
        adt: Per-token decay A_t * dt_t (batch_size, seq_len, num_heads)
        scale: K-row scale gamma + shifted_gamma (batch_size, seq_len, num_heads)
        skip: Diagonal band gamma * <q, k> (batch_size, seq_len, num_heads)
        D: Skip connection weights (num_heads,)
        chunk_size: Size of chunks for efficient computation
        initial_states: Optional initial SSM states (batch, 1, heads, head_dim, state_size)
        return_final_states: Whether to return final SSM states

    Returns:
        y: Output tensor (batch_size, seq_len, num_heads, head_dim)
        final_state: Optional final states (batch_size, num_heads, head_dim, state_size)
    """
    _B_size, seq_len, num_heads, _head_dim = x.shape

    # Fast path: single chunk, skip inter-chunk machinery
    if seq_len <= chunk_size:
        return _ssd_single_chunk(x, q, k, adt, scale, skip, D, initial_states, return_final_states)

    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

    # Pad tensors along sequence dimension
    x_padded = _pad_seq_dim(x, pad_size)
    q_padded = _pad_seq_dim(q, pad_size)
    k_padded = _pad_seq_dim(k, pad_size)
    adt_padded = _pad_seq_dim(adt, pad_size)
    scale_padded = _pad_seq_dim(scale, pad_size)
    skip_padded = _pad_seq_dim(skip, pad_size)

    # Scale K, D residual, and diagonal (i, i) skip band
    k_scaled = k_padded * scale_padded[..., None]
    D_residual = D.reshape(1, 1, num_heads, 1) * x_padded
    skip_residual = skip_padded[..., None] * x_padded

    # Chunk everything
    def chunk_tensor(t):
        b, cl, *remaining = t.shape
        return t.reshape(b, cl // chunk_size, chunk_size, *remaining)

    x_blk = chunk_tensor(x_padded)
    q_blk = chunk_tensor(q_padded)
    k_blk = chunk_tensor(k_scaled)
    adt_blk = chunk_tensor(adt_padded)

    # A cumsum over intra-chunk time dimension
    A_blk2 = jnp.transpose(adt_blk, (0, 3, 1, 2))
    A_cumsum = jnp.cumsum(A_blk2, axis=-1)

    # Decay matrices (cast to activation dtype for matmuls)
    dtype = x.dtype

    # 1. Intra-chunk (strictly lower-triangular; the diagonal is the skip band)
    l = A_cumsum.shape[-1]
    diff = A_cumsum[..., :, None] - A_cumsum[..., None, :]
    strict_lower = jnp.tril(jnp.ones((l, l), dtype=bool), k=-1)
    L_mat = jnp.where(strict_lower, jnp.exp(jnp.minimum(diff, 0.0)), 0.0).astype(dtype)
    Y_diag = jnp.einsum(
        "bclhn,bcshn,bhcls,bcshp->bclhp", q_blk, k_blk, L_mat, x_blk, precision=jax.lax.Precision.HIGHEST
    )

    # 2. States within each chunk
    decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum).astype(dtype)
    states = jnp.einsum("bclhn,bhcl,bclhp->bchpn", k_blk, decay_states, x_blk, precision=jax.lax.Precision.HIGHEST)

    # 3. Inter-chunk recurrence
    if initial_states is None:
        initial_states = jnp.zeros_like(states[:, :1, ...])
    states = jnp.concatenate([initial_states, states], axis=1)

    A_end = A_cumsum[..., -1]
    A_end_padded = jnp.pad(A_end, ((0, 0), (0, 0), (1, 0)))
    decay_chunk = jnp.exp(segsum(A_end_padded)).astype(dtype)
    new_states = jnp.einsum("bhzc,bchpn->bzhpn", decay_chunk, states, precision=jax.lax.Precision.HIGHEST)
    states, final_state = new_states[:, :-1, ...], new_states[:, -1, ...]

    # 4. Convert states -> outputs
    state_decay_out = jnp.exp(A_cumsum).astype(dtype)
    Y_off = jnp.einsum("bclhn,bchpn,bhcl->bclhp", q_blk, states, state_decay_out, precision=jax.lax.Precision.HIGHEST)

    y = Y_diag + Y_off
    b, c, l, h, p = y.shape
    y = y.reshape(b, c * l, h, p)
    y = y + D_residual + skip_residual

    # Remove padding
    if pad_size > 0:
        y = y[:, :seq_len, :, :]

    return (y, final_state) if return_final_states else (y, None)
