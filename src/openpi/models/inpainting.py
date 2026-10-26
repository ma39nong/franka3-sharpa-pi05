"""Parameter-free correction inside the Euler sampler (t=1 noise, t=0 actions)."""

import jax
import jax.numpy as jnp


def correct_inpainting(x, reference, noise, time_new, *, correction_matrix=None, threshold=0.3):
    """Return corrected state and [applied_steps, skipped_propagations]."""
    count = reference.shape[1]

    def apply(value):
        desired = (1.0 - time_new) * reference + time_new * noise[:, :count]
        delta = desired - value[:, :count]
        value = value.at[:, :count].set(desired)
        skipped = jnp.int32(0)
        if correction_matrix is not None:
            propagated = delta.reshape(delta.shape[0], -1) @ correction_matrix.T
            valid = jnp.all(jnp.isfinite(propagated)) & (jnp.max(jnp.abs(propagated)) <= 1.0)
            value = jax.lax.cond(
                valid, lambda v: v.at[:, count:].add(propagated.reshape(v[:, count:].shape)), lambda v: v, value
            )
            skipped = (~valid).astype(jnp.int32)
        return value, jnp.stack([jnp.int32(1), skipped])

    # Explicit tolerance makes the threshold independent of float32 accumulation.
    return jax.lax.cond(time_new > threshold + 1e-6, apply, lambda value: (value, jnp.zeros(2, dtype=jnp.int32)), x)
