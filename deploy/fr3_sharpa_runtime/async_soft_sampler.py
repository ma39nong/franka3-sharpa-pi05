"""New sampler only: four phase-aligned reference nodes at the next segment start."""

import einops
import jax
import jax.numpy as jnp

from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask


def sample_async_soft(self, rng, observation, *, noise=None, reference=None, reference_start=0, num_steps=10):
    """Bind on a new service instance; never replace Pi0.sample_actions globally."""
    observation = model_lib.preprocess_observation(None, observation, train=False)
    batch = observation.state.shape[0]
    if self.action_horizon != 50 or self.action_dim != 58:
        raise ValueError("Async Soft sampler requires native 50x58 actions")
    if noise is None:
        noise = jax.random.normal(rng, (batch, 50, 58))
    if reference is not None and reference.shape != (batch, 4, 58):
        raise ValueError("Async Soft reference must be batch x4x58")
    dt = -1.0 / num_steps
    tokens, mask, ar = self.embed_prefix(observation)
    _, cache = self.PaliGemma.llm([tokens, None], mask=make_attn_mask(mask, ar), positions=jnp.cumsum(mask, axis=1) - 1)
    indices = jnp.asarray(reference_start, jnp.int32) + jnp.arange(4)

    def step(carry):
        x, time, count = carry
        suffix, suffix_mask, suffix_ar, cond = self.embed_suffix(observation, x, jnp.broadcast_to(time, batch))
        attention = jnp.concatenate(
            [
                einops.repeat(mask, "b p -> b s p", s=suffix.shape[1]),
                make_attn_mask(suffix_mask, suffix_ar),
            ],
            axis=-1,
        )
        positions = jnp.sum(mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, output), _ = self.PaliGemma.llm(
            [None, suffix],
            mask=attention,
            positions=positions,
            kv_cache=cache,
            adarms_cond=[None, cond],
        )
        updated = x + dt * self.action_out_proj(output[:, -50:])
        time_new = time + dt
        if reference is not None:
            apply = time_new > 0.3 + 1e-6
            desired = (1 - time_new) * reference + time_new * noise[:, indices]
            updated = jax.lax.cond(apply, lambda v: v.at[:, indices].set(desired), lambda v: v, updated)
            count += apply.astype(jnp.int32)
        return updated, time_new, count

    actions, _, count = jax.lax.while_loop(
        lambda carry: carry[1] >= -dt / 2,
        step,
        (noise, 1.0, jnp.int32(0)),
    )
    return actions, count
