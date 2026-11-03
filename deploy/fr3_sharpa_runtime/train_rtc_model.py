"""Train RTC inference from training commit c135c79; parameter layout unchanged."""

from dataclasses import dataclass

import einops
from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0_config import Pi0Config
from openpi.shared import array_typing as at


@dataclass(frozen=True)
class TrainRTCPi0Config(Pi0Config):
    rtc_max_delay: int = 0

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05 or type(self.rtc_max_delay) is not int or not 0 < self.rtc_max_delay < self.action_horizon:
            raise ValueError("Train RTC requires pi05 and 0 < rtc_max_delay < horizon")

    def create(self, rng):
        return TrainRTCPi0(self, nnx.Rngs(rng))


class TrainRTCPi0(Pi0):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        self.rtc_max_delay = config.rtc_max_delay

    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        prefix_actions: at.Float[at.Array, "b ah ad"] | None = None,
        prefix_length: int = 0,
    ) -> _model.Actions:
        """Sample a chunk, optionally keeping a committed RTC prefix fixed.

        prefix_actions must be in this observation's normalized model action
        space, padded to the full horizon. prefix_length is a static integer;
        pass static_argnames=("prefix_length",) when JIT-compiling this method.
        Only a checkpoint trained with RTC supports a nonzero prefix.
        """
        if type(prefix_length) is not int or not 0 <= prefix_length <= self.rtc_max_delay:
            raise ValueError("prefix_length must be a static integer in [0, rtc_max_delay]")
        if prefix_length and prefix_actions is None:
            raise ValueError("prefix_actions is required for a nonzero prefix_length")
        if prefix_actions is not None and prefix_actions.shape != (
            observation.state.shape[0],
            self.action_horizon,
            self.action_dim,
        ):
            raise ValueError("prefix_actions must have shape (batch, action_horizon, action_dim)")
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        action_prefix_mask = jnp.arange(self.action_horizon)[None, :] < prefix_length
        if prefix_length:
            noise = jnp.where(action_prefix_mask[..., None], prefix_actions, noise)

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            timestep = jnp.broadcast_to(time, batch_size)
            if prefix_length:
                timestep = jnp.where(action_prefix_mask, 0.0, timestep[:, None])
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, timestep)
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            x_next = x_t + dt * v_t
            if prefix_length:
                x_next = jnp.where(action_prefix_mask[..., None], prefix_actions, x_next)
            return x_next, time + dt

        def cond(carry):
            _x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
