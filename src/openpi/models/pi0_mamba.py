"""Pi05 VLM-only bridge plus Mamba-3 expert, sharing existing training/checkpoint APIs."""

from flax import nnx
from flax.nnx import bridge
import jax
import jax.numpy as jnp

from openpi.models import gemma
from openpi.models import model as model_lib
from openpi.models import rtc
from openpi.models import siglip
from openpi.models.mamba3.action_expert import MambaActionExpert
from openpi.models.mamba3.context_compressor import ContextCompressor
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask


class MambaPi0(model_lib.BaseModel):
    def __init__(self, config, rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.config = config
        self.rtc_max_delay = config.rtc_max_delay
        vlm = gemma.get_config(config.paligemma_variant)
        llm = bridge.ToNNX(gemma.Module(configs=[vlm], embed_dtype=config.dtype, adarms=True))
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False])
        img = bridge.ToNNX(
            siglip.Module(
                num_classes=vlm.width, variant="So400m/14", pool_type="none", scan=True, dtype_mm=config.dtype
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.context_compressor = ContextCompressor(
            vlm.width,
            config.mamba.hidden_size,
            config.mamba_mlp_dim,
            config.context_heads,
            vision_tokens=config.context_vision_tokens,
            language_tokens=config.context_language_tokens,
            global_tokens=config.context_global_tokens,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.action_expert = MambaActionExpert(
            config.mamba, config.action_dim, config.mamba_depth, config.mamba_mlp_dim, config.film_mlp_dim, rngs=rngs
        )
        self.deterministic = True

    # This stateless helper only reads PaliGemma, never old action projections/cache.
    embed_prefix = Pi0.embed_prefix

    # Same standard-normal prior as training Pi0.noise; deployment Pi0 has no helper.
    def noise(self, rng, shape):
        return jax.random.normal(rng, shape)

    def prefix_hidden(self, observation):
        if tuple(observation.images) != model_lib.IMAGE_KEYS:
            raise ValueError("Mamba requires fixed base/left/right camera slot order")
        tokens, mask, ar = self.embed_prefix(observation)
        positions = jnp.cumsum(mask, axis=1) - 1
        (hidden,), _ = self.PaliGemma.llm(
            [tokens], mask=make_attn_mask(mask, ar), positions=positions, deterministic=self.deterministic
        )
        language_length = 0 if observation.tokenized_prompt is None else observation.tokenized_prompt.shape[1]
        return hidden, mask, hidden.shape[1] - language_length

    def encode_condition(self, observation):
        hidden, mask, vision_length = self.prefix_hidden(observation)
        z = self.context_compressor(hidden, mask, vision_length)
        return self.action_expert.prepare_condition(z)

    def velocity(self, noisy_actions, time, condition):
        return self.action_expert(noisy_actions, time, condition)

    def compute_loss(self, rng, observation, actions, *, train=False, augmentation_rng=None):
        if self.rtc_max_delay:
            preprocess_rng, noise_rng, time_rng, delay_rng = jax.random.split(rng, 4)
        else:
            # Preserve the RTC=0 random streams and scalar-time path exactly.
            preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = model_lib.preprocess_observation(
            preprocess_rng if augmentation_rng is None else augmentation_rng, observation, train=train
        )
        epsilon = self.noise(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1.0, actions.shape[:-2]) * 0.999 + 0.001
        if self.rtc_max_delay:
            time, prefix_mask = rtc.token_times(time, delay_rng, self.action_horizon, self.rtc_max_delay)
            expanded = time[..., None]
        else:
            expanded = time[..., None, None]
        x_t = expanded * epsilon + (1 - expanded) * actions
        v_t = self.velocity(x_t, time, self.encode_condition(observation))
        loss = jnp.mean((v_t.astype(jnp.float32) - (epsilon - actions).astype(jnp.float32)) ** 2, axis=-1)
        return rtc.suffix_loss(loss, prefix_mask) if self.rtc_max_delay else loss

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None, prefix_actions=None, prefix_length=0):
        """Sample with a static committed prefix in the CURRENT model action space."""
        rtc.validate_prefix(
            prefix_actions,
            prefix_length,
            (observation.state.shape[0], self.action_horizon, self.action_dim),
            self.rtc_max_delay,
        )
        if isinstance(num_steps, int) and num_steps <= 0:
            raise ValueError("num_steps must be positive")
        observation = model_lib.preprocess_observation(None, observation, train=False)
        condition = self.encode_condition(observation)  # local to this query only
        shape = (observation.state.shape[0], self.action_horizon, self.action_dim)
        x = self.noise(rng, shape) if noise is None else noise
        if x.shape != shape:
            raise ValueError(f"Expected initial noise shape {shape}, got {x.shape}")
        prefix_mask = jnp.arange(self.action_horizon)[None, :] < prefix_length
        if prefix_length:
            x = jnp.where(prefix_mask[..., None], prefix_actions, x)
        dt = -1.0 / num_steps

        def step(i, actions):
            time = jnp.full((actions.shape[0],), 1.0 + i * dt)
            if prefix_length:
                time = jnp.where(prefix_mask, 0.0, time[:, None])
            updated = actions + dt * self.velocity(actions, time, condition)
            # Every velocity call starts a fresh SSM scan; no carry is reused
            # across ODE steps, whose noisy action inputs are different.
            return jnp.where(prefix_mask[..., None], prefix_actions, updated) if prefix_length else updated

        return jax.lax.fori_loop(0, num_steps, step, x)
