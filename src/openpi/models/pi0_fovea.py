"""Sharpa RGB FoveaVLA: shared prefix, isolated flow and location-token branches."""

from dataclasses import dataclass
from dataclasses import replace

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import model as base
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0_config import Pi0Config
from openpi.shared import array_typing as at

LOCATION_START = 256000
BOS = 2


def branch_mask(prefix_mask, box_length, action_length):
    """Prefix is bidirectional; both output branches see only prefix and themselves."""
    b, p = prefix_mask.shape
    n = p + box_length + action_length
    mask = jnp.zeros((b, n, n), dtype=bool)
    mask = mask.at[:, :p, :p].set(prefix_mask[:, :, None] & prefix_mask[:, None, :])
    mask = mask.at[:, p:, :p].set(jnp.broadcast_to(prefix_mask[:, None, :], (b, n - p, p)))
    mask = mask.at[:, p : p + box_length, p : p + box_length].set(jnp.tril(jnp.ones((box_length, box_length), bool)))
    return mask.at[:, p + box_length :, p + box_length :].set(True)


@dataclass(frozen=True)
class FoveaConfig(Pi0Config):
    pi05: bool = True
    rtc_max_delay: int = 0
    box_loss_weight: float = 0.1
    execute_steps: int = 8

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05 or self.rtc_max_delay:
            raise ValueError("Fovea v1 requires pi05 without RTC")
        if type(self.execute_steps) is not int or not 0 < self.execute_steps <= self.action_horizon:
            raise ValueError("execute_steps must be in [1, action_horizon]")
        if not 0 < self.box_loss_weight < float("inf"):
            raise ValueError("Positive finite box_loss_weight required")

    def create(self, rng):
        return FoveaPi0(self, rngs=nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size=1):
        obs, actions = super().inputs_spec(batch_size=batch_size)
        with at.disable_typechecking():
            obs = replace(obs, foveation_tokens=jax.ShapeDtypeStruct((batch_size, 4), jnp.int32))
        return obs, actions


class FoveaPi0(Pi0):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        self.box_loss_weight = config.box_loss_weight
        self.execute_steps = config.execute_steps

    def compute_loss_components(self, rng, observation, actions, *, train=False, augmentation_rng=None):
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = base.preprocess_observation(
            preprocess_rng if augmentation_rng is None else augmentation_rng,
            observation,
            train=train,
            image_keys=self.image_keys,
            geometric_augmentation=False,
        )
        if observation.foveation_tokens is None or observation.foveation_tokens.shape != (actions.shape[0], 4):
            raise ValueError("Four future location-token labels required")
        target = observation.foveation_tokens
        teacher = jnp.concatenate([jnp.full((actions.shape[0], 1), BOS, jnp.int32), target[:, :-1]], axis=1)
        box = self.PaliGemma.llm(teacher, method="embed")
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        prefix, prefix_valid, _ = self.embed_prefix(observation)
        suffix, _, _, cond = self.embed_suffix(observation, x_t, time)
        p = prefix.shape[1]
        count = prefix_valid.sum(axis=-1)[:, None]
        positions = jnp.concatenate(
            [
                jnp.cumsum(prefix_valid, axis=1) - 1,
                count + jnp.arange(4)[None],
                count + jnp.arange(suffix.shape[1])[None],
            ],
            axis=1,
        )
        (vlm, expert), _ = self.PaliGemma.llm(
            [jnp.concatenate([prefix, box], axis=1), suffix],
            mask=branch_mask(prefix_valid, 4, suffix.shape[1]),
            positions=positions,
            adarms_cond=[None, cond],
        )
        velocity = self.action_out_proj(expert[:, -self.action_horizon :])
        flow = jnp.mean(jnp.square(velocity - (noise - actions)), axis=(-2, -1))
        logits = self.PaliGemma.llm(vlm[:, p : p + 4], method="decode_locations")
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        box_loss = -jnp.take_along_axis(log_probs, (target - LOCATION_START)[..., None], axis=-1)[..., 0].mean(axis=-1)
        return {"loss": flow + self.box_loss_weight * box_loss, "action_loss": flow, "box_loss": box_loss}

    def compute_loss(self, rng, observation, actions, **kwargs):
        return self.compute_loss_components(rng, observation, actions, **kwargs)["loss"]

    def sample_joint(self, rng, observation, *, num_steps=10, noise=None):
        """Encode once; reuse the immutable prefix cache for both output branches."""
        observation = base.preprocess_observation(None, observation, image_keys=self.image_keys)
        prefix, valid, ar = self.embed_prefix(observation)
        positions = jnp.cumsum(valid, axis=1) - 1
        _, prefix_cache = self.PaliGemma.llm([prefix, None], positions=positions, mask=make_attn_mask(valid, ar))
        batch = observation.state.shape[0]
        count = valid.sum(axis=-1)[:, None]
        if noise is None:
            noise = jax.random.normal(rng, (batch, self.action_horizon, self.action_dim))
        dt = -1.0 / num_steps

        def step(i, x):
            suffix, sm, sar, cond = self.embed_suffix(observation, x, jnp.full((batch,), 1.0 + i * dt))
            mask = jnp.concatenate(
                [
                    jnp.broadcast_to(valid[:, None, :], (batch, suffix.shape[1], valid.shape[1])),
                    make_attn_mask(sm, sar),
                ],
                axis=-1,
            )
            (_, output), _ = self.PaliGemma.llm(
                [None, suffix],
                positions=count + jnp.arange(suffix.shape[1])[None],
                mask=mask,
                kv_cache=prefix_cache,
                adarms_cond=[None, cond],
            )
            return x + dt * self.action_out_proj(output[:, -self.action_horizon :])

        actions = jax.lax.fori_loop(0, num_steps, step, noise)
        cache = prefix_cache
        token = jnp.full((batch, 1), BOS, jnp.int32)
        generated = []
        for i in range(4):
            embedded = self.PaliGemma.llm(token, method="embed")
            mask = jnp.concatenate([valid, jnp.ones((batch, i + 1), bool)], axis=1)[:, None, :]
            (hidden, _), cache = self.PaliGemma.llm([embedded, None], positions=count + i, mask=mask, kv_cache=cache)
            logits = self.PaliGemma.llm(hidden, method="decode_locations")
            token = jnp.argmax(logits, axis=-1).astype(jnp.int32) + LOCATION_START
            generated.append(token)
        return {"actions": actions, "foveation_tokens": jnp.concatenate(generated, axis=1)}

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
        return self.sample_joint(rng, observation, num_steps=num_steps, noise=noise)["actions"]
