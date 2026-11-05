"""Shared train-RTC prefix semantics for Transformer and Mamba experts."""

import jax
import jax.numpy as jnp


def token_times(time, delay_rng, horizon, max_delay):
    """Sample 0..max_delay clean actions; t=0 denotes clean data."""
    delay = jax.random.randint(delay_rng, time.shape, 0, max_delay + 1)
    prefix_mask = jnp.arange(horizon) < delay[..., None]
    return jnp.where(prefix_mask, 0.0, time[..., None]), prefix_mask


def suffix_loss(loss, prefix_mask):
    # The trainer averages all tokens; normalize by the supervised count.
    postfix_mask = ~prefix_mask
    return jnp.where(postfix_mask, loss, 0.0) * (loss.size / jnp.sum(postfix_mask))


def validate_prefix(prefix_actions, prefix_length, expected_shape, max_delay):
    if type(prefix_length) is not int or not 0 <= prefix_length <= max_delay:
        raise ValueError("prefix_length must be a static integer in [0, rtc_max_delay]")
    if prefix_length and prefix_actions is None:
        raise ValueError("prefix_actions is required for a nonzero prefix_length")
    if prefix_actions is not None and prefix_actions.shape != expected_shape:
        raise ValueError("prefix_actions must have shape (batch, action_horizon, action_dim)")
