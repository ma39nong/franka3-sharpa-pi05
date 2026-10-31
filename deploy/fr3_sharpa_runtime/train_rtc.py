"""Bridge the existing absolute queue protocol to trained hard-prefix conditioning."""

import jax.numpy as jnp

from openpi.policies.policy import Policy
from openpi.shared import nnx_utils


class TrainRTCPolicy(Policy):
    def __init__(self, model, **kwargs):
        if not 0 < getattr(model, "rtc_max_delay", 0) < 50 or model.action_dim != 58 or model.action_horizon != 50:
            raise ValueError("Train RTC requires a trained native 50x58 model")
        super().__init__(model, **kwargs)
        self._sample_actions = nnx_utils.module_jit(model.sample_actions, static_argnames=("prefix_length",))

    def _rtc_sample_kwargs(self, context, target):
        delay = context.inference_delay_steps
        if delay > self._model.rtc_max_delay or delay > len(context.previous_actions):
            raise ValueError("RTC committed prefix exceeds training max delay or available actions")
        return {"prefix_actions": jnp.asarray(target)[None, ...], "prefix_length": delay}
