"""Paper RTC: normalize variable-length absolute references on the NEW time axis."""

from dataclasses import dataclass

import numpy as np

from openpi.policies.policy import Policy


@dataclass(frozen=True)
class PaperRTCPrefixTransform:
    transform: object

    def __call__(self, inputs):
        if "actions" not in inputs:
            return self.transform(inputs)
        actions = np.asarray(inputs["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 58 or not 1 <= len(actions) <= 50:
            raise ValueError("Paper RTC requires a 1..50 x 58 action reference")
        if not np.isfinite(actions).all():
            raise ValueError("Paper RTC reference contains non-finite actions")
        length = len(actions)
        padded = np.zeros((50, 58), dtype=np.float32)
        padded[:length] = actions
        # Run the SAME absolute->delta->quantiles->Paper bridge as training.
        # Indices 0..K-1 belong to the new window, not old chunk indices 15..49.
        transformed = self.transform({**inputs, "actions": padded})
        normalized = np.asarray(transformed["actions"])
        if normalized.shape != (50, 58) or not np.isfinite(normalized).all():
            raise ValueError("Paper RTC input transforms must produce finite 50x58 actions")
        # Base Policy retains the real reference length and zero guidance outside it.
        return {**transformed, "actions": normalized[:length]}


class PaperRTCPolicy(Policy):
    """Keep the RTC sampler/receipts while adapting Paper's time-indexed statistics."""

    def __init__(self, model, **kwargs):
        if model.action_dim != 58 or model.action_horizon != 50:
            raise ValueError("Paper RTC requires a native 50x58 model")
        super().__init__(model, **kwargs)
        self._input_transform = PaperRTCPrefixTransform(self._input_transform)
