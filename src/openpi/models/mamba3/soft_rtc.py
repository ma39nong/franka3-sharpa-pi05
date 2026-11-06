"""Input contracts for deployment-only Mamba guidance; no model state."""

import jax
import numpy as np


def validate_soft_inputs(actions, weights, maximum, shape):
    if actions.shape != shape or weights is None or weights.shape != (*shape[:2], 1):
        raise ValueError("Mamba Soft RTC expects actions (B,H,D) and weights (B,H,1)")
    if np.shape(maximum) != ():
        raise ValueError("Mamba Soft RTC guidance cap must be scalar")
    # Policy calls this before JIT on concrete arrays. During tracing only
    # structural checks are possible; outputs are checked again on the host.
    for value, name, lower, upper in (
        (actions, "actions", None, None),
        (weights, "weights", 0.0, 1.0),
        (maximum, "guidance cap", 0.0, 100.0),
    ):
        if isinstance(value, jax.core.Tracer):
            continue
        concrete = np.asarray(value)
        if (
            not np.isfinite(concrete).all()
            or (lower is not None and (concrete < lower).any())
            or (upper is not None and (concrete > upper).any())
        ):
            raise ValueError(f"Invalid Mamba Soft RTC {name}")
