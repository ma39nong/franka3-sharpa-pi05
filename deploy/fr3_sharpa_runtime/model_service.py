"""Stateless experimental service adapter; references stay with the execution run."""

import time

import numpy as np

from .rolling import CAPABILITY
from .rolling import CONTEXT_KEY
from .rolling import RollingOptions
from .rolling import prepare_context

ARMS = np.r_[0:7, 29:36]


def encode_reference(absolute, state, *, delta, normalizer):
    values = np.asarray(absolute, dtype=np.float32).copy()
    state = np.asarray(state, dtype=np.float32)
    if (
        values.ndim != 2
        or values.shape[1] != 58
        or state.shape != (58,)
        or not np.isfinite(values).all()
        or not np.isfinite(state).all()
    ):
        raise ValueError("Expected finite canonical reference and measured state")
    if delta:
        values[:, ARMS] -= state[ARMS]
    values = np.asarray(normalizer({"actions": values})["actions"], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite normalized reference")
    return values


class ExperimentalPolicy:
    def __init__(self, timed_policy, metadata, *, normalizer, delta, matrix=None, covariance_info=None):
        self.policy = timed_policy
        self.metadata = metadata
        self.normalizer = normalizer
        self.delta = delta
        self.matrix = matrix
        self.covariance_info = covariance_info

    @property
    def received_ns(self):
        return self.policy.received_ns

    @received_ns.setter
    def received_ns(self, value):
        self.policy.received_ns = value

    def infer(self, obs):
        clean = dict(obs)
        context = clean.pop(CONTEXT_KEY, None)
        if context is None:
            return self.policy.infer(clean)
        started = time.monotonic_ns()
        context = prepare_context(context, self.metadata, now_ns=started)
        options = RollingOptions(**context["options"])
        if options.inpainting == "correlation" and self.matrix is None:
            raise ValueError("correlation mode requires matching covariance asset at server startup")
        kwargs = {}
        reference = context["reference"]
        if reference is not None and options.inpainting != "off":
            kwargs["initial_actions"] = encode_reference(
                reference["absolute_actions"], clean["observation/state"], delta=self.delta, normalizer=self.normalizer
            )
            if options.inpainting == "correlation":
                kwargs["correction_matrix"] = self.matrix
        result = self.policy.infer(clean, **kwargs)
        result["inference_experiment"] = {
            "version": CAPABILITY,
            "run_id": context["run_id"],
            "chunk_index": context["chunk_index"],
            "alignment": context["alignment"],
            "inpainting": options.inpainting,
            "reference_applied": bool(kwargs),
            "reference_chunk_index": reference["chunk_index"] if reference else None,
            "reference_completed_monotonic_ns": reference["completed_monotonic_ns"] if reference else None,
            "invalidated_reason": context.get("invalidated_reason"),
            "options": options.as_dict(),
            "inference_start_monotonic_ns": started,
            "inference_end_monotonic_ns": time.monotonic_ns(),
            "covariance": self.covariance_info if options.inpainting == "correlation" else None,
            "sampler": result.get("inpainting_diagnostics", {}),
        }
        return result
