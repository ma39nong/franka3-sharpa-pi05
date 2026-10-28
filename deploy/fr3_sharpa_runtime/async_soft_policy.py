"""Service-local policy adapter; existing Policy/RTC/serial Soft are untouched."""
# ruff: noqa: SLF001 -- New adapter reuses one loaded policy's transforms/weights.

import time
from types import MethodType

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.shared.nnx_utils import module_jit

from .async_soft_sampler import sample_async_soft
from .async_soft_stream import CONTEXT_KEY
from .async_soft_stream import RECEIPT_KEY
from .async_soft_stream import handover_window
from .async_soft_stream import receipt
from .async_soft_stream import validate_context
from .model_service import encode_reference
from .rolling import model_identity


class AsyncSoftPolicy:
    def __init__(self, policy, metadata, *, normalizer, delta):
        if policy._is_pytorch_model:
            raise ValueError("Async Soft supports JAX full/full-delta only")
        self.policy, self.metadata = policy, metadata
        self.normalizer, self.delta = normalizer, delta
        self.sample = module_jit(MethodType(sample_async_soft, policy._model))
        self.rng = jax.random.key(0)
        self.received_ns = None

    def infer(self, obs, *, rtc_context=None, noise=None):
        if rtc_context is not None or "_inference_context" in obs:
            raise ValueError("Async Soft does not mix with RTC guidance or serial Soft")
        clean = dict(obs)
        context = clean.pop(CONTEXT_KEY, None)
        kwargs = {}
        if context is not None:
            context = validate_context(context)
            if time.monotonic() > handover_window(context)[1]:
                raise TimeoutError("Async Soft request handover window expired")
            kwargs = {
                "reference": jnp.asarray(
                    encode_reference(
                        context["previous_actions"],
                        clean["observation/state"],
                        delta=self.delta,
                        normalizer=self.normalizer,
                    )
                )[None],
                "reference_start": jnp.asarray(context["reference_start"], jnp.int32),
            }
        inputs = self.policy._input_transform(clean)
        inputs = jax.tree.map(lambda value: jnp.asarray(value)[None], inputs)
        self.rng, key = jax.random.split(self.rng)
        if noise is not None:
            noise = np.asarray(noise, np.float32)
            if noise.shape != (50, 58) or not np.isfinite(noise).all():
                raise ValueError("Async Soft noise must be finite 50x58")
            kwargs["noise"] = jnp.asarray(noise)[None]
        started = time.monotonic()
        actions, count = self.sample(key, model_lib.Observation.from_dict(inputs), **kwargs)
        actions, count = np.asarray(actions[0]), int(count)
        output = self.policy._output_transform({"state": np.asarray(inputs["state"][0]), "actions": actions})
        if context is not None:
            if time.monotonic() > handover_window(context)[1]:
                raise TimeoutError("Async Soft inference missed handover window")
            output[RECEIPT_KEY] = {
                "context": receipt(context),
                "server_instance_id": self.metadata["server_instance_id"],
                "model_identity": model_identity(self.metadata),
            }
        output["async_soft_diagnostics"] = {
            "reference_start": context["reference_start"] if context else None,
            "applied_steps": count,
            "correlation": False,
            "infer_ms": (time.monotonic() - started) * 1000,
        }
        return output
