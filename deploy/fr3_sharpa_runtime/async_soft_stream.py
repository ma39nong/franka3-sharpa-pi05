"""Isolated async-Soft protocol and observation-anchored timeline; no devices."""

import math
import re
import uuid

import numpy as np

from .protocol import as_finite_action_chunk
from .rtc_stream import RTCTimeline

CAPABILITY = "sharpa-async-soft-v1"
CONTEXT_KEY = "_async_soft_context"
RECEIPT_KEY = "async_soft_receipt"
_FIELDS = {
    "version",
    "token",
    "source_number",
    "source_sha256",
    "observation_monotonic_s",
    "snapshot_monotonic_s",
    "node_hz",
    "reference_start",
    "previous_actions",
}


def validate_context(value):
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise ValueError("Invalid async Soft context fields")
    if value["version"] != CAPABILITY or not isinstance(value["token"], str) or not value["token"]:
        raise ValueError("Invalid async Soft protocol/token")
    for key, low, high in (("source_number", 1, 2**53), ("reference_start", 1, 45)):
        if type(value[key]) is not int or not low <= value[key] <= high:
            raise ValueError(f"Invalid async Soft {key}")
    digest = value["source_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("Invalid async Soft source digest")
    for key in ("observation_monotonic_s", "snapshot_monotonic_s", "node_hz"):
        if isinstance(value[key], bool) or not isinstance(value[key], int | float) or not math.isfinite(value[key]):
            raise ValueError(f"Invalid async Soft {key}")
    captured, sampled, hz = (value[k] for k in ("observation_monotonic_s", "snapshot_monotonic_s", "node_hz"))
    if not 0 <= captured <= sampled or sampled - captured > 0.2 or not 0 < hz <= 30:
        raise ValueError("Async Soft snapshot observation is stale or time scale invalid")
    if captured + value["reference_start"] / hz < sampled:
        raise ValueError("Async Soft handover is already past at snapshot")
    reference = np.array(value["previous_actions"], dtype=np.float32, copy=True)
    if reference.shape != (4, 58) or not np.isfinite(reference).all():
        raise ValueError("Async Soft requires four finite absolute 58D references")
    return {**value, "previous_actions": reference}


def receipt(context):
    context = validate_context(context)
    return {key: value for key, value in context.items() if key != "previous_actions"}


def handover_window(context):
    return (
        context["observation_monotonic_s"] + context["reference_start"] / context["node_hz"],
        context["observation_monotonic_s"] + (context["reference_start"] + 3) / context["node_hz"],
    )


def require_capabilities(metadata):
    if (
        metadata.get("async_soft_version") != CAPABILITY
        or metadata.get("async_soft_warmed_up") is not True
        or metadata.get("model_profile") not in ("full", "full-delta")
        or not isinstance(metadata.get("server_instance_id"), str)
        or not metadata["server_instance_id"]
    ):
        raise ValueError("Async Soft needs its own warmed service: serve_async_soft.sh")


class AsyncSoftTimeline(RTCTimeline):
    """Reuse read-only playback math; snapshot/install use an independent protocol.

    Predicted model node zero is the OBSERVATION time, not the later IPC snapshot.
    Old commands continue until a result is ready AND the planned handover arrives.
    """

    def snapshot(self, now, *, delay_steps, observation_monotonic_s):
        if self.pending is not None:
            raise ValueError("Only one async Soft prediction may be outstanding")
        if type(delay_steps) is not int or not 1 <= delay_steps <= 20:
            raise ValueError("Async Soft delay estimate must be in [1, 20]")
        observed = observation_monotonic_s
        if not isinstance(observed, int | float) or not math.isfinite(observed) or not 0 <= now - observed <= 0.2:
            raise ValueError("Async Soft snapshot observation is stale")
        self.sample(now)
        origin = self.phase(observed)
        start = math.ceil((now - observed) * self.node_hz - 1e-9) + delay_steps
        if origin + start + 3 > 49:
            raise TimeoutError("Old trajectory has insufficient future overlap")
        context = validate_context(
            {
                "version": CAPABILITY,
                "token": uuid.uuid4().hex,
                "source_number": self.generation,
                "source_sha256": self.digest,
                "observation_monotonic_s": observed,
                "snapshot_monotonic_s": now,
                "node_hz": self.node_hz,
                "reference_start": start,
                "previous_actions": np.stack([self.sample_phase(origin + start + i) for i in range(4)]),
            }
        )
        self.pending = context
        return {**context, "previous_actions": context["previous_actions"].copy()}

    def install(self, actions, acknowledgement, now):
        if self.pending is None:
            raise ValueError("No outstanding async Soft snapshot")
        context = self.pending
        if acknowledgement != receipt(context):
            raise ValueError("Async Soft response does not match outstanding snapshot")
        if context["source_number"] != self.generation or context["source_sha256"] != self.digest:
            raise ValueError("Async Soft source changed")
        self.sample(now)  # Never revive an exhausted old plan.
        earliest, latest = handover_window(context)
        if now < earliest - 1e-9:
            raise ValueError("Async Soft result arrived before scheduled handover")
        if now > latest + 1e-9:
            raise TimeoutError("Async Soft result missed the four-node handover window")
        actions = as_finite_action_chunk(actions)
        # Fractional phase preserves the observation clock; no ceil-induced time jump.
        skipped = (now - context["observation_monotonic_s"]) * self.node_hz
        generation = self.generation
        self._install(actions, now, skipped)
        return {
            "skipped": skipped,
            "generation": self.generation,
            "source_generation": generation,
            "actual_delay_s": now - context["snapshot_monotonic_s"],
            "reference_start": context["reference_start"],
            "observation_anchored": True,
        }
