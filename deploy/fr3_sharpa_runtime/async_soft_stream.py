"""Fixed-boundary async Soft: 50 -> 30 -> 26+4 -> 20 execution targets.

Prediction phases are retimed to the next segment, rather than skipped by wall
clock latency. Four old future goals anchor new phases 0..3. No RTC guidance.
"""

import hashlib
import math
import re
import uuid

import numpy as np

from .protocol import as_finite_action_chunk

CAPABILITY = "sharpa-async-soft-rolling-v3"
CONTEXT_KEY = "_async_soft_context"
RECEIPT_KEY = "async_soft_receipt"
_FIELDS = {
    "version",
    "token",
    "source_number",
    "source_sha256",
    "reference_sha256",
    "observation_monotonic_s",
    "snapshot_monotonic_s",
    "node_hz",
    "handover_monotonic_s",
    "reference_start",
    "previous_actions",
}


def digest(values):
    return hashlib.sha256(np.asarray(values, dtype=np.float32).tobytes()).hexdigest()


def validate_context(value):
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise ValueError("Invalid async Soft context fields")
    if value["version"] != CAPABILITY or not isinstance(value["token"], str) or not value["token"]:
        raise ValueError("Invalid async Soft protocol/token")
    if type(value["source_number"]) is not int or not 1 <= value["source_number"] <= 2**53:
        raise ValueError("Invalid async Soft source_number")
    if type(value["reference_start"]) is not int or value["reference_start"] != 0:
        raise ValueError("Async Soft reference_start must be zero for phase alignment")
    for key in ("source_sha256", "reference_sha256"):
        if not isinstance(value[key], str) or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None:
            raise ValueError("Invalid async Soft digest")
    for key in ("observation_monotonic_s", "snapshot_monotonic_s", "node_hz", "handover_monotonic_s"):
        if isinstance(value[key], bool) or not isinstance(value[key], int | float) or not math.isfinite(value[key]):
            raise ValueError(f"Invalid async Soft {key}")
    captured, sampled, hz, end = (
        value[k] for k in ("observation_monotonic_s", "snapshot_monotonic_s", "node_hz", "handover_monotonic_s")
    )
    if not 0 <= captured <= sampled or sampled - captured > 0.2 or not 0 < hz <= 30 or end <= sampled:
        raise ValueError("Async Soft snapshot observation is stale or handover invalid")
    reference = np.array(value["previous_actions"], dtype=np.float32, copy=True)
    if reference.shape != (4, 58) or not np.isfinite(reference).all():
        raise ValueError("Async Soft requires four finite absolute 58D references")
    if digest(reference) != value["reference_sha256"]:
        raise ValueError("Async Soft reference digest mismatch")
    return {**value, "previous_actions": reference}


def receipt(context):
    context = validate_context(context)
    return {key: value for key, value in context.items() if key != "previous_actions"}


def handover_window(context):
    # A staging deadline, NOT permission to execute a new segment early.
    return context["snapshot_monotonic_s"], context["handover_monotonic_s"]


def require_capabilities(metadata):
    if (
        metadata.get("async_soft_version") != CAPABILITY
        or metadata.get("async_soft_warmed_up") is not True
        or metadata.get("model_profile") not in ("full", "full-delta")
        or not isinstance(metadata.get("server_instance_id"), str)
        or not metadata["server_instance_id"]
    ):
        raise ValueError("Async Soft needs its own warmed service: serve_async_soft.sh")


def execution_targets(actions):
    """Shape-preserving cubic 26 -> 20; endpoints preserved, no overshoot."""
    from scipy.interpolate import PchipInterpolator

    source = as_finite_action_chunk(actions)[:26]
    targets = PchipInterpolator(np.arange(26), source, axis=0)(np.linspace(0, 25, 20)).astype(np.float32)
    targets[[0, -1]] = source[[0, -1]]
    return targets


class AsyncSoftTimeline:
    def __init__(self, actions, now, speed_scale, *, terminal=False):
        if not np.isfinite(speed_scale) or not 0 < speed_scale <= 1:
            raise ValueError("Invalid async Soft speed scale")
        self.node_hz = 30.0 / math.ceil(1.0 / speed_scale)
        self.generation = 0
        self.finished = False
        self._activate(self.prepare(actions), now, terminal)

    @staticmethod
    def prepare(actions, anchor=None):
        # Discarded nodes 30..49 cannot affect command generation or identity.
        kept = as_finite_action_chunk(actions)[:30].copy()
        padded = np.concatenate([kept, np.repeat(kept[-1:], 20, axis=0)])
        targets = execution_targets(padded)
        if anchor is not None:
            gap = float(np.max(np.abs(targets[0] - anchor)))
            if gap > 0.05:
                raise ValueError(f"Async Soft boundary mismatch {gap:.6f} rad exceeds 0.05 rad; stopped")
            # Blend within the existing 20 targets, with no added commands or tail.
            # Convex weights preserve per-joint bounds. Never hide a large mismatch.
            u = np.linspace(0, 1, 5, dtype=np.float32)
            weights = (u * u * (3 - 2 * u))[:, None]
            targets[:5] = (1 - weights) * anchor + weights * targets[:5]
        return kept, targets

    def _activate(self, plan, now, terminal):
        self.actions, self.targets = plan
        self.digest = digest(self.actions)
        self.generation += 1
        self.installed_at = now
        self.boundary = now + 20 / self.node_hz
        self.terminal = terminal
        self.pending = self.queued = None

    def _advance(self, now):
        if not math.isfinite(now) or now < self.installed_at:
            raise ValueError("Async Soft clock moved backwards")
        if now >= self.boundary:
            if self.terminal:
                self.finished = True
            elif self.queued is None:
                raise TimeoutError("Async Soft next prediction missed fixed 20-target handover")
            else:
                plan, terminal = self.queued
                boundary = self.boundary
                self._activate(plan, boundary, terminal)
                if now >= self.boundary:
                    raise TimeoutError("Async Soft playback skipped an entire segment")

    def phase(self, now):
        self._advance(now)
        return min(19.0, (now - self.installed_at) * self.node_hz) * 25 / 19

    def sample(self, now):
        self._advance(now)
        index = min(19.0, max(0.0, (now - self.installed_at) * self.node_hz))
        i = math.floor(index)
        return self.targets[i] + (self.targets[min(19, i + 1)] - self.targets[i]) * (index - i)

    def status(self, now):
        phase = self.phase(now)
        return {
            "generation": self.generation,
            "phase": phase,
            "sampled_monotonic_s": now,
            "consumed_since_install": phase,
            "source_sha256": self.digest,
            "handover_monotonic_s": self.boundary,
            "finished": self.finished,
            "queued": self.queued is not None,
            "execution_target_count": 20,
            "consumed_model_steps": 26,
            "discarded_model_steps": 20,
        }

    def snapshot(self, now, *, observation_monotonic_s):
        self._advance(now)
        if self.pending is not None or self.terminal:
            raise ValueError("Only one outstanding prediction per nonterminal segment")
        context = validate_context(
            {
                "version": CAPABILITY,
                "token": uuid.uuid4().hex,
                "source_number": self.generation,
                "source_sha256": self.digest,
                "reference_sha256": digest(self.actions[26:30]),
                "observation_monotonic_s": observation_monotonic_s,
                "snapshot_monotonic_s": now,
                "node_hz": self.node_hz,
                "handover_monotonic_s": self.boundary,
                "reference_start": 0,
                "previous_actions": self.actions[26:30].copy(),
            }
        )
        self.pending = context
        return context

    def install(self, actions, acknowledgement, now, *, terminal=False):
        if self.pending is None or self.queued is not None:
            raise ValueError("No outstanding async Soft snapshot or result already queued")
        context = self.pending
        if acknowledgement != receipt(context):
            raise ValueError("Async Soft response does not match outstanding snapshot")
        if context["source_number"] != self.generation or context["source_sha256"] != self.digest:
            raise ValueError("Async Soft source changed")
        if now < context["snapshot_monotonic_s"] or now >= self.boundary:
            raise TimeoutError("Async Soft staging missed handover deadline")
        self.queued = self.prepare(actions, anchor=self.targets[-1]), terminal
        return {
            "queued": True,
            "source_generation": self.generation,
            "handover_monotonic_s": self.boundary,
            "reference_start": 0,
            "skipped": 0,
            "boundary_gap_rad": float(np.max(np.abs(as_finite_action_chunk(actions)[0] - self.targets[-1]))),
            "alignment": "trajectory_phase",
            "execution_target_count": 20,
        }
