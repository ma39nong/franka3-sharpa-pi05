"""Opt-in, trajectory-phase rolling references. No device or ML framework imports."""

from dataclasses import asdict
from dataclasses import dataclass
import time
import uuid

import numpy as np

from .protocol import as_finite_action_chunk

CONTEXT_KEY = "_inference_context"
CAPABILITY = "paper-rolling-v1"
IDENTITY_KEYS = ("model_profile", "checkpoint", "checkpoint_manifest_sha256", "norm_stats", "normalization_sha256")


def model_identity(metadata):
    return {key: metadata.get(key) for key in IDENTITY_KEYS}


@dataclass(frozen=True)
class RollingOptions:
    inpainting: str = "off"
    compression: str = "off"
    reference_max_age_s: float = 1.0
    hand_variation_rad: float = 0.05
    effective_horizon: int = 30
    execute_horizon: int = 26
    overlap_horizon: int = 4
    compressed_steps: int = 20

    def __post_init__(self):
        if self.inpainting not in ("off", "soft", "correlation") or self.compression not in ("off", "cubic"):
            raise ValueError("Invalid rolling algorithm")
        if (self.effective_horizon, self.execute_horizon, self.overlap_horizon, self.compressed_steps) != (
            30,
            26,
            4,
            20,
        ):
            raise ValueError("paper-rolling requires 50 -> 30 -> 26+4, optionally 26 -> 20")
        if not np.isfinite(self.reference_max_age_s) or self.reference_max_age_s <= 0:
            raise ValueError("reference_max_age_s must be finite and positive")
        if not np.isfinite(self.hand_variation_rad) or self.hand_variation_rad < 0:
            raise ValueError("hand_variation_rad must be finite and nonnegative")

    def as_dict(self):
        return asdict(self)


class RollingReference:
    def __init__(self, options, *, clock_ns=time.monotonic_ns):
        self.options = options
        self.clock_ns = clock_ns
        self.run_id = uuid.uuid4().hex
        self.reference = None
        self.invalidated_reason = "first_chunk"

    def clear(self, reason):
        self.reference = None
        self.invalidated_reason = reason

    def context(self, chunk_index):
        if self.reference is not None:
            age = self.clock_ns() - self.reference["completed_monotonic_ns"]
            if not 0 <= age <= self.options.reference_max_age_s * 1e9:
                self.clear("reference_expired")
        return {
            "version": CAPABILITY,
            "run_id": self.run_id,
            "chunk_index": chunk_index,
            "options": self.options.as_dict(),
            "reference": self.reference,
            "invalidated_reason": self.invalidated_reason,
            "alignment": "trajectory_phase_not_wall_clock",
        }

    def commit(self, actions, metadata, report, chunk_index):
        target_count = report.get("execution_target_count")
        if (
            report.get("status") != "PASS"
            or report.get("policy_frames") != 26
            or report.get("completion") != "commands_sent"
            or target_count not in (20, 26)
            or type(report.get("frames")) is not int
            or report["frames"] < target_count
        ):
            self.clear("incomplete_execution")
            raise RuntimeError("Rolling reference requires successful consumption of 26 policy frames")
        stamp = report.get("completed_monotonic_ns")
        if type(stamp) is not int or not 0 <= self.clock_ns() - stamp <= self.options.reference_max_age_s * 1e9:
            self.clear("invalid_completion_timestamp")
            raise RuntimeError("Invalid rolling completion timestamp")
        self.reference = {
            "absolute_actions": as_finite_action_chunk(actions)[26:30].tolist(),
            "chunk_index": chunk_index,
            "run_id": self.run_id,
            "source_indices": [26, 30],
            "consumed_model_steps": 26,
            "completed_monotonic_ns": stamp,
            "model_identity": model_identity(metadata),
            "server_instance_id": metadata["server_instance_id"],
        }
        self.invalidated_reason = None


def prepare_context(context, metadata, *, now_ns=None):
    """Validate per-request context; a new socket alone never invalidates a reference."""
    if context.get("version") != CAPABILITY or CAPABILITY not in metadata.get("inference_capabilities", []):
        raise ValueError("Policy server does not support paper-rolling-v1")
    if metadata.get("model_profile") not in ("full", "full-delta"):
        raise ValueError("paper-rolling supports full and full-delta only")
    if not isinstance(context.get("run_id"), str) or not context["run_id"]:
        raise ValueError("Rolling run_id required")
    if type(context.get("chunk_index")) is not int or context["chunk_index"] < 0:
        raise ValueError("Invalid rolling chunk index")
    if context.get("alignment") != "trajectory_phase_not_wall_clock":
        raise ValueError("Rolling references require explicit trajectory-phase alignment")
    options = RollingOptions(**context["options"])
    result = dict(context)
    reference = context.get("reference")
    if reference is None:
        return result
    reason = None
    now = time.monotonic_ns() if now_ns is None else now_ns
    stamp = reference.get("completed_monotonic_ns")
    if type(stamp) is not int or not 0 <= now - stamp <= options.reference_max_age_s * 1e9:
        reason = "reference_expired"
    elif reference.get("server_instance_id") != metadata.get("server_instance_id"):
        reason = "server_restarted"
    elif reference.get("model_identity") != model_identity(metadata):
        reason = "model_identity_changed"
    elif (
        reference.get("run_id") != context["run_id"]
        or reference.get("chunk_index") != context["chunk_index"] - 1
        or reference.get("source_indices") != [26, 30]
        or reference.get("consumed_model_steps") != 26
    ):
        reason = "execution_progress_changed"
    if reason:
        result.update(reference=None, invalidated_reason=reason)
    else:
        values = np.asarray(reference["absolute_actions"], dtype=np.float32)
        if values.shape != (4, 58) or not np.isfinite(values).all():
            raise ValueError("Rolling reference must contain four finite absolute 58D actions")
    return result
