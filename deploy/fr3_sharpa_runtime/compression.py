"""Synchronous 58D trajectory compression; SDK validation remains authoritative."""
# ruff: noqa: PLC0415
# Lazy imports keep pure helpers hardware/framework-free and test patches local.

import numpy as np

from .protocol import as_finite_action_chunk
from .sdk_direct import chunk_command_frames


def validate_targets(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 58 or not 1 <= len(values) <= 50 or not np.isfinite(values).all():
        raise ValueError("Expected 1..50 finite absolute 58D execution targets")
    return values.copy()


def padded_targets(values):
    values = validate_targets(values)
    return np.concatenate([values, np.repeat(values[-1:], 50 - len(values), axis=0)])


def hand_variation(source, feedback):
    state = np.asarray(feedback, dtype=np.float32)
    if state.shape != (58,) or not np.isfinite(state).all():
        raise ValueError("Compression requires finite measured feedback")
    hands = np.r_[7:29, 36:58]
    return float(np.max(np.ptp(np.vstack([state, source])[:, hands], axis=0)))


def recheck_compression(actions, candidate, report, feedback):
    """Fresh worker feedback gate, deliberately independent of SciPy/Python model env."""
    report = dict(report)
    threshold = report.get("hand_variation_threshold_rad")
    if threshold is None or not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid hand variation threshold")
    variation = hand_variation(as_finite_action_chunk(actions)[:26], feedback)
    report["worker_hand_variation_rad"] = variation
    if report.get("applied") and variation > threshold:
        report.update(applied=False, target_count=26, fallback_reason="fresh_hand_variation")
        return as_finite_action_chunk(actions)[:26], report
    return candidate, report


def compression_candidate(actions, feedback, options):
    source = as_finite_action_chunk(actions)[:26].copy()
    variation = hand_variation(source, feedback)
    report = {
        "requested": options.compression,
        "applied": False,
        "source_count": 26,
        "target_count": 26,
        "hand_variation_rad": variation,
        "hand_variation_threshold_rad": options.hand_variation_rad,
        "fallback_reason": None,
    }
    if options.compression == "off":
        return source, report
    if variation > options.hand_variation_rad:
        report["fallback_reason"] = "hand_variation"
        return source, report
    from scipy.interpolate import interp1d

    indices = np.linspace(0, 25, 26)
    candidate = interp1d(indices, source, kind="cubic", axis=0)(np.linspace(0, 25, 20)).astype(np.float32)
    candidate[[0, -1]] = source[[0, -1]]
    report.update(applied=True, target_count=20)
    return candidate, report


def plan_execution(
    actions,
    execution_targets,
    compression,
    state,
    *,
    speed_scale,
    approach_speed_rad_s,
    range_validator,
    prediction_clip_rad=0.05,
):
    """Pure planning before any send. Validate cubic samples strictly; fall back once."""
    from .sdk_direct import clip_hand_predictions

    original = as_finite_action_chunk(actions)
    report = dict(compression)
    candidate = validate_targets(execution_targets)
    if report.get("source_count") != 26 or report.get("requested") not in ("off", "cubic"):
        raise ValueError("Invalid rolling execution plan")
    applied = report.get("applied")
    if type(applied) is not bool or len(candidate) != (20 if applied else 26):
        raise ValueError("Compression target count mismatch")
    if applied and report["requested"] != "cubic":
        raise ValueError("Compressed sequence requires cubic mode")

    def build(targets, strict):
        padded = padded_targets(targets)
        if strict:
            # Cubic overshoot may not consume the prediction clipping allowance.
            range_validator(padded, check_slew=False)
            audit = {
                "corrected_values": 0,
                "corrections": [],
                "max_applied_correction_rad": 0.0,
                "max_allowed_correction_rad": 0.0,
            }
        else:
            padded, audit = clip_hand_predictions(
                padded, execute_steps=len(targets), max_correction_rad=prediction_clip_rad
            )
            padded[len(targets) :] = padded[len(targets) - 1]
            range_validator(padded, check_slew=False)
        frames, bridge = chunk_command_frames(
            padded,
            state,
            speed_scale=speed_scale,
            execute_steps=len(targets),
            approach_speed_rad_s=approach_speed_rad_s,
        )
        for offset in range(0, len(frames), 50):
            range_validator(padded_targets(frames[offset : offset + 50]), check_slew=False)
        return frames, bridge, padded[: len(targets)], audit

    if not applied and not np.array_equal(candidate, original[:26]):
        raise ValueError("Uncompressed targets must equal original policy prefix")
    try:
        frames, bridge, targets, audit = build(candidate, strict=applied)
    except ValueError as error:
        if not applied:
            raise
        report.update(
            applied=False, target_count=26, fallback_reason="validation_rejected", validation_error=str(error)
        )
        frames, bridge, targets, audit = build(original[:26], strict=False)
    return frames, bridge, targets, audit, report
