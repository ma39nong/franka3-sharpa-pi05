"""Fixed-Home boundary recovery using SDK interpolation and URDF position limits.

Recovery does not override device speed/current settings or add distance,
convergence, stability, progress, or recovery-duration thresholds. The caller
provides the runtime's existing validated feedback and owns stop/shutdown.
"""

import time

import numpy as np

from .protocol import DT_NS, LEFT_SHARPA_SLICE, RIGHT_SHARPA_SLICE, SHARPA_JOINT_NAMES
from .sdk_direct import _hand_bounds_float32, hand_joint_limits, validate_hand_command


def recovery_target(measured):
    """Project out-of-range hands to representable URDF boundaries, not feedback."""
    state = np.asarray(measured, dtype=np.float64)
    if state.shape != (58,) or not np.isfinite(state).all():
        raise ValueError("Home recovery requires finite measured 58D feedback")
    limits = hand_joint_limits()
    lower, upper = _hand_bounds_float32(limits)
    target = state.copy()
    changed = []
    for side, part in (("left", LEFT_SHARPA_SLICE), ("right", RIGHT_SHARPA_SLICE)):
        values = state[part]
        for joint in np.flatnonzero((values < limits[:, 0]) | (values > limits[:, 1])):
            # Round only toward the legal interval; there is no extra angle margin.
            goal = float(lower[joint] if values[joint] < limits[joint, 0] else upper[joint])
            target[part.start + joint] = goal
            changed.append(dict(side=side, joint=SHARPA_JOINT_NAMES[joint],
                measured_rad=float(values[joint]), target_rad=goal,
                displacement_rad=abs(goal - float(values[joint])), legal_range_rad=limits[joint].tolist()))
        validate_hand_command(target[part], side, limits=limits)
    return target, changed


def recover_before_home(executor, feedback, *, clock=time.monotonic, sleep=time.sleep, emit=lambda report: None):
    """Wait for actual URDF-legal feedback; SDK errors and operator stop propagate.

    No SDK settings are written here. No arm command, recorder gate or policy
    method is invoked. The caller's feedback validation remains authoritative.
    """
    target, changed = recovery_target(feedback().state)
    if not changed:
        return {"status": "SKIPPED", "reason": "initial hand feedback already inside command limits"}
    if any(getattr(getattr(hand, "config", None), "interpolate", None) is not True
           for hand in (executor.left, executor.right)):
        raise RuntimeError("Home recovery needs native interpolation")
    report = dict(status="RUNNING", joints=changed, device_settings_preserved=True)
    limits = hand_joint_limits()
    started = clock()
    emit({"event": "home_recovery_start", **report})
    executor.resume_stream()
    next_tick = clock()
    while True:
        now = clock()
        if now < next_tick:
            sleep(min(0.01, next_tick - now))
            continue
        measured = feedback().state
        legal = all(np.all((measured[part] >= limits[:, 0]) & (measured[part] <= limits[:, 1]))
                    for part in (LEFT_SHARPA_SLICE, RIGHT_SHARPA_SLICE))
        if legal:
            report.update(status="PASS", elapsed_s=clock() - started)
            executor.resume_stream()
            emit({"event": "home_recovery_complete", **report})
            return report
        executor.send(target[LEFT_SHARPA_SLICE], target[RIGHT_SHARPA_SLICE])
        # Keep the existing 30 Hz pacing without a recovery-specific deadline.
        # A delayed tick rebases instead of sending a burst to catch up.
        next_tick = max(next_tick + DT_NS / 1e9, clock())
