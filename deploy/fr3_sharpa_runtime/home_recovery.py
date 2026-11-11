"""Native interpolated boundary recovery, exclusively before a fixed Home.

Command limits and task send paths remain unchanged. Firmware speed coefficients
are dimensionless, not a certified rad/s bound. Unsupported SDKs fail closed.
"""

# ruff: noqa: SLF001 -- Narrow adapter to the installed driver's verified setting methods.

import math
import time

import numpy as np

from .protocol import DT_NS
from .protocol import LEFT_SHARPA_SLICE
from .protocol import RIGHT_SHARPA_SLICE
from .protocol import SHARPA_JOINT_NAMES
from .sdk_direct import PREPARE_SLEW_LIMIT_RAD
from .sdk_direct import _hand_bounds_float32
from .sdk_direct import hand_joint_limits
from .sdk_direct import validate_hand_command

RECOVERY_SPEED_COEFFICIENT = 0.05
RECOVERY_TIMEOUT_S = 10.0
RECOVERY_FEEDBACK_WAIT_S = 5.0
RECOVERY_STABLE_S = 0.3
RECOVERY_MARGIN_RAD = 0.005
RECOVERY_TOLERANCE_RAD = 0.003


def recovery_target(measured):
    """Return legal two-hand targets and diagnostics without changing feedback."""
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
            # Aim just inside the interval so feedback noise need not sit on a bound.
            margin = min(RECOVERY_MARGIN_RAD, (float(upper[joint]) - float(lower[joint])) / 4)
            goal = float(lower[joint]) + margin if values[joint] < limits[joint, 0] else float(upper[joint]) - margin
            if abs(goal - values[joint]) >= PREPARE_SLEW_LIMIT_RAD:
                raise ValueError(
                    f"Home recovery {side} {SHARPA_JOINT_NAMES[joint]} displacement "
                    f"{abs(goal - values[joint]):.6f} rad exceeds existing SDK slew bound"
                )
            target[part.start + joint] = goal
            changed.append(
                {
                    "side": side,
                    "joint": SHARPA_JOINT_NAMES[joint],
                    "measured_rad": float(values[joint]),
                    "target_rad": goal,
                }
            )
        validate_hand_command(target[part], side, limits=limits)
    return target, changed


def _speed_adapter(hand):
    """Use the installed driver's read/write/readback contract on owned handles."""
    if (
        getattr(getattr(hand, "config", None), "interpolate", None) is not True
        or not callable(getattr(hand, "_read_setting", None))
        or not callable(getattr(hand, "_ensure_coefficient", None))
    ):
        raise RuntimeError("Home recovery needs native interpolation and verified speed readback")
    value = float(hand._read_setting("get_speed_coeff"))
    if not np.isfinite(value) or not 0 < value <= 1:
        raise ValueError("Home recovery requires a positive finite original speed coefficient")
    return value


def _set_verified_speed(hand, desired):
    observed = float(hand._read_setting("get_speed_coeff"))
    if not np.isfinite(observed):
        raise RuntimeError("Home recovery speed readback is nonfinite")
    hand._ensure_coefficient("speed", observed, desired)
    actual = float(hand._read_setting("get_speed_coeff"))
    if not math.isclose(actual, desired, rel_tol=0, abs_tol=1e-6):
        raise RuntimeError(f"Home recovery speed verification failed: wanted {desired}, got {actual}")


def wait_recovery_feedback(feedback, *, after, phase, clock, sleep, emit):
    """No motion here: require new receipts AFTER settings finished, not old cache.

    `feedback` still applies the original source/receipt freshness and alignment
    checks. ValueError is retried only inside this stationary preparation gate.
    """
    deadline = clock() + RECOVERY_FEEDBACK_WAIT_S
    last_error = "waiting for new joint feedback after speed settings"
    emit({"event": "home_recovery_feedback_wait", "phase": phase})
    while clock() < deadline:
        try:
            sample = feedback()
        except ValueError as error:
            last_error = str(error)
        else:
            received = sample.received_monotonic_ns
            if len(received) != 4:
                raise RuntimeError("Home recovery requires four real receipt timestamps")
            if min(received) > after * 1e9:
                emit(
                    {
                        "event": "home_recovery_feedback_ready",
                        "phase": phase,
                        "receipt_ages_ms": [(clock() * 1e9 - value) / 1e6 for value in received],
                    }
                )
                return sample
            last_error = "cached joint receipts predate speed setting completion"
        sleep(0.02)
    raise TimeoutError(f"Home recovery {phase}: no new valid feedback within 5s; {last_error}")


def recover_before_home(executor, feedback, *, clock=time.monotonic, sleep=time.sleep, emit=lambda report: None):
    """Recover small initial excursions; restore firmware settings on every exit.

    `feedback` must be the worker's existing fresh_feedback validator. No arm
    commands, recorder gate, policy inference or task methods are invoked here.
    """
    initial = feedback().state.copy()
    target, changed = recovery_target(initial)
    if not changed:
        return {"status": "SKIPPED", "reason": "initial hand feedback already inside command limits"}
    hands = (executor.left, executor.right)
    original = [_speed_adapter(hand) for hand in hands]
    attempted = []
    report = {
        "status": "RUNNING",
        "joints": changed,
        "original_speed_coefficients": original,
        "recovery_speed_coefficients": [min(value, RECOVERY_SPEED_COEFFICIENT) for value in original],
    }
    limits = hand_joint_limits()
    hand_indices = np.r_[np.arange(7, 29), np.arange(36, 58)]
    try:
        for index, hand in enumerate(hands):
            attempted.append(index)
            _set_verified_speed(hand, min(original[index], RECOVERY_SPEED_COEFFICIENT))
        sample = wait_recovery_feedback(
            feedback, after=clock(), phase="before_motion", clock=clock, sleep=sleep, emit=emit
        )
        # State may change while SDK settings are being written. Revalidate the
        # full target/slew bound from the new real state, never from old cache.
        initial = sample.state.copy()
        target, changed = recovery_target(initial)
        report["joints"] = changed
        if changed:
            started = clock()
            initial_distance = np.abs(target[hand_indices] - initial[hand_indices])
            previous_error = float(np.max(initial_distance))
            progressed_at = started
            stable_since = None
            emit({"event": "home_recovery_start", **report})
            executor.resume_stream()
            next_tick = clock()
            while True:
                now = clock()
                if now - started >= RECOVERY_TIMEOUT_S:
                    raise TimeoutError("Home recovery timed out; ordinary Home/task blocked")
                if now < next_tick:
                    sleep(min(0.01, next_tick - now))
                    continue
                if now - next_tick > DT_NS / 1e9:
                    raise RuntimeError("Home recovery missed 30Hz deadline")
                # No retries after entering the motion phase.
                measured = feedback().state
                if clock() - next_tick > DT_NS / 1e9:
                    raise RuntimeError("Home recovery feedback missed 30Hz deadline")
                distance = np.abs(target[hand_indices] - measured[hand_indices])
                if np.any(distance >= PREPARE_SLEW_LIMIT_RAD) or np.any(distance > initial_distance + RECOVERY_MARGIN_RAD):
                    raise RuntimeError("Home recovery feedback moved away from recovery target")
                error = float(np.max(distance))
                if error < previous_error - 0.0005:
                    previous_error, progressed_at = error, now
                legal = all(
                    np.all((measured[part] >= limits[:, 0]) & (measured[part] <= limits[:, 1]))
                    for part in (LEFT_SHARPA_SLICE, RIGHT_SHARPA_SLICE)
                )
                if legal and error <= RECOVERY_TOLERANCE_RAD:
                    stable_since = now if stable_since is None else stable_since
                    if now - stable_since >= RECOVERY_STABLE_S:
                        report.update(status="PASS", elapsed_s=now - started)
                        break
                else:
                    stable_since = None
                    if now - progressed_at >= 2.0:
                        raise TimeoutError("Home recovery feedback made no progress; task blocked")
                executor.send(target[LEFT_SHARPA_SLICE], target[RIGHT_SHARPA_SLICE])
                next_tick += DT_NS / 1e9
        else:
            report.update(status="SKIPPED", reason="fresh feedback became legal during setup")
    finally:
        restore_errors = []
        for index in reversed(attempted):
            try:
                _set_verified_speed(hands[index], original[index])
            except BaseException as error:
                restore_errors.append(f"{'left' if index == 0 else 'right'}: {error}")
        if restore_errors:
            raise RuntimeError("Home recovery speed restoration failed; task blocked: " + "; ".join(restore_errors))
    # Restoration can also delay feedback. Wait without issuing commands, then
    # require NEW feedback before entering the ordinary Home phase.
    actual = wait_recovery_feedback(
        feedback, after=clock(), phase="after_restore", clock=clock, sleep=sleep, emit=emit
    ).state
    for side, part in (("left", LEFT_SHARPA_SLICE), ("right", RIGHT_SHARPA_SLICE)):
        validate_hand_command(actual[part], side, limits=limits)
    executor.resume_stream()
    emit({"event": "home_recovery_complete", **report, "speed_restored": True})
    return report
