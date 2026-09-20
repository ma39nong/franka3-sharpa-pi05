"""Bounded single-shot lifecycle, separate from the legacy policy freshness contract.

A fresh inference may be admitted once, then planned into a finite trajectory.
Its original timestamps never change. Every generated command carries a separate
short deadline and is checked again by the output consumer. No device imports.
"""

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np

from .limits import ARM_TRACKING_TOLERANCE_RAD
from .limits import HAND_CONTACT_SECONDS
from .limits import HAND_SPEED_RAD_S
from .limits import check_hand_control

ARM = np.r_[0:7, 27:34]
HAND = np.r_[7:27, 34:54]
SPEED = np.full(54, HAND_SPEED_RAD_S)
SPEED[ARM] = 2.0
SPEED.flags.writeable = False


def checked_arm_speed(value):
    """Medium-only per-session arm limit, at most 2.0 rad/s."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value <= 2.0
    ):
        raise ValueError("Arm speed must be positive, finite and at most 2.0 rad/s")
    return float(value)


def speed_caps(arm_speed_rad_s):
    result = SPEED.copy()
    result[ARM] = checked_arm_speed(arm_speed_rad_s)
    result.flags.writeable = False
    return result


def vector(value):
    a = np.array(value, dtype=np.float64, copy=True)
    if a.shape != (54,) or not np.isfinite(a).all():
        raise ValueError("Expected 54 finite measured/commanded joint values")
    a.flags.writeable = False
    return a


def finite(*values):
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Non-finite timestamp or limit")


@dataclass(frozen=True)
class Feedback:
    positions: np.ndarray
    velocities: np.ndarray
    # Four groups, in model order. Both source and receipt ages are checked.
    source_times: tuple[float, ...]
    receipt_times: tuple[float, ...]
    epoch: int = 0
    healthy: bool = True
    hand_control: str = "strict"
    hand_targets_reached: bool = True
    hand_soft_limits: tuple = ()
    hand_contacts: tuple = ()
    arm_speed_rad_s: float = 2.0

    def __post_init__(self):
        object.__setattr__(self, "positions", vector(self.positions))
        object.__setattr__(self, "velocities", vector(self.velocities))
        check_hand_control(self.hand_control)
        object.__setattr__(self, "_speed", speed_caps(self.arm_speed_rad_s))
        limits = tuple(tuple(item) for item in self.hand_soft_limits)
        indices = set()
        for item in limits:
            if len(item) != 3:
                raise ValueError("Invalid hand soft limit")
            index, direction, bound = item
            if (
                type(index) is not int
                or index not in HAND
                or index in indices
                or type(direction) is not int
                or direction not in (-1, 1)
                or not math.isfinite(bound)
            ):
                raise ValueError("Invalid hand soft limit")
            indices.add(index)
        if limits and self.hand_control != "slider":
            raise ValueError("Stall soft limits require slider mode")
        object.__setattr__(self, "hand_soft_limits", limits)
        contacts = tuple(tuple(item) for item in self.hand_contacts)
        indices = set()
        for item in contacts:
            if len(item) != 4:
                raise ValueError("Invalid hand contact")
            index, direction, bound, started = item
            if (
                type(index) is not int
                or index not in HAND
                or index in indices
                or type(direction) is not int
                or direction not in (-1, 1)
                or not math.isfinite(bound)
                or not math.isfinite(started)
            ):
                raise ValueError("Invalid hand contact")
            indices.add(index)
        if contacts and (self.hand_control != "slider" or limits):
            raise ValueError("Timed hand contacts require slider mode without legacy soft limits")
        object.__setattr__(self, "hand_contacts", contacts)
        if len(self.source_times) != 4 or len(self.receipt_times) != 4:
            raise ValueError("Need timestamps for all four device groups")
        object.__setattr__(self, "source_times", tuple(self.source_times))
        object.__setattr__(self, "receipt_times", tuple(self.receipt_times))
        finite(*self.source_times, *self.receipt_times)

    def effective_target(self, target):
        if not self.hand_soft_limits:
            return target
        result = np.asarray(target, dtype=float).copy()
        for index, direction, bound in self.hand_soft_limits:
            if direction * (result[index] - bound) > 0:
                result[index] = bound
        return result

    def contact_indices(self, target):
        # Only the diagnosed loading direction can count as contact. An opening
        # target must still be physically reached; a latched warning is not enough.
        return tuple(
            index
            for index, direction, bound, _ in self.hand_contacts
            if direction * (target[index] - bound) >= 0 and direction * (target[index] - self.positions[index]) > 0.005
        )

    def hands_settled(self, target, tolerance):
        errors = np.abs(self.positions - self.effective_target(target))
        for index in self.contact_indices(target):
            if abs(self.velocities[index]) <= 0.02:
                errors[index] = 0.0
        return bool(np.all(errors[HAND] <= tolerance))

    def check(self, now, epoch):
        finite(now)
        if any(not 0 <= now - started < HAND_CONTACT_SECONDS for _, _, _, started in self.hand_contacts):
            raise ValueError("Hand contact duration exceeded or timestamp is in the future")
        if self.epoch != epoch or not self.healthy:
            raise ValueError("Feedback clock epoch changed or device unhealthy")
        if any(not 0 <= now - t <= 0.15 for t in (*self.source_times, *self.receipt_times)):
            raise ValueError("Missing, future or stale device feedback")
        indices = ARM if self.hand_control == "slider" else np.arange(54)
        exceeded = indices[np.abs(self.velocities[indices]) > self._speed[indices] + 1e-9]
        if exceeded.size:
            details = []
            for index in exceeded:
                group, offset = next(
                    (name, start)
                    for name, start, end in (
                        ("left_arm", 0, 7),
                        ("left_hand", 7, 27),
                        ("right_arm", 27, 34),
                        ("right_hand", 34, 54),
                    )
                    if start <= index < end
                )
                details.append(
                    f"{group}[{index - offset}] action_index={index} "
                    f"velocity={self.velocities[index]:.9f} rad/s "
                    f"limit={self._speed[index]:.9f} rad/s"
                )
            raise ValueError("Measured joint speed exceeds software ceiling: " + "; ".join(details))


@dataclass(frozen=True)
class Admission:
    observation_time: float
    sent_time: float
    received_time: float
    request_id: str
    checkpoint: str
    epoch: int

    def check(self):
        finite(self.observation_time, self.sent_time, self.received_time)
        if not self.request_id or not self.checkpoint:
            raise ValueError("Missing inference provenance")
        if not 0 <= self.sent_time - self.observation_time <= 0.07:
            raise ValueError("Inference observation exceeded 70 ms send budget")
        if not self.sent_time <= self.received_time <= self.observation_time + 1.0:
            raise ValueError("Inference response exceeded original 1 s admission deadline")


@dataclass(frozen=True)
class Frame:
    run_id: str
    plan_hash: str
    sequence: int
    created: float
    valid_until: float
    plan_elapsed: float
    positions: np.ndarray
    velocities: np.ndarray
    phase: str

    def __post_init__(self):
        object.__setattr__(self, "positions", vector(self.positions))
        object.__setattr__(self, "velocities", vector(self.velocities))
        finite(self.created, self.valid_until, self.plan_elapsed)
        if self.sequence < 0 or not self.run_id or not self.plan_hash:
            raise ValueError("Invalid output frame identity")
        if not 0 < self.valid_until - self.created <= 0.020000001:
            raise ValueError("Frame lifetime must be at most 20 ms")


def plan_digest(plan):
    h = hashlib.sha256()
    for name in ("raw", "knots", "start"):
        h.update(np.asarray(getattr(plan, name), dtype="<f8").tobytes())
    # Bind actual spline coefficients as well as config; changing a cached
    # interpolator must invalidate the reviewed plan.
    for phase in (plan.approach, plan.playback):
        h.update(np.asarray(phase.spline.t, dtype="<f8").tobytes())
        h.update(np.asarray(phase.spline.c, dtype="<f8").tobytes())
        h.update(np.asarray([phase.scale], dtype="<f8").tobytes())
    h.update(json.dumps(plan.config, sort_keys=True, allow_nan=False).encode())
    return h.hexdigest()


def checked_arm_tracking_tolerance(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Arm tracking tolerance must be positive and finite")
    return float(value)


def check_tracking(target, feedback, indices, label, *, arm_tracking_rad=ARM_TRACKING_TOLERANCE_RAD):
    errors = np.abs(target[indices] - feedback.positions[indices])
    limits = np.where(np.isin(indices, ARM), checked_arm_tracking_tolerance(arm_tracking_rad), 0.05)
    bad = np.flatnonzero(errors > limits)
    if bad.size:
        j = bad[np.argmax(errors[bad])]
        index = int(indices[j])
        group, offset = next(
            (name, start)
            for name, start, end in (
                ("left_arm", 0, 7),
                ("left_hand", 7, 27),
                ("right_arm", 27, 34),
                ("right_hand", 34, 54),
            )
            if start <= index < end
        )
        raise ValueError(
            f"{label} error exceeds {limits[j]:.2f} rad: {group}[{index-offset}], "
            f"target={target[index]:.6f}, measured={feedback.positions[index]:.6f}, "
            f"error={errors[j]:.6f} rad"
        )


class ConsumerGuard:
    """Must run at the final send boundary, not just when a frame is queued.

    Commit only after submission succeeds; partial device submission is a fault.
    A guard may arm once and may not be reused after stop/failure.
    """

    def __init__(
        self, lower, upper, *, hand_control="strict", arm_speed_rad_s=2.0, arm_tracking_rad=ARM_TRACKING_TOLERANCE_RAD
    ):
        self.arm_tracking_rad = checked_arm_tracking_tolerance(arm_tracking_rad)
        self.hand_control = check_hand_control(hand_control)
        self.checked = ARM if hand_control == "slider" else np.arange(54)
        self.arm_speed_rad_s = checked_arm_speed(arm_speed_rad_s)
        self.speed = speed_caps(self.arm_speed_rad_s)
        self.lower, self.upper = vector(lower), vector(upper)
        if np.any(self.lower >= self.upper):
            raise ValueError("Invalid position bounds")
        self.state = "new"
        self.last = None

    def arm(self, run_id, digest, feedback, now):
        if self.state != "new":
            raise ValueError("Output consumer is single-use")
        feedback.check(now, feedback.epoch)
        if feedback.arm_speed_rad_s != self.arm_speed_rad_s:
            raise ValueError("Arm speed mode mismatch")
        if feedback.hand_control != self.hand_control:
            raise ValueError("Hand control mode mismatch")
        if not run_id or not digest or np.any(np.abs(feedback.velocities[self.checked]) > 0.02):
            raise ValueError("Acquisition requires identity and stationary feedback")
        self.run_id, self.digest, self.epoch = run_id, digest, feedback.epoch
        self.initial = feedback.positions.copy()
        self.state = "armed"

    def validate(self, frame, feedback, now):
        if self.state != "armed":
            raise ValueError("Output consumer is not armed")
        feedback.check(now, self.epoch)
        if feedback.arm_speed_rad_s != self.arm_speed_rad_s:
            raise ValueError("Arm speed mode changed")
        if feedback.hand_control != self.hand_control:
            raise ValueError("Hand control mode changed")
        if (frame.run_id, frame.plan_hash) != (self.run_id, self.digest):
            raise ValueError("Foreign or superseded frame")
        if not frame.created <= now < frame.valid_until:
            raise ValueError("Expired or future command at final consumer")
        if np.any(frame.positions < self.lower) or np.any(frame.positions > self.upper):
            raise ValueError("Command position exceeds joint limits")
        if np.any(np.abs(frame.velocities[self.checked]) > self.speed[self.checked] + 1e-9):
            raise ValueError("Command derivative exceeds joint speed ceiling")
        check_tracking(frame.positions, feedback, self.checked, "Tracking", arm_tracking_rad=self.arm_tracking_rad)
        if self.last is None:
            if (
                frame.sequence != 0
                or np.max(np.abs(frame.positions[self.checked] - self.initial[self.checked])) > 0.005
            ):
                raise ValueError("First output must be the checked starting pose")
        else:
            if frame.sequence != self.last.sequence + 1:
                raise ValueError("Duplicate, out-of-order or skipped command")
            dt = frame.created - self.last.created
            if not 0 < dt <= 0.030000001:
                raise ValueError("Consumer command stream stalled; no catch-up")
            if np.any(
                np.abs(frame.positions[self.checked] - self.last.positions[self.checked])
                > self.speed[self.checked] * dt + 1e-8
            ):
                raise ValueError(f"Command slew exceeds arm {self.arm_speed_rad_s:g} rad/s or hand 90 deg/s")

    def commit(self, frame):
        self.last = frame

    def stop(self):
        self.state = "stopped"
