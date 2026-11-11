"""Direct Sharpa Wave SDK backend.

This module intentionally has no dependency on the historical litchibot
validation, permit, acquisition, or handoff modules.  The SDK import is lazy
so dry-run and unit tests never touch hardware.
"""

from __future__ import annotations

import os
import time
import json
import math
from dataclasses import dataclass
from typing import Callable, Any

import numpy as np

from .protocol import DT_NS, LEFT_SHARPA_SLICE, RIGHT_SHARPA_SLICE, SHARPA_JOINT_NAMES, as_finite_action_chunk

SDK_SLEW_LIMIT_RAD = 0.20
PREPARE_SLEW_LIMIT_RAD = 0.10
WATCHDOG_TIMEOUT_S = 0.25
DEFAULT_PREDICTION_CLIP_RAD = 0.05


def scaled_action_frames(actions: np.ndarray, speed_scale: float, *, execute_steps=50):
    """Time-stretch an absolute chunk while retaining 30 Hz output."""
    chunk = action_prefix(actions, execute_steps)
    if not np.isfinite(speed_scale) or not 0.0 < speed_scale <= 1.0:
        raise ValueError("execute speed scale must be finite and in (0, 1]")
    steps = max(1, int(math.ceil(1.0 / float(speed_scale))))
    previous = chunk[0].copy()
    yield previous
    for target in chunk[1:]:
        delta = target - previous
        for index in range(1, steps + 1):
            yield previous + delta * (index / steps)
        previous = target.copy()


def action_prefix(actions, execute_steps):
    if type(execute_steps) is not int or not 1 <= execute_steps <= 50:
        raise ValueError("execute_steps must be an integer in [1, 50]")
    return as_finite_action_chunk(actions)[:execute_steps]


def chunk_command_frames(actions, start_state, *, speed_scale=1.0, approach_speed_rad_s=0.03, execute_steps=50):
    """Join measured state to a new prediction, preserving every selected target.

    The bridge is joint-space interpolation, not another policy inference.
    Check slew on the emitted stream (including its first transition), rather
    than rejecting raw nodes that become executable through time stretching.
    Joint limits on the policy targets and start state are checked by ROS.
    """
    chunk = action_prefix(actions, execute_steps)
    start = np.asarray(start_state, dtype=np.float32)
    if start.shape != (58,) or not np.isfinite(start).all():
        raise ValueError("Chunk bridge requires finite measured 58D state")
    if not np.isfinite(approach_speed_rad_s) or approach_speed_rad_s <= 0:
        raise ValueError("Chunk bridge speed must be finite and positive")
    delta = chunk[0] - start
    distance = float(np.max(np.abs(delta)))
    bridge_steps = int(math.ceil(distance / (approach_speed_rad_s * DT_NS / 1e9)))
    bridge = [start + delta * (i / bridge_steps) for i in range(1, bridge_steps + 1)]
    frames = np.asarray([*bridge, *scaled_action_frames(actions, speed_scale, execute_steps=execute_steps)], dtype=np.float32)
    changes = np.diff(np.vstack([start, frames]), axis=0)
    for part in (LEFT_SHARPA_SLICE, RIGHT_SHARPA_SLICE):
        if np.max(np.abs(changes[:, part])) >= SDK_SLEW_LIMIT_RAD:
            raise ValueError("Interpolated Sharpa stream exceeds slew limit; reduce execute speed scale")
    return frames, bridge_steps


def play_command_frames(frames, send, feedback, *, clock=time.monotonic, sleep=time.sleep):
    """Schedule 30Hz against an absolute clock, without accumulating send time.

    Never burst stale frames to catch up. Real feedback is checked before every
    command; the native SDK watchdog and FR3 gateway remain authoritative.
    """
    period = DT_NS / 1e9
    started = clock()
    for index, frame in enumerate(frames):
        due = started + index * period
        remaining = due - clock()
        if remaining > 0:
            sleep(remaining)
        if clock() - due > period:
            raise RuntimeError("Chunk 30Hz deadline missed; stopped")
        feedback()
        if clock() - due > period:
            raise RuntimeError("Chunk feedback missed 30Hz deadline; stopped")
        send(frame)
    remaining = started + len(frames) * period - clock()
    if remaining > 0:
        sleep(remaining)


def hand_joint_limits():
    """Use exactly the same static range as the lower-level hand driver."""
    try:
        from litchi_hardware.hardware.sharpa.constants import SHARPA_JOINT_LIMITS
    except ImportError:
        schema_path = os.getenv(
            "SHARPA_SCHEMA_PATH", "/home/user/gello_upper_body_teleop/config/litchibot/sharpa_schema.json")
        with open(schema_path, encoding="utf-8") as stream:
            limits = json.load(stream)["joint_limits"]
        SHARPA_JOINT_LIMITS = limits
    limits = np.asarray(SHARPA_JOINT_LIMITS, dtype=np.float64)
    if limits.shape != (22, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError("Invalid Sharpa joint limit schema")
    return limits


def _hand_bounds_float32(limits):
    """Represent the driver's float64 interval strictly from the inside."""
    lower, upper = limits[:, 0].astype(np.float32), limits[:, 1].astype(np.float32)
    lower = np.where(lower.astype(np.float64) < limits[:, 0], np.nextafter(lower, np.float32(np.inf)), lower)
    upper = np.where(upper.astype(np.float64) > limits[:, 1], np.nextafter(upper, np.float32(-np.inf)), upper)
    return lower, upper


def validate_hand_command(values, side, *, limits=None):
    """Validate each hand before either SDK call; include the rejected angle."""
    hand = np.asarray(values, dtype=np.float64)
    if hand.shape != (22,) or not np.isfinite(hand).all():
        raise ValueError(f"{side} Sharpa command requires 22 finite joint positions")
    limits = hand_joint_limits() if limits is None else limits
    outside = np.flatnonzero((hand < limits[:, 0]) | (hand > limits[:, 1]))
    if outside.size:
        joint = int(outside[0])
        raise ValueError(
            f"{side} Sharpa {SHARPA_JOINT_NAMES[joint]} command={hand[joint]:.10f} rad "
            f"outside [{limits[joint, 0]:.10f}, {limits[joint, 1]:.10f}] rad; no hand command sent")
    return tuple(float(x) for x in hand)


def prepare_command_with_limits(command, previous, *, speed_rad_s):
    """Keep prepare output inside driver limits and the existing per-tick speed.

    Feedback is never altered. A boundary projection is allowed only when the
    resulting command remains reachable from the previous command (or initial
    measured state) within this prepare tick. Larger excursions fail closed.
    This is separate from the model prediction clipping allowance.
    """
    candidate = np.asarray(command, dtype=np.float32).copy()
    previous = np.asarray(previous, dtype=np.float64)
    if (candidate.shape != (58,) or previous.shape != (58,)
            or not np.isfinite(candidate).all() or not np.isfinite(previous).all()
            or not np.isfinite(speed_rad_s) or speed_rad_s <= 0):
        raise ValueError("Prepare requires finite 58D commands/feedback and positive speed")
    limits = hand_joint_limits()
    lower, upper = _hand_bounds_float32(limits)
    step_budget = min(float(speed_rad_s) * DT_NS / 1e9, PREPARE_SLEW_LIMIT_RAD)
    corrections = []
    for side, part in (("left", LEFT_SHARPA_SLICE), ("right", RIGHT_SHARPA_SLICE)):
        raw = candidate[part].copy()
        adjusted = np.clip(raw, lower, upper)
        for joint in np.flatnonzero(adjusted != raw):
            dimension = part.start + int(joint)
            movement = abs(float(adjusted[joint]) - previous[dimension])
            if movement > step_budget or movement >= PREPARE_SLEW_LIMIT_RAD:
                raise ValueError(
                    f"Prepare {side} Sharpa {SHARPA_JOINT_NAMES[joint]}: "
                    f"previous_or_initial_feedback={previous[dimension]:.10f} rad, "
                    f"generated_command={float(raw[joint]):.10f} rad, "
                    f"limits=[{limits[joint, 0]:.10f}, {limits[joint, 1]:.10f}] rad; "
                    f"nearest legal command requires {movement:.10f} rad > allowed prepare step "
                    f"{step_budget:.10f} rad (or SDK slew limit). "
                    "Home stopped before sending this frame; inspect measured hand position/calibration")
            corrections.append({"side": side, "joint": SHARPA_JOINT_NAMES[joint],
                                "previous_rad": float(previous[dimension]), "original_rad": float(raw[joint]),
                                "command_rad": float(adjusted[joint]), "movement_rad": movement})
        validate_hand_command(adjusted, side, limits=limits)
        candidate[part] = adjusted
    return candidate, corrections


def clip_hand_predictions(actions, *, execute_steps=15, max_correction_rad=DEFAULT_PREDICTION_CLIP_RAD):
    """Project only selected model hand targets; never clip Home or measured state."""
    if not np.isfinite(max_correction_rad) or max_correction_rad < 0:
        raise ValueError("prediction clip allowance must be finite and nonnegative")
    corrected = as_finite_action_chunk(actions)
    action_prefix(corrected, execute_steps)
    limits = hand_joint_limits()
    # Round bounds inward: float32(upper) can otherwise exceed the float64 driver limit.
    lower, upper = _hand_bounds_float32(limits)
    corrections = []
    for side, part in (("left", LEFT_SHARPA_SLICE), ("right", RIGHT_SHARPA_SLICE)):
        raw = corrected[:execute_steps, part].copy()
        clipped = np.clip(raw, lower, upper)
        differences = np.abs(clipped - raw)
        if np.any(differences > np.float32(max_correction_rad)):
            frame, joint = np.unravel_index(np.argmax(differences), differences.shape)
            raise ValueError(
                f"{side} {SHARPA_JOINT_NAMES[joint]} action[{frame}]={raw[frame, joint]:.8f} rad "
                f"requires correction {differences[frame, joint]:.8f} rad > allowance {max_correction_rad:.8f} rad")
        for frame, joint in np.argwhere(differences > 0):
            corrections.append({"action_index": int(frame), "side": side,
                                "joint": SHARPA_JOINT_NAMES[joint], "action_dimension": part.start + int(joint),
                                "original_rad": float(raw[frame, joint]), "corrected_rad": float(clipped[frame, joint]),
                                "correction_rad": float(differences[frame, joint])})
        corrected[:execute_steps, part] = clipped
    return corrected, {"max_allowed_correction_rad": float(max_correction_rad),
                       "corrected_values": len(corrections),
                       "max_applied_correction_rad": max((row["correction_rad"] for row in corrections), default=0.0),
                       "corrections": corrections}


def validate_direct_hand_actions(actions: np.ndarray, *, check_slew: bool = True) -> None:
    """Validate finite positions, native Wave limits and per-frame slew."""
    chunk = as_finite_action_chunk(actions)
    limits = hand_joint_limits()
    for side, part in (("left", LEFT_SHARPA_SLICE), ("right", RIGHT_SHARPA_SLICE)):
        hand = chunk[:, part]
        if np.any(hand < limits[:, 0]) or np.any(hand > limits[:, 1]):
            raise ValueError(f"{side} Sharpa SDK command exceeds Wave joint limits")
        if check_slew and len(hand) > 1 and np.max(np.abs(np.diff(hand, axis=0))) >= SDK_SLEW_LIMIT_RAD:
            raise ValueError(f"{side} Sharpa SDK command exceeds slew limit")


@dataclass
class DirectSdkConfig:
    left_serial: str | None = None
    right_serial: str | None = None
    watchdog_timeout_s: float = WATCHDOG_TIMEOUT_S


class SharpaSdkDirectExecutor:
    """Two-hand executor around litchi_hardware SharpaSdkHand.

    ``hand_factory`` is injectable for unit tests.  Real SDK objects are only
    constructed by ``connect`` and only in the explicit hardware path.
    """

    def __init__(self, cfg: DirectSdkConfig | None = None,
                 hand_factory: Callable[..., Any] | None = None) -> None:
        self.cfg = cfg or DirectSdkConfig(
            left_serial=os.getenv("SHARPA_LEFT_SERIAL"),
            right_serial=os.getenv("SHARPA_RIGHT_SERIAL"),
        )
        self._factory = hand_factory
        self.left = self.right = None
        self._last_send = None

    def connect(self) -> None:
        sdk_factory = self._factory is None
        if self._factory is None:
            from litchi_hardware.hardware.sharpa import create_hand
            from litchi_hardware.core.types import HandSide
            from litchi_hardware.hardware.sharpa.config import SharpaConfig
            def factory(side, serial):
                # Some installed litchi_hardware wheels expose ``HandSide``
                # as a plain string for compatibility.  Normalize at this
                # boundary so the SDK config always receives the enum it
                # expects, while avoiding a fragile ``side.value`` access.
                resolved_side = side if isinstance(side, HandSide) else HandSide(str(side))
                return create_hand(SharpaConfig(
                    name=f"sharpa-{resolved_side.value}", side=resolved_side, serial=serial))
            self._factory = factory
        if not sdk_factory:
            # Test factories may accept opaque side values and must not need
            # the litchi_hardware package installed.
            left_side, right_side = "left", "right"
        else:
            from litchi_hardware.core.types import HandSide
            left_side, right_side = HandSide.LEFT, HandSide.RIGHT
        self.left = self._factory(left_side, self.cfg.left_serial)
        self.left.connect()
        # Do not expose an unstarted right-hand object as an owned connection
        # if startup is interrupted while connecting the left hand.
        self.right = self._factory(right_side, self.cfg.right_serial)
        try:
            self.right.connect()
        except Exception:
            self.left.disconnect()
            self.left = None
            raise
        # The transport may reserve a short synchronized start delay after
        # connect; watchdog starts with the first command, not connection.
        self._last_send = None

    def send(self, left: Any, right: Any) -> None:
        if self.left is None or self.right is None:
            raise RuntimeError("Sharpa SDK backend is not connected")
        now = time.monotonic()
        if self._last_send is not None and now - self._last_send > self.cfg.watchdog_timeout_s:
            self.stop()
            raise RuntimeError("Sharpa SDK watchdog expired")
        limits = hand_joint_limits()
        left = validate_hand_command(left, "left", limits=limits)
        right = validate_hand_command(right, "right", limits=limits)
        self.left.command_joint_position(left)
        self.right.command_joint_position(right)
        self._last_send = now

    def resume_stream(self) -> None:
        """Start a new authorized command phase without reconnecting.

        Policy inference and live observation happen between 30 Hz command
        phases.  That gap is intentional, so the next phase must establish a
        fresh watchdog epoch; once its first frame is sent, the normal
        watchdog continues to protect every subsequent frame.
        """
        if self.left is None or self.right is None:
            raise RuntimeError("Sharpa SDK backend is not connected")
        self._last_send = None

    def stop(self) -> list[str]:
        """Best-effort stop and disconnect for both hands.

        A transport failure on one hand must not prevent the other hand from
        receiving its stop/disable attempt.  Errors are returned so callers
        can surface a cleanup warning without aborting the already completed
        motion result.
        """
        errors = []
        for side, hand in (("left", self.left), ("right", self.right)):
            if hand is None:
                continue
            try:
                hand.stop()
            except Exception as error:
                errors.append(f"{side}.stop: {error}")
            try:
                hand.disconnect()
            except Exception as error:
                errors.append(f"{side}.disconnect: {error}")
        self.left = self.right = None
        self._last_send = None
        return errors

    def shutdown_confirmed(self):
        """Final-exit disable verification on owned handles, without reconnecting."""
        from .shutdown import disable_and_confirm
        try:
            return disable_and_confirm({"left": self.left, "right": self.right})
        finally:
            self.left = self.right = None
            self._last_send = None
