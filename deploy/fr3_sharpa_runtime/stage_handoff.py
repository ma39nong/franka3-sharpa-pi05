"""Bounded online trajectory handover; ordinary within-stage RTC is unchanged.

The worker remains the sole 30Hz owner. A switch revokes old receipts, permits
only a bounded tail, and blends two time-aligned ABSOLUTE command trajectories.
This is a command continuity contract, not a physical stop/hold certificate.
"""

import math
import uuid

import numpy as np

from .protocol import as_finite_action_chunk
from .rtc_stream import RTCTimeline

SWITCH_MAX_SECONDS = 0.75
BRIDGE_STEPS = 6
DT = 1 / 30


def aligned_candidate(actions, captured, start, epoch):
    """Fractional phase preserves sensor-time alignment without rounding ahead."""
    candidate = RTCTimeline(as_finite_action_chunk(actions), start, 1.0, task_epoch=epoch)
    candidate.skipped = max(0.0, start - captured) * candidate.node_hz
    return candidate


def trajectory_metrics(commands):
    """Command-space diagnostics only: no invented acceleration/jerk limits."""
    commands = np.asarray(commands)
    groups = {"left_arm": slice(0, 7), "left_hand": slice(7, 29),
              "right_arm": slice(29, 36), "right_hand": slice(36, 58)}
    result = {}
    for name, part in groups.items():
        result[name] = {}
        for order, label in ((1, "velocity_rad_s"), (2, "acceleration_rad_s2"), (3, "jerk_rad_s3")):
            delta = np.diff(commands[:, part], n=order, axis=0) / DT**order
            result[name][label] = float(np.max(np.abs(delta))) if delta.size else 0.0
    return result


class BlendCurve:
    """Pure time sampling, also used to supersede an already armed bridge."""

    def __init__(self, source, candidate, start):
        self.source, self.candidate, self.start = source, candidate, start

    def sample(self, now):
        alpha = np.clip((now - self.start) / (BRIDGE_STEPS * DT), 0.0, 1.0)
        weight = alpha**3 * (10 - 15 * alpha + 6 * alpha**2)
        if now >= self.start + BRIDGE_STEPS * DT:
            return self.candidate.sample(now)
        return (1 - weight) * self.source.sample(now) + weight * self.candidate.sample(now)


def validate_command_step(previous, command, arm_speed):
    """Additional conservative gate; never replaces gateway or SDK checks."""
    if not math.isfinite(arm_speed) or arm_speed <= 0:
        raise ValueError("Unknown FR3 gateway speed; online handover blocked")
    delta = np.abs(np.asarray(command) - np.asarray(previous))
    arms = np.r_[delta[:7], delta[29:36]]
    if np.any(arms > arm_speed * DT + 1e-7):
        raise ValueError("Online handover exceeds live FR3 gateway per-tick speed")
    from .sdk_direct import SDK_SLEW_LIMIT_RAD

    if np.any(np.r_[delta[7:29], delta[36:]] >= SDK_SLEW_LIMIT_RAD):
        raise ValueError("Online handover exceeds existing Sharpa slew limit")


class StageTimeline(RTCTimeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.switch = None
        self.bridge = None
        self.applied_token = None
        self.source = None

    def source_curve(self):
        if self.source is not None:
            return self.source
        curve = RTCTimeline(self.actions, self.installed_at, 1.0, task_epoch=self.task_epoch)
        curve.skipped = self.skipped
        return curve

    def begin_switch(self, epoch, request_id, now):
        self.status(now)  # Inspect only; bridge completion belongs to a device tick.
        if self.node_hz != 30:
            raise ValueError("Online handover requires the native 30Hz timeline")
        if type(epoch) is not int or self.task_epoch is None or epoch <= self.task_epoch:
            raise ValueError("Switch epoch must advance monotonically")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("Switch requires an operator request ID")
        if self.bridge is not None:
            self.source = BlendCurve(self.source_curve(), self.bridge["candidate"], self.bridge["start"])
            self.bridge = None
        deadline = (
            self.switch["deadline"]
            if self.switch
            else min(now + SWITCH_MAX_SECONDS, now + (50 - self.phase(now)) / self.node_hz - DT)
        )
        if deadline - now <= (BRIDGE_STEPS + 2) * DT:
            raise TimeoutError("Insufficient old trajectory budget for an online switch")
        self.source = self.source_curve()  # Freeze/copy/hash once, not on each 30Hz tick.
        self.pending = None
        self.generation += 1
        self.task_epoch = epoch
        self.switch = dict(
            token=uuid.uuid4().hex,
            task_epoch=epoch,
            request_id=request_id,
            source_generation=self.generation,
            source_sha256=self.digest,
            requested_at=now,
            deadline=deadline,
        )
        return dict(self.switch)

    def sample(self, now):
        if self.switch and now >= self.switch["deadline"]:
            raise TimeoutError("Online stage switch deadline exceeded; stop required")
        if self.bridge is None:
            return self.source.sample(now) if self.source is not None else super().sample(now)
        bridge = self.bridge
        alpha = np.clip((now - bridge["start"]) / (BRIDGE_STEPS * DT), 0.0, 1.0)
        target = bridge["candidate"].sample(now)
        if now >= bridge["start"] + BRIDGE_STEPS * DT:
            candidate = bridge["candidate"]
            token = self.switch["token"]
            # Move the candidate's original time anchor; never restart its clock
            # after spending time executing the bridge.
            self._install(candidate.actions, candidate.installed_at, candidate.skipped)
            self.applied_token = token
            self.bridge = self.switch = None
            self.source = None
            return target
        weight = alpha**3 * (10 - 15 * alpha + 6 * alpha**2)
        source = self.source_curve().sample(now)
        return (1 - weight) * source + weight * target

    def status(self, now):
        # Only actual device ticks call sample() and commit bridge completion.
        status = super().status(now)
        if self.switch and now >= self.switch["deadline"]:
            raise TimeoutError("Online stage switch deadline exceeded")
        return {
            **status,
            "task_epoch": self.task_epoch,
            "handoff": ({**self.switch, "state": "bridging" if self.bridge else "waiting"} if self.switch else None),
            "applied_token": self.applied_token,
            "applied_monotonic_s": getattr(self, "applied_monotonic_s", None),
        }

    def prepare_handoff(self, actions, ticket, now, next_tick, last_command, arm_speed, validate):
        if not self.switch or self.bridge:
            raise ValueError("No pending online handover or duplicate installation")
        expected = self.switch
        if any(
            ticket.get(key) != expected[key]
            for key in ("token", "task_epoch", "request_id", "source_generation", "source_sha256")
        ):
            raise ValueError("Stale online handover token/epoch/source")
        captured = ticket.get("captured_monotonic_s")
        if not isinstance(captured, (int, float)) or not 0 <= now - captured <= 0.2:
            raise ValueError("Online handover observation/result is stale")
        start = max(now, next_tick)
        end = start + BRIDGE_STEPS * DT
        if end + DT >= expected["deadline"]:
            raise TimeoutError("New stage arrived too late for the bounded bridge")
        candidate = aligned_candidate(actions, captured, start, self.task_epoch)
        commands = [np.asarray(last_command)]
        for i in range(BRIDGE_STEPS + 1):
            t = start + i * DT
            alpha = i / BRIDGE_STEPS
            weight = alpha**3 * (10 - 15 * alpha + 6 * alpha**2)
            commands.append((1 - weight) * self.source_curve().sample(t) + weight * candidate.sample(t))
        bridge_command_count = len(commands)
        # Admit the actual bridge against the gateway speed. Ordinary RTC may
        # replace the predicted suffix before it is played; its future speed
        # is diagnostic, rather than an extra cross-stage installation gate.
        for previous, command in zip(commands, commands[1:]):
            validate_command_step(previous, command, arm_speed)
        # Also validate candidate continuation and the bridge's exit boundary.
        for i in range(BRIDGE_STEPS + 1, math.ceil(50 - candidate.skipped)):
            commands.append(candidate.sample(start + i * DT))
        from .sdk_direct import SDK_SLEW_LIMIT_RAD
        continuation = np.asarray(commands[bridge_command_count - 1 :])
        hand_delta = np.diff(continuation[:, np.r_[7:29, 36:58]], axis=0)
        if np.any(np.abs(hand_delta) >= SDK_SLEW_LIMIT_RAD):
            raise ValueError("Online handover exceeds existing Sharpa slew limit")
        # Reuse native full-chunk range/hand slew validator, padded by holding
        # the last command only for validation; no held tail is executed.
        for offset in range(0, len(commands) - 1, 49):
            window = np.asarray(commands[offset : offset + 50])
            padded = np.repeat(window[-1][None, :], 50, axis=0)
            padded[: len(window)] = window
            validate(padded)
        self.bridge = dict(start=start, candidate=candidate)
        arm_delta = np.diff(continuation[:, np.r_[0:7, 29:36]], axis=0)
        continuation_arm_speed = float(np.max(np.abs(arm_delta)) / DT) if arm_delta.size else 0.0
        return dict(
            accepted=True,
            applied=False,
            token=expected["token"],
            generation=self.generation,
            skipped=candidate.skipped,
            bridge_start=start,
            bridge_end=end,
            observation_age_s=now - captured,
            alignment_rounding_s=0.0,
            trajectory_metrics=trajectory_metrics(commands),
            speed_validation_scope="bridge_and_entry",
            continuation_arm_speed_rad_s=continuation_arm_speed,
            continuation_exceeds_gateway_speed=continuation_arm_speed > arm_speed,
        )

    def snapshot(self, *args, **kwargs):
        if self.switch:
            raise ValueError("Ordinary RTC prefix forbidden during stage handover")
        return super().snapshot(*args, **kwargs)

    def install(self, *args, **kwargs):
        if self.switch:
            raise ValueError("Old RTC result cannot install during stage handover")
        return super().install(*args, **kwargs)

    def invalidate(self):
        self.switch = self.bridge = None
        self.source = None
        super().invalidate()
