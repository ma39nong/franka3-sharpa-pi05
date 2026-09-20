"""Medium trajectory lifecycle, independently maintained from slow OneShot."""
from dataclasses import dataclass, field
import uuid
import numpy as np
from .runtime.core import (
    ARM, HAND, Admission, Frame, check_tracking, finite, plan_digest, speed_caps,
)
from .runtime.limits import (
    ARM_ENDPOINT_TOLERANCE_RAD, HAND_ENDPOINT_TOLERANCE_RAD,
    HAND_CONTACT_SETTLE_SECONDS, SLIDER_POSITION_TOLERANCE_RAD, check_hand_control,
)
from .profile import ARM_SPEED_RAD_S
SPEED = speed_caps(ARM_SPEED_RAD_S)

@dataclass
class OneShot:
    plan: object
    admission: Admission
    prepared_at: float
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: str = "prepared"
    reason: str | None = None
    transitions: list = field(default_factory=list)

    def __post_init__(self):
        self.hand_control = check_hand_control(self.plan.config.get("hand_control", "strict"))
        self.checked = ARM if self.hand_control == "slider" else np.arange(54)
        self.admission.check()
        finite(self.prepared_at)
        if self.prepared_at < self.admission.received_time:
            raise ValueError("Plan preparation predates inference")
        if self.plan.raw.shape != (self.plan.config.get("execution_steps", 50), 54):
            raise ValueError("This player executes the configured prediction prefix")
        for phase in (self.plan.approach, self.plan.playback):
            speed = np.maximum(np.abs(phase.lower[1]), np.abs(phase.upper[1])) / phase.scale
            if np.any(speed[self.checked] > SPEED[self.checked] + 1e-8):
                raise ValueError("Planned motion exceeds agreed arm/hand speed ceiling")
        if self.plan.report["total_seconds"] > 60:
            raise ValueError("One-shot plan exceeds 60-second motion budget")
        self.digest = plan_digest(self.plan)
        self.sequence = 0
        self.last_tick = None
        self.settled_since = None
        self.phase_started = None

    def transition(self, state, now, reason):
        self.transitions.append({"from": self.state, "to": state, "at": now, "reason": reason})
        self.state, self.phase_started, self.settled_since = state, now, None

    def check_start(self, feedback, now):
        if self.state != "prepared":
            raise ValueError("A one-shot plan cannot restart or resume")
        self._check_plan()
        feedback.check(now, self.admission.epoch)
        if feedback.arm_speed_rad_s != ARM_SPEED_RAD_S:
            raise ValueError("Medium player requires matching arm speed mode")
        if feedback.hand_control != self.hand_control:
            raise ValueError("Player and device hand control mismatch")
        if not self.prepared_at <= now <= self.prepared_at + 30:
            raise ValueError("Reviewed one-shot plan must start within 30 seconds of preparation")
        if np.max(np.abs(feedback.positions[self.checked] - self.plan.start[self.checked])) > (ARM_ENDPOINT_TOLERANCE_RAD if self.plan.config.get("continuation") else 0.005):
            raise ValueError("Start pose changed; rebuild from fresh feedback")
        if np.max(np.abs(feedback.velocities[self.checked])) > 0.02:
            raise ValueError("Start requires stationary feedback")

    def start(self, feedback, now):
        self.check_start(feedback, now)
        self.started = now
        # The finite committed plan is distinct from a streaming policy.
        # 1 s still governed admission above, and is never re-stamped.
        self.deadline = now + self.plan.report["total_seconds"] + (2 * HAND_CONTACT_SETTLE_SECONDS if self.hand_control == "slider" else 10)
        self.transition("approach", now, "single_plan_started")

    def _check_plan(self):
        if plan_digest(self.plan) != self.digest:
            raise ValueError("Plan changed since preparation")

    def request_stop(self, now, reason="operator_stop"):
        if self.state not in {"complete", "stopped", "fault"}:
            self.reason = reason
            self.transition("stopping", now, reason)

    def stopped(self, feedback, now):
        if self.state != "stopping":
            raise ValueError("No stop is pending")
        feedback.check(now, self.admission.epoch)
        if feedback.arm_speed_rad_s != ARM_SPEED_RAD_S:
            raise ValueError("Medium player requires matching arm speed mode")
        if feedback.hand_control != self.hand_control:
            raise ValueError("Player and device hand control mismatch")
        if np.max(np.abs(feedback.velocities)) > 0.02:
            self.settled_since = None
            return False
        if self.settled_since is None:
            self.settled_since = now
        if now - self.settled_since >= 0.5:
            self.transition("stopped", now, "measured_stop_confirmed")
            return True
        return False

    def tick(self, feedback, now):
        if self.state in {"prepared", "complete", "stopping", "stopped", "fault"}:
            return None
        try:
            self._check_plan()
            feedback.check(now, self.admission.epoch)
            if feedback.arm_speed_rad_s != ARM_SPEED_RAD_S:
                raise ValueError("Medium player requires matching arm speed mode")
            if feedback.hand_control != self.hand_control:
                raise ValueError("Player and device hand control mismatch")
            if now > self.deadline:
                raise ValueError("One-shot execution exceeded finite lifetime")
            if self.last_tick is not None and not 0 < now - self.last_tick <= 0.030000001:
                raise ValueError("Scheduler clock jump or missed ticks; no catch-up")
            self.last_tick = now
            if self.state in {"approach", "playback"}:
                phase = self.plan.approach if self.state == "approach" else self.plan.playback
                elapsed = min(now - self.phase_started, phase.duration)
                q, dq = phase.sample(elapsed), phase.sample(elapsed, 1)
                if self.hand_control == "slider":
                    index = 0 if self.state == "approach" else min(len(self.plan.raw) - 1, int(elapsed / phase.duration * (len(self.plan.raw) - 1)))
                    q[HAND], dq[HAND] = self.plan.raw[index, HAND], 0.0
                if elapsed >= phase.duration:
                    self.transition(
                        ("playback" if self.plan.config.get("continuation") else "settle_start")
                        if self.state == "approach" else "settle_end", now, "endpoint"
                    )
            else:
                q = self.plan.raw[0] if self.state == "settle_start" else self.plan.raw[-1]
                dq = np.zeros(54)
                settled = (
                    np.max(np.abs(feedback.positions[ARM] - q[ARM])) <= ARM_ENDPOINT_TOLERANCE_RAD
                    and np.max(np.abs(feedback.velocities[self.checked])) <= 0.02
                )
                hand_tolerance = (HAND_ENDPOINT_TOLERANCE_RAD if self.state == "settle_end"
                                  else SLIDER_POSITION_TOLERANCE_RAD if self.hand_control == "slider" else HAND_ENDPOINT_TOLERANCE_RAD)
                if self.hand_control == "strict":
                    settled = settled and feedback.hands_settled(q, hand_tolerance)
                if self.hand_control == "slider":
                    settled = (
                        settled
                        and feedback.hand_targets_reached
                        and feedback.hands_settled(q, hand_tolerance)
                    )
                if settled:
                    if self.settled_since is None:
                        self.settled_since = now
                    if now - self.settled_since >= (self.plan.config.get("final_settle_seconds", 0.5) if self.state == "settle_end" and self.plan.config.get("final_chunk", True) else self.plan.config["settle_seconds"]):
                        if self.state == "settle_end":
                            self.transition("complete", now, "final_contact_settle" if feedback.contact_indices(q) else "final_measured_settle")
                            return None
                        self.transition("playback", now, "initial_measured_settle")
                else:
                    self.settled_since = None
                settle_timeout = HAND_CONTACT_SETTLE_SECONDS if feedback.contact_indices(q) else 5
                if now - self.phase_started > settle_timeout:
                    raise ValueError("Measured endpoint settling timed out")
            check_tracking(q, feedback, self.checked, "Measured tracking")
            frame = Frame(
                self.run_id,
                self.digest,
                self.sequence,
                now,
                min(now + 0.02, self.deadline),
                now - self.started,
                q,
                dq,
                self.state,
            )
            self.sequence += 1
            return frame
        except ValueError as error:
            self.reason = str(error)
            self.transition("fault", now, self.reason)
            return None
