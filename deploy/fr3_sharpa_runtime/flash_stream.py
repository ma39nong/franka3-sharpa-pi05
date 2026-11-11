"""FLASH execution timeline, independent of model RTC prefix guidance.

All phases use ceil on the host's monotonic observation anchor. Installation
only schedules a future control boundary; issued/committed frames stay intact.
"""
from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .protocol import as_finite_action_chunk


@dataclass(frozen=True)
class FlashSettings:
    seam_position_rad: float
    seam_velocity_rad_s: float
    feedback_error_rad: float
    min_remaining_nodes: int = 20
    reserve_s: float = 0.1
    trigger_nodes: int = 15

    def __post_init__(self):
        # Deliberately no fabricated hardware seam/feedback defaults.
        for value in (self.seam_position_rad, self.seam_velocity_rad_s, self.feedback_error_rad):
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError("flashRTC requires positive configured seam position/velocity and feedback limits")
        if type(self.min_remaining_nodes) is not int or not 2 <= self.min_remaining_nodes <= 49:
            raise ValueError("Invalid FLASH minimum remaining nodes")
        if not np.isfinite(self.reserve_s) or not 1 / 30 <= self.reserve_s <= 0.5:
            raise ValueError("Invalid FLASH stop reserve")
        if type(self.trigger_nodes) is not int or not 1 <= self.trigger_nodes <= 24:
            raise ValueError("Invalid FLASH replan trigger")

    def required_suffix(self, observed_latency_nodes):
        # Reserve the next trigger interval, the latency just measured, stop
        # reserve, and two committed control intervals. Slower later requests
        # still have their own hard deadlines; this is no latency guarantee.
        return max(self.min_remaining_nodes, self.trigger_nodes + observed_latency_nodes
                   + math.ceil(self.reserve_s * 30) + 2)


def observation_anchor(observation, now):
    ctx = observation["flash_context"]
    captured = ctx["captured_monotonic_ns"]
    age = ctx["anchor_age_at_capture_ns"]
    if (type(captured) is not int or captured <= 0 or type(age) is not int or age < 0
            or not 0 <= now * 1e9 - captured <= 200_000_000):
        raise ValueError("FLASH request observation is stale or has an invalid clock")
    return (captured - age) / 1e9


def phase_index(anchor, now):
    if not np.isfinite([anchor, now]).all() or anchor <= 0 or now < anchor:
        raise ValueError("FLASH clock/observation anchor invalid")
    return math.ceil((now - anchor) * 30 - 1e-7)


def action_digest(actions):
    return hashlib.sha256(actions.tobytes()).hexdigest()


def validate_execution_plan(plan, identity, now, max_age_ms=None):
    from .flash_contract import check_plan_identity
    ctx = plan["flash_context"]
    check_plan_identity(ctx)
    expected = {key: ctx[key] for key in
                ("protocol_version", "request_id", "session_id", "observation_anchor_source_ns")}
    expected["model_identity"] = identity
    captured, age = ctx["captured_monotonic_ns"], ctx["anchor_age_at_capture_ns"]
    if type(captured) is not int or captured <= 0 or type(age) is not int or age < 0:
        raise ValueError("Invalid FLASH execution clock")
    anchor = (captured - age) / 1e9
    phase_index(anchor, now)
    if (plan["receipt"] != expected or plan["model_identity"] != identity
            or plan["request_id"] != ctx["request_id"] or plan["history_session"] != ctx["session_id"]
            or plan["anchor_s"] != anchor):
        raise ValueError("FLASH worker receipt/model/anchor mismatch")
    if max_age_ms is not None and (now - anchor) * 1000 > max_age_ms:
        raise ValueError("FLASH candidate exceeds optional plan age budget")


class FlashAsyncTimeline:
    def __init__(self, actions, anchor, now, session, model_identity, history_session, settings):
        self.settings = settings
        self.actions = as_finite_action_chunk(actions).copy()
        self.anchor = anchor
        self.installed_at = now
        self.session = session
        self.model_identity = model_identity
        self.history_session = history_session
        self.generation = 1
        self.digest = action_digest(self.actions)
        self.pending = None
        self.request = None
        self.closed = False
        self.last_now = now
        self.sample(now)
        if 50 - self.phase(now) < settings.required_suffix(self.phase(now)):
            raise TimeoutError("FLASH initial plan has insufficient remaining nodes")

    def _active(self):
        if self.closed:
            raise ValueError("FLASH stream is stopped")

    def _advance(self, now):
        self._active()
        if not np.isfinite(now) or now < self.last_now:
            raise ValueError("FLASH monotonic clock moved backwards")
        self.last_now = now
        if self.pending is not None and now >= self.pending["switch_s"]:
            plan = self.pending
            self.actions, self.anchor = plan["actions"], plan["anchor"]
            self.installed_at = plan["switch_s"]
            self.digest = action_digest(self.actions)
            self.generation += 1
            self.pending = None

    def phase(self, now):
        self._advance(now)
        return phase_index(self.anchor, now)

    def sample(self, now):
        index = self.phase(now)
        if index >= 50:
            raise TimeoutError("FLASH accepted trajectory exhausted")
        return self.actions[index].copy()

    def status(self, now):
        self.sample(now)
        return dict(generation=self.generation, source_sha256=self.digest,
                    phase=self.phase(now), sampled_monotonic_s=now,
                    consumed_since_install=(now - self.installed_at) * 30,
                    remaining_s=(49 / 30 + self.anchor - now),
                    pending_install=self.pending is not None)

    def snapshot(self, request, now):
        self.sample(now)
        if self.pending is not None or self.request is not None:
            raise ValueError("Only one FLASH request/installation may be outstanding")
        expected = dict(run_session_id=self.session, source_generation=self.generation,
                        source_sha256=self.digest, model_identity=self.model_identity,
                        history_session=self.history_session)
        if any(request.get(key) != value for key, value in expected.items()):
            raise ValueError("FLASH request session/source/model/history mismatch")
        if (not request.get("request_id") or not np.isfinite(request.get("deadline_s", float("nan")))
                or not now < request["deadline_s"] <= self.anchor + 49 / 30):
            raise ValueError("FLASH request deadline invalid")
        anchor = request.get("anchor_s", 0)
        phase_index(anchor, now)
        self.request = dict(request)

    def install(self, actions, request, now, switch_s, last_command, measured, validate, *, next_tick=None):
        self.sample(now)  # Never revive an exhausted old stream.
        if request != self.request or request is None:
            raise ValueError("FLASH response has no matching outstanding request")
        if (self.pending is not None or request["source_generation"] != self.generation
                or request["source_sha256"] != self.digest or now >= request["deadline_s"]):
            raise ValueError("FLASH candidate is stale")
        # Commit at least two control periods ahead, beyond any issued frame.
        if switch_s < now + 1 / 30 or switch_s >= request["deadline_s"]:
            raise ValueError("FLASH switch violates commitment/deadline")
        actions = as_finite_action_chunk(actions).copy()
        index = phase_index(request["anchor_s"], switch_s)
        required = self.settings.required_suffix(phase_index(request["anchor_s"], now))
        if index >= 50 or 50 - index < required:
            raise TimeoutError("FLASH candidate has insufficient remaining suffix")
        old_index = phase_index(self.anchor, switch_s)
        if old_index >= 49:
            raise TimeoutError("FLASH old stream cannot cover commitment")
        old = self.actions[old_index]
        new = actions[index]
        position = float(np.max(np.abs(new - old)))
        velocity = float(np.max(np.abs((actions[min(index + 1, 49)] - new)
                            - (self.actions[old_index + 1] - old))) * 30)
        feedback_error = float(np.max(np.abs(np.asarray(measured) - np.asarray(last_command))))
        if (not np.isfinite([position, velocity, feedback_error]).all()
                or position > self.settings.seam_position_rad
                or velocity > self.settings.seam_velocity_rad_s
                or feedback_error > self.settings.feedback_error_rad):
            raise ValueError("FLASH seam position/velocity/feedback rejected")
        # Check all committed old frames, seam and candidate suffix BEFORE mutation.
        commands = [np.asarray(last_command)]
        tick = now if next_tick is None else next_tick
        if not now <= tick < now + 1 / 30 + 1e-7:
            raise ValueError("FLASH next committed tick is stale/invalid")
        while tick < switch_s - 1e-9:
            commands.append(self.actions[phase_index(self.anchor, tick)].copy())
            tick += 1 / 30
        commands.extend(actions[index:])
        if np.max(np.abs(np.diff(commands, axis=0))) > self.settings.seam_position_rad:
            raise ValueError("FLASH committed/candidate command step exceeds configured limit")
        validate(np.asarray(commands))
        result = dict(switch_s=switch_s, skipped=index, source_generation=self.generation,
                      generation=self.generation + 1, seam_position_rad=position,
                      seam_velocity_rad_s=velocity, feedback_error_rad=feedback_error,
                      required_remaining_nodes=required, remaining_nodes=50-index)
        self.pending = dict(actions=actions, anchor=request["anchor_s"], switch_s=switch_s)
        self.request = None
        return result

    def cancel_request(self):
        self.request = None

    def stop(self):
        self.closed = True
        self.pending = self.request = None
