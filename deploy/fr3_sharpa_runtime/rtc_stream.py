"""Hardware-independent RTC timeline; time units are original 30 Hz model nodes.

The device clock remains 30 Hz. Playback speed scales model-node consumption,
including latency compensation, so slow is not treated as fast for RTC skips.
"""

from collections import deque
from dataclasses import dataclass
import hashlib
import math
import uuid

import numpy as np
from openpi_client import rtc

from .protocol import as_finite_action_chunk


@dataclass(frozen=True)
class RTCSettings:
    initial_delay_steps: int = 5
    delay_buffer_size: int = 10
    prefix_attention_schedule: str = "exp"
    max_guidance_weight: float = 5.0

    def __post_init__(self):
        if type(self.initial_delay_steps) is not int or not 0 <= self.initial_delay_steps <= 25:
            raise ValueError("rtc initial delay must be an integer in [0, 25]")
        if type(self.delay_buffer_size) is not int or not 1 <= self.delay_buffer_size <= 1000:
            raise ValueError("rtc delay buffer size must be an integer in [1, 1000]")
        if self.prefix_attention_schedule not in rtc.PREFIX_ATTENTION_SCHEDULES:
            raise ValueError("Invalid RTC prefix attention schedule")
        if not np.isfinite(self.max_guidance_weight) or not 0 < self.max_guidance_weight <= rtc.MAX_GUIDANCE_WEIGHT:
            raise ValueError("Invalid RTC max guidance weight")

    def history(self):
        return deque([self.initial_delay_steps], maxlen=self.delay_buffer_size)


def require_capabilities(metadata):
    if (
        metadata.get("rtc_supported") is not True
        or metadata.get("rtc_warmed_up") is not True
        or metadata.get("rtc_protocol_version") != rtc.PROTOCOL_VERSION
        or metadata.get("rtc_action_dim") != 58
        or metadata.get("rtc_algorithm") != "soft_prefix_guidance"
    ):
        raise ValueError("RTC requires a warmed 58D JAX soft-prefix-guidance service; start serve with --enable-rtc")


def canonical_prefix(actions):
    """The deployed Sharpa adapter already consumes canonical absolute 58D actions.

    Return a copy: DeltaActions mutates its action array. The policy then uses
    the NEW observation to re-anchor arms, followed by this checkpoint's stats.
    """
    actions = np.array(actions, dtype=np.float32, copy=True)
    if actions.ndim != 2 or actions.shape[1] != 58 or not np.isfinite(actions).all():
        raise ValueError("RTC prefix requires finite canonical (K, 58) actions")
    return actions


class RTCTimeline:
    def __init__(self, actions, now, speed_scale):
        if not np.isfinite(speed_scale) or not 0 < speed_scale <= 1:
            raise ValueError("Invalid RTC speed scale")
        self.node_hz = 30.0 / math.ceil(1.0 / speed_scale)
        self.generation = 0
        self.pending = None
        self._install(actions, now, 0)

    def _install(self, actions, now, skipped):
        self.actions = as_finite_action_chunk(actions).copy()
        self.digest = hashlib.sha256(self.actions.tobytes()).hexdigest()
        self.generation += 1
        self.installed_at = now
        self.skipped = skipped
        self.pending = None

    def phase(self, now):
        if not np.isfinite(now) or now < self.installed_at:
            raise ValueError("RTC clock moved backwards")
        return self.skipped + (now - self.installed_at) * self.node_hz

    def sample_phase(self, phase):
        if phase >= 50:
            raise TimeoutError("RTC accepted chunk exhausted before a replacement arrived")
        phase = min(max(0.0, phase), 49.0)
        index = int(math.floor(phase))
        return self.actions[index] + (self.actions[min(49, index + 1)] - self.actions[index]) * (phase - index)

    def sample(self, now):
        return self.sample_phase(self.phase(now))

    def status(self, now):
        phase = self.phase(now)
        self.sample_phase(phase)  # Also enforce exhaustion on IPC calls.
        return {
            "generation": self.generation,
            "phase": phase,
            "consumed_since_install": phase - self.skipped,
            "skipped": self.skipped,
            "source_sha256": self.digest,
        }

    def snapshot(self, now, *, delay_steps, schedule, weight):
        if self.pending is not None:
            raise ValueError("Only one RTC prediction may be outstanding")
        phase = self.phase(now)
        self.sample_phase(phase)
        length = math.ceil(50 - phase)
        previous = np.stack([self.sample_phase(min(49.0, phase + i)) for i in range(length)])
        context = rtc.validate_context(
            {
                "protocol_version": rtc.PROTOCOL_VERSION,
                "token": uuid.uuid4().hex,
                "source_number": self.generation,
                "source_sha256": self.digest,
                "previous_actions": previous,
                "inference_delay_steps": delay_steps,
                "prefix_attention_horizon": min(25, length),
                "prefix_attention_schedule": schedule,
                "max_guidance_weight": weight,
            }
        )
        self.pending = (now, context)
        return context.to_dict()

    def install(self, actions, receipt, now):
        if self.pending is None:
            raise ValueError("No outstanding RTC snapshot")
        sent, context = self.pending
        rtc.validate_receipt(receipt, context)
        if context.source_number != self.generation or context.source_sha256 != self.digest:
            raise ValueError("RTC result belongs to a stale source chunk")
        self.sample(now)  # Never revive an exhausted old plan.
        skipped = math.ceil(max(0.0, now - sent) * self.node_hz - 1e-9)
        if skipped > context.prefix_attention_horizon or skipped >= 50:
            raise TimeoutError("RTC latency exceeded the guided prefix horizon")
        self._install(actions, now, skipped)
        return {
            "skipped": skipped,
            "source_generation": context.source_number,
            "generation": self.generation,
            "actual_delay_s": now - sent,
        }
