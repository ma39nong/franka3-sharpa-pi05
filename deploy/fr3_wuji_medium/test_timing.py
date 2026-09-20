"""Allow bounded acknowledged jitter, preserving deadlines and overload stops."""

from dataclasses import replace
import queue

import numpy as np
import pytest

from .continuous import CONTROL_OVERRUN_GRACE_SECONDS
from .continuous import control_loop
from .test_medium import Clock
from .test_medium import Devices
from .test_medium import make_player


def run(clock, devices, events):
    control_loop(
        devices,
        make_player(final=True),
        queue.Queue(),
        events.append,
        lambda *args: None,
        lambda: False,
        1,
        clock=clock,
        sleep=clock.sleep,
    )


@pytest.mark.parametrize("elapsed", [0.012526, 0.0149, 0.015])
def test_acknowledged_isolated_overrun_continues_without_burst(elapsed):
    clock, events = Clock(), []
    devices = Devices(clock)
    original = devices.submit

    def submit(frame, now):
        result = original(frame, now)
        if frame.sequence == 5:
            clock.now += elapsed
        return result

    devices.submit = submit
    run(clock, devices, events)
    warnings = [e for e in events if e["event"] == "control_tick_overrun"]
    assert CONTROL_OVERRUN_GRACE_SECONDS == 0.005
    assert len(warnings) == 1
    assert warnings[0]["overrun_ms"] == pytest.approx((elapsed - 0.01) * 1000)
    assert devices.finishes == 1
    assert min(np.diff([f.created for f in devices.frames])) >= 0.01 - 1e-9
    assert max(f.valid_until - f.created for f in devices.frames) <= 0.02 + 1e-9
    assert [f.sequence for f in devices.frames] == list(range(len(devices.frames)))


@pytest.mark.parametrize("fault", ["over_grace", "expired", "stale", "missing", "unconfirmed"])
def test_relaxed_grace_does_not_hide_faults(fault):
    clock, events = Clock(), []
    devices = Devices(clock)
    original = devices.submit

    def submit(frame, now):
        result = original(frame, now)
        clock.now += {"over_grace": 0.0151, "expired": 0.0201}.get(fault, 0.012526)
        if fault == "stale":
            return replace(result, source_times=(now - 0.2,) * 4)
        if fault == "missing":
            return None
        if fault == "unconfirmed":
            raise TimeoutError("unconfirmed submission")
        return result

    devices.submit = submit
    pattern = {
        "over_grace": "exceeded 100 Hz",
        "expired": "frame deadline",
        "stale": "stale",
        "missing": "lacks submission feedback",
        "unconfirmed": "unconfirmed",
    }[fault]
    with pytest.raises((RuntimeError, ValueError, TimeoutError), match=pattern):
        run(clock, devices, events)
    assert devices.finishes == 0
    assert not any(e["event"] == "control_tick_overrun" for e in events)


@pytest.mark.parametrize("slow_frames,count", [(set(range(3)), 3), (set(range(0, 11, 2)), 11)])
def test_sustained_or_frequent_overruns_still_stop(slow_frames, count):
    clock, events = Clock(), []
    devices = Devices(clock)
    original = devices.submit

    def submit(frame, now):
        result = original(frame, now)
        if frame.sequence in slow_frames:
            clock.now += 0.012526
        return result

    devices.submit = submit
    with pytest.raises(RuntimeError, match="Repeated control tick overruns"):
        run(clock, devices, events)
    assert len(devices.frames) == count
    assert devices.finishes == 0


def test_slow_mode_grace_is_unchanged():
    from experiments.weight_motion_eval.oneshot.continuous import CONTROL_OVERRUN_GRACE_SECONDS as slow

    assert slow == 0.002
