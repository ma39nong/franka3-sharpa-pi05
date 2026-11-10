"""No hardware: time-aligned bridges, bounded old motion and fail-closed limits."""

from types import SimpleNamespace
import sys

import numpy as np
import pytest
from openpi_client import rtc

from deploy.fr3_sharpa_runtime.stage_handoff import StageTimeline, BRIDGE_STEPS, DT, validate_command_step


def actions(offset=0.0):
    return np.repeat((np.arange(50) * 0.0001 + offset)[:, None], 58, axis=1).astype(np.float32)


def timeline():
    return StageTimeline(actions(), 1.0, 1.0, task_epoch=0)


def arm(t, *, now=1.55, start=1.56, captured=1.5, offset=0.002, validate=lambda a: None):
    boundary = dict(t.switch, captured_monotonic_s=captured)
    last = t.sample(start - DT)
    return t.prepare_handoff(actions(offset), boundary, now, start, last, 0.7, validate)


def test_switch_revokes_receipt_and_forbids_old_prefix():
    t = timeline()
    ctx = t.snapshot(1.3, delay_steps=5, schedule="exp", weight=5)
    b = t.begin_switch(1, "operator-1", 1.5)
    assert t.pending is None and b["source_generation"] > ctx["source_number"]
    with pytest.raises(ValueError, match="handover"):
        t.install(actions(), rtc.make_receipt(ctx), 1.6, task_epoch=1)
    with pytest.raises(ValueError, match="prefix"):
        t.snapshot(1.6, delay_steps=5, schedule="exp", weight=5)


def test_bridge_both_boundaries_and_clock_do_not_restart():
    t = timeline()
    source = timeline()
    b = t.begin_switch(1, "next", 1.5)
    validated = []
    result = arm(t, validate=lambda a: validated.append(a.copy()))
    assert result["accepted"] and not result["applied"] and validated
    assert t.status(1.56)["applied_token"] is None
    np.testing.assert_allclose(t.sample(1.56), source.sample(1.56), atol=1e-8)
    commands = []
    for i in range(BRIDGE_STEPS + 1):
        commands.append(t.sample(1.56 + i * DT))
    assert t.status(1.56 + BRIDGE_STEPS * DT)["applied_token"] == b["token"]
    assert t.switch is None and t.bridge is None
    assert t.phase(1.56 + BRIDGE_STEPS * DT) == pytest.approx(result["skipped"] + BRIDGE_STEPS)
    np.testing.assert_allclose(commands[-1], actions(0.002)[result["skipped"] + BRIDGE_STEPS], atol=1e-8)
    for a, c in zip(commands, commands[1:]):
        validate_command_step(a, c, 0.7)
    assert not np.allclose(commands[1], actions(0.002)[0])


def test_quintic_blend_matches_endpoint_velocities():
    t = timeline()
    t.begin_switch(1, "next", 1.5)
    arm(t)
    source = t.source_curve()
    candidate = t.bridge["candidate"]
    h = 1e-5
    start = 1.56
    end = start + BRIDGE_STEPS * DT
    v0 = (t.sample(start + h) - t.sample(start)) / h
    vsource = (source.sample(start + h) - source.sample(start)) / h
    vend = (candidate.sample(end) - t.sample(end - h)) / h
    vtarget = (candidate.sample(end) - candidate.sample(end - h)) / h
    np.testing.assert_allclose(v0, vsource, atol=3e-5)
    np.testing.assert_allclose(vend, vtarget, atol=3e-5)


def test_retarget_before_and_during_bridge_never_extends_deadline_or_jumps():
    t = timeline()
    first = t.begin_switch(1, "one", 1.5)
    second = t.begin_switch(2, "two", 1.53)
    assert second["deadline"] == first["deadline"] and second["token"] != first["token"]
    arm(t)
    before = t.sample(1.62).copy()
    third = t.begin_switch(3, "three", 1.62)
    np.testing.assert_array_equal(t.sample(1.62), before)
    assert third["deadline"] == first["deadline"]
    with pytest.raises(ValueError, match="Stale"):
        t.prepare_handoff(actions(), {**second, "captured_monotonic_s": 1.6}, 1.63, 1.64, before, 0.7, lambda a: None)
    result = arm(t, now=1.65, start=1.66, captured=1.63, offset=0.003)
    end = result["bridge_end"]
    t.sample(end)
    assert t.status(end)["applied_token"] == third["token"]


def test_old_motion_has_a_hard_deadline_and_exhaustion_margin():
    t = timeline()
    b = t.begin_switch(1, "one", 1.5)
    assert b["deadline"] - 1.5 <= 0.75
    with pytest.raises(TimeoutError):
        t.sample(b["deadline"])
    with pytest.raises(TimeoutError):
        t.begin_switch(2, "two", b["deadline"] - 0.1)
    t = timeline()
    with pytest.raises(TimeoutError, match="Insufficient"):
        t.begin_switch(1, "late", 2.5)


@pytest.mark.parametrize(
    "failure", ["epoch", "token", "source", "stale", "future", "nan", "arm_speed", "hand_slew", "range", "late"]
)
def test_invalid_handover_never_arms(failure):
    t = timeline()
    b = t.begin_switch(1, "one", 1.5)
    ticket = {**b, "captured_monotonic_s": 1.5}
    a = actions(0.002)
    now = 1.55
    validate = lambda a: None
    if failure == "epoch":
        ticket["task_epoch"] = 9
    if failure == "token":
        ticket["token"] = "old"
    if failure == "source":
        ticket["source_generation"] = 999
    if failure == "stale":
        ticket["captured_monotonic_s"] = 1.0
    if failure == "future":
        ticket["captured_monotonic_s"] = 1.7
    if failure == "nan":
        a[5, 0] = np.nan
    if failure == "arm_speed":
        a[25, 0] = 2.0
    if failure == "hand_slew":
        a[25, 7] = 2.0
    if failure == "range":

        def validate(a):
            raise ValueError("native joint limits reject")

    if failure == "late":
        now = b["deadline"] - 0.1
        ticket["captured_monotonic_s"] = now - 0.05
    with pytest.raises((ValueError, TimeoutError)):
        t.prepare_handoff(a, ticket, now, now + 0.01, t.sample(now - DT), 0.7, validate)
    assert t.bridge is None and t.switch is not None


def test_unknown_speed_and_non_native_timeline_are_blocked():
    with pytest.raises(ValueError):
        validate_command_step(np.zeros(58), np.zeros(58), float("nan"))
    t = StageTimeline(actions(), 1.0, 0.5, task_epoch=0)
    with pytest.raises(ValueError, match="30Hz"):
        t.begin_switch(1, "one", 1.1)


def test_invalidate_clears_bridge_and_pending_context():
    t = timeline()
    t.begin_switch(1, "one", 1.5)
    arm(t)
    t.invalidate()
    assert t.switch is None and t.bridge is None and t.pending is None
    with pytest.raises(ValueError):
        t.sample(1.6)


@pytest.mark.parametrize(
    "value,kind,done", [(0.7, 3, True), (0.0, 3, True), (-1.0, 3, True), (0.7, 1, True), (0.7, 3, False)]
)
def test_gateway_speed_is_read_not_invented(monkeypatch, value, kind, done):
    from deploy.fr3_sharpa_runtime.ros_transport import read_gateway_joint_speed

    class Request:
        pass

    monkeypatch.setitem(sys.modules, "rclpy", SimpleNamespace(spin_until_future_complete=lambda *a, **kw: None))
    monkeypatch.setitem(
        sys.modules, "rcl_interfaces.srv", SimpleNamespace(GetParameters=SimpleNamespace(Request=Request))
    )
    future = SimpleNamespace(
        done=lambda: done,
        exception=lambda: None,
        result=lambda: SimpleNamespace(values=[SimpleNamespace(type=kind, double_value=value)]),
    )

    class Node:
        destroyed = False

        def create_client(self, *a):
            return SimpleNamespace(wait_for_service=lambda **kw: True, call_async=lambda r: future)

        def destroy_client(self, c):
            self.destroyed = True

    node = Node()
    if value > 0 and kind == 3 and done:
        assert read_gateway_joint_speed(node) == value
    else:
        with pytest.raises((RuntimeError, ValueError)):
            read_gateway_joint_speed(node)
    assert node.destroyed


def test_sdk_backend_online_rpcs_keep_existing_worker_and_connection():
    from deploy.fr3_sharpa_runtime.task_rtc import SDKTaskBackend

    calls = []

    class Worker:
        closed = False

        def request(self, payload, **options):
            calls.append((payload, options))
            if payload["operation"] == "stream_switch":
                return {"status": "PASS", "boundary": {"token": "bound", "task_epoch": 1, "request_id": "operator"}}
            return {"status": "PASS", "accepted": True, "applied": False, "token": "bound"}

        def close(self):
            self.closed = True

    worker = Worker()
    backend = SDKTaskBackend(worker, SimpleNamespace(prediction_clip_rad=0.05))
    b = backend.begin_switch(1, "operator")
    backend.handoff(actions(), b, 1.5)
    assert [v[0]["operation"] for v in calls] == ["stream_switch", "stream_handoff"]
    assert calls[1][0]["ticket"]["captured_monotonic_s"] == 1.5
    assert calls[1][0]["ticket"]["token"] == "bound"
    assert not worker.closed and not backend.stopped
