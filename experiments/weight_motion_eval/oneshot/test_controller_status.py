"""Controller failure must stop the session even with fresh stationary feedback."""

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from .controller_status import ControllerStatus, ControllerStatusUnavailable
from .ros_devices import RosArms
from .test_devices import make_session, prepare, frame


def status(fault=False, reason=None, stamp=110_000_000_000, **fields):
    return json.dumps({"version": 1, "stamp_ns": stamp, "faulted": fault,
                       "reason": reason or ("previous_command_expired" if fault else "none"), **fields})


def monitor():
    result = ControllerStatus()
    for side in ("left", "right"):
        result.receive(side, status(), 10, 110)
    return result


def test_native_first_fault_and_expiry_replay(tmp_path):
    folder = Path(__file__).parent
    binary = tmp_path / "controller_fault"
    subprocess.run(["g++", "-std=c++17", "-pthread", "-Wall", "-Wextra", "-Werror",
                    "-I", str(folder), str(folder / "test_controller_fault.cpp"), "-o", str(binary)], check=True)
    value = json.loads(subprocess.check_output([str(binary)], text=True))
    assert value["reason"] == "previous_command_expired"
    assert value["received_sequence"] == 503 and value["expected_sequence"] == 504
    assert value["command_age_ms"] == 20
    assert value["deadline_remaining_ms"] == 0


@pytest.mark.parametrize("side", ["left", "right"])
def test_first_fault_survives_later_healthy_status(side):
    m = monitor()
    m.receive(side, status(True, received_sequence=503), 10, 110)
    first = m.first_fault
    m.receive(side, status(), 10, 110)
    assert m.first_fault == first
    with pytest.raises(RuntimeError, match="previous_command_expired"):
        m.check(10)


@pytest.mark.parametrize("data", ["{}", "[]", "not json", status(stamp=109_000_000_000), status(stamp=111_000_000_000)])
def test_bad_or_stale_status_fails_closed(data):
    m = monitor()
    m.receive("left", data, 10, 110)
    with pytest.raises(RuntimeError, match="invalid_status"):
        m.check(10)


def test_status_missing_or_expired_is_not_healthy():
    with pytest.raises(ControllerStatusUnavailable, match="Missing"):
        ControllerStatus().check(10)
    m = monitor()
    m.check(10.1)
    with pytest.raises(ControllerStatusUnavailable, match="Stale"):
        m.check(10.151)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("path", ["submit", "idle"])
def test_controller_fault_stops_hands_and_arms_without_waiting_for_tracking(side, path):
    session = make_session()
    prepare(session)
    m = monitor()
    old = session.arms.feedback
    def feedback(now):
        m.check(now)
        return old(now)
    session.arms.feedback = feedback
    m.receive(side, status(True), 10, 110)
    if path == "submit":
        with pytest.raises(RuntimeError, match="previous_command_expired"):
            session.submit(frame())
    else:
        session.last_health_check = 0
        session.service()
    assert session.state == "stopped"
    assert "previous_command_expired" in session.fault
    assert session.arms.stops == 1 and not session.arms.sent
    assert all(hand.stops == 1 and not hand.sent for hand in session.hands.values())


def test_stop_confirmation_still_reads_measured_feedback(monkeypatch):
    from . import ros_devices as module
    monkeypatch.setattr(module.time, "monotonic", lambda: 10)
    monkeypatch.setattr(module.time, "time", lambda: 110)
    arms = RosArms.__new__(RosArms)
    arms.spin = lambda *_: None
    arms.offset, arms.output, arms.stopped = 100, object(), False
    arms.controller_status = monitor()
    arms.controller_status.receive("left", status(True), 10, 110)
    arms.samples = {s: (np.zeros(7), np.zeros(7), 10, 10) for s in ("left", "right")}
    with pytest.raises(RuntimeError, match="previous_command_expired"):
        arms.feedback(10)
    arms.stop()
    assert arms.feedback(10)[0].shape == (14,)
    with pytest.raises(RuntimeError, match="cannot restart"):
        arms.begin_submit(frame())


def test_fault_is_checked_before_gateway_ack_can_mask_it(monkeypatch):
    from . import ros_devices as module
    monkeypatch.setattr(module.time, "monotonic", lambda: 10)
    arms = RosArms.__new__(RosArms)
    arms.output, arms.stopped = object(), False
    arms.controller_status = monitor()
    f = frame()
    ack = SimpleNamespace(header="header", faults=[], accepted_sides=["left", "right"])
    arms.statuses = {(f.run_id, f.sequence): ack}
    arms.spin = lambda *_: arms.controller_status.receive("right", status(True), 10, 110)
    with pytest.raises(RuntimeError, match="previous_command_expired"):
        arms.confirm_submit(f, ack)


def test_post_ack_fault_is_preserved_in_next_ipc_reply(monkeypatch):
    from . import bridge
    from .ipc import wire_frame
    session = make_session()
    prepare(session)
    requests = iter([{"id": 1, "operation": "submit", "frame": wire_frame(frame())},
                     {"id": 2, "operation": "feedback"}])
    replies = []
    conn = SimpleNamespace(settimeout=lambda *_: None, sendall=lambda data: replies.append(json.loads(data)))
    monkeypatch.setattr(bridge, "receive", lambda _: next(requests))
    monkeypatch.setattr(bridge.select, "select", lambda *args: ([], [], []))
    def pump(*args):
        raise RuntimeError("Final arm controller fault: previous_command_expired")
    bridge.serve_connection(conn, session, stopping=lambda: len(replies) == 2, pump=pump)
    assert replies[0]["ok"] and not replies[1]["ok"]
    assert "previous_command_expired" in replies[1]["error"]
    assert session.state == "stopped"
    assert all(hand.stops == 1 for hand in session.hands.values())
