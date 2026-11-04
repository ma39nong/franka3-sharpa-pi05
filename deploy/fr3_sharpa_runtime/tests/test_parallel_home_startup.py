"""Parallel camera/Home admission, cancellation and post-Home observation; no hardware."""

import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime import hardware_session
from deploy.fr3_sharpa_runtime import run
from deploy.fr3_sharpa_runtime.camera_monitor import CameraMonitor
from deploy.fr3_sharpa_runtime.startup_gate import CAMERA_GATE_ENV
from deploy.fr3_sharpa_runtime.startup_gate import wait_for_cameras


def test_camera_gate_requires_explicit_pass_and_consumes_descriptor(monkeypatch):
    read, write = os.pipe()
    monkeypatch.setenv(CAMERA_GATE_ENV, str(read))
    os.write(write, b"PASS\n")
    os.close(write)
    wait_for_cameras()
    assert CAMERA_GATE_ENV not in os.environ
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(read)


@pytest.mark.parametrize("reply", [b"", b"FAIL\n"])
def test_camera_gate_eof_or_invalid_reply_never_grants_task(monkeypatch, reply):
    read, write = os.pipe()
    monkeypatch.setenv(CAMERA_GATE_ENV, str(read))
    os.write(write, reply)
    os.close(write)
    with pytest.raises(RuntimeError):
        wait_for_cameras()


def test_monitor_pending_is_not_a_health_pass_and_timeout_is_failure(monkeypatch):
    monitor = CameraMonitor("unused")
    monitor.process = SimpleNamespace(poll=lambda: None)
    monitor.startup_deadline = time.monotonic() + 10
    monkeypatch.setattr(monitor, "_reports", list)
    assert monitor.startup_ready() is False
    assert monitor.last_seen is None
    monitor.startup_deadline = time.monotonic() - 1
    with pytest.raises(TimeoutError):
        monitor.startup_ready()


@pytest.mark.parametrize("camera_first", [True, False])
def test_real_child_can_home_while_cameras_pending_but_task_waits_for_both(tmp_path, camera_first):
    events_file = tmp_path / "events.jsonl"
    # The child simulates only lifecycle events and sleeps. No ROS/SDK imports.
    script = """
import json, sys, time
from deploy.fr3_sharpa_runtime.startup_gate import wait_for_cameras
path = sys.argv[1]
def event(name):
    with open(path, 'a') as f: f.write(json.dumps([name, time.monotonic()]) + '\\n')
event('home_start')
time.sleep(float(sys.argv[2]))
event('home_done')
wait_for_cameras()
event('fresh_observation')
event('task')
"""
    started = time.monotonic()
    events = []

    class Monitor:
        def startup_ready(self):
            # Ensure the child has actually started Home before releasing cameras.
            if (
                events_file.exists()
                and "home_start" in events_file.read_text()
                and time.monotonic() - started >= (0.05 if camera_first else 0.3)
            ):
                events.append(("camera_pass", time.monotonic()))
                return True
            return False

        def healthy(self):
            return True

        def close(self):
            events.append(("monitor_closed", time.monotonic()))

    hardware = SimpleNamespace(cleanup=lambda: events.append(("cleanup", time.monotonic())))
    status = hardware_session.run_session(
        [sys.executable, "-c", script, str(events_file), ".4" if camera_first else ".01"],
        hardware,
        camera_monitor=Monitor(),
        parallel_home=True,
    )
    assert status == 0
    times = dict(json.loads(line) for line in events_file.read_text().splitlines()) | dict(events)
    assert times["home_start"] < times["camera_pass"]
    assert times["fresh_observation"] >= max(times["home_done"], times["camera_pass"])
    assert times["task"] >= times["fresh_observation"]
    assert times["monitor_closed"] <= times["cleanup"]
    if camera_first:
        assert times["camera_pass"] < times["home_done"]
    else:
        assert times["home_done"] < times["camera_pass"]


def test_camera_failure_interrupts_pending_home_once_and_waits_for_disable(monkeypatch):
    events = []
    interrupted = False

    def kill(pid, number):
        nonlocal interrupted
        interrupted = True
        events.append("interrupt_home")

    def wait(**kwargs):
        assert interrupted
        events.append("disable_confirmed")
        return 130

    def pending_failure():
        raise RuntimeError("cam1 failed")

    monkeypatch.setattr(
        hardware_session.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=12, poll=lambda: None, wait=wait)
    )
    monkeypatch.setattr(hardware_session.os, "killpg", kill)
    monitor = SimpleNamespace(startup_ready=pending_failure, close=lambda: events.append("camera_cancelled"))
    status = hardware_session.run_session(
        ["unused"],
        SimpleNamespace(cleanup=lambda: events.append("cleanup")),
        camera_monitor=monitor,
        parallel_home=True,
    )
    assert status == 1
    assert events == ["interrupt_home", "disable_confirmed", "camera_cancelled", "cleanup"]


def test_home_failure_cancels_pending_camera_validation(monkeypatch):
    events = []
    monkeypatch.setattr(
        hardware_session.subprocess,
        "Popen",
        lambda *a, **kw: SimpleNamespace(pid=12, poll=lambda: 1, wait=lambda **kw: 1),
    )
    monitor = SimpleNamespace(startup_ready=lambda: False, close=lambda: events.append("camera_cancelled"))
    status = hardware_session.run_session(
        ["unused"],
        SimpleNamespace(cleanup=lambda: events.append("cleanup")),
        camera_monitor=monitor,
        parallel_home=True,
    )
    assert status == 1
    assert events == ["camera_cancelled", "cleanup"]


@pytest.mark.parametrize("termination", [signal.SIGINT, signal.SIGTERM])
def test_gui_stop_cancels_home_and_camera_wait_before_cleanup(monkeypatch, termination):
    handlers, events = {}, []

    def install(number, handler):
        old = handlers.get(number)
        handlers[number] = handler
        return old

    def wait(**kwargs):
        handlers[termination](termination, None)
        events.append("disable_confirmed")
        return 128 + termination

    monkeypatch.setattr(hardware_session.signal, "signal", install)
    monkeypatch.setattr(hardware_session.os, "killpg", lambda *a: events.append("stop_home"))
    monkeypatch.setattr(
        hardware_session.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=12, poll=lambda: None, wait=wait)
    )
    monitor = SimpleNamespace(startup_ready=lambda: False, close=lambda: events.append("camera_cancelled"))
    status = hardware_session.run_session(
        ["unused"],
        SimpleNamespace(cleanup=lambda: events.append("cleanup")),
        camera_monitor=monitor,
        parallel_home=True,
    )
    assert status == 128 + termination
    assert events == ["stop_home", "disable_confirmed", "camera_cancelled", "cleanup"]


@pytest.mark.parametrize("broker", ["serial", "rtc"])
def test_home_uses_joint_feedback_and_no_images_before_prepare(monkeypatch, tmp_path, broker):
    from deploy.fr3_sharpa_runtime import home_pose
    from deploy.fr3_sharpa_runtime import motion_worker
    from deploy.fr3_sharpa_runtime import rtc_runtime

    events = []
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "setup.bash").touch()
    state = np.zeros(58, dtype=np.float32)
    completed = {"status": "PASS", "completed_ros_ns": 10, "completed_monotonic_ns": 10}

    class Worker:
        def __init__(self, *a, **kw):
            pass

        def request(self, payload):
            operation = payload["operation"]
            events.append(operation)
            if operation == "ready":
                return {"status": "PASS", "operation": "ready"}
            if operation == "feedback":
                return {"status": "PASS", "state": state.tolist()}
            if operation == "prepare":
                return dict(completed)
            return {"status": "PASS", "published": True, "completed_ros_ns": 10, "completed_monotonic_ns": 10}

        def close(self):
            events.append("worker_closed")

    class Runtime:
        def __init__(self, *a, **kw):
            pass

        def inspect_metadata(self, **kw):
            return {}

        def infer_chunk(self, observation):
            events.append("infer")
            from deploy.fr3_sharpa_runtime.action_adapter import SharpaActionAdapter

            actions = np.zeros((50, 58), dtype=np.float32)
            chunk = SharpaActionAdapter().adapt_chunk(actions, start_timestamp_ns=1_000_000_000, session_id="test")
            return {"checkpoint": "fake", "norm_stats": "fake"}, actions, chunk

    def observe(*a, **kw):
        assert "prepare" in events
        events.append("fresh_observation")
        return {
            "observation/state": state.copy(),
            "_topic_stamp_ns": dict.fromkeys(range(7), 11),
            "_captured_monotonic_ns": time.monotonic_ns(),
        }

    def gate():
        assert "prepare" in events
        events.append("camera_gate_pass")

    monkeypatch.setattr(motion_worker, "MotionWorker", Worker)
    monkeypatch.setattr(
        home_pose,
        "load_home",
        lambda *a, **kw: (
            {"left": np.zeros(7), "right": np.zeros(7), "left_hand": np.zeros(22), "right_hand": np.zeros(22)},
            {},
        ),
    )
    monkeypatch.setattr(run, "_start_observation_worker", lambda *a: None)
    monkeypatch.setattr(run, "live_observation", observe)
    monkeypatch.setattr(run, "SharpaRuntime", Runtime)
    monkeypatch.setattr(run.atexit, "register", lambda *a: None)
    monkeypatch.setattr("deploy.fr3_sharpa_runtime.startup_gate.wait_for_cameras", gate)
    monkeypatch.setattr(rtc_runtime, "run_rtc", lambda *a: events.append("rtc_task") or {})
    args = SimpleNamespace(
        ros_overlay=overlay,
        sharpa_backend="sdk_direct",
        initialization="home",
        broker_mode=broker,
        execution_trace_dir=tmp_path,
        policy_url="unused",
        prompt="tomato",
        num_chunks=1,
        prepare_speed_rad_s=0.03,
        execute_speed_scale=1.0,
        execute_steps=15,
        home_poses=Path("unused"),
        hand_home=Path("unused"),
        home_task="powder_weighing",
    )
    run.execute_policy_prepare(args)
    assert events.index("feedback") < events.index("prepare") < events.index("camera_gate_pass")
    assert events.index("camera_gate_pass") < events.index("fresh_observation") < events.index("infer")
    assert events[-1] == "worker_closed"
    assert events.count("prepare") == 1
