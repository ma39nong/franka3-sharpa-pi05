import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from deploy.fr3_sharpa_runtime.motion_worker import MotionWorker
from deploy.fr3_sharpa_runtime.shutdown import DisableSettings
from deploy.fr3_sharpa_runtime.shutdown import disable_and_confirm
from deploy.fr3_sharpa_runtime.shutdown import protect_disable_confirmation
from deploy.fr3_sharpa_runtime.shutdown import read_enable_state
from deploy.fr3_sharpa_runtime.shutdown import save_shutdown_report


class Hand:
    def __init__(self, states=(False,), *, stop_error=False):
        self.states = iter(states)
        self.last = True
        self.requests = 0
        self.stop_error = stop_error
        self.disconnected = threading.Event()

    def stop(self):
        self.requests += 1
        if self.stop_error:
            raise RuntimeError("disable transport error")

    def read_enable_state(self):
        self.last = next(self.states, self.last)
        if isinstance(self.last, Exception):
            raise self.last
        return self.last

    def disconnect(self):
        self.disconnected.set()


def settings(**overrides):
    return DisableSettings(**{"timeout_s": 1, "max_attempts": 3, "retry_interval_s": 0.001, **overrides})


def test_success_requires_readback_not_just_disable_call():
    left, right = Hand((True, False)), Hand()
    messages = []
    report = disable_and_confirm({"left": left, "right": right}, settings(), emit=messages.append)
    assert report["status"] == "PASS"
    assert report["hands"]["left"]["attempts"] == 2
    assert report["hands"]["right"]["attempts"] == 1
    assert all(hand.disconnected.is_set() for hand in (left, right))
    assert all(state["last_enabled"] is False for state in report["hands"].values())
    assert any("确认失能" in message for message in messages)


def test_exhausted_left_does_not_prevent_right_disable_and_warns():
    left, right = Hand((True,)), Hand()
    messages = []
    report = disable_and_confirm({"left": left, "right": right}, settings(), emit=messages.append)
    assert report["status"] == "FAIL"
    assert left.requests == 3
    assert right.requests == 1
    assert report["hands"]["right"]["confirmed_disabled"]
    assert report["hands"]["left"]["last_enabled"] is True
    assert any("ERROR left" in message and "未确认失能" in message for message in messages)


def test_sdk_error_or_invalid_getter_value_is_not_confirmation():
    report = disable_and_confirm(
        {"left": Hand((RuntimeError("read failed"),), stop_error=True), "right": Hand(("false",))},
        settings(),
        emit=lambda message: None,
    )
    assert report["status"] == "FAIL"
    assert not any(state["confirmed_disabled"] for state in report["hands"].values())
    assert "read failed" in report["hands"]["left"]["last_error"]
    assert "must be bool" in report["hands"]["right"]["last_error"]


def test_verified_false_counts_even_if_redundant_disable_write_errors():
    report = disable_and_confirm(
        {"left": Hand(stop_error=True), "right": Hand()}, settings(), emit=lambda message: None
    )
    assert report["status"] == "PASS"
    assert report["hands"]["left"]["confirmed_disabled"]


def test_disconnect_failure_does_not_hide_verified_disable():
    class DisconnectFailure(Hand):
        def disconnect(self):
            raise RuntimeError("disconnect failed")

    messages = []
    report = disable_and_confirm({"left": DisconnectFailure(), "right": Hand()}, settings(), emit=messages.append)
    assert report["status"] == "FAIL"
    assert all(state["confirmed_disabled"] for state in report["hands"].values())
    assert "disconnect failed" in report["hands"]["left"]["disconnect_error"]
    assert any("失能已读回确认" in message for message in messages)


def test_blocked_sdk_call_has_shared_deadline_and_does_not_block_other_hand():
    release = threading.Event()

    class BlockedHand(Hand):
        def stop(self):
            self.requests += 1
            release.wait(2)

    left, right = BlockedHand(), Hand()
    started = time.monotonic()
    try:
        report = disable_and_confirm(
            {"left": left, "right": right}, settings(timeout_s=0.05), emit=lambda message: None
        )
        assert time.monotonic() - started < 1
        assert report["status"] == "FAIL"
        assert report["hands"]["left"]["status"] == "UNCONFIRMED"
        assert report["hands"]["right"]["confirmed_disabled"]
    finally:
        release.set()
        assert left.disconnected.wait(1)


def test_no_handle_does_not_connect_or_claim_device_disabled():
    report = disable_and_confirm({"left": None, "right": None}, settings(), emit=lambda message: None)
    assert report["status"] == "NOT_CONNECTED"
    assert not any(state["confirmed_disabled"] for state in report["hands"].values())


def test_only_one_handle_cannot_claim_both_hands_disabled():
    report = disable_and_confirm({"left": None, "right": Hand()}, settings(), emit=lambda message: None)
    assert report["status"] == "FAIL"
    assert report["hands"]["right"]["confirmed_disabled"]
    assert not report["hands"]["left"]["confirmed_disabled"]


def test_installed_hal_read_adapter_does_not_call_connect():
    class ExistingHandle:
        def connect(self):
            pytest.fail("Never reconnect during shutdown")

        def _read_setting(self, key):
            assert key == "get_enable_state"
            return False

    assert read_enable_state(ExistingHandle()) is False


def test_repeated_interrupt_is_deferred_and_original_handler_restored():
    original = signal.getsignal(signal.SIGINT)
    messages = []
    with protect_disable_confirmation(messages.append):
        handler = signal.getsignal(signal.SIGINT)
        handler(signal.SIGINT, None)
        handler(signal.SIGINT, None)
    assert signal.getsignal(signal.SIGINT) == original
    assert len(messages) == 2


@pytest.mark.parametrize("error", [BrokenPipeError(), ValueError("closed stream"), RuntimeError("logger failed")])
def test_logging_failure_cannot_prevent_either_hand_disable(error):
    def broken_output(message):
        raise error

    left, right = Hand(), Hand()
    report = disable_and_confirm({"left": left, "right": right}, settings(), emit=broken_output)
    assert report["status"] == "PASS"
    assert left.requests == right.requests == 1
    assert left.disconnected.is_set()
    assert right.disconnected.is_set()
    with protect_disable_confirmation(broken_output):
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)


def test_actual_closed_stderr_pipe_still_disables_and_persists_result(tmp_path):
    script = """
import os, sys
from deploy.fr3_sharpa_runtime.shutdown import disable_and_confirm, save_shutdown_report
class Hand:
    def __init__(self): self.stopped = False
    def stop(self): self.stopped = True
    def read_enable_state(self): return not self.stopped
    def disconnect(self): pass
report = disable_and_confirm({"left": Hand(), "right": Hand()})
save_shutdown_report(report)
# Avoid an unrelated interpreter-final-flush error after the safety work.
sys.stderr = open(os.devnull, "w")
"""
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[3],
            env={**os.environ, "SHARPA_SHUTDOWN_REPORT_DIR": str(tmp_path)},
            stderr=write_fd,
            timeout=5,
            check=False,
        )
    finally:
        os.close(write_fd)
    assert result.returncode == 0
    report = json.loads(next(tmp_path.glob("shutdown-*.json")).read_text())
    assert report["status"] == "PASS"
    assert all(state["confirmed_disabled"] for state in report["hands"].values())


def test_report_persistence_does_not_promote_failure_to_success(monkeypatch, tmp_path):
    monkeypatch.setenv("SHARPA_SHUTDOWN_REPORT_DIR", str(tmp_path))
    report = disable_and_confirm({"left": Hand((True,)), "right": Hand()}, settings(), emit=lambda message: None)
    path = save_shutdown_report(report)
    stored = json.loads(path.read_text())
    assert stored["status"] == "FAIL"
    assert not stored["hands"]["left"]["confirmed_disabled"]
    assert stored["hands"]["right"]["confirmed_disabled"]


@pytest.mark.parametrize(
    "overrides", [{"timeout_s": 0}, {"timeout_s": float("inf")}, {"max_attempts": 0}, {"retry_interval_s": -1}]
)
def test_invalid_bounds_rejected(overrides):
    with pytest.raises(ValueError, match="positive"):
        settings(**overrides)


def test_direct_worker_waits_for_final_sdk_ack_and_restores_signals(monkeypatch, tmp_path):
    monkeypatch.setenv("SHARPA_DISABLE_TIMEOUT_S", "1")
    original_sigterm = signal.getsignal(signal.SIGTERM)
    command = """
import json, os, signal, time
stream = os.fdopen(int(os.environ["SHARPA_PROTOCOL_FD"]), "w", buffering=1)
def stop(number, frame):
    raise KeyboardInterrupt()
signal.signal(signal.SIGINT, stop)
try:
    print(json.dumps({"status": "PASS", "operation": "ready"}), file=stream, flush=True)
    while True: time.sleep(1)
except KeyboardInterrupt:
    time.sleep(0.15)
    print(json.dumps({"status": "PASS", "operation": "shutdown"}), file=stream, flush=True)
"""
    worker = MotionWorker([sys.executable, "-u", "-c", command], cwd=tmp_path, protocol_fd=True)
    try:
        assert worker.request({"operation": "ready"})["status"] == "PASS"
        started = time.monotonic()
        worker.close()
        assert time.monotonic() - started >= 0.1
        assert worker.process.poll() == 0
        assert signal.getsignal(signal.SIGTERM) == original_sigterm
        worker.close()  # atexit callback must not repeat cleanup.
    finally:
        if worker.process.poll() is None:
            worker.close()


def test_closed_wrapper_still_waits_for_sdk_child_ack(monkeypatch, tmp_path):
    monkeypatch.setenv("SHARPA_DISABLE_TIMEOUT_S", "1")
    command = """
import json, os, subprocess, sys
fd = int(os.environ["SHARPA_PROTOCOL_FD"])
child = 'import os,time,json; time.sleep(0.2); os.write(' + str(fd) + ', (json.dumps({"status":"PASS","operation":"shutdown"})+chr(10)).encode())'
subprocess.Popen([sys.executable, "-c", child], pass_fds=(fd,))
os.write(fd, (json.dumps({"status":"PASS","operation":"ready"})+"\\n").encode())
"""
    worker = MotionWorker([sys.executable, "-c", command], cwd=tmp_path, protocol_fd=True)
    assert worker.request({"operation": "ready"})["status"] == "PASS"
    worker.process.wait(timeout=2)
    started = time.monotonic()
    worker.close()
    assert time.monotonic() - started >= 0.1


def test_missing_sdk_ack_is_reported_not_silently_successful(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SHARPA_DISABLE_TIMEOUT_S", "1")
    command = "import os,json; os.write(int(os.environ['SHARPA_PROTOCOL_FD']), (json.dumps({'status':'PASS','operation':'ready'})+chr(10)).encode())"
    worker = MotionWorker([sys.executable, "-c", command], cwd=tmp_path, protocol_fd=True)
    assert worker.request({"operation": "ready"})["status"] == "PASS"
    worker.process.wait(timeout=2)
    worker.close()
    assert "未收到 SDK 失能读回确认" in capsys.readouterr().err


def test_parent_worker_budget_covers_full_disable_window(monkeypatch, tmp_path):
    import io
    from types import SimpleNamespace

    from deploy.fr3_sharpa_runtime import motion_worker

    monkeypatch.setenv("SHARPA_DISABLE_TIMEOUT_S", "120")
    waits = []
    process = SimpleNamespace(
        pid=12345, stdin=io.StringIO(), poll=lambda: None, wait=lambda **kwargs: waits.append(kwargs["timeout"])
    )
    monkeypatch.setattr(motion_worker.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(motion_worker.os, "killpg", lambda *args: None)
    monkeypatch.setattr(MotionWorker, "_await_shutdown", lambda self, deadline: True)
    worker = MotionWorker(["fake"], cwd=tmp_path, protocol_fd=True)
    worker.close()
    assert waits[0] > 120


def test_outer_session_waits_for_configured_disable_grace_before_cleanup(monkeypatch):
    import subprocess
    from types import SimpleNamespace

    from deploy.fr3_sharpa_runtime import hardware_session

    handlers, waits, events = {}, [], []

    def install(number, handler):
        old = handlers.get(number)
        handlers[number] = handler
        return old

    def wait(*, timeout):
        waits.append(timeout)
        if len(waits) == 1:
            handlers[signal.SIGINT](signal.SIGINT, None)
            raise subprocess.TimeoutExpired("fake", timeout)
        events.append("disable_finished")
        return 130

    process = SimpleNamespace(pid=12345, poll=lambda: None, wait=wait)
    monkeypatch.setattr(hardware_session.signal, "signal", install)
    monkeypatch.setattr(hardware_session.os, "killpg", lambda *args: None)
    monkeypatch.setattr(hardware_session.subprocess, "Popen", lambda *args, **kwargs: process)
    hardware_session.run_session(
        ["fake"],
        SimpleNamespace(cleanup=lambda: events.append("container_cleanup")),
        shutdown_grace_s=DisableSettings().session_grace_s,
    )
    assert waits == [1, 165]
    assert events == ["disable_finished", "container_cleanup"]
