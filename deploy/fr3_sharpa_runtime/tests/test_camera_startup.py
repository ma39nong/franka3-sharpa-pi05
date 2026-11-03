"""No hardware: ownership decisions, ordering, leases and loss shutdown."""

import copy
import json
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import yaml

from deploy.fr3_sharpa_runtime import hardware_session
from deploy.fr3_sharpa_runtime.camera_lease import LEASE_ENV
from deploy.fr3_sharpa_runtime.camera_lease import CameraLease
from deploy.fr3_sharpa_runtime.camera_monitor import CameraMonitor
from deploy.fr3_sharpa_runtime.camera_probe import check_serials
from deploy.fr3_sharpa_runtime.camera_startup import ROLES
from deploy.fr3_sharpa_runtime.camera_startup import CameraStartup
from deploy.fr3_sharpa_runtime.camera_startup import camera_contract
from deploy.fr3_sharpa_runtime.camera_startup import classify


@pytest.fixture
def config(tmp_path):
    data = {name: {"serial_number": name + "-serial", "semantic": role} for name, role in ROLES.items()}
    path = tmp_path / "cameras.yaml"
    path.write_text(yaml.safe_dump({"camera_bringup": data}))
    return path


def container(path, service="teleop-cameras", identifier="known-camera"):
    return {
        "Id": identifier,
        "Name": service,
        "State": {"Running": True, "Pid": 123},
        "Config": {
            "Cmd": [
                "ros2",
                "launch",
                "teleop_camera_bringup",
                "triple_camera.launch.py",
                "deployment_config:=/camera.yaml",
            ],
            "Labels": {"com.docker.compose.service": service},
        },
        "Mounts": [{"Type": "bind", "Source": str(path), "Destination": "/camera.yaml"}],
    }


def test_known_services_require_serials_mapping_and_launch_identity(config):
    expected = camera_contract(config)
    assert classify(container(config), expected) == "known"
    assert classify(container(config, "openpi-cameras"), expected) == "known"
    with pytest.raises(RuntimeError, match="未知"):
        classify(container(config, "data-collection"), expected)
    wrong = container(config)
    wrong["Config"]["Cmd"] = ["arbitrary-driver"]
    with pytest.raises(RuntimeError, match="无法核对"):
        classify(wrong, expected)
    other = copy.deepcopy(expected)
    other["cam1"]["serial_number"] = "another-wrist"
    with pytest.raises(RuntimeError, match="未知"):
        classify(container(config), other)


def test_unrelated_group_is_preserved_and_actual_wrong_serial_is_rejected(config):
    expected = camera_contract(config)
    other = copy.deepcopy(expected)
    for entry in other.values():
        entry["serial_number"] += "other"
    assert classify(container(config), other) == "unrelated"
    check_serials(expected, {key: value["serial_number"] for key, value in expected.items()})
    with pytest.raises(ValueError, match="实际序列号"):
        check_serials(expected, dict.fromkeys(expected, "different"))


class FakeStartup(CameraStartup):
    def __init__(self, path):
        self.config, self.expected = path, camera_contract(path)
        self.original = container(path)
        self.deployed = container(path, "openpi-cameras", "new-deployment-camera")
        self.started, self.stopped, self.events = None, [], []
        self.hardware = SimpleNamespace(compose=["compose"], is_openpi=lambda item, service: True)
        self.inspect_count = 0
        self.ready = False

    def command(self, args, **kwargs):
        self.events.append(tuple(args))
        if args[:2] == ["docker", "inspect"]:
            item = copy.deepcopy(self.original if args[-1] == self.original["Id"] else self.deployed)
            if ("stopped", args[-1]) in self.events:
                item["State"] = {"Running": False, "Pid": 0}
            return json.dumps([item])
        if args[:2] == ["docker", "stop"]:
            self.events.append(("stopped", args[-1]))
        if args[-2:] == ["franka-control", "teleop-control"]:
            assert self.ready  # No arm dependency startup before the camera gate passes.
        return ""

    def inspect(self):
        self.inspect_count += 1
        return [self.original] if self.inspect_count == 1 else []

    def release(self):
        assert ("stopped", self.original["Id"]) in self.events
        self.events.append(("released",))

    def containers(self):
        return [self.deployed]

    def probe(self):
        self.events.append(("probe",))
        self.ready = True
        return {"status": "PASS"}


def test_takeover_waits_for_exit_release_and_readiness_before_arm_start(config):
    startup = FakeStartup(config)
    report = startup.up(exclusive=True)
    assert report["status"] == "PASS"
    assert report["stopped"] == ["teleop-cameras"]
    order = startup.events
    assert order.index(("stopped", "known-camera")) < order.index(("released",))
    assert order.index(("released",)) < order.index(("probe",))
    assert order[-1][-2:] == ("franka-control", "teleop-control")


def test_no_explicit_takeover_does_not_stop_existing_services(config):
    startup = FakeStartup(config)
    with pytest.raises(RuntimeError, match="独占接管"):
        startup.up(exclusive=False)
    assert not any(event[:2] == ("docker", "stop") for event in startup.events)


def test_readiness_failure_cleans_only_this_camera_without_starting_arms(config, monkeypatch):
    startup = FakeStartup(config)
    monkeypatch.setattr(startup, "probe", lambda: (_ for _ in ()).throw(RuntimeError("wrong serial")))
    with pytest.raises(RuntimeError, match="wrong serial"):
        startup.up(exclusive=True)
    assert ("stopped", "new-deployment-camera") in startup.events
    assert not any("franka-control" in event for event in startup.events)


def test_unknown_occupancy_prevents_stopping_even_known_containers(config, monkeypatch):
    startup = FakeStartup(config)
    monkeypatch.setattr(
        startup, "containers", lambda: [startup.original, container(config, "unknown-viewer", "unknown")]
    )
    with pytest.raises(RuntimeError, match="未知"):
        CameraStartup.inspect(startup)
    assert startup.events == []


def test_driver_still_running_is_not_treated_as_released(config, monkeypatch):
    startup = FakeStartup(config)
    real_command = startup.command

    def command(args, **kwargs):
        if args[:2] == ["docker", "inspect"]:
            return json.dumps([startup.original])
        return real_command(args, **kwargs)

    monkeypatch.setattr(startup, "command", command)
    clock = iter([0.0, 0.0, 11.0])
    monkeypatch.setattr("deploy.fr3_sharpa_runtime.camera_startup.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("deploy.fr3_sharpa_runtime.camera_startup.time.sleep", lambda _: None)
    with pytest.raises(TimeoutError, match="驱动未退出"):
        startup.stop(startup.original)
    assert not startup.stopped


def test_lease_is_shared_across_launchers_and_borrowed_child_cannot_unlock(tmp_path, monkeypatch):
    monkeypatch.delenv(LEASE_ENV, raising=False)
    path = tmp_path / "camera.lock"
    owner = CameraLease(path).acquire()
    try:
        with pytest.raises(RuntimeError, match="禁止重复"):
            CameraLease(path).acquire()
        monkeypatch.setenv(LEASE_ENV, str(owner.fd))
        child = CameraLease(path).acquire()
        child.close()
        monkeypatch.delenv(LEASE_ENV)
        with pytest.raises(RuntimeError, match="禁止重复"):
            CameraLease(path).acquire()
    finally:
        owner.close()
    with CameraLease(path):
        pass


def test_monitor_checks_nonblocking_heartbeat_and_eof(config):
    read, write = os.pipe()
    with os.fdopen(read, "rb", buffering=0) as stream:
        monitor = CameraMonitor(config)
        monitor.process = SimpleNamespace(stdout=stream, poll=lambda: None)
        os.write(write, (json.dumps({"status": "PASS", "checked_monotonic_ns": time.monotonic_ns()}) + "\n").encode())
        assert monitor.healthy()
        os.write(write, b'{"status":')  # Incomplete reply must not block the command/shutdown loop.
        assert monitor.healthy()
        monitor.last_seen -= 1
        assert not monitor.healthy()
        os.close(write)
        assert not monitor.healthy()


def test_lease_survives_shell_exec_without_child_unlocking_parent(tmp_path, monkeypatch):
    monkeypatch.delenv(LEASE_ENV, raising=False)
    path = tmp_path / "inherited-camera.lock"
    with CameraLease(path) as lease:
        result = subprocess.run(
            [
                "bash",
                "-c",
                'exec "$@"',
                "--",
                sys.executable,
                "-c",
                "import sys; from deploy.fr3_sharpa_runtime.camera_lease import CameraLease; "
                "c=CameraLease(sys.argv[1]).acquire(); assert c.borrowed; c.close(); print('borrowed')",
                str(path),
            ],
            pass_fds=(lease.fd,),
            env={**os.environ, LEASE_ENV: str(lease.fd)},
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "borrowed"
        with pytest.raises(RuntimeError, match="禁止重复"):
            CameraLease(path).acquire()


def test_camera_loss_interrupts_once_waits_for_runtime_then_cleans_without_restart(monkeypatch):
    events = []
    state = {"interrupted": False}

    def kill(pid, number):
        state["interrupted"] = True
        events.append(("interrupt", pid, number))

    def wait(**kwargs):
        if not state["interrupted"]:
            raise subprocess.TimeoutExpired("runtime", kwargs["timeout"])
        events.append(("runtime_shutdown_confirmed",))
        return 130

    monkeypatch.setattr(
        hardware_session.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=123, poll=lambda: None, wait=wait)
    )
    monkeypatch.setattr(hardware_session.os, "killpg", kill)
    monitor = SimpleNamespace(healthy=lambda: False, close=lambda: events.append(("monitor_closed",)))
    hardware = SimpleNamespace(cleanup=lambda: events.append(("hardware_cleanup",)))
    status = hardware_session.run_session(["runtime"], hardware, camera_monitor=monitor)
    assert status == 1
    assert events == [
        ("interrupt", 123, signal.SIGINT),
        ("runtime_shutdown_confirmed",),
        ("monitor_closed",),
        ("hardware_cleanup",),
    ]
