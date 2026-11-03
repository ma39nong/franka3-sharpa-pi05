import copy
import signal
import subprocess
from types import SimpleNamespace

import pytest

from deploy.fr3_sharpa_runtime import hardware_session as session


def container(service, identifier):
    commands = {
        "teleop-control": ["config:=/etc/openpi/teleop_control.yaml"],
        "openpi-cameras": ["triple_camera.launch.py"],
        "franka-control": ["robot_control.launch.py"],
    }
    return {
        "Id": identifier,
        "Config": {
            "Cmd": commands[service],
            "Labels": {
                "com.docker.compose.service": service,
                "com.docker.compose.project.config_files": "/base.yaml,/deployment/compose.hardware.yaml",
            },
        },
        "Mounts": [{"Source": "/deployment/gateway.yaml", "Destination": "/etc/openpi/teleop_control.yaml"}],
    }


@pytest.fixture
def hardware(monkeypatch):
    obj = session.HardwareSession.__new__(session.HardwareSession)
    obj.overlay = "/deployment/compose.hardware.yaml"
    obj.gateway = "/deployment/gateway.yaml"
    obj.owned = {
        name: container(name, name + "-original") for name in ("openpi-cameras", "teleop-control", "franka-control")
    }
    current = copy.deepcopy(obj.owned)
    monkeypatch.setattr(obj, "inventory", lambda: current)
    calls = []
    monkeypatch.setattr(session.subprocess, "run", lambda command, **kwargs: calls.append(command))
    return obj, current, calls


def test_cleanup_removes_openpi_configuration_and_camera_owners(hardware):
    obj, _, calls = hardware
    obj.cleanup()
    assert calls == [
        ["docker", "stop", "--time", "30", "openpi-cameras-original"],
        ["docker", "rm", "openpi-cameras-original"],
        ["docker", "stop", "--time", "30", "teleop-control-original"],
        ["docker", "rm", "teleop-control-original"],
        ["docker", "stop", "--time", "30", "franka-control-original"],
    ]


def test_cleanup_never_stops_replacements_or_their_arm_controller(hardware):
    obj, current, calls = hardware
    for item in current.values():
        item["Id"] += "-replacement"
    obj.cleanup()
    assert calls == []


def test_cleanup_does_not_touch_a_teleop_configuration(hardware):
    obj, current, calls = hardware
    current["teleop-control"]["Config"]["Cmd"] = ["config:=/workspace/config/modes/teleop_control.yaml"]
    del current["openpi-cameras"]
    obj.cleanup()
    assert calls == []


def test_cleanup_rejects_another_deployments_same_named_services(hardware):
    obj, current, calls = hardware
    for item in current.values():
        item["Config"]["Labels"]["com.docker.compose.project.config_files"] = "/other/compose.hardware.yaml"
    obj.cleanup()
    assert calls == []


def test_cleanup_of_already_removed_containers_is_noop(hardware):
    obj, current, calls = hardware
    current.clear()
    obj.cleanup()
    assert calls == []


def test_cleanup_errors_are_visible_and_other_resources_still_released(hardware, monkeypatch):
    obj, _, calls = hardware

    def fail_camera(command, **kwargs):
        calls.append(command)
        if command[-1] == "openpi-cameras-original":
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(session.subprocess, "run", fail_camera)
    with pytest.raises(RuntimeError, match="openpi-cameras"):
        obj.cleanup()
    assert ["docker", "rm", "teleop-control-original"] in calls


@pytest.mark.parametrize("exit_status", [0, 1])
def test_normal_or_failed_runtime_always_cleans_up(monkeypatch, exit_status):
    cleaned = []
    monkeypatch.setattr(
        session.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(wait=lambda **kw: exit_status)
    )
    assert session.run_session(["unused"], SimpleNamespace(cleanup=lambda: cleaned.append(True))) == exit_status
    assert cleaned == [True]


@pytest.mark.parametrize("termination", [signal.SIGINT, signal.SIGTERM])
def test_termination_forwards_to_runtime_tree_then_cleans_up(monkeypatch, termination):
    handlers, events = {}, []

    def install(number, handler):
        old = handlers.get(number)
        handlers[number] = handler
        return old

    def wait(**kwargs):
        handlers[termination](termination, None)
        events.append("child_stopped")
        return 128 + termination

    monkeypatch.setattr(session.signal, "signal", install)
    monkeypatch.setattr(session.os, "killpg", lambda pid, number: events.append((pid, number)))
    monkeypatch.setattr(
        session.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(pid=123, poll=lambda: None, wait=wait)
    )
    status = session.run_session(["unused"], SimpleNamespace(cleanup=lambda: events.append("cleanup")))
    assert status == 128 + termination
    assert events == [(123, termination), "child_stopped", "cleanup"]


def test_handover_during_cleanup_preserves_the_controller(hardware, monkeypatch):
    obj, current, calls = hardware
    after = copy.deepcopy(current)
    after["teleop-control"]["Id"] = "teleop-took-over"
    inventories = iter([current, after])
    monkeypatch.setattr(obj, "inventory", lambda: next(inventories))
    obj.cleanup()
    assert ["docker", "rm", "teleop-control-original"] in calls
    assert not any(command[-1] == "franka-control-original" for command in calls)


def test_runtime_launch_failure_still_releases_hardware(monkeypatch):
    cleaned = []

    def fail(*args, **kwargs):
        raise OSError("failed to start")

    monkeypatch.setattr(session.subprocess, "Popen", fail)
    with pytest.raises(OSError, match="failed to start"):
        session.run_session(["unused"], SimpleNamespace(cleanup=lambda: cleaned.append(True)))
    assert cleaned == [True]


@pytest.mark.parametrize("variable", ("SHARPA_CAMERA_CAM0_PARAMS", "SHARPA_CAMERA_CAM1_PARAMS"))
def test_missing_camera_profile_is_rejected_before_docker_inventory(monkeypatch, tmp_path, variable):
    monkeypatch.setenv("SHARPA_GELLO_REPO", str(tmp_path))
    monkeypatch.setenv("SHARPA_GATEWAY_CONFIG", str(tmp_path / "gateway.yaml"))
    monkeypatch.setenv(variable, str(tmp_path / "missing.yaml"))
    monkeypatch.setattr(session.HardwareSession, "inventory", lambda self: pytest.fail("No Docker calls for invalid profiles"))
    with pytest.raises(ValueError, match="existing absolute file"):
        session.HardwareSession()
