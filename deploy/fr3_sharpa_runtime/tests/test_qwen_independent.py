from dataclasses import asdict
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import urllib.request

import pytest

from deploy.fr3_sharpa_control.agent import AgentSupervisor
from deploy.fr3_sharpa_control.agent_settings import AgentSettings
from deploy.fr3_sharpa_control.qwen_supervisor import QwenSettings
from deploy.fr3_sharpa_control.qwen_supervisor import QwenSupervisor
from deploy.fr3_sharpa_control.ui import server_for


def ready(settings):
    return {"ready": True, "provider": "qwen", "checks": [], "note": "fixture", "settings": asdict(settings)}


def wait_until_stopped(self, settings):
    self.update(status="waiting", message="fixture independent Qwen")
    self.closed.wait(5)
    self.update(status="stopped")


def test_qwen_never_parses_or_loads_vjepa_and_switch_requires_stop(tmp_path, monkeypatch):
    owner = SimpleNamespace(root=tmp_path, stop=lambda: pytest.fail("Qwen stop must not stop execution"))
    agent = AgentSupervisor(owner)
    monkeypatch.setattr(AgentSettings, "parse", lambda *args: pytest.fail("Qwen must not require V-JEPA settings"))
    monkeypatch.setattr(QwenSettings, "check", ready)
    monkeypatch.setattr(QwenSupervisor, "_run", wait_until_stopped)
    result = agent.start({"provider": "qwen", "model": "qwen3.8-max", "quotas": "4"})
    try:
        assert result["provider"] == "qwen"
        assert agent.snapshot()["control_eligible"] is False
        assert agent.qwen.settings.model == "qwen3.8-max"
        assert not agent.settings_path.exists()
        assert not (tmp_path / ".deployment/agents").exists()
        with pytest.raises(ValueError, match="先停止"):
            agent.start({"provider": "mengxiang"})
        with pytest.raises(ValueError, match="自动执行"):
            agent.set_mode("auto")
    finally:
        agent.stop()
        agent.thread.join(2)
    assert not agent.thread.is_alive()
    assert agent.snapshot()["status"] == "stopped"


def test_qwen_settings_and_selection_restore_without_touching_vjepa_settings(tmp_path, monkeypatch):
    root = tmp_path / ".deployment"
    root.mkdir()
    legacy = root / "agent_settings.json"
    legacy.write_text(json.dumps({"provider": "mengxiang", "checkpoint": "keep-me"}))
    before = legacy.read_bytes()
    agent = AgentSupervisor(SimpleNamespace(root=tmp_path))
    monkeypatch.setattr(QwenSettings, "check", ready)
    monkeypatch.setattr(QwenSupervisor, "_run", wait_until_stopped)
    agent.start({"provider": "qwen", "quotas": "3,1"})
    agent.stop()
    agent.thread.join(2)
    restored = AgentSupervisor(SimpleNamespace(root=tmp_path))
    assert restored.qwen_active is True
    assert restored.settings["quotas"] == "3,1"
    assert legacy.read_bytes() == before
    assert restored.presets()["mengxiang"]["checkpoint"].endswith("keep-me")


def test_qwen_collector_owns_ros_without_model_command(tmp_path):
    settings = QwenSettings.parse({}, tmp_path)
    gello = Path(settings.gello_repository)
    settings.gello_repository = str(tmp_path / "gello")
    gello = Path(settings.gello_repository)
    (gello / "docker").mkdir(parents=True)
    (gello / "docker/.env").write_text("TELEOP_ROS_DOMAIN_ID=28\n")
    supervisor = QwenSupervisor(SimpleNamespace(root=tmp_path))
    supervisor.directory = tmp_path / "independent-history"
    command, cwd, env = supervisor.collector_command(settings)
    assert "agent_history" in command[-1]
    assert "--native-agent-roi" in command[-1]
    assert "serve_tomato" not in str(command)
    assert "vjepa" not in str(command)
    assert env["ROS_DOMAIN_ID"] == "28"
    assert cwd == gello / "portable_deps/litchi_hardware"


def test_broken_qwen_preferences_do_not_break_vjepa_startup(tmp_path):
    root = tmp_path / ".deployment"
    root.mkdir()
    for invalid in (None, [], 0):
        (root / "qwen_settings.json").write_text(json.dumps(invalid))
        (root / "agent_choice.json").write_text(json.dumps(invalid))
        agent = AgentSupervisor(SimpleNamespace(root=tmp_path))
        assert agent.qwen_active is False
        assert agent.settings["provider"] == "mengxiang"


def test_http_qwen_controls_are_separate_and_do_not_expose_key_contents(tmp_path, monkeypatch):
    monkeypatch.setattr(QwenSettings, "check", ready)
    monkeypatch.setattr(QwenSupervisor, "_run", wait_until_stopped)
    server = server_for(tmp_path, port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}/api/"
    try:
        with urllib.request.urlopen(base + "info") as response:
            info = json.load(response)
        assert set(info["agent_presets"]) == {"mengxiang", "tairan", "qwen"}
        secret = tmp_path / "key"
        secret.write_text("not-for-frontend")
        config = {"provider": "qwen", "key_path": str(secret), "quotas": "2,2"}

        def post(path, data):
            request = urllib.request.Request(
                base + path,
                data=json.dumps(data).encode(),
                headers={"Content-Type": "application/json", "X-Control-Token": info["token"]},
            )
            with urllib.request.urlopen(request) as response:
                text = response.read().decode()
                assert "not-for-frontend" not in text
                return json.loads(text)

        assert post("agent-check", config)["provider"] == "qwen"
        assert post("start-agent", config)["provider"] == "qwen"
        assert server.controller.agent.qwen_active is True
        with urllib.request.urlopen(base + "status") as response:
            state = json.load(response)
        assert state["agent"]["provider"] == "qwen"
        assert state["agent"]["control_eligible"] is False
        post("stop-agent", {})
        server.controller.agent.thread.join(2)
    finally:
        server.controller.agent.stop()
        server.shutdown()
        worker.join(2)
        server.server_close()


def test_independent_engine_reads_frames_and_updates_own_memory_without_vjepa(tmp_path, monkeypatch):
    import time

    from deploy.fr3_sharpa_control import qwen_supervisor

    owner = SimpleNamespace(root=tmp_path)
    qwen = QwenSupervisor(owner)
    now = time.time()
    task = {
        "run_id": "qwen-only-run",
        "task_profile": "tomato-in-hand119",
        "execution_state": "RUNNING",
        "active_stage": "pickup",
        "requested_stage": "pickup",
        "stage_version": 0,
        "active_applied_unix_s": now - 2,
    }
    monkeypatch.setattr(qwen, "task", lambda: dict(task))
    monkeypatch.setattr(QwenSettings, "check", ready)
    monkeypatch.setattr(qwen, "collector_command", lambda settings: (["fixture-collector"], tmp_path, {}))
    interrupted = []

    class Collector:
        pid = 123456789

        def __init__(self, *args, **kwargs):
            assert args[0] == ["fixture-collector"]

        def poll(self):
            return None

        def wait(self, **kwargs):
            return 0

    def capture(self, directory, closed):
        with self.lock:
            for index in range(4):
                frame = {"id": str(index), "camera": "cam0", "timestamp": now - 1.5 + index * 0.5}
                self.frames[frame["id"]] = (frame, b"\xff\xd8fixture")
        closed.wait(5)

    def recognize(payload, **kwargs):
        assert payload["model"] == "qwen3.8-flash"
        assert payload["enable_thinking"] is False
        return {
            "answer": {
                "events": [],
                "ready": True,
                "ready_evidence": ["2", "3"],
                "left_empty": True,
                "counting_complete": True,
                "uncertainties": [],
            },
            "elapsed_s": 0.01,
        }

    original_update = qwen.update

    def update(**fields):
        original_update(**fields)
        if fields.get("status") == "ready":
            qwen.closed.set()

    monkeypatch.setattr(qwen, "update", update)
    monkeypatch.setattr(qwen_supervisor.subprocess, "Popen", Collector)
    monkeypatch.setattr(qwen_supervisor.os, "killpg", lambda *args: interrupted.append(args))
    monkeypatch.setattr(qwen_supervisor.FrameBuffer, "collect", capture)
    monkeypatch.setattr(qwen_supervisor, "recognize", recognize)
    qwen.start({"provider": "qwen"})
    qwen.thread.join(3)
    assert not qwen.thread.is_alive()
    assert qwen.snapshot()["evaluation"]["proposed_stage"] == "harvest_1"
    assert qwen.snapshot()["control_eligible"] is False
    saved = list((tmp_path / ".deployment/qwen_memories").glob("*.json"))
    assert json.loads(saved[0].read_text())["run_id"] == "qwen-only-run"
    assert len(interrupted) == 1
