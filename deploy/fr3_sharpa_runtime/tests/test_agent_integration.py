"""Agent authority, real IPC, causal histories and HTTP integration without hardware."""

from types import SimpleNamespace
import time
import math

import pytest
from deploy.fr3_sharpa_control.agent import DecisionGate, AgentSupervisor
from deploy.fr3_sharpa_runtime.agent_history import select_history, JOINT_NAMES
from deploy.fr3_sharpa_runtime.ros_observation import TOPICS
from deploy.fr3_sharpa_runtime.task_control import TaskControl, TaskServer, task_rpc


def packet(t=100.0, label=1):
    p = [0.01] * 4
    p[label] = 0.97
    z = [math.log(v) for v in p]
    return dict(
        protocol="tomato-stage-prediction-v1",
        labels=["pickup", "harvest", "return", "adjust"],
        valid=True,
        error=None,
        model_version="b0",
        prediction_mode="joint",
        input_timestamp=t,
        result_timestamp=t + 0.01,
        future_horizon_s=1.5,
        current_probabilities=p,
        current_logits=z,
        future_probabilities=p,
        future_logits=z,
    )


def task(**kw):
    return dict(
        run_id="run",
        stage_version=0,
        active_stage="pickup_1",
        requested_stage="pickup_1",
        execution_state="RUNNING",
        **kw,
    )


def test_inhand_observer_displays_both_horizons_without_task(tmp_path):
    from deploy.fr3_sharpa_control.agent_settings import TAIRAN_PROVIDER

    agent = AgentSupervisor(SimpleNamespace(root=tmp_path))
    agent.settings["provider"] = TAIRAN_PROVIDER
    agent.state["control_eligible"] = True
    result = packet()
    result.update(
        future_horizon_s=1.0, control_eligible=False, latency_ms=400.0, roi_fallback_frames=2,
        future_probabilities=packet(label=3)["future_probabilities"],
    )
    agent._display_inhand(result, dict(timestamp=100.0, observation_id="100", source_age_s=0.1), "b0", agent.epoch)
    state = agent.snapshot()
    assert state["status"] == "ready" and state["mode"] == "manual"
    assert state["current"]["harvest"] == 0.97 and state["future"]["adjust"] == 0.97
    assert state["roi_fallback_frames"] == 2


def test_inhand_auto_handoff_keeps_old_prompt_until_execution_ack(tmp_path, monkeypatch):
    from deploy.fr3_sharpa_control.agent import InHandGate
    from deploy.fr3_sharpa_control.agent_settings import TAIRAN_PROVIDER
    from deploy.fr3_sharpa_runtime.task_stages import IN_HAND_AGENT_SOURCE, IN_HAND_PROFILE, IN_HAND_PROMPTS

    c = TaskControl(initial_stage="pickup", profile=IN_HAND_PROFILE)
    c.ready()
    server = TaskServer(tmp_path / "task.sock", c)
    owner = SimpleNamespace(root=tmp_path, task_endpoint=server.path, task_run_id=c.run_id,
                            state={"running": True}, operation="execute")
    agent = AgentSupervisor(owner)
    agent.settings.update(provider=TAIRAN_PROVIDER, profile="workstation")
    agent.gate = InHandGate()
    agent.state.update(status="ready", control_eligible=True, received_unix=time.time())
    try:
        old_contact = c.last_contact
        agent.set_mode("auto")
        with pytest.raises(RuntimeError, match="稳定运行"):
            task_rpc(server.path, c.run_id, "stage", request_id="premature", expected_version=0,
                     target="harvest_1", source=IN_HAND_AGENT_SOURCE)
        assert c.last_contact == old_contact
        c.command("start", "start", expected_version=0)
        first = c.ticket()
        c.activate(first, "first")
        stamp = c.snapshot()["active_applied_unix_s"]
        monkeypatch.setattr(time, "time", lambda: stamp + 2)
        old_contact = c.last_contact

        def predict(offset, future=1, epoch=None):
            timestamp = stamp + offset
            result = packet(timestamp)
            result.update(future_horizon_s=1.0, control_eligible=False, latency_ms=250, roi_fallback_frames=0,
                          future_probabilities=packet(label=future)["future_probabilities"])
            meta = dict(timestamp=timestamp, captured_unix=timestamp, ros_now=timestamp,
                        observation_id=str(timestamp), source_age_s=0)
            agent._display_inhand(result, meta, "b0", agent.epoch if epoch is None else epoch)

        predict(0.1, epoch=agent.epoch - 1)
        predict(0.2, future=2)
        assert c.snapshot()["requested_stage"] == "pickup"
        predict(0.7)
        pending = c.snapshot()
        assert pending["requested_stage"] == "harvest_1"
        assert pending["active_stage"] == "pickup" and pending["active_prompt"] == IN_HAND_PROMPTS["pickup"]
        assert c.last_contact == old_contact
        c.transition("SWITCHING", "test execution owner")
        c.activate(c.ticket(), "next")
        assert c.snapshot()["active_prompt"] == IN_HAND_PROMPTS["harvest"]
    finally:
        server.close()


def test_inhand_input_keeps_crop_state_canonical_and_reorders_head_state(tmp_path):
    import numpy as np
    from deploy.fr3_sharpa_control.inhand119_agent_service import model_inputs

    stamps = np.arange(1, 17, dtype=np.int64) * 100_000_000
    path = tmp_path / "history.npz"
    joints = np.arange(16 * 58, dtype=np.float32).reshape(16, 58)
    np.savez(path, rgb=np.zeros((3, 16, 256, 256, 3), np.uint8), joints=joints,
             joint_names=JOINT_NAMES, native_cam0=np.zeros((16, 40, 40, 3), np.uint8), cam0_stamp_ns=stamps)

    class Cropper:
        def reset(self):
            self.contexts = []

        def apply(self, observation):
            index = len(self.contexts)
            np.testing.assert_array_equal(observation["observation/state"], joints[index])
            self.contexts.append(observation["_crop_context"])
            self.last = {"local_roi_valid": index != 0}
            return {"observation/interaction_image": np.full((224, 224, 3), index, np.uint8)}

    cropper = Cropper()
    views, state, fallback = model_inputs(path, JOINT_NAMES[::-1], cropper, stamps[-1], "run")
    assert views.shape == (1, 4, 16, 256, 256, 3)
    assert views[0, 3, -1, 0, 0, 0] == 15 and fallback == 1
    assert [c["source_monotonic_ns"] for c in cropper.contexts] == stamps.tolist()
    np.testing.assert_array_equal(state, joints[None, :, ::-1])
    with pytest.raises(ValueError, match="历史格式无效"):
        model_inputs(path, JOINT_NAMES, cropper, stamps[-2], "run")


def test_gate_current_k2_future_cannot_command():
    g = DecisionGate()
    a = packet()
    a["future_probabilities"] = packet(label=3)["future_probabilities"]
    a["future_logits"] = packet(label=3)["future_logits"]
    assert g.update(a, task(), now=100.1, model_version="b0") is None
    assert g.update(packet(100.5), task(), now=100.6, model_version="b0") == "harvest_1"
    g.reset()
    for t in [100, 100.5, 101]:
        p = packet(t, 0)
        p["future_probabilities"] = packet()["future_probabilities"]
        p["future_logits"] = packet()["future_logits"]
        assert g.update(p, task(), now=t + 0.1, model_version="b0") is None


@pytest.mark.parametrize(
    "mutation",
    [
        {"valid": False},
        {"model_version": "other"},
        {"input_timestamp": 90.0},
        {"input_timestamp": 101.0},
        {"labels": ["pickup"]},
        {"future_horizon_s": 1.0},
        {"current_probabilities": [float("nan")] * 4},
        {"current_logits": [0.0] * 4},
    ],
)
def test_bad_packet_breaks_confirmation(mutation):
    g = DecisionGate()
    g.update(packet(), task(), now=100.1, model_version="b0")
    bad = {**packet(100.5), **mutation}
    with pytest.raises((ValueError, KeyError)):
        g.update(bad, task(), now=100.6, model_version="b0")
    assert g.count == 0


def test_pending_ack_and_version_change_reset_streak():
    g = DecisionGate()
    g.update(packet(), task(), now=100.1, model_version="b0")
    pending = task()
    pending.update(requested_stage="harvest_1", execution_state="SWITCHING", stage_version=1)
    assert g.update(packet(100.5), pending, now=100.6, model_version="b0") is None
    assert g.count == 0
    fresh = task()
    fresh["stage_version"] = 2
    assert g.update(packet(101), fresh, now=101.1, model_version="b0") is None
    assert g.count == 1


@pytest.mark.parametrize("delta", [0.1, 1.0, 3.0])
def test_wrong_cadence_cannot_join_k2(delta):
    g = DecisionGate()
    g.update(packet(), task(), now=100.1, model_version="b0")
    assert g.update(packet(100 + delta), task(), now=100 + delta + 0.1, model_version="b0") is None


def test_duplicate_and_final_phase():
    g = DecisionGate()
    g.update(packet(), task(), now=100.1, model_version="b0")
    with pytest.raises(ValueError):
        g.update(packet(), task(), now=100.1, model_version="b0")
    final = task()
    final.update(active_stage="return_stem", requested_stage="return_stem")
    for t in [101, 101.5, 102]:
        assert g.update(packet(t), final, now=t + 0.1, model_version="b0") is None


def entry(ns):
    sec, nanosec = divmod(ns, 1_000_000_000)
    return (SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec))), ns)


def test_real_source_history_causality_and_missing_frames():
    anchor = 5_000_000_000
    history = {k: [entry(anchor - i * 100_000_000) for i in range(18)] for k in TOPICS}
    history["cam0"].append(entry(anchor + 1))
    rows = select_history(history, anchor)
    assert len(rows) == 16 and rows[-1]["cam0"][1] == anchor
    history["cam1"] = [entry(anchor)]
    with pytest.raises(ValueError):
        select_history(history, anchor)


def test_history_receiver_can_fill_buffer_while_processing_snapshot():
    import threading
    from deploy.fr3_sharpa_runtime.agent_history import HistoryBuffer

    buffer = HistoryBuffer(1024)
    first = object()
    buffer.remember("cam0", first)
    latest, histories = buffer.snapshot()

    def receive():
        for _ in range(100):
            buffer.remember("cam0", object())

    worker = threading.Thread(target=receive)
    worker.start()
    worker.join(timeout=1)
    assert not worker.is_alive()
    current, rows = buffer.snapshot()
    assert len(rows["cam0"]) == 101
    assert histories["cam0"] == ((first, latest["cam0"][1]),)
    assert latest["cam0"][0] is first and current["cam0"][0] is not first
    assert len(JOINT_NAMES) == len(set(JOINT_NAMES)) == 58


def test_agent_ipc_does_not_renew_operator_lease(tmp_path):
    tmp_path.chmod(0o700)
    c = TaskControl()
    c.ready()
    c.command("start", "start", expected_version=0)
    c.activate(c.ticket(), "initial")
    c.last_contact = 0.0
    server = TaskServer(tmp_path / "task.sock", c)
    try:
        task_rpc(server.path, c.run_id, renew_contact=False)
        assert c.last_contact == 0.0
        s = task_rpc(
            server.path, c.run_id, "stage", request_id="agent-1", expected_version=0, target="harvest_1", source="agent"
        )
        assert c.last_contact == 0.0 and s["active_stage"] == "pickup_1" and s["requested_stage"] == "harvest_1"
        with pytest.raises(RuntimeError):
            task_rpc(server.path, c.run_id, "start", request_id="agent-2", expected_version=1, source="agent")
        task_rpc(server.path, c.run_id)
        assert c.last_contact > 0
    finally:
        server.close()


def test_stop_auto_requests_execution_stop_and_resets(tmp_path):
    calls = []
    controller = SimpleNamespace(root=tmp_path, stop=lambda: calls.append("stop"))
    agent = AgentSupervisor(controller)
    agent.mode = "auto"
    agent.gate.count = 1
    agent.stop()
    assert calls == ["stop"] and agent.mode == "manual" and agent.gate.count == 0


def test_mode_requires_live_ready_and_task_and_resets(tmp_path):
    agent = AgentSupervisor(SimpleNamespace(root=tmp_path))
    with pytest.raises(ValueError):
        agent.set_mode("auto")
    agent.settings["profile"] = "workstation"
    agent.state.update(status="ready", received_unix=time.time())
    agent._task = lambda: task()
    agent.set_mode("auto")
    agent.gate.count = 1
    epoch = agent.epoch
    agent.set_mode("manual")
    assert agent.epoch > epoch and agent.gate.count == 0


def test_http_mode_authority_and_source_forgery(tmp_path):
    import threading
    import urllib.request
    import urllib.error
    import json
    from deploy.fr3_sharpa_control.ui import server_for

    tmp_path.chmod(0o700)
    c = TaskControl()
    c.ready()
    c.command("start", "start", expected_version=0)
    c.activate(c.ticket(), "initial")
    server = TaskServer(tmp_path / "task.sock", c)
    web = server_for(tmp_path / "repo", port=0)
    ctl = web.controller
    ctl.task_endpoint = server.path
    ctl.task_run_id = c.run_id
    ctl.operation = "execute"
    ctl.state["running"] = True
    ctl.agent.settings["profile"] = "workstation"
    ctl.agent.state.update(status="ready", received_unix=time.time())
    thread = threading.Thread(target=web.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{web.server_port}"

    def post(path, data, token):
        req = urllib.request.Request(
            base + "/api/" + path,
            json.dumps(data).encode(),
            headers={"X-Control-Token": token, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as r:
            return json.load(r)

    try:
        with urllib.request.urlopen(base + "/api/info") as r:
            token = json.load(r)["token"]
        with pytest.raises(urllib.error.HTTPError) as err:
            post("agent-mode", {"mode": "auto"}, "wrong")
        assert err.value.code == 403
        assert post("agent-mode", {"mode": "auto"}, token)["mode"] == "auto"
        with pytest.raises(urllib.error.HTTPError):
            post("task-command", dict(command="next", expected_version=0, request_id="manual"), token)
        assert c.snapshot()["stage_version"] == 0
        with pytest.raises(urllib.error.HTTPError):
            post(
                "task-command",
                dict(command="stage", expected_version=0, request_id="spoof", target="harvest_1", source="agent"),
                token,
            )
        assert post("agent-mode", {"mode": "manual"}, token)["mode"] == "manual"
        after = post("task-command", dict(command="next", expected_version=0, request_id="manual-ok"), token)
        assert after["requested_stage"] == "harvest_1" and after["active_stage"] == "pickup_1"
        with urllib.request.urlopen(base + "/api/status") as r:
            state = json.load(r)
        assert state["agent"]["mode"] == "manual"
    finally:
        web.shutdown()
        web.server_close()
        thread.join()
        server.close()


def test_fault_keeps_error_after_execution_stop_resets_mode(tmp_path):
    holder = {}

    def stop():
        holder["agent"].set_mode("manual")

    a = AgentSupervisor(SimpleNamespace(root=tmp_path, stop=stop))
    holder["agent"] = a
    a.mode = "auto"
    a.state["status"] = "ready"
    a._fail("camera disconnected")
    assert a.snapshot()["status"] == "failed" and a.snapshot()["message"] == "camera disconnected"


def test_development_cannot_enable_auto_even_with_ready_packet(tmp_path):
    a = AgentSupervisor(SimpleNamespace(root=tmp_path))
    a.state.update(status="ready", received_unix=time.time())
    a._task = lambda: task()
    with pytest.raises(ValueError, match="开发机"):
        a.set_mode("auto")


def test_portable_settings_resolve_paths_on_target_machine(tmp_path):
    from deploy.fr3_sharpa_control.agent_settings import AgentSettings

    repo = tmp_path / "different-user" / "agent"
    deploy = tmp_path / "deploy"
    s = AgentSettings.parse(
        dict(repository=str(repo), python="env/bin/python", checkpoint="models/b0.pt", ros_overlay="overlay"), deploy
    )
    assert s.profile == "development" and s.checkpoint == str(repo / "models/b0.pt")
    assert s.python == str(repo / "env/bin/python") and s.ros_overlay == str(deploy / "overlay")
    with pytest.raises(ValueError):
        AgentSettings.parse(dict(profile="remote"), deploy)
    with pytest.raises(ValueError):
        AgentSettings.parse(dict(command="unknown"), deploy)


def test_development_check_never_reads_dds_or_ros(tmp_path, monkeypatch):
    import subprocess, json
    from pathlib import Path
    from deploy.fr3_sharpa_control.agent_settings import AgentSettings

    repo = tmp_path / "agent"
    deploy = tmp_path / "deploy"
    s = AgentSettings.parse(dict(repository=str(repo)), deploy)
    for p in [
        repo / "tools/serve_tomato_agent.py",
        Path(s.python),
        Path(s.checkpoint),
        Path(s.pretrained) / "model.safetensors",
        Path(s.pretrained) / "config.json",
        Path(s.sample_npz),
    ]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("fixture")
    Path(s.python).chmod(0o700)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=0, stdout=json.dumps(dict(torch="fixture", cuda=True, bf16=True)), stderr=""
        ),
    )
    original = Path.read_text

    def read(p, *args, **kwargs):
        assert str(p) != "/proc/sys/net/core/rmem_max", "Development unexpectedly checked DDS"
        return original(p, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    report = s.check()
    assert report["ready"] and not any("DDS" in row["name"] or "ROS" in row["name"] for row in report["checks"])
    s.profile = "workstation"
    with pytest.raises(AssertionError, match="DDS"):
        s.check()


def test_machine_settings_persist_without_auto_authorization(tmp_path):
    import json

    machine = tmp_path / ".deployment"
    machine.mkdir()
    (machine / "agent_settings.json").write_text(
        json.dumps(dict(profile="workstation", repository=str(tmp_path / "agent")))
    )
    a = AgentSupervisor(SimpleNamespace(root=tmp_path))
    assert a.settings["profile"] == "workstation" and a.mode == "manual"


def test_python_virtualenv_symlink_is_preserved(tmp_path):
    from deploy.fr3_sharpa_control.agent_settings import AgentSettings

    repo = tmp_path / "agent"
    interpreter = repo / ".venv-tomato/bin/python"
    interpreter.parent.mkdir(parents=True)
    system = tmp_path / "system-python"
    system.write_text("fixture")
    interpreter.symlink_to(system)
    s = AgentSettings.parse(dict(repository=str(repo)), tmp_path)
    assert s.python == str(interpreter) and s.python != str(system)
