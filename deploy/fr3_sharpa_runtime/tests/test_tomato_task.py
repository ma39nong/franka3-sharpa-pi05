"""Hardware-free acceptance: real state machine, sockets, RTC protocol and HTTP."""

from concurrent.futures import Future
from contextlib import contextmanager
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import urllib.error
import urllib.request
import uuid

import numpy as np
from openpi_client import rtc
import pytest

from deploy.fr3_sharpa_runtime.task_stages import ACTION_PROMPTS, PHASES, STAGES
from deploy.fr3_sharpa_runtime.task_control import TaskControl, TaskServer, task_rpc
from deploy.fr3_sharpa_runtime.task_rtc import TaskRTC, SDKTaskBackend, safe_stop_proof
from deploy.fr3_sharpa_runtime.rtc_stream import RTCTimeline
from deploy.fr3_sharpa_runtime.stage_handoff import StageTimeline, DT


def command(c, op, **fields):
    return c.command(op, uuid.uuid4().hex, expected_version=c.snapshot()["stage_version"], **fields)


def running(c):
    c.ready()
    command(c, "start")
    ticket = c.ticket()
    c.activate(ticket, "0:1")
    return ticket


def test_training_contract_exact_and_independent():
    # Compare actual training source without importing its runtime dependencies.
    import ast

    source = Path("/home/user/tomato-openpi/src/openpi/subtasks/tomato.py")
    if source.exists():
        tree = ast.parse(source.read_text())
        definitions = {
            n.targets[0].id: ast.literal_eval(n.value)
            for n in tree.body
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in ("ACTION_PROMPTS", "PHASES")
        }
        assert definitions == dict(ACTION_PROMPTS=ACTION_PROMPTS, PHASES=PHASES)
    assert [s.action for s in STAGES] == ["pickup", "harvest", "return", "adjust", "pickup", "harvest", "return"]
    assert STAGES[0].prompt == STAGES[4].prompt
    assert STAGES[2].prompt == STAGES[6].prompt


def test_stage_request_does_not_fake_activation_and_is_idempotent():
    c = TaskControl()
    old = running(c)
    args = dict(expected_version=0, target="harvest_1")
    first = c.command("stage", "same", **args)
    assert first["active_stage"] == "pickup_1" and first["requested_stage"] == "harvest_1"
    assert first["stage_version"] == 1 and not c.valid(old)
    assert c.command("stage", "same", **args) == first
    with pytest.raises(ValueError, match="重复"):
        c.command("pause", "same", expected_version=1)
    with pytest.raises(ValueError, match="版本"):
        c.command("next", "stale", expected_version=0)


@pytest.mark.parametrize("target", ["invalid", "pickup_1", "return_stem", "adjust"])
def test_invalid_jump_or_rollback_rejected(target):
    c = TaskControl()
    running(c)
    with pytest.raises(ValueError):
        command(c, "stage", target=target)
    assert c.snapshot()["stage_version"] == 0


def test_same_prompt_recovery_and_rapid_clicks_never_merge_progress():
    c = TaskControl()
    ticket = running(c)
    command(c, "pause")
    c.transition("PAUSED", "simulated verified pause")
    command(c, "recover", target="pickup_2", confirmed=True)
    assert c.snapshot()["stage_version"] == 2 and not c.valid(ticket)
    assert STAGES[4].prompt == ticket["prompt"]
    command(c, "next")
    command(c, "next")
    assert c.snapshot()["requested_stage"] == "return_stem"
    assert c.snapshot()["active_stage"] == "pickup_1"
    assert c.snapshot()["stage_version"] == 4


def test_complete_requires_actual_final_stage_and_explicit_operator():
    c = TaskControl(initial_stage="return_stem")
    running(c)
    assert c.snapshot()["execution_state"] == "RUNNING"
    command(c, "complete")
    assert c.snapshot()["execution_state"] == "STOPPING"
    d = TaskControl()
    running(d)
    with pytest.raises(ValueError):
        command(d, "complete")


def test_no_false_pause_no_unapproved_resume_and_legal_transitions():
    c = TaskControl()
    running(c)
    command(c, "pause")
    assert c.snapshot()["execution_state"] == "PAUSING"
    assert not c.snapshot()["safe_to_resume"]
    with pytest.raises(ValueError):
        command(c, "resume", confirmed=True)
    with pytest.raises(ValueError):
        c.transition("RUNNING", "illegal")
    c.fault("disable failed")
    with pytest.raises(ValueError):
        command(c, "resume", confirmed=True)
    command(c, "stop")
    assert c.snapshot()["execution_state"] == "STOPPING"


@contextmanager
def ipc(tmp_path):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    c = TaskControl()
    c.ready()
    server = TaskServer(root / "task.sock", c)
    try:
        yield c, server
    finally:
        server.close()


def test_real_unix_ipc_identity_permissions_idempotency_and_controller(tmp_path):
    from deploy.fr3_sharpa_control.controller import DeploymentController

    with ipc(tmp_path) as (c, s):
        assert s.path.stat().st_mode & 0o777 == 0o600
        assert task_rpc(s.path, c.run_id)["execution_state"] == "READY"
        with pytest.raises(RuntimeError, match="身份"):
            task_rpc(s.path, "wrong")
        with pytest.raises(RuntimeError):
            task_rpc(s.path, c.run_id, "bad", request_id="x", expected_version=0)
        ctl = DeploymentController(tmp_path / "repo")
        ctl.task_endpoint = s.path
        ctl.task_run_id = c.run_id
        ctl.operation = "execute"
        ctl.state["running"] = True
        a = ctl.task_command(dict(command="start", request_id="start", expected_version=0))
        assert a["execution_state"] == "RESUMING"
        assert ctl.snapshot()["task"]["execution_state"] == "RESUMING"
        assert (
            ctl.task_command(dict(command="start", request_id="start", expected_version=0))["execution_state"]
            == "RESUMING"
        )
        assert (
            ctl.task_command(dict(command="stop", request_id="stop", expected_version=-99))["execution_state"]
            == "STOPPING"
        )
    with pytest.raises(OSError):
        task_rpc(s.path, c.run_id)


def test_socket_insecure_parent_rejected(tmp_path):
    tmp_path.chmod(0o755)
    with pytest.raises(PermissionError):
        TaskServer(tmp_path / "bad.sock", TaskControl())


class FakeBackend:
    def __init__(self, safe=True):
        self.plan = None
        self.calls = []
        self.safe = safe
        self.epoch = None

    def reauthorize(self):
        self.calls.append("reauthorize")

    def start(self, actions, ticket, path):
        self.calls.append(("start", ticket.copy()))
        self.epoch = ticket["stage_version"]
        self.plan = StageTimeline(actions, time.monotonic(), 1, task_epoch=self.epoch)
        self.next_tick = time.monotonic()
        self.last_command = actions[0].copy()
        self.frames = []
        return self.plan.status(time.monotonic())

    def status(self):
        now = time.monotonic()
        while self.next_tick <= now:
            self.last_command = self.plan.sample(self.next_tick)
            self.frames.append((self.next_tick, self.last_command.copy()))
            self.next_tick += DT
        return self.plan.status(now)

    def begin_switch(self, epoch, request_id):
        self.calls.append(("switch", epoch))
        return self.plan.begin_switch(epoch, request_id, time.monotonic())

    def handoff(self, actions, boundary, captured):
        self.calls.append(("handoff", boundary.copy()))
        return self.plan.prepare_handoff(
            actions,
            {**boundary, "captured_monotonic_s": captured},
            time.monotonic(),
            self.next_tick,
            self.last_command,
            0.7,
            lambda a: None,
        )

    def snapshot(self, settings, delay):
        return self.plan.snapshot(
            time.monotonic(),
            delay_steps=delay,
            schedule=settings.prefix_attention_schedule,
            weight=settings.max_guidance_weight,
        )

    def install(self, actions, receipt, ticket):
        result = self.plan.install(actions, receipt, time.monotonic(), task_epoch=ticket["stage_version"])
        self.calls.append(("install", ticket.copy()))
        return result

    def stop(self):
        self.calls.append("stop")
        if self.plan:
            self.plan.invalidate()
            self.plan = None
        return dict(
            status="PASS" if self.safe else "BLOCKED",
            hands_disabled=self.safe,
            arms_safe=self.safe,
            old_stream_invalidated=True,
        )


class FakeRuntime:
    def __init__(self):
        self.calls = []
        self.failure = None
        self.last_receipt = None
        self.bad_shape = False

    def infer_chunk(self, obs, *, rtc_context=None, response_timeout=5):
        self.calls.append((obs.copy(), rtc_context))
        if self.failure:
            raise self.failure
        self.last_receipt = rtc.make_receipt(rtc_context) if rtc_context else None
        return {"checkpoint": "simulation-only"}, np.zeros((49 if self.bad_shape else 50, 58), np.float32), None


@pytest.fixture
def engine(tmp_path):
    c = TaskControl()
    events = []
    c.emit = lambda **e: events.append(e)
    runtime = FakeRuntime()
    backend = FakeBackend()
    observations = []

    def observe(**options):
        if options.get("on_poll"):
            options["on_poll"]()
        obs = {
            "observation/state": np.full(58, len(observations), np.float32),
            "_captured_monotonic_ns": time.monotonic_ns(),
        }
        observations.append(obs)
        return obs

    args = SimpleNamespace(
        rtc_initial_delay_steps=5,
        rtc_delay_buffer_size=10,
        rtc_prefix_attention_schedule="exp",
        rtc_max_guidance_weight=5,
        execute_steps=15,
        num_chunks=0,
    )
    e = TaskRTC(c, runtime, backend, observe, args, tmp_path)
    c.ready()
    yield e, c, runtime, backend, observations, events
    e.pool.shutdown(wait=True, cancel_futures=True)
    e.observation_pool.shutdown(wait=True, cancel_futures=True)


def until(e, predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        assert time.monotonic() < deadline, "simulation timeout"
        e.tick()
        time.sleep(0.001)


def start_engine(parts):
    e, c, *_ = parts
    command(c, "start")
    until(e, lambda: c.snapshot()["execution_state"] == "RUNNING")


def test_full_seven_stage_simulation_fresh_boundaries_and_no_auto_progress(engine):
    e, c, r, b, obs, events = engine
    start_engine(engine)
    for expected in PHASES[1:]:
        old_count = len(obs)
        command(c, "next")
        e.tick()
        assert c.snapshot()["execution_state"] == "SWITCHING"
        assert c.snapshot()["active_stage"] != expected
        until(e, lambda: c.snapshot()["execution_state"] == "RUNNING")
        assert c.snapshot()["active_stage"] == expected
        assert len(obs) > old_count and r.calls[-1][1] is None
        assert r.calls[-1][0]["prompt"] == STAGES[PHASES.index(expected)].prompt
    assert c.snapshot()["execution_state"] == "RUNNING"
    assert sum(isinstance(x, tuple) and x[0] == "start" for x in b.calls) == 1
    assert "stop" not in b.calls
    command(c, "complete")
    assert e.tick() is False
    assert c.snapshot()["execution_state"] == "COMPLETED"
    assert any(x["event"] == "inference_accepted" for x in events)


def test_same_stage_rtc_real_protocol_delay_and_delta_reference(engine):
    e, c, r, b, obs, _ = engine
    start_engine(engine)
    b.plan.installed_at -= 0.51
    e.tick()
    until(e, lambda: e.future is None)
    assert b.calls[-1][0] == "install"
    assert c.snapshot()["active_stage"] == "pickup_1"
    assert len(obs) == 2
    assert r.calls[-1][1] is not None
    np.testing.assert_array_equal(r.calls[-1][0]["observation/state"], obs[-1]["observation/state"])
    assert c.snapshot()["chunk_progress"]["phase"] >= 15


@pytest.mark.parametrize("op", ["pause", "next", "stage"])
def test_late_future_never_installs_after_invalidating_command(engine, op):
    e, c, r, b, obs, events = engine
    start_engine(engine)
    pending = Future()
    pending.set_running_or_notify_cancel()  # Simulate an already running GPU call.
    ticket = c.ticket()
    e.future = pending
    e.ticket = ticket
    e.first_deadline = time.monotonic() + 5
    command(c, op, **({"target": "harvest_1"} if op == "stage" else {}))
    e.tick()
    assert c.snapshot()["execution_state"] == ("PAUSED" if op == "pause" else "SWITCHING")
    pending.set_exception(RuntimeError("old GPU error should be discarded"))
    e.tick()
    assert not any(isinstance(x, tuple) and x[0] == "install" for x in b.calls)
    assert any(x["event"] == "inference_discarded" for x in events)
    if op == "pause":
        command(c, "resume", confirmed=True)
    until(e, lambda: c.snapshot()["execution_state"] == "RUNNING")
    assert r.calls[-1][1] is None


@pytest.mark.parametrize("kind", ["timeout", "inference", "shape", "stale", "exhausted", "receipt"])
def test_task_errors_cannot_install(engine, kind):
    e, c, r, b, obs, _ = engine
    start_engine(engine)
    installed = len([x for x in b.calls if isinstance(x, tuple)])
    if kind == "exhausted":
        b.plan.installed_at -= 2
    elif kind == "timeout":
        e.future = Future()
        e.ticket = c.ticket()
        e.first_deadline = time.monotonic() - 1
    else:
        if kind == "inference":
            r.failure = RuntimeError("model failed")
        if kind == "shape":
            r.bad_shape = True
        if kind == "stale":
            e.observe = lambda **kw: {"observation/state": np.zeros(58), "_captured_monotonic_ns": 0}
        if kind == "receipt":
            original = b.install

            def corrupt(actions, receipt, ticket):
                return original(actions, {**receipt, "token": "wrong"}, ticket)

            b.install = corrupt
        b.plan.installed_at -= 0.51
        e.tick()
    with pytest.raises((TimeoutError, RuntimeError, ValueError)):
        until(e, lambda: False)
    assert len([x for x in b.calls if isinstance(x, tuple)]) == installed


def test_unverified_hardware_boundary_is_blocked_not_paused(engine):
    e, c, r, b, obs, _ = engine
    start_engine(engine)
    b.safe = False
    command(c, "pause")
    with pytest.raises(RuntimeError, match="安全边界"):
        e.tick()
    assert c.snapshot()["execution_state"] == "PAUSING" and not c.snapshot()["safe_to_resume"]
    c.fault("stop unconfirmed")
    with pytest.raises(ValueError):
        command(c, "resume", confirmed=True)


def test_timeline_epoch_invalidate_and_old_receipt():
    a = np.zeros((50, 58), np.float32)
    t = RTCTimeline(a, 1.0, 1, task_epoch=2)
    ctx = t.snapshot(1.1, delay_steps=5, schedule="exp", weight=5)
    with pytest.raises(ValueError, match="epoch"):
        t.install(a, rtc.make_receipt(ctx), 1.2, task_epoch=1)
    t.invalidate()
    assert t.pending is None
    with pytest.raises(ValueError):
        t.install(a, rtc.make_receipt(ctx), 1.2, task_epoch=2)


def test_real_backend_never_claims_robot_hold_from_hand_disable():
    class Worker:
        shutdown_report = {"status": "PASS"}

        def close(self):
            pass

    b = SDKTaskBackend(Worker(), SimpleNamespace())
    proof = b.stop()
    assert proof["hands_disabled"] and not proof["arms_safe"] and not safe_stop_proof(proof)
    with pytest.raises(RuntimeError):
        b.reauthorize()


def test_ui_lease_expiry_stops_without_waiting_for_future(engine):
    e, c, r, b, obs, _ = engine
    start_engine(engine)
    e.future = Future()
    e.ticket = c.ticket()
    e.first_deadline = time.monotonic() + 10
    c.last_contact = time.monotonic() - 6
    assert e.tick() is False
    assert c.snapshot()["execution_state"] == "STOPPED"
    assert "stop" in b.calls


def test_request_id_replacement_same_epoch_cannot_install():
    c = TaskControl()
    running(c)
    old = c.ticket()
    new = c.ticket()
    assert not c.valid(old) and c.valid(new)
    with pytest.raises(ValueError, match="旧阶段"):
        c.activate(old, "old")


def test_real_http_ui_ipc_and_refresh_no_duplicate_start(tmp_path):
    from deploy.fr3_sharpa_control.ui import server_for

    with ipc(tmp_path) as (c, ipc_server):
        web = server_for(tmp_path / "repo", port=0)
        ctl = web.controller
        ctl.task_endpoint = ipc_server.path
        ctl.task_run_id = c.run_id
        ctl.operation = "execute"
        ctl.state["running"] = True
        thread = threading.Thread(target=web.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{web.server_port}"

        def get(path):
            with urllib.request.urlopen(base + path) as response:
                return json.loads(response.read())

        def post(data, token):
            req = urllib.request.Request(
                base + "/api/task-command",
                json.dumps(data).encode(),
                headers={"X-Control-Token": token, "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as response:
                return json.loads(response.read())

        try:
            with urllib.request.urlopen(base) as response:
                html = response.read().decode()
            assert "番茄采摘任务状态机" in html and "task-command" in html
            token = get("/api/info")["token"]
            assert get("/api/status")["task"]["execution_state"] == "READY"
            data = dict(command="start", request_id="http-start", expected_version=0)
            with pytest.raises(urllib.error.HTTPError) as error:
                post(data, "wrong")
            assert error.value.code == 403
            assert post(data, token)["execution_state"] == "RESUMING"
            assert post(data, token)["execution_state"] == "RESUMING"
            assert get("/api/status")["task"]["execution_state"] == "RESUMING"
            data = dict(command="stop", request_id="http-stop", expected_version=0)
            assert post(data, token)["execution_state"] == "STOPPING"
        finally:
            web.shutdown()
            web.server_close()
            thread.join(timeout=2)


def test_run_saves_fault_and_stops_on_inference_failure(engine, tmp_path):
    e, c, r, b, obs, events = engine
    r.failure = RuntimeError("inference failure")
    # run() owns READY; trigger the operator from its real concurrent control path.
    c.data["execution_state"] = "IDLE"

    def operator():
        deadline = time.monotonic() + 2
        while c.snapshot()["execution_state"] != "READY":
            assert time.monotonic() < deadline
            time.sleep(0.001)
        command(c, "start")

    thread = threading.Thread(target=operator)
    thread.start()
    with pytest.raises(RuntimeError, match="inference failure"):
        e.run()
    thread.join()
    assert c.snapshot()["execution_state"] == "FAULT"
    assert "stop" in b.calls
    report = json.loads((tmp_path / "task-run.json").read_text())
    assert report["status"]["last_error"] == "inference failure"


def test_stop_can_register_during_slow_disable_confirmation(engine):
    e, c, r, b, obs, events = engine
    start_engine(engine)
    entered = threading.Event()
    release = threading.Event()
    original = b.stop

    def slow_stop():
        entered.set()
        assert release.wait(2)
        return original()

    b.stop = slow_stop
    command(c, "pause")
    result = []
    thread = threading.Thread(target=lambda: result.append(e.tick()))
    thread.start()
    assert entered.wait(1)
    # No GPU or device RPC owns the state lock during disable/readback.
    command(c, "stop")
    release.set()
    thread.join(2)
    assert result == [False] and c.snapshot()["execution_state"] == "STOPPED"


def test_reproducibility_typed_stats_exact_unwrap(tmp_path):
    from deploy.fr3_sharpa_runtime.train_rtc_contract import load_training_record

    fields = {"mean": [0.0] * 58, "std": [1.0] * 58, "q01": [-1.0] * 58, "q99": [1.0] * 58}
    expected = {k: {"type": "openpi.shared.normalize.NormStats", "fields": fields} for k in ["state", "actions"]}
    (tmp_path / "norm_stats.json").write_text(json.dumps(expected))
    checkpoint = tmp_path / "stats.json"
    checkpoint.write_text(json.dumps({"norm_stats": {k: fields for k in expected}}))
    config = tmp_path / "effective_config.json"
    config.write_text(json.dumps({"fields": {"model": {"fields": {"rtc_max_delay": 0}}, "data": {"fields": {}}}}))
    assert load_training_record(config, checkpoint)["config"]["fields"]["model"]["fields"]["rtc_max_delay"] == 0
    expected["actions"]["fields"] = {**fields, "mean": [0.000001] * 58}
    (tmp_path / "norm_stats.json").write_text(json.dumps(expected))
    with pytest.raises(ValueError, match="统计"):
        load_training_record(config, checkpoint)


def _isolated_task_process(path, run_id, ready, done):
    c = TaskControl(run_id)
    c.ready()
    server = TaskServer(path, c)
    ready.set()
    try:
        done.wait(5)
    finally:
        server.close()


def test_ipc_real_separate_execution_process(tmp_path):
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    ready, done = ctx.Event(), ctx.Event()
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    path = directory / "task.sock"
    run_id = uuid.uuid4().hex
    process = ctx.Process(target=_isolated_task_process, args=(path, run_id, ready, done))
    process.start()
    try:
        assert ready.wait(3)
        assert task_rpc(path, run_id)["execution_state"] == "READY"
        assert task_rpc(path, run_id, "start", request_id="start", expected_version=0)["execution_state"] == "RESUMING"
        assert task_rpc(path, run_id, "stop", request_id="stop", expected_version=0)["execution_state"] == "STOPPING"
    finally:
        done.set()
        process.join(5)
    assert process.exitcode == 0
    with pytest.raises(OSError):
        task_rpc(path, run_id)


def test_peer_credentials_rejected(tmp_path, monkeypatch):
    from deploy.fr3_sharpa_runtime import task_control

    with ipc(tmp_path) as (c, s):
        monkeypatch.setattr(task_control.os, "getuid", lambda: -1)
        with pytest.raises(RuntimeError, match="同一用户"):
            task_rpc(s.path, c.run_id)


def test_task_plan_uses_approved_stage_and_correct_rtc(tmp_path, monkeypatch):
    from deploy.fr3_sharpa_control import plan as plans

    root = Path(__file__).resolve().parents[3]
    identity = dict(
        profile="full-delta",
        semantics="delta",
        identity="simulated",
        tomato_model={"name": "test"},
        train_rtc=None,
        paper_algorithm=None,
        supported_modes=["rtc"],
        finetune_mode="full",
        checkpoint="fake",
        norm_stats="fake",
    )
    monkeypatch.setattr(plans, "inspect_checkpoint", lambda *a, **kw: identity)
    settings = plans.DeploymentSettings(str(root), "fake", task_mode="tomato-b", mode="rtc", speed="fast")
    first = settings.plan()
    assert "--task-mode" in first["execute"] and "--enable-rtc" in first["service"]
    assert "--train-rtc" not in first["execute"]
    from dataclasses import replace

    with pytest.raises(ValueError, match="恢复"):
        replace(settings, initial_stage="pickup_2").plan()
    recovered = replace(settings, initial_stage="pickup_2", recovery_confirmed=True).plan()
    assert recovered["approval"] != first["approval"]
    assert recovered["execute"][recovered["execute"].index("--initial-stage") + 1] == "pickup_2"
    with pytest.raises(ValueError):
        replace(settings, mode="serial").plan()


def test_audit_failure_cannot_leave_new_prompt_on_old_running_stream():
    c = TaskControl()
    old = running(c)

    def disk_full(**event):
        raise OSError("disk full")

    c.emit = disk_full
    with pytest.raises(RuntimeError, match="audit"):
        command(c, "next")
    assert c.snapshot()["execution_state"] == "FAULT"
    assert not c.snapshot()["safe_to_resume"] and not c.valid(old)


def test_epoch_requires_matching_request_and_stage(engine):
    e, c, *_ = engine
    start_engine(engine)
    old = c.ticket()
    fake = {**old, "stage_id": "harvest_1"}
    assert not c.valid(fake)
    assert not c.valid({**old, "request_id": "wrong"})
    assert not c.valid({**old, "run_id": "wrong"})


def test_ui_javascript_syntax(tmp_path):
    from deploy.fr3_sharpa_control.ui import HTML
    import shutil
    import subprocess

    node = shutil.which("node") or str(
        Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
    )
    if not Path(node).is_file():
        pytest.skip("Node.js required for frontend syntax check")
    script = tmp_path / "ui.js"
    script.write_text(HTML.split("<script>")[1].split("</script>")[0])
    subprocess.run([node, "--check", str(script)], check=True, capture_output=True, text=True)


def test_known_model_identity_requires_all_three_fingerprints():
    from deploy.fr3_sharpa_runtime.tomato_model import identify, STATS_SHA256, MANIFEST_SHA256, METADATA_SHA256

    assert identify(STATS_SHA256, MANIFEST_SHA256, METADATA_SHA256)["subtask_variant"] == "b_coarse"
    assert identify(STATS_SHA256, "wrong", METADATA_SHA256) is None
    assert identify("wrong", MANIFEST_SHA256, METADATA_SHA256) is None
    assert identify(STATS_SHA256, MANIFEST_SHA256, "wrong") is None


def test_lease_covers_blocking_home_and_startup_without_device_commands():
    from deploy.fr3_sharpa_runtime.task_control import TaskLease

    c = TaskControl()
    interrupted = []
    lease = TaskLease(c, lambda: interrupted.append("owner interrupted"), clock=lambda: c.last_contact + 6)
    assert c.snapshot()["execution_state"] == "IDLE"
    assert lease.poll() and interrupted == ["owner interrupted"]
    # Explicit stop also interrupts a blocking owner even while UI is alive.
    c = TaskControl()
    command(c, "stop")
    lease = TaskLease(c, lambda: interrupted.append("stop"), clock=lambda: c.last_contact)
    assert lease.poll() and interrupted[-1] == "stop"


def test_execution_entry_without_private_control_never_connects_devices(tmp_path, monkeypatch):
    from deploy.fr3_sharpa_runtime.task_execution import execute_task

    monkeypatch.delenv("SHARPA_TASK_SOCKET", raising=False)
    monkeypatch.delenv("SHARPA_TASK_RUN_ID", raising=False)
    with pytest.raises(ValueError, match="socket"):
        execute_task(SimpleNamespace(execution_trace_dir=tmp_path))


def test_execution_wrapper_fault_closes_owner_and_socket(tmp_path, monkeypatch):
    from deploy.fr3_sharpa_runtime import task_execution, run, runtime, motion_worker

    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "setup.bash").write_text("# offline fixture")
    path = root / "task.sock"
    monkeypatch.setenv("SHARPA_TASK_SOCKET", str(path))
    monkeypatch.setenv("SHARPA_TASK_RUN_ID", "execution-wrapper")
    lifecycle = []

    class Policy:
        def __init__(self, *args):
            pass

        def inspect_metadata(self, **kw):
            return dict(tomato_model={"name": "simulation"}, model_profile="full-delta")

    class Worker:
        shutdown_report = {"status": "PASS"}

        def __init__(self, *args, **kw):
            lifecycle.append("worker")

        def request(self, payload, **kw):
            assert payload["operation"] in ("ready", "feedback")
            return {"status": "PASS"}

        def close(self):
            lifecycle.append("closed")

    class Engine:
        def __init__(self, *args):
            pass

        def run(self):
            raise RuntimeError("forced execution failure")

    class Lease:
        def __init__(self, *args):
            pass

        def start(self):
            lifecycle.append("lease")

        def close(self):
            lifecycle.append("lease_closed")

    monkeypatch.setattr(runtime, "SharpaRuntime", Policy)
    monkeypatch.setattr(motion_worker, "MotionWorker", Worker)
    monkeypatch.setattr(task_execution, "TaskRTC", Engine)
    monkeypatch.setattr(task_execution, "TaskLease", Lease)
    monkeypatch.setattr(run, "_start_observation_worker", lambda *a: None)
    monkeypatch.setattr(run, "live_observation", lambda *a, **kw: {"observation/state": np.zeros(58)})
    args = SimpleNamespace(
        execution_trace_dir=tmp_path / "logs",
        initial_stage="pickup_1",
        policy_url="offline",
        ros_overlay=overlay,
        initialization="current",
    )
    with pytest.raises(RuntimeError, match="forced"):
        task_execution.execute_task(args)
    assert lifecycle == ["lease", "worker", "lease_closed", "closed"]
    assert not path.exists()
    final = json.loads((tmp_path / "logs/task-final.json").read_text())
    assert final["execution_state"] == "FAULT" and not final["safe_to_resume"]


def test_rapid_online_requests_install_only_latest_in_same_session(engine):
    e, c, r, b, obs, events = engine
    start_engine(engine)
    command(c, "next")
    e.tick()
    command(c, "next")
    until(e, lambda: c.snapshot()["execution_state"] == "RUNNING")
    assert c.snapshot()["active_stage"] == "return_truss"
    assert c.snapshot()["stage_version"] == 2
    assert sum(isinstance(x, tuple) and x[0] == "start" for x in b.calls) == 1
    assert "stop" not in b.calls
    assert r.calls[-1][0]["prompt"] == STAGES[2].prompt and r.calls[-1][1] is None


def test_new_request_during_bridge_retargets_without_reconnect(engine):
    e, c, r, b, obs, events = engine
    start_engine(engine)
    command(c, "next")
    until(e, lambda: e.bridge_ticket is not None)
    previous_deadline = e.boundary["deadline"]
    command(c, "next")
    e.tick()
    assert e.boundary["deadline"] == previous_deadline
    assert c.snapshot()["active_stage"] == "pickup_1"
    until(e, lambda: c.snapshot()["execution_state"] == "RUNNING")
    assert c.snapshot()["active_stage"] == "return_truss" and "stop" not in b.calls


def test_recovery_requires_separate_pause_and_ready_selection_does_not_start(engine):
    e, c, r, b, obs, events = engine
    command(c, "next")
    assert c.snapshot()["execution_state"] == "READY" and not b.calls
    start_engine(engine)
    with pytest.raises(ValueError, match="先独立暂停"):
        command(c, "recover", target="pickup_1", confirmed=True)
    assert c.snapshot()["execution_state"] == "RUNNING"


@pytest.mark.parametrize("failure", ["feedback", "invalid_action", "inference", "timeout"])
def test_online_failure_runs_fault_cleanup_not_redeployment(engine, failure):
    e, c, r, b, obs, events = engine
    c.data["execution_state"] = "IDLE"
    original_status = b.status
    original_begin = b.begin_switch

    def status():
        if failure == "feedback" and b.plan.switch is not None:
            raise RuntimeError("feedback abnormal")
        return original_status()

    b.status = status

    def begin(*args):
        boundary = original_begin(*args)
        if failure == "timeout":
            boundary["deadline"] = time.monotonic() - 1
        if failure == "inference":
            r.failure = RuntimeError("new prompt inference failed")
        return boundary

    b.begin_switch = begin
    if failure == "invalid_action":

        def invalid(*args):
            raise ValueError("new trajectory violates native bounds")

        b.handoff = invalid

    def operator():
        deadline = time.monotonic() + 2
        while c.snapshot()["execution_state"] != "READY":
            assert time.monotonic() < deadline
            time.sleep(0.001)
        command(c, "start")
        while c.snapshot()["execution_state"] != "RUNNING":
            assert time.monotonic() < deadline
            time.sleep(0.001)
        command(c, "next")

    thread = threading.Thread(target=operator)
    thread.start()
    with pytest.raises((RuntimeError, ValueError, TimeoutError)):
        e.run()
    thread.join(2)
    assert not thread.is_alive() and c.snapshot()["execution_state"] == "FAULT"
    assert "stop" in b.calls
    assert sum(isinstance(x, tuple) and x[0] == "start" for x in b.calls) == 1


def test_same_prompt_stage_versions_are_isolated_online(engine):
    e, c, r, b, obs, events = engine
    start_engine(engine)
    original_prompt = r.calls[0][0]["prompt"]
    old = c.ticket()
    for _ in range(4):
        command(c, "next")
        until(e, lambda: c.snapshot()["execution_state"] == "RUNNING")
    assert c.snapshot()["active_stage"] == "pickup_2"
    assert c.snapshot()["stage_version"] == 4
    assert r.calls[-1][0]["prompt"] == original_prompt
    assert not c.valid(old) and "stop" not in b.calls


@pytest.mark.parametrize("bad", ["epoch", "generation", "token"])
def test_bad_online_worker_ack_never_activates_stage(engine, bad):
    e, c, r, b, obs, events = engine
    start_engine(engine)
    if bad == "token":
        original = b.handoff

        def handoff(*a):
            result = original(*a)
            result["token"] = "wrong"
            return result

        b.handoff = handoff
    else:
        original = b.status

        def status():
            result = original()
            if result.get("applied_token"):
                if bad == "epoch":
                    result["task_epoch"] = -1
                else:
                    result["generation"] = 0
            return result

        b.status = status
    command(c, "next")
    with pytest.raises(ValueError, match="acknowledgement"):
        until(e, lambda: c.snapshot()["active_stage"] == "harvest_1")
    assert c.snapshot()["active_stage"] == "pickup_1"


def test_recover_after_verified_pause_keeps_real_stop_proof_and_requires_resume(engine):
    e, c, r, b, obs, events = engine
    start_engine(engine)
    command(c, "pause")
    e.tick()
    assert c.snapshot()["safe_to_resume"]
    count = len(r.calls)
    command(c, "recover", target="pickup_2", confirmed=True)
    assert c.snapshot()["execution_state"] == "PAUSED" and c.snapshot()["safe_to_resume"]
    assert len(r.calls) == count
    command(c, "resume", confirmed=True)
    until(e, lambda: c.snapshot()["execution_state"] == "RUNNING")
    assert c.snapshot()["active_stage"] == "pickup_2" and r.calls[-1][1] is None
