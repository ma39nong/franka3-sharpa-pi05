"""Audit regressions use real coordinators/timelines and a real read-only IPC child."""

from concurrent.futures import Future
import json
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.observation_worker import ObservationWorker
from deploy.fr3_sharpa_runtime.rtc_stream import RTCTimeline
from deploy.fr3_sharpa_runtime.stage_handoff import StageTimeline, aligned_candidate, trajectory_metrics, DT
from deploy.fr3_sharpa_runtime.switch_audit import simulate, summarize
from deploy.fr3_sharpa_runtime.task_rtc import TaskRTC
from deploy.fr3_sharpa_runtime.tests import test_tomato_task as helpers
from deploy.fr3_sharpa_runtime.tests.test_tomato_task import start_engine, command, until


@pytest.fixture
def engine(tmp_path):
    yield from helpers.engine.__wrapped__(tmp_path)


@pytest.mark.parametrize("phase,window,budget", [(10, .75, .483333), (15, .75, .483333),
                         (25, .75, .483333), (35, .466667, .2), (40, .3, .033333), (45, .133333, -.133333)])
def test_phase_budgets_are_from_accepted_timeline(phase, window, budget):
    t = StageTimeline(np.zeros((50, 58)), 1, 1, task_epoch=0)
    now = 1 + phase / 30
    if phase == 45:
        with pytest.raises(TimeoutError, match="Insufficient"):
            t.begin_switch(1, "late", now)
    else:
        b = t.begin_switch(1, "next", now)
        assert b["deadline"] - now == pytest.approx(window, abs=1e-6)
        assert b["deadline"] - now - 8 * DT == pytest.approx(budget, abs=1e-6)


@pytest.mark.parametrize("delay", [.001, .035, .067, .099, .133])
def test_fractional_alignment_has_no_one_tick_rounding_lead(delay):
    a = np.repeat(np.arange(50)[:, None] * .001, 58, axis=1)
    candidate = aligned_candidate(a, 1, 1 + delay, 1)
    reference = RTCTimeline(a, 1, 1)
    for step in range(12):
        now = 1 + delay + step * DT
        np.testing.assert_allclose(candidate.sample(now), reference.sample(now), atol=1e-8)
    assert candidate.skipped == pytest.approx(delay * 30)


def test_frozen_source_is_reused_not_copied_each_tick():
    t = StageTimeline(np.zeros((50, 58)), 1, 1, task_epoch=0)
    t.begin_switch(1, "next", 1.3)
    first = t.source_curve()
    for i in range(10):
        t.sample(1.3 + i * DT)
        assert t.source_curve() is first


def test_new_observation_overlaps_old_policy_without_concurrent_inference(engine):
    e, c, r, b, observations, events = engine
    start_engine(engine)
    release, entered = threading.Event(), threading.Event()
    ticket = c.ticket()

    def old_gpu():
        entered.set()
        assert release.wait(1)
        raise RuntimeError("obsolete GPU result")

    e.ticket, e.first_deadline = ticket, time.monotonic() + 1
    e.future = e.pool.submit(old_gpu)
    assert entered.wait(1)
    count, calls = len(observations), len(r.calls)
    try:
        command(c, "next")
        e.tick()
        until(e, lambda: len(observations) > count)
        assert not e.future.done() and len(r.calls) == calls
        release.set()
        until(e, lambda: c.snapshot()["execution_state"] == "RUNNING")
        assert r.calls[-1][1] is None
        assert not any(isinstance(x, tuple) and x[0] == "install" for x in b.calls)
    finally:
        release.set()


def test_queued_obsolete_job_cancels_without_gpu_claim(engine):
    e, c, *_ = engine
    start_engine(engine)
    pending = Future()
    e.ticket, e.future = c.ticket(), pending
    e.first_deadline = time.monotonic() + 1
    command(c, "next")
    e.tick()
    assert pending.cancelled()
    until(e, lambda: c.snapshot()["execution_state"] == "RUNNING")


def test_slow_previous_policy_cannot_cause_infinite_observation_refresh(engine):
    e, c, *_ = engine
    start_engine(engine)
    e.policy_duration = .4  # Advisory prediction must never create a zero-age budget.
    command(c, "next")
    until(e, lambda: c.snapshot()["execution_state"] == "RUNNING")


@pytest.mark.parametrize("ordinary", [False, True])
def test_stage_change_does_not_kill_persistent_sensor_child(engine, tmp_path, ordinary):
    e, c, _, b, *_ = engine
    start_engine(engine)
    code = '''import sys,json,time
for line in sys.stdin:
 req=json.loads(line)
 if req.get('command')=='close':break
 time.sleep(.025)
 print(json.dumps({'observation/state':[0]*58,'_captured_monotonic_ns':time.monotonic_ns()}),flush=True)
'''
    worker = ObservationWorker([sys.executable, "-u", "-c", code], cwd=tmp_path)
    fired = False

    def observe(**options):
        def poll():
            nonlocal fired
            if not fired:
                fired = True
                command(c, "next")
            options.get("on_poll", lambda: None)()
        return json.loads(worker.snapshot(timeout_s=.3, on_poll=poll))

    e.observe = observe
    try:
        if ordinary:
            b.plan.installed_at = time.monotonic() - .55
            b.next_tick = time.monotonic()
            e.tick()  # Version changes inside the real child's on_poll.
        else:
            ticket = c.ticket()
            e.capture(ticket, .3)
            assert not c.valid(ticket)
        assert fired and worker.process.poll() is None
        until(e, lambda: c.snapshot()["execution_state"] == "RUNNING")
        assert worker.process.poll() is None
    finally:
        worker.close()


def test_synthetic_stale_prefetch_refreshes_and_keeps_one_policy():
    result = simulate(TaskRTC, StageTimeline, phase=25, old_ms=220, observation_ms=20, infer_ms=60)
    assert result["status"] == "PASS", result["reason"]
    assert result["max_concurrent_policy"] == 1 and not result["old_stage_installs"]
    assert any(e["event"] == "observation_discarded" for e in result["events"])


def test_derivatives_are_command_diagnostics_not_unverified_limits():
    a = np.arange(12)[:, None] * np.ones((1, 58)) * .001
    metrics = trajectory_metrics(a)
    assert metrics["right_hand"]["velocity_rad_s"] == pytest.approx(.03)
    assert metrics["right_arm"]["acceleration_rad_s2"] < 1e-10
    assert metrics["left_arm"]["jerk_rad_s3"] < 1e-8


def test_event_summary_distinguishes_armed_applied_and_missing(tmp_path):
    path = tmp_path / "events.jsonl"
    data = [dict(event=event, monotonic_ns=stamp, stage_version=1, request_id="next")
            for event, stamp in [("stage_requested", 1000000), ("online_switch_armed", 11000000),
                                 ("online_switch_applied", 211000000)]]
    path.write_text("\n".join(json.dumps(e) for e in data))
    row = summarize(path)["switches"][0]["milestones_ms"]
    assert row["online_switch_armed"] == 10 and row["online_switch_applied"] == 210
    assert row["observation_start"] is None and row["ordinary_rtc_resumed"] is None


@pytest.mark.parametrize("late", [False, True])
def test_task_transport_shares_one_total_budget_but_ordinary_is_unchanged(monkeypatch, late):
    from deploy.fr3_sharpa_runtime import runtime as mod
    from openpi_client import msgpack_numpy
    import websockets.sync.client

    now = [1000000000]
    received, connected = [], []
    class WS:
        calls = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            now[0] += (100 if late else 1) * 1000000
        def recv(self, timeout):
            received.append(timeout)
            now[0] += 20_000_000
            self.calls += 1
            return msgpack_numpy.Packer().pack({} if self.calls == 1 else {"actions": np.zeros((50, 58))})
        def send(self, payload):
            pass
    def connect(url, **options):
        connected.append(options)
        return WS()
    monkeypatch.setattr(websockets.sync.client, "connect", connect)
    monkeypatch.setattr(mod, "time", SimpleNamespace(monotonic_ns=lambda: now[0], time_ns=lambda: now[0]))
    monkeypatch.setattr(mod.config, "require_checkpoint", lambda: None)
    monkeypatch.setattr(mod.config, "validate_policy_metadata", lambda metadata: None)
    r = mod.SharpaRuntime.__new__(mod.SharpaRuntime)
    r.policy_url, r.train_rtc, r.paper_rtc = "ws://127.0.0.1:8000", False, False
    r.adapter = SimpleNamespace(adapt_chunk=lambda *args, **kwargs: None)
    obs = {"observation/state": np.zeros(58), "prompt": "test",
           **{f"observation/{name}": np.zeros((2, 2, 3), np.uint8)
              for name in ("image", "left_wrist_image", "right_wrist_image")}}
    if late:
        with pytest.raises(TimeoutError, match="total transport"):
            r.infer_task_chunk(obs, response_timeout=.1)
    else:
        r.infer_task_chunk(obs, response_timeout=.1)
    assert connected[0]["open_timeout"] <= .1 and connected[0]["close_timeout"] <= .05
    assert received == pytest.approx([.1, .08])
    received.clear()
    r.infer_chunk(obs, response_timeout=.1)
    assert connected[-1]["open_timeout"] == 10 and connected[-1]["close_timeout"] == 5
    assert received == [10, .1]


def test_ordinary_rtc_replans_between_cross_stage_handoffs(engine):
    e, c, _, b, _, events = engine
    start_engine(engine)
    for _ in range(3):
        command(c, "next")
        until(e, lambda: c.snapshot()["execution_state"] == "RUNNING")
        assert b.plan.source is None
        previous = len([x for x in b.calls if isinstance(x, tuple) and x[0] == "install"])
        # Simulate advancing the accepted chunk; do not synthesize more actions.
        b.plan.installed_at -= .51
        b.next_tick = time.monotonic()
        until(e, lambda: len([x for x in b.calls if isinstance(x, tuple) and x[0] == "install"]) > previous)
        assert b.plan.task_epoch == c.snapshot()["stage_version"]
    assert len([x for x in events if x["event"] == "ordinary_rtc_resumed"]) == 3


def test_lightweight_task_poll_does_not_read_log_files(tmp_path, monkeypatch):
    from deploy.fr3_sharpa_control.controller import DeploymentController
    from deploy.fr3_sharpa_runtime.task_control import TaskControl, TaskServer
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    c = TaskControl()
    c.ready()
    server = TaskServer(private / "task.sock", c)
    ctl = DeploymentController(tmp_path / "repo")
    ctl.task_endpoint, ctl.task_run_id = server.path, c.run_id
    try:
        with monkeypatch.context() as patch:
            patch.setattr(type(tmp_path), "open", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("log I/O")))
            assert ctl.task_status()["execution_state"] == "READY"
    finally:
        server.close()
