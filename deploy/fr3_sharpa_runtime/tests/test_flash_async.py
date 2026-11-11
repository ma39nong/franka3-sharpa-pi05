"""Deterministic FLASH phase, commitment, faults and overlapping coordinator tests."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.flash_stream import (
    FlashAsyncTimeline, FlashSettings, observation_anchor, phase_index, validate_execution_plan,
)


SETTINGS = FlashSettings(0.05, 0.5, 0.1)


def timeline(settings=SETTINGS):
    return FlashAsyncTimeline(np.zeros((50, 58)), 10., 10., "run", "model", "history", settings)


def request(stream, anchor=10.2, now=10.3):
    value = dict(run_session_id="run", source_generation=stream.generation,
                 source_sha256=stream.digest, model_identity="model", history_session="history",
                 request_id="request", anchor_s=anchor, deadline_s=11.5)
    stream.snapshot(value, now)
    return value


def install(stream, req, actions=None, **kwargs):
    return stream.install(np.zeros((50, 58)) if actions is None else actions, req,
                          kwargs.pop("now", 10.4), kwargs.pop("switch_s", 10.5),
                          kwargs.pop("last_command", np.zeros(58)),
                          kwargs.pop("measured", np.zeros(58)), kwargs.pop("validate", lambda _: None))


@pytest.mark.parametrize("index", [0, 1, 48, 49, 50])
def test_phase_boundaries(index):
    assert phase_index(10, 10 + index / 30) == index
    assert phase_index(10, 10 + (index + 0.1) / 30) == index + 1


def test_capture_age_is_part_of_phase_and_request_freshness_is_separate():
    obs = {"flash_context": dict(captured_monotonic_ns=10_000_000_000, anchor_age_at_capture_ns=100_000_000)}
    anchor = observation_anchor(obs, 10.01)
    assert phase_index(anchor, 10.4) == 15
    with pytest.raises(ValueError, match="stale"):
        observation_anchor(obs, 10.3)


def test_future_install_preserves_committed_and_skips_from_observation_not_send():
    stream = timeline()
    req = request(stream)
    actions = np.full((50, 58), 0.01)
    result = install(stream, req, actions)
    assert result["skipped"] == 9  # 300ms since observation, not 200ms since request.
    assert stream.generation == 1
    np.testing.assert_array_equal(stream.sample(10.49), np.zeros(58))
    np.testing.assert_allclose(stream.sample(10.5), 0.01)
    assert stream.generation == 2
    with pytest.raises(ValueError):
        install(stream, req, now=10.51, switch_s=10.6)


@pytest.mark.parametrize("field,value", [("run_session_id", "wrong"), ("model_identity", "wrong"),
    ("history_session", "wrong"), ("source_generation", 99), ("source_sha256", "wrong")])
def test_identity_is_enforced_before_outstanding_slot(field, value):
    stream = timeline()
    req = request(stream)
    stream.cancel_request()
    with pytest.raises(ValueError, match="mismatch"):
        stream.snapshot({**req, field: value}, 10.31)
    assert stream.request is None


@pytest.mark.parametrize("fault", ["position", "velocity", "feedback", "range", "nonfinite", "stale_source", "late", "commit"])
def test_rejection_is_atomic(fault):
    stream = timeline()
    req = request(stream)
    actions = np.zeros((50, 58))
    options = {}
    if fault == "position":
        actions[:] = 0.2
    elif fault == "velocity":
        actions[10:] = 0.04
    elif fault == "feedback":
        options["measured"] = np.ones(58)
    elif fault == "range":
        def reject(_):
            raise ValueError("range")
        options["validate"] = reject
    elif fault == "nonfinite":
        actions[0, 0] = np.nan
    elif fault == "stale_source":
        stream.digest = "new source"
    elif fault == "late":
        options.update(now=11.51, switch_s=11.6)
    elif fault == "commit":
        options["switch_s"] = 10.41
    before = stream.actions.copy()
    with pytest.raises((ValueError, TimeoutError)):
        install(stream, req, actions, **options)
    np.testing.assert_array_equal(stream.actions, before)
    assert stream.generation == 1
    assert stream.pending is None


@pytest.mark.parametrize("index,allowed", [(48, False), (49, False), (50, False)])
def test_candidate_suffix_boundary(index, allowed):
    stream = timeline(replace(SETTINGS, min_remaining_nodes=2))
    # Candidate can be older than the currently playing source, but is never re-dated.
    req = request(stream, anchor=10.5 - index / 30)
    if allowed:
        assert install(stream, req)["skipped"] == index
    else:
        with pytest.raises(TimeoutError, match="suffix"):
            install(stream, req)


def test_single_outstanding_stop_and_exhaustion():
    stream = timeline()
    req = request(stream)
    with pytest.raises(ValueError, match="Only one"):
        stream.snapshot(req, 10.31)
    with pytest.raises(TimeoutError, match="exhausted"):
        stream.sample(12)
    stream.stop()
    with pytest.raises(ValueError, match="stopped"):
        stream.sample(12.1)


def test_backward_clock_rejected():
    stream = timeline()
    stream.sample(10.2)
    with pytest.raises(ValueError, match="backwards"):
        stream.sample(10.1)


def test_worker_receipt_is_distinct_from_rtc_and_cannot_reanchor_a_late_response():
    ctx = dict(protocol_version=1, request_id="req", session_id="history",
               observation_anchor_source_ns=123, captured_monotonic_ns=10_000_000_000,
               anchor_age_at_capture_ns=100_000_000)
    receipt = {key: ctx[key] for key in ("protocol_version", "request_id", "session_id", "observation_anchor_source_ns")}
    receipt["model_identity"] = "model"
    plan = dict(flash_context=ctx, receipt=receipt, model_identity="model", history_session="history",
                request_id="req", anchor_s=9.9)
    validate_execution_plan(plan, "model", 10.4)  # >200ms is legal under async phase checks.
    for key, value in (("receipt", {"rtc": True}), ("anchor_s", 10.4), ("model_identity", "other"),
                       ("history_session", "other"), ("request_id", "other")):
        with pytest.raises(ValueError, match="mismatch"):
            validate_execution_plan({**plan, key: value}, "model", 10.4)
    with pytest.raises(ValueError, match="age budget"):
        validate_execution_plan(plan, "model", 10.4, max_age_ms=200)


@pytest.mark.parametrize("values", [(0, .1, .1), (.1, 0, .1), (.1, .1, 0), (np.nan, .1, .1)])
def test_no_unvalidated_default_limits(values):
    with pytest.raises(ValueError):
        FlashSettings(*values)


class Clock:
    def __init__(self):
        self.now = 10.
        self.hooks = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        for hook in self.hooks:
            hook()


def exercise_coordinator(monkeypatch, tmp_path, *, delay=.3, chunks=4, sensor_failure=False, trigger=15):
    from deploy.fr3_sharpa_runtime import flash_runtime as module
    clock = Clock()
    sent = []
    requests = []
    observations = []
    monkeypatch.setattr(module, "validate_flash_observation", lambda obs: obs)

    def observe(**options):
        if options.get("on_poll"):
            options["on_poll"]()
        if sensor_failure and clock() > 10.7:
            raise TimeoutError("camera stale")
        obs = {"observation/state": np.zeros(58), "observation/history": np.zeros((8, 58)),
            "_captured_monotonic_ns": int(clock() * 1e9),
            "flash_context": dict(captured_monotonic_ns=int(clock() * 1e9), anchor_age_at_capture_ns=50_000_000,
                session_id="history", request_id=str(len(observations)))}
        observations.append(obs)
        return obs

    def infer(obs, **options):
        obs["_flash_receipt"] = {"test": True}
        return {"flash_manifest_sha256": "model"}, np.zeros((50, 58)), None

    class Pool:
        def __init__(self, **_):
            self.active = False

        def submit(self, fn, *args):
            assert not self.active
            self.active = True
            ready = clock() + delay
            value = fn(*args)
            owner = self
            class Future:
                def done(self):
                    return clock() >= ready
                def result(self):
                    owner.active = False
                    return value
                def cancel(self):
                    owner.active = False
            return Future()

        def shutdown(self, **_):
            pass

    class Worker:
        stream = None
        next_tick = None
        stopped = False

        def tick(self):
            while self.stream is not None and self.next_tick <= clock():
                command = self.stream.sample(self.next_tick)
                sent.append((self.next_tick, self.stream.generation, command))
                self.next_tick += 1 / 30

        def request(self, payload):
            op = payload["operation"]
            if op == "flash_prepare":
                clock.sleep(.1)
                return {"completed_monotonic_ns": int(clock() * 1e9)}
            if op == "flash_start":
                p = payload["plan"]
                self.stream = FlashAsyncTimeline(payload["actions"], p["anchor_s"], clock(),
                    payload["run_session_id"], p["model_identity"], p["history_session"], FlashSettings(**payload["settings"]))
                self.next_tick = clock()
                self.tick()
            elif op == "flash_snapshot":
                self.stream.snapshot(payload["request"], clock())
                requests.append((clock(), payload["request"]))
            elif op == "flash_install":
                receipt = self.stream.install(payload["actions"], payload["request"], clock(),
                    clock() + 2 / 30, np.zeros(58), np.zeros(58), lambda _: None)
                return {"accepted": True, "installation": receipt}
            elif op == "stream_status":
                return self.stream.status(clock())
            elif op == "stream_stop":
                self.stopped = True
                self.stream.stop()
                self.stream = None
            return {"status": "PASS"}

    monkeypatch.setattr(module, "ThreadPoolExecutor", Pool)
    worker = Worker()
    clock.hooks.append(worker.tick)
    args = SimpleNamespace(flash_seam_position_rad=.05, flash_seam_velocity_rad_s=.5,
        flash_feedback_error_rad=.1, execution_trace_dir=tmp_path, prediction_clip_rad=.05,
        prepare_speed_rad_s=.03, execute_steps=trigger, num_chunks=chunks)
    first_obs = observe()
    result = module.run_flash_async(SimpleNamespace(infer_chunk=infer), observe, worker,
        infer(first_obs), first_obs, args, clock=clock, sleep=clock.sleep)
    return result, sent, requests, observations, worker


def test_coordinator_nonzero_latency_keeps_old_30hz_stream_and_runs_many_generations(monkeypatch, tmp_path):
    result, sent, requests, observations, worker = exercise_coordinator(monkeypatch, tmp_path)
    assert result["status"] == "COMPLETE" and result["accepted_chunks"] == 4
    assert worker.stopped
    assert len(requests) == 3
    assert len(observations) > 10  # Sensor watchdog runs during inference too.
    stamps = [item[0] for item in sent]
    np.testing.assert_allclose(np.diff(stamps), 1 / 30, atol=1e-12)
    for stamp, req in requests:
        assert len([frame for frame in sent if stamp < frame[0] < stamp + .3]) >= 8
    events = [json.loads(line) for line in (tmp_path / ("flashRTC-" + result["run_session_id"]) / "requests.jsonl").read_text().splitlines()]
    assert all(e["installation"]["skipped"] >= 11 for e in events if e["event"] == "accepted")


def test_one_chunk_stops_at_trigger_without_request(monkeypatch, tmp_path):
    result, _, requests, _, worker = exercise_coordinator(monkeypatch, tmp_path, chunks=1)
    assert result["accepted_chunks"] == 1 and not requests and worker.stopped


def test_500ms_candidate_can_run_with_earlier_trigger(monkeypatch, tmp_path):
    result, _, requests, _, _ = exercise_coordinator(monkeypatch, tmp_path, delay=.5, trigger=6)
    assert result["accepted_chunks"] == 4 and len(requests) == 3


@pytest.mark.parametrize("sensor_failure", [False, True])
def test_timeout_or_sensor_loss_cannot_extend_old_plan(monkeypatch, tmp_path, sensor_failure):
    with pytest.raises(TimeoutError, match="deadline|reserve|camera stale"):
        exercise_coordinator(monkeypatch, tmp_path, delay=2, sensor_failure=sensor_failure)
    reports = list(tmp_path.glob("*/run.json"))
    assert json.loads(reports[0].read_text())["status"] == "FAILED"
