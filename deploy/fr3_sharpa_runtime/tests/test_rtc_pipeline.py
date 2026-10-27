"""Deterministic asynchronous broker tests using the real RTC timeline/protocol."""

from types import SimpleNamespace
import json
import time

import numpy as np
from openpi_client import rtc
import pytest

from deploy.fr3_sharpa_runtime.rtc_stream import RTCTimeline
from deploy.fr3_sharpa_runtime import rtc_runtime


def prediction(base=0.1):
    return np.repeat((base + np.arange(50, dtype=np.float32) * 0.001)[:, None], 58, axis=1)


@pytest.mark.parametrize("scale", [0.1, 0.5, 1.0])
def test_async_broker_keeps_old_motion_running_and_accepts_five_blocks(monkeypatch, tmp_path, scale):
    simulate(monkeypatch, tmp_path, scale, False)


def test_late_guided_result_is_rejected_and_failure_recorded(monkeypatch, tmp_path):
    simulate(monkeypatch, tmp_path, 1.0, True)


@pytest.mark.parametrize("failure", ["observation_timeout", "worker_failure"])
def test_observation_wait_is_bounded_and_preserves_worker_error(monkeypatch, tmp_path, failure):
    simulate(monkeypatch, tmp_path, 1.0, False, failure=failure)


def simulate(monkeypatch, tmp_path, scale, late, *, failure=None):
    clock = [1.0]
    log = []

    class Worker:
        plan = None

        def request(self, payload):
            op = payload["operation"]
            log.append(op)
            if op == "stream_start":
                self.plan = RTCTimeline(payload["actions"], clock[0], payload["execute_speed_scale"])
                return {"status": "PASS"}
            if op == "stream_status":
                if failure == "worker_failure" and self.plan.phase(clock[0]) >= 20:
                    raise RuntimeError("original device feedback failure")
                # Samples are still consumed while a future is outstanding.
                sent.append(self.plan.sample(clock[0]))
                return {"status": "PASS", **self.plan.status(clock[0])}
            if op == "stream_snapshot":
                context = self.plan.snapshot(
                    clock[0], delay_steps=payload["delay_steps"], schedule=payload["schedule"], weight=payload["weight"]
                )
                context["previous_actions"] = context["previous_actions"].tolist()
                return {"status": "PASS", "context": context}
            if op == "stream_install":
                return {"status": "PASS", **self.plan.install(payload["actions"], payload["receipt"], clock[0])}
            if op == "stream_stop":
                return {"status": "PASS", **self.plan.status(clock[0])}
            raise AssertionError(op)

    class Future:
        def __init__(self, fn, *args):
            self.ready = clock[0] + (0.9 if late else 0.1)
            self.fn = fn
            self.args = args

        def done(self):
            return clock[0] >= self.ready

        def result(self):
            return self.fn(*self.args)

    class Pool:
        def __init__(self, **kwargs):
            pass

        def submit(self, fn, *args):
            return Future(fn, *args)

        def shutdown(self, **kwargs):
            pass

    class Runtime:
        def infer_chunk(self, obs, *, rtc_context, response_timeout):
            assert response_timeout > 0
            self.last_receipt = rtc.make_receipt(rtc_context)
            return {"checkpoint": "fake"}, prediction(0.1 + rtc_context["source_number"] * 0.001), None

    sent = []

    def sleep(delay):
        clock[0] += delay

    def observe(**options):
        if failure and options:
            assert 0 < options["timeout_s"] < 35 / 30
            options["on_diagnostic"]({"event": "observation_wait", "topics": {"cam0": {"missing": True}}})
            while True:
                clock[0] += 0.01
                options["on_poll"]()
        return {"observation/state": prediction()[0], "_captured_monotonic_ns": int(clock[0] * 1e9)}

    monkeypatch.setattr(rtc_runtime, "ThreadPoolExecutor", Pool)
    monkeypatch.setattr(rtc_runtime.time, "monotonic_ns", lambda: int(clock[0] * 1e9))
    args = SimpleNamespace(
        rtc_initial_delay_steps=5,
        rtc_delay_buffer_size=10,
        rtc_prefix_attention_schedule="exp",
        rtc_max_guidance_weight=5.0,
        execution_trace_dir=tmp_path,
        prediction_clip_rad=0.05,
        execute_speed_scale=scale,
        prepare_speed_rad_s=0.03,
        execute_steps=15,
        num_chunks=5,
    )
    first = ({"checkpoint": "fake"}, prediction(), None)
    call = lambda: rtc_runtime.run_rtc(
        Runtime(), observe, Worker(), first, observe(), args, clock=lambda: clock[0], sleep=sleep
    )
    if failure:
        error = TimeoutError if failure == "observation_timeout" else RuntimeError
        match = "observation deadline" if failure == "observation_timeout" else "original device feedback failure"
        with pytest.raises(error, match=match):
            call()
        report = json.loads(next(tmp_path.glob("rtc-*/run.json")).read_text())
        assert report["accepted_chunks"] == 1 and report["stage"] == "observing"
        assert report["last_observation_diagnostic"]["topics"]["cam0"]["missing"]
        assert "stream_snapshot" not in log and "stream_install" not in log
        assert clock[0] < 1 + 50 / 30  # Stop before old motion is exhausted.
    elif late:
        with pytest.raises(TimeoutError, match="guided prefix"):
            call()
        report = json.loads(next(tmp_path.glob("rtc-*/run.json")).read_text())
        assert report["status"] == "FAILED" and report["accepted_chunks"] == 1
    else:
        result = call()
        assert result["status"] == "COMPLETE" and result["accepted_chunks"] == 5
        assert log.count("stream_snapshot") == log.count("stream_install") == 4
        assert log[-1] == "stream_stop"
        events = [json.loads(line) for line in next(tmp_path.glob("rtc-*/requests.jsonl")).read_text().splitlines()]
        assert len([e for e in events if e["event"] == "accepted"]) == 4
        assert len(sent) > 4 * 15  # Old commands continued during each pending inference.
