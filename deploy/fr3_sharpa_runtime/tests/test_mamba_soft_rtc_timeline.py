"""Virtual time and fake worker only; no physical motion or real-time claims."""

# ruff: noqa: SLF001
import json
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from deploy.fr3_sharpa_runtime import rtc_runtime
from deploy.fr3_sharpa_runtime.mamba_model import MambaSoftRTCPolicy
from deploy.fr3_sharpa_runtime.rtc_stream import RTCTimeline
from deploy.fr3_sharpa_runtime.rtc_stream import require_capabilities
from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import tiny_config
from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import tiny_network


@pytest.mark.parametrize("failure", [None, "late", "stale_observation", "wrong_receipt", "exhausted"])
def test_soft_mamba_five_virtual_chunks_and_stop_paths(tmp_path, monkeypatch, failure):
    clock = [1.0]
    sent, operations = [], []
    with tiny_network():
        model = tiny_config().create(jax.random.key(42))
        policy = MambaSoftRTCPolicy(model, rtc_action_repack=lambda a: a, sample_kwargs={"num_steps": 2})
        obs = jax.tree.map(lambda v: np.asarray(v)[0], model.config.fake_obs().to_dict())
        first_actions = policy.infer(obs)["actions"]
        metadata = {
            "checkpoint": "SYNTHETIC-MAMBA",
            "model_architecture": "pi05_mamba3_jax",
            "rtc_supported": True,
            "rtc_warmed_up": True,
            "rtc_action_dim": 58,
            "rtc_protocol_version": 1,
            "rtc_algorithm": "soft_prefix_guidance",
        }
        require_capabilities(metadata)

        class Worker:
            timeline = None

            def request(self, payload):
                op = payload["operation"]
                operations.append(op)
                if op == "stream_start":
                    self.timeline = RTCTimeline(payload["actions"], clock[0], payload["execute_speed_scale"])
                    return {}
                if op == "stream_status":
                    sent.append(self.timeline.sample(clock[0]))
                    return self.timeline.status(clock[0])
                if op == "stream_snapshot":
                    context = self.timeline.snapshot(
                        clock[0],
                        delay_steps=payload["delay_steps"],
                        schedule=payload["schedule"],
                        weight=payload["weight"],
                    )
                    context["previous_actions"] = context["previous_actions"].tolist()
                    return {"context": context}
                if op == "stream_install":
                    return self.timeline.install(payload["actions"], payload["receipt"], clock[0])
                if op == "stream_stop":
                    return self.timeline.status(clock[0])
                raise AssertionError(op)

        class Future:
            def __init__(self, fn, *args):
                self.ready = clock[0] + (2.0 if failure == "exhausted" else 0.9 if failure == "late" else 0.1)
                self.fn, self.args = fn, args

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
            def infer_chunk(self, observation, *, rtc_context, response_timeout):
                require_capabilities(metadata)
                result = policy.infer(obs, rtc_context=rtc_context)
                self.last_receipt = result["rtc"]
                if failure == "wrong_receipt":
                    self.last_receipt = {**self.last_receipt, "token": "obsolete"}
                return metadata, result["actions"], None

        def observe(**kwargs):
            return {
                "observation/state": np.zeros(58),
                "_captured_monotonic_ns": int((clock[0] - (0.3 if failure == "stale_observation" else 0)) * 1e9),
            }

        def sleep(delay):
            clock[0] += delay

        monkeypatch.setattr(rtc_runtime, "ThreadPoolExecutor", Pool)
        monkeypatch.setattr(rtc_runtime.time, "monotonic_ns", lambda: int(clock[0] * 1e9))
        args = SimpleNamespace(
            train_rtc=False,
            rtc_initial_delay_steps=5,
            rtc_delay_buffer_size=10,
            rtc_prefix_attention_schedule="exp",
            rtc_max_guidance_weight=0.2,
            execution_trace_dir=tmp_path,
            prediction_clip_rad=0.05,
            execute_speed_scale=1.0,
            prepare_speed_rad_s=0.03,
            execute_steps=15,
            num_chunks=5,
        )
        if failure:
            with pytest.raises((ValueError, TimeoutError)):
                rtc_runtime.run_rtc(
                    Runtime(),
                    observe,
                    Worker(),
                    (metadata, first_actions, None),
                    observe(),
                    args,
                    clock=lambda: clock[0],
                    sleep=sleep,
                )
            report = json.loads(next(tmp_path.glob("rtc-*/run.json")).read_text())
            assert report["accepted_chunks"] == 1
            assert report["status"] == "FAILED"
        else:
            result = rtc_runtime.run_rtc(
                Runtime(),
                observe,
                Worker(),
                (metadata, first_actions, None),
                observe(),
                args,
                clock=lambda: clock[0],
                sleep=sleep,
            )
            assert result["accepted_chunks"] == 5
            assert result["status"] == "COMPLETE"
            assert operations.count("stream_install") == 4
            assert operations[-1] == "stream_stop"
            assert len(sent) > 4 * 15  # The old chunk was consumed while each Future waited.
            events = [json.loads(v) for v in next(tmp_path.glob("rtc-*/requests.jsonl")).read_text().splitlines()]
            assert len([e for e in events if e["event"] == "accepted"]) == 4
