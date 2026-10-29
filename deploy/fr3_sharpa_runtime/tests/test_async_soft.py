"""Fourth-mode tests with synthetic clocks/models/devices; never connect hardware."""
# ruff: noqa: PLC0415, SLF001

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import MethodType
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.async_soft_stream import CAPABILITY
from deploy.fr3_sharpa_runtime.async_soft_stream import CONTEXT_KEY
from deploy.fr3_sharpa_runtime.async_soft_stream import RECEIPT_KEY
from deploy.fr3_sharpa_runtime.async_soft_stream import AsyncSoftTimeline
from deploy.fr3_sharpa_runtime.async_soft_stream import handover_window
from deploy.fr3_sharpa_runtime.async_soft_stream import receipt
from deploy.fr3_sharpa_runtime.async_soft_stream import require_capabilities
from deploy.fr3_sharpa_runtime.async_soft_stream import validate_context

ROOT = Path(__file__).resolve().parents[3]


def trajectory(offset=0):
    return np.broadcast_to((offset + np.arange(50) * 0.001)[:, None], (50, 58)).astype(np.float32).copy()


def snapshot(scale=1.0):
    timeline = AsyncSoftTimeline(trajectory(), 10.0, scale)
    context = timeline.snapshot(10.4, observation_monotonic_s=10.35)
    return timeline, context


@pytest.mark.parametrize("scale", [0.1, 0.5, 1.0])
def test_fixed_prefix_compression_references_and_boundary_install(scale):
    timeline, context = snapshot(scale)
    hz = 30 / math.ceil(1 / scale)
    assert timeline.actions.shape == (30, 58)
    assert timeline.targets.shape == (20, 58)
    np.testing.assert_array_equal(timeline.targets[[0, -1]], trajectory()[[0, 25]])
    np.testing.assert_array_equal(context["previous_actions"], trajectory()[26:30])
    assert context["reference_start"] == 0
    boundary = 10 + 20 / hz
    assert handover_window(context)[1] == pytest.approx(boundary)
    old = timeline.sample(10.4).copy()
    staged = timeline.install(trajectory(0.026), receipt(context), 10.4, terminal=True)
    assert staged["queued"]
    assert staged["skipped"] == 0
    np.testing.assert_array_equal(timeline.sample(10.4), old)
    np.testing.assert_allclose(timeline.sample(boundary), trajectory(0.026)[0])
    assert timeline.generation == 2
    assert timeline.status(boundary + 20 / hz)["finished"]


@pytest.mark.parametrize("failure", ["early", "late", "receipt", "source", "exhausted"])
def test_invalid_install_never_replaces_old_trajectory(failure):
    timeline, context = snapshot()
    at, ack = 10.5, receipt(context)
    if failure == "early":
        at = 10.3
    elif failure == "late":
        at = context["handover_monotonic_s"]
    elif failure == "receipt":
        ack["token"] = "another-request"
    elif failure == "source":
        timeline.generation += 1
    else:
        at = 12
    before = timeline.actions.copy()
    with pytest.raises((ValueError, TimeoutError)):
        timeline.install(trajectory(0.2), ack, at)
    np.testing.assert_array_equal(timeline.actions, before)
    assert timeline.queued is None


def test_snapshot_rejects_stale_observations_duplicate_requests_and_short_tail():
    timeline, context = snapshot()
    with pytest.raises(ValueError, match="outstanding"):
        timeline.snapshot(10.41, observation_monotonic_s=10.4)
    timeline = AsyncSoftTimeline(trajectory(), 10, 1)
    with pytest.raises(ValueError, match="stale"):
        timeline.snapshot(10.5, observation_monotonic_s=10)
    with pytest.raises(TimeoutError, match="handover"):
        timeline.snapshot(11.5, observation_monotonic_s=11.5)
    changed = copy.deepcopy(context)
    changed["previous_actions"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_context(changed)
    changed = {**context, "reference_start": True}
    with pytest.raises(ValueError, match="reference_start"):
        validate_context(changed)
    changed = copy.deepcopy(context)
    changed["previous_actions"][0, 0] += 0.1
    with pytest.raises(ValueError, match="digest"):
        validate_context(changed)
    with pytest.raises(ValueError, match="own warmed service"):
        require_capabilities({"rtc_supported": True, "rtc_warmed_up": True})


def test_discarded_tail_never_changes_compressed_targets_references_or_digest():
    original = AsyncSoftTimeline(trajectory(), 10, 1)
    changed = trajectory()
    changed[30:] = 10000
    new = AsyncSoftTimeline(changed, 10, 1)
    np.testing.assert_array_equal(new.targets, original.targets)
    assert new.digest == original.digest
    np.testing.assert_array_equal(new.actions[26:30], original.actions[26:30])
    with pytest.raises(TimeoutError):
        original.sample(10 + 20 / 30)


def tiny(monkeypatch):
    from deploy.fr3_sharpa_runtime.tests.test_paper_sampler import tiny_sampler

    return tiny_sampler(monkeypatch)


def test_actual_jitted_sampler_only_guides_delayed_nodes_and_releases_late_steps(monkeypatch):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from deploy.fr3_sharpa_runtime.async_soft_sampler import sample_async_soft
    from openpi.shared.nnx_utils import module_jit

    model, obs = tiny(monkeypatch)
    sample = module_jit(MethodType(sample_async_soft, model))
    noise = jax.random.normal(jax.random.key(8), (1, 50, 58))
    baseline, count = sample(jax.random.key(4), obs, noise=noise)
    original = module_jit(model.sample_actions)(jax.random.key(4), obs, noise=noise)
    np.testing.assert_array_equal(baseline, original)
    assert int(count) == 0
    reference = jnp.full((1, 4, 58), 0.25)
    for start in (0, 5, 11):
        guided, count = sample(
            jax.random.key(4), obs, noise=noise, reference=reference, reference_start=jnp.int32(start)
        )
        assert int(count) == 6
        np.testing.assert_array_equal(guided[:, :start], baseline[:, :start])
        np.testing.assert_array_equal(guided[:, start + 4 :], baseline[:, start + 4 :])
        assert not np.allclose(guided[:, start : start + 4], baseline[:, start : start + 4])
        assert not np.allclose(guided[:, start : start + 4], reference)


def test_new_policy_preserves_old_policy_and_encodes_delta_from_observation(monkeypatch):
    pytest.importorskip("jax")
    import deploy.fr3_sharpa_runtime.async_soft_policy as module
    from deploy.fr3_sharpa_runtime.async_soft_policy import AsyncSoftPolicy
    from openpi.policies.policy import Policy

    model, _ = tiny(monkeypatch)
    base = Policy(model, transforms=[lambda obs: {"state": obs["observation/state"], "image": {}, "image_mask": {}}])
    original_sampler = base._sample_actions
    metadata = {"server_instance_id": "boot"}
    encoded = []

    def normalize(data):
        encoded.append(data["actions"].copy())
        return data

    new = AsyncSoftPolicy(base, metadata, normalizer=normalize, delta=True)
    timeline, context = snapshot()
    monkeypatch.setattr(module.time, "monotonic", lambda: context["snapshot_monotonic_s"])
    state = np.arange(58, dtype=np.float32) * 0.01
    obs = {"observation/state": state}
    noise = np.ones((50, 58), np.float32)
    expected = base.infer(obs, noise=noise)
    plain = new.infer(obs, noise=noise)
    np.testing.assert_array_equal(plain["actions"], expected["actions"])
    result = new.infer({**obs, CONTEXT_KEY: context}, noise=noise)
    arms = np.r_[0:7, 29:36]
    np.testing.assert_allclose(encoded[0][:, arms], context["previous_actions"][:, arms] - state[arms])
    np.testing.assert_array_equal(encoded[0][:, 7:29], context["previous_actions"][:, 7:29])
    assert result["async_soft_diagnostics"]["applied_steps"] == 6
    assert result[RECEIPT_KEY]["context"] == receipt(context)
    assert base._sample_actions is original_sampler
    np.testing.assert_array_equal(base.infer(obs, noise=noise)["actions"], expected["actions"])
    with pytest.raises(ValueError, match="does not mix"):
        new.infer(obs, rtc_context={})


def test_client_msgpack_identity_receipt_and_restart_rejection(monkeypatch):
    from openpi_client import msgpack_numpy
    import websockets.sync.client

    from deploy.fr3_sharpa_runtime import async_soft_client as module
    from deploy.fr3_sharpa_runtime.rolling import model_identity

    _, context = snapshot()
    monkeypatch.setattr(module.time, "monotonic", lambda: context["snapshot_monotonic_s"])
    monkeypatch.setattr(module.config, "validate_policy_metadata", lambda metadata: None)
    metadata = {
        "async_soft_version": CAPABILITY,
        "async_soft_warmed_up": True,
        "server_instance_id": "boot1",
        "model_profile": "full-delta",
    }
    sent = []

    class Socket:
        def __enter__(self):
            self.calls = 0
            return self

        def __exit__(self, *args):
            pass

        def send(self, wire):
            sent.append(msgpack_numpy.unpackb(wire))

        def recv(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return msgpack_numpy.Packer().pack(metadata)
            return msgpack_numpy.Packer().pack(
                {
                    "actions": trajectory(),
                    RECEIPT_KEY: {
                        "context": receipt(context),
                        "server_instance_id": "boot1",
                        "model_identity": model_identity(metadata),
                    },
                }
            )

    monkeypatch.setattr(websockets.sync.client, "connect", lambda *a, **kw: Socket())
    obs = {"observation/state": np.zeros(58, np.float32), "prompt": "test"}
    obs["_captured_monotonic_ns"] = int(context["observation_monotonic_s"] * 1e9)
    for key in ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image"):
        obs[key] = np.zeros((8, 8, 3), np.uint8)
    client = module.AsyncSoftClient()
    _, actions, _ = client.infer_chunk(obs, context=context)
    np.testing.assert_array_equal(actions, trajectory())
    assert CONTEXT_KEY in sent[0]
    assert "rtc" not in sent[0]
    assert client.last_receipt == receipt(context)
    metadata["server_instance_id"] = "boot2"
    with pytest.raises(ValueError, match="changed"):
        client.infer_chunk(obs, context=context)
    assert client.last_receipt is None


@pytest.mark.parametrize("scale", [0.1, 0.5, 1])
@pytest.mark.parametrize("failure", [None, "late", "observation", "worker", "interrupt"])
def test_async_coordinator_keeps_old_commands_running_and_rejects_failures(monkeypatch, tmp_path, scale, failure):
    from deploy.fr3_sharpa_runtime import async_soft_runtime as module

    clock = [10.0]
    log, sent = [], []

    class Worker:
        timeline = None

        def request(self, request):
            op = request["operation"]
            log.append(op)
            if op == "stream_start":
                self.timeline = AsyncSoftTimeline(request["actions"], clock[0], scale, terminal=request["terminal"])
                return {"status": "PASS"}
            if op == "stream_status":
                if failure == "worker" and clock[0] > 10.03:
                    raise RuntimeError("original device feedback failure")
                if failure == "interrupt" and clock[0] > 10.03:
                    raise KeyboardInterrupt()
                sent.append(self.timeline.sample(clock[0]))
                return self.timeline.status(clock[0])
            if op == "stream_snapshot":
                context = self.timeline.snapshot(
                    clock[0],
                    observation_monotonic_s=request["observation_monotonic_s"],
                )
                return {"context": {**context, "previous_actions": context["previous_actions"].tolist()}}
            if op == "stream_install":
                return self.timeline.install(
                    request["actions"], request["receipt"], clock[0], terminal=request["terminal"]
                )
            if op == "stream_stop":
                return self.timeline.status(clock[0])
            raise AssertionError(op)

    class Client:
        def infer_chunk(self, obs, *, context, response_timeout):
            assert response_timeout > 0
            self.last_receipt = receipt(context)
            return {"checkpoint": "fake"}, trajectory(0.01), {"applied_steps": 6}

    class Future:
        def __init__(self, fn, *args):
            self.ready = clock[0] + (100 if failure == "late" else 0.06)
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

    def observe(**options):
        assert options["timeout_s"] > 0
        if failure == "observation":
            while True:
                clock[0] += 0.01
                options["on_poll"]()
        clock[0] += 0.02
        options["on_poll"]()
        return {"observation/state": np.zeros(58), "_captured_monotonic_ns": int((clock[0] - 0.01) * 1e9)}

    def sleep(delay):
        clock[0] += delay

    monkeypatch.setattr(module, "ThreadPoolExecutor", Pool)
    args = SimpleNamespace(
        soft_initial_delay_steps=5,
        soft_delay_buffer_size=10,
        execution_trace_dir=tmp_path,
        execute_speed_scale=scale,
        prediction_clip_rad=0.05,
        prepare_speed_rad_s=0.03,
        execute_steps=15,
        num_chunks=4,
    )
    first = ({"checkpoint": "fake"}, trajectory(), {})

    def call():
        return module.run_async_soft(
            Client(),
            observe,
            Worker(),
            first,
            {"observation/state": np.zeros(58)},
            args,
            clock=lambda: clock[0],
            sleep=sleep,
        )

    if failure is None:
        result = call()
        assert result["accepted_chunks"] == 4
        assert log.count("stream_install") == 3
        assert log[-1] == "stream_stop"
        rows = [
            json.loads(line) for line in next(tmp_path.glob("async-soft-*/requests.jsonl")).read_text().splitlines()
        ]
        for row in [r for r in rows if r["event"] == "queued"]:
            assert row["installation"]["skipped"] == 0
            assert row["installation"]["queued"]
        assert len(sent) > 4  # Old commands were sampled while prediction was pending.
        assert result["rtc_guidance"] is False
    else:
        error = KeyboardInterrupt if failure == "interrupt" else RuntimeError if failure == "worker" else TimeoutError
        with pytest.raises(error):
            call()
        result = json.loads(next(tmp_path.glob("async-soft-*/run.json")).read_text())
        assert result["accepted_chunks"] == 1
        assert "stream_install" not in log
        assert result["status"] == ("INTERRUPTED" if failure == "interrupt" else "FAILED")


@pytest.mark.parametrize("model", ["full", "full-delta"])
def test_fourth_cli_is_offline_and_old_options_do_not_enable_it(model):
    env = {key: value for key, value in os.environ.items() if not key.startswith("SHARPA_")}
    env["SHARPA_PYTHON"] = sys.executable
    process = subprocess.run(
        ["bash", "deploy/fr3_sharpa_full/runtime_async_soft.sh", "--model", model, "--check", "--speed", "slow"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(process.stdout)
    assert report["broker_mode"] == "async-soft"
    assert report["execute_steps"] == 26
    assert report["execution_target_count"] == 20
    assert report["compression"] == "cubic"
    assert report["node_hz"] == 3
    assert report["model_profile"] == model
    assert not report["hardware_output"]
    assert report["policy_url"].endswith(":18865" if model == "full" else ":18867")
    rejected = subprocess.run(
        ["bash", "deploy/fr3_sharpa_full/runtime_async_soft.sh", "--model", model, "--check", "--broker-mode", "rtc"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode == 2


def test_worker_isolated_and_strict_compression():
    text = (ROOT / "deploy/fr3_sharpa_runtime/async_soft_motion_ros.py").read_text()
    assert "from .async_soft_stream import AsyncSoftTimeline" in text
    assert "from .rtc_stream import RTCTimeline" not in text
    assert "targets = execution_targets(actions)" in text
    assert "transition[0] = stream.targets[-1]" in text


@pytest.mark.parametrize("fail_after", [None, 2])
def test_isolated_worker_retains_confirmed_shutdown_on_partial_execution(monkeypatch, capsys, fail_after):
    import deploy.fr3_sharpa_runtime as package
    from deploy.fr3_sharpa_runtime import async_soft_motion_ros
    from deploy.fr3_sharpa_runtime.tests.test_paper_worker import (
        test_real_worker_compressed_stream_and_partial_failure_audit as verify_worker,
    )

    # Run the existing real-loop/fake-ROS validation against the independent worker.
    monkeypatch.setattr(package, "sdk_direct_motion_ros", async_soft_motion_ros, raising=False)
    verify_worker(monkeypatch, capsys, fail_after)


@pytest.mark.parametrize("scale", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("failure", [None, "late", "splice", "overshoot"])
def test_real_async_worker_snapshot_install_and_confirmed_shutdown(monkeypatch, scale, failure):
    import io
    from types import ModuleType

    from deploy.fr3_sharpa_runtime import sdk_direct
    import deploy.fr3_sharpa_runtime.async_soft_motion_ros as worker
    from deploy.fr3_sharpa_runtime.feedback import LiveFeedback

    clock = [10.0]
    state = np.zeros(58, np.float32)
    sent, stopped = [], []
    protocol = io.StringIO()

    class Stamp:
        @property
        def nanoseconds(self):
            return worker.time.monotonic_ns()

        def to_msg(self):
            return None

    class Node:
        def get_clock(self):
            return SimpleNamespace(now=Stamp)

        def get_node_names(self):
            return ["teleop_safety_gateway"]

        def get_publishers_info_by_topic(self, topic):
            return []

        def destroy_node(self):
            pass

        def create_publisher(self, message, topic, depth):
            return SimpleNamespace(publish=lambda value: None)

    class Message:
        def __init__(self):
            self.header = SimpleNamespace(stamp=None)

    class Feedback:
        def __init__(self, node):
            pass

        def snapshot(self):
            now = worker.time.monotonic_ns()
            return LiveFeedback(state.copy(), (now,) * 4, (now,) * 4)

        def close(self):
            pass

    class SDK:
        def __init__(self):
            self.left = self.right = SimpleNamespace(
                read_joint_state=lambda: SimpleNamespace(names=[], position=[], velocity=[], effort=[])
            )

        def connect(self):
            pass

        def resume_stream(self):
            pass

        def send(self, left, right):
            sent.append(np.r_[left, right])

        def shutdown_confirmed(self):
            stopped.append(True)
            return {"status": "PASS", "simulation": True}

    class Executor:
        def __init__(self, **kwargs):
            pass

        def add_node(self, node):
            pass

        def spin(self):
            pass

        def shutdown(self, **kwargs):
            pass

    hz = 30 / math.ceil(1 / scale)

    class Input:
        count = 0

        def readline(self):
            self.count += 1
            if self.count == 1:
                payload = {
                    "operation": "stream_start",
                    "actions": trajectory().tolist(),
                    "execute_speed_scale": scale,
                    "events_path": str(events),
                }
            elif self.count == 2:
                payload = {"operation": "stream_snapshot", "observation_monotonic_s": clock[0]}
            elif self.count == 3:
                rows = [json.loads(line) for line in protocol.getvalue().splitlines()]
                context = next(row["context"] for row in rows if "context" in row)
                clock[0] += 0.0001
                payload = {
                    "operation": "stream_install",
                    "actions": trajectory(0.2 if failure == "splice" else 0.026).tolist(),
                    "receipt": receipt(context),
                    "terminal": True,
                }
            elif self.count == 4:
                payload = {"operation": "stream_status"}
            elif self.count == 5:
                payload = {"operation": "stream_stop"}
            else:
                payload = {"operation": "close"}
            if self.count == 1 and failure == "overshoot":
                raw = trajectory()
                raw[:, 0] = 0
                raw[1:3, 0] = 0.4
                payload["actions"] = raw.tolist()
            return json.dumps(payload) + "\n"

    def select_ready(*args):
        count = sys.stdin.count
        waiting = (failure == "late" and count == 2) or (count == 3 and clock[0] < 10 + 40 / hz + 0.05)
        if waiting:
            clock[0] += 0.01
            return [], [], []
        return [sys.stdin], [], []

    def validate_ranges(values, *, check_slew=True):
        if np.max(np.abs(values[:, 0])) > 0.400001:
            raise ValueError("cubic overshoot rejected")
        if check_slew and np.max(np.abs(np.diff(values, axis=0))) > 0.03:
            raise ValueError("future splice slew rejected")

    # No actual ROS, SDK, processes, network or hardware are used.
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        events = Path(directory) / "control.jsonl"
        ros = ModuleType("rclpy")
        ros.init = ros.shutdown = lambda: None
        ros.ok = lambda: True
        ros.create_node = lambda name: Node()
        executors = ModuleType("rclpy.executors")
        executors.MultiThreadedExecutor = Executor
        monkeypatch.setitem(sys.modules, "rclpy", ros)
        monkeypatch.setitem(sys.modules, "rclpy.executors", executors)
        for name, symbol in (("sensor_msgs", "JointState"), ("teleop_interfaces", "ArmCommand")):
            parent, child = ModuleType(name), ModuleType(name + ".msg")
            setattr(child, symbol, Message)
            parent.msg = child
            monkeypatch.setitem(sys.modules, name, parent)
            monkeypatch.setitem(sys.modules, name + ".msg", child)
        monkeypatch.setattr(worker, "ROSFeedbackProvider", Feedback)
        monkeypatch.setattr(worker, "SharpaSdkDirectExecutor", SDK)
        monkeypatch.setattr(worker, "validate_gateway_admission", lambda node: None)
        monkeypatch.setattr(worker, "validate_action_ranges", validate_ranges)
        monkeypatch.setattr(sdk_direct, "hand_joint_limits", lambda: np.tile([-1.0, 1.0], (22, 1)))
        monkeypatch.setattr(worker, "construct_messages", lambda frame: (SimpleNamespace(), None, None))
        monkeypatch.setattr(worker, "play_command_frames", lambda frames, send, feedback: [send(f) for f in frames])
        monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(worker.select, "select", select_ready)
        monkeypatch.delenv("SHARPA_PROTOCOL_FD", raising=False)
        monkeypatch.setattr(sys, "stdin", Input())
        monkeypatch.setattr(sys, "stdout", protocol)
        monkeypatch.setattr(sys, "argv", ["worker", "--enable-hardware"])
        if failure is None:
            worker.main()
        else:
            with pytest.raises(SystemExit):
                worker.main()
        rows = [json.loads(line) for line in protocol.getvalue().splitlines()]
        assert rows[-1]["operation"] == "shutdown"
        assert stopped == [True]
        if failure is None:
            assert all(row["status"] == "PASS" for row in rows)
            assert any(row.get("queued") for row in rows)
            final = next(row for row in rows if row.get("finished"))
            assert final["generation"] == 2
            events_rows = [json.loads(line) for line in events.read_text().splitlines()]
            frames = [row for row in events_rows if row["event"] == "frame"]
            expected = 20 * math.ceil(1 / scale)
            assert sum(row["generation"] == 1 for row in frames) == expected
            assert sum(row["generation"] == 2 for row in frames) == expected
            assert max(row["phase"] for row in frames) <= 25
            np.testing.assert_allclose(frames[-1]["command"], trajectory(0.026)[25], atol=1e-6)
        else:
            failed = next(row for row in rows if row["status"] == "FAIL")
            assert {"late": "handover", "splice": "splice", "overshoot": "overshoot"}[failure] in failed["reason"]
            if failure == "overshoot":
                assert sent == []


@pytest.mark.parametrize(
    "flags",
    [
        ["--execute-steps", "15"],
        ["--execute-steps", "20"],
        ["--soft-initial-delay-steps", "5"],
        ["--soft-delay-buffer-size", "10"],
        ["--compression", "off"],
    ],
)
def test_obsolete_async_schedule_cannot_silently_run(flags):
    process = subprocess.run(
        [sys.executable, "-m", "deploy.fr3_sharpa_runtime.async_soft_run", "--check", *flags],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 2
