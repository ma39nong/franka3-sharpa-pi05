"""Hardware-free integration of prepare, observations, policy and stream playback."""
from deploy.fr3_sharpa_runtime.protocol import DT_NS, FPS
import json
import sys
import time

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.action_adapter import SharpaActionAdapter
from deploy.fr3_sharpa_runtime.execution_trace import ExecutionTraceWriter
from deploy.fr3_sharpa_runtime.feedback import LiveFeedback
from deploy.fr3_sharpa_runtime.policy_prepare import SlowApproach, prepare_and_run_chunks
from deploy.fr3_sharpa_runtime.sdk_direct import chunk_command_frames, play_command_frames


class SimulatedRobot:
    def __init__(self):
        self.now = 1.0
        self.state = np.zeros(58, dtype=np.float32)
        self.operations = []
        self.sent = []

    def clock(self): return self.now
    def sleep(self, duration): self.now += duration
    def feedback(self):
        stamp = int(self.now * 1e9)
        return LiveFeedback(self.state.copy(), (stamp,) * 4, (stamp,) * 4)

    def observe(self, report=None):
        return {"observation/state": self.state.copy(), "prompt": "tomato",
                "_captured_monotonic_ns": int(self.now * 1e9)}

    def request(self, payload):
        self.operations.append(payload["operation"])
        if payload["operation"] == "prepare":
            planner = SlowApproach(payload["target"], self.state,
                                   speed_rad_s=payload["speed_rad_s"])
            for _ in range(10000):
                stamp = int(self.now * 1e9)
                self.state, settled, _ = planner.step(self.feedback(), now_ros_ns=stamp,
                    now_monotonic_ns=stamp, enforce_execute_delta=False)
                self.now += DT_NS/1e9
                if settled:
                    break
            else:
                pytest.fail("prepare did not settle")
        else:
            frames, _ = chunk_command_frames(payload["actions"], self.state,
                speed_scale=payload["execute_speed_scale"],
                execute_steps=payload.get("execute_steps", 50),
                approach_speed_rad_s=payload["approach_speed_rad_s"])
            def send(frame):
                self.state = frame.copy()
                self.sent.append(frame.copy())
            play_command_frames(frames, send, self.feedback, clock=self.clock, sleep=self.sleep)
        return {"status": "PASS", "published": True,
                "completed_ros_ns": int(self.now * 1e9),
                "completed_monotonic_ns": int(self.now * 1e9)}


def policy(obs):
    actions = np.repeat(obs["observation/state"][None, :], 50, axis=0)
    actions += np.linspace(0.005, 0.03, 50, dtype=np.float32)[:, None]
    chunk = SharpaActionAdapter().adapt_chunk(actions, start_timestamp_ns=1_000_000_000,
                                             session_id="test", source="openpi")
    return {"checkpoint": "test", "norm_stats": "test"}, actions, chunk


def test_current_state_restart_reinfers_measured_grasp_without_any_prepare(tmp_path):
    robot = SimulatedRobot()
    robot.state = np.linspace(0.1, 0.3, 58, dtype=np.float32)
    observed = []
    acquired = []
    def infer(obs):
        observed.append(obs["observation/state"].copy())
        return policy(obs)
    def acquire(completed):
        assert completed["status"] == "PASS"
        acquired.append(completed)
        return robot.observe()

    for attempt in range(2):
        start = robot.state.copy()
        initial = robot.observe()
        traces = ExecutionTraceWriter(tmp_path / str(attempt))
        report = prepare_and_run_chunks(infer, acquire, robot, initial,
            prepare_initial=False, num_chunks=2, execute_steps=15, trace_writer=traces)
        np.testing.assert_array_equal(observed[attempt * 2], start)
        assert report["prepare"] is None and report["initialization"] == "current"
        assert report["published"] is True and report["num_chunks"] == 2
        record = json.loads(open(traces.paths[0]).read())
        assert record["planned_execute_steps"] == 15
        assert json.loads(traces.run_path.read_text())["initialization"] == "current"
    assert robot.operations == ["chunk"] * 4
    assert len(observed) == 4  # No discarded startup prediction or Home inference.
    assert len(acquired) == 4  # Only post-chunk observations; never acquire_after(None).
    assert np.all(observed[2] > observed[0])  # Restart sees the new state, not the previous start.


def test_current_state_inference_failure_never_sends_a_target(tmp_path):
    robot = SimulatedRobot()
    traces = ExecutionTraceWriter(tmp_path)
    def fail(obs):
        raise RuntimeError("policy unavailable")
    with pytest.raises(RuntimeError, match="policy unavailable"):
        prepare_and_run_chunks(fail, robot.observe, robot, robot.observe(),
            prepare_initial=False, trace_writer=traces)
    assert robot.operations == []
    report = json.loads(traces.run_path.read_text())
    assert report["initialization"] == "current" and report["prepare"] is None
    assert report["status"] == "FAILED" and report["stage"] == "inference"


def test_live_execute_current_mode_never_loads_home_and_releases_worker(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from deploy.fr3_sharpa_runtime import run, motion_worker, home_pose
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "setup.bash").write_text("")
    robot = SimulatedRobot()
    robot.state[:] = 0.1
    closed, observations = [], []
    class Worker:
        def __init__(self, *args, **kwargs): pass
        def request(self, request):
            if request["operation"] == "ready":
                return {"operation": "ready", "status": "PASS"}
            return robot.request(request)
        def close(self): closed.append(True)
    class Policy:
        def __init__(self, *args): pass
        def infer_chunk(self, observation):
            observations.append(observation["observation/state"].copy())
            return policy(observation)
    def observe(*args):
        result = robot.observe()
        result["_captured_monotonic_ns"] = time.monotonic_ns()
        return result
    monkeypatch.setattr(motion_worker, "MotionWorker", Worker)
    monkeypatch.setattr(home_pose, "load_home", lambda *a, **k: pytest.fail("current mode must not load Home"))
    monkeypatch.setattr(run, "_start_observation_worker", lambda *a: None)
    monkeypatch.setattr(run, "live_observation", observe)
    monkeypatch.setattr(run, "observation_after_completion", observe)
    monkeypatch.setattr(run, "SharpaRuntime", Policy)
    monkeypatch.setattr(run.atexit, "register", lambda *a: None)
    args = SimpleNamespace(ros_overlay=overlay, sharpa_backend="sdk_direct", initialization="current",
        execution_trace_dir=tmp_path, policy_url="test", prompt="tomato", num_chunks=2,
        prepare_speed_rad_s=0.03, execute_speed_scale=1.0, execute_steps=15,
        home_poses=tmp_path / "absent.yaml", hand_home=tmp_path / "absent.json", home_task="powder_weighing")
    report = run.execute_policy_prepare(args)
    assert robot.operations == ["chunk", "chunk"] and len(observations) == 2
    np.testing.assert_allclose(observations[0], 0.1)
    assert report["initialization"] == "current" and report["prepare"] is None
    assert closed == [True]


@pytest.mark.parametrize("options, mode, hands", [
    ([], "current", "current_measured_positions"),
    (["--home"], "home", "fixed_named_hand_home"),
    (["--initialization", "home"], "home", "fixed_named_hand_home"),
    (["--initialization", "policy"], "policy", "policy_first_target"),
])
def test_start_mode_check_does_not_require_home_files(monkeypatch, capsys, options, mode, hands):
    from deploy.fr3_sharpa_runtime import run
    monkeypatch.setattr(sys, "argv", ["run", "--check", "--home-poses", "/absent/poses.yaml",
        "--hand-home", "/absent/hands.json", *options])
    run.main()
    report = json.loads(capsys.readouterr().out)
    assert report["initialization"] == mode and report["hand_initialization"] == hands


@pytest.mark.parametrize("options", [
    ["--home", "--initialization", "current"],
    ["--home", "--prepare-policy-first"],
    ["--initialization", "current", "--prepare-policy-first"],
    ["--home", "--sharpa-backend", "litchibot"],
])
def test_conflicting_start_modes_are_rejected_before_hardware(monkeypatch, options):
    from deploy.fr3_sharpa_runtime import run
    monkeypatch.setattr(sys, "argv", ["run", "--check", *options])
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2


def test_fixed_home_precedes_inference_and_each_round_executes_only_fifteen(tmp_path):
    robot = SimulatedRobot()
    home = np.full(58, 0.02, dtype=np.float32)
    observations = []
    traces = ExecutionTraceWriter(tmp_path)
    def infer(obs):
        observations.append(obs["observation/state"].copy())
        return policy(obs)
    report = prepare_and_run_chunks(infer, robot.observe, robot, robot.observe(),
        initial_target=home, home_metadata={"kind": "fixed_arm_home"},
        num_chunks=2, execute_steps=15, trace_writer=traces)
    increment = np.linspace(0.005, 0.03, 50, dtype=np.float32)[14]
    assert robot.operations == ["prepare", "chunk", "chunk"]
    assert len(observations) == 2  # No inference before Home.
    np.testing.assert_allclose(observations[0], home)
    np.testing.assert_allclose(observations[1], home + increment)
    np.testing.assert_allclose(robot.state, home + 2 * increment)
    assert report["execute_steps"] == 15
    for path in traces.paths:
        record = json.loads(open(path).read())
        assert len(record["policy_output_unnormalized"]) == 50
        assert record["planned_execute_steps"] == 15
        assert record["planned_action_indices"] == [0, 14]


def test_continuous_home_prefix_execution_reaches_five_rounds_and_stops_on_interrupt(tmp_path):
    robot = SimulatedRobot()
    traces = ExecutionTraceWriter(tmp_path)
    predictions = 0
    def infer(obs):
        nonlocal predictions
        if predictions == 5:
            raise KeyboardInterrupt()
        predictions += 1
        return policy(obs)
    with pytest.raises(KeyboardInterrupt):
        prepare_and_run_chunks(infer, robot.observe, robot, robot.observe(),
            initial_target=np.zeros(58), num_chunks=0, execute_steps=15, trace_writer=traces)
    assert robot.operations == ["prepare"] + ["chunk"] * 5
    report = json.loads(traces.run_path.read_text())
    assert report["num_chunks"] == 5 and report["status"] == "INTERRUPTED"
    assert len(traces.paths) == 5
    assert all(json.loads(open(path).read())["planned_execute_steps"] == 15 for path in traces.paths)
    increment = np.linspace(0.005, 0.03, 50, dtype=np.float32)[14]
    np.testing.assert_allclose(robot.state, 5 * increment)


def test_home_failure_prevents_any_policy_inference():
    robot = SimulatedRobot()
    def infer(obs):
        pytest.fail("No inference before Home is settled")
    class RejectHome:
        def request(self, payload):
            assert payload["operation"] == "prepare"
            return {"status": "FAIL"}
    with pytest.raises(RuntimeError, match="Prepare"):
        prepare_and_run_chunks(infer, robot.observe, RejectHome(), robot.observe(),
            initial_target=np.zeros(58), execute_steps=15)


def test_prepare_then_five_complete_chunks_reobserve_actual_end_state(tmp_path):
    robot = SimulatedRobot()
    traces = ExecutionTraceWriter(tmp_path)
    observations = []
    def infer(obs):
        observations.append(obs["observation/state"].copy())
        return policy(obs)
    report = prepare_and_run_chunks(infer, robot.observe, robot, robot.observe(),
        num_chunks=5, execute_speed_scale=0.1, trace_writer=traces)
    assert robot.operations == ["prepare"] + ["chunk"] * 5
    assert len(observations) == 6
    assert report["num_chunks"] == 5 and report["status"] == "COMPLETE"
    assert np.allclose(observations[1], 0.005)  # First inference only sets prepare.
    assert np.allclose(observations[-1], 0.125)
    assert np.allclose(robot.state, 0.155)
    assert len(traces.paths) == 5
    for index, path in enumerate(traces.paths):
        record = json.loads(open(path).read())
        assert record["status"] == "COMPLETE"
        assert record["chunk_index"] == index
        assert record["feedback_after"] is not None
        assert len(record["policy_output_unnormalized"]) == 50
    assert json.loads(traces.run_path.read_text())["num_chunks"] == 5


@pytest.mark.parametrize("prepare_initial", [True, False])
def test_continuous_mode_runs_beyond_two_chunks_and_records_interrupt(tmp_path, prepare_initial):
    robot = SimulatedRobot()
    traces = ExecutionTraceWriter(tmp_path)
    def infer(obs):
        if robot.operations.count("chunk") == 3:
            raise KeyboardInterrupt
        return policy(obs)
    with pytest.raises(KeyboardInterrupt):
        prepare_and_run_chunks(infer, robot.observe, robot, robot.observe(),
                               num_chunks=0, trace_writer=traces, prepare_initial=prepare_initial)
    report = json.loads(traces.run_path.read_text())
    assert report["status"] == "INTERRUPTED" and report["num_chunks"] == 3
    assert robot.operations == (["prepare"] if prepare_initial else []) + ["chunk"] * 3
    # A failure before the next chunk must not overwrite the last completed trace.
    assert all(json.loads(open(path).read())["status"] == "COMPLETE" for path in traces.paths)


def test_motion_failure_records_reason_and_stops_next_inference(tmp_path):
    robot = SimulatedRobot()
    traces = ExecutionTraceWriter(tmp_path)
    robot_request = robot.request
    # Preserve the real prepare implementation while failing the chunk IPC.
    class BrokenTransport:
        def request(self, payload):
            if payload["operation"] == "chunk":
                raise RuntimeError("feedback link lost")
            return robot_request(payload)
    with pytest.raises(RuntimeError, match="feedback link lost"):
        prepare_and_run_chunks(policy, robot.observe, BrokenTransport(), robot.observe(),
                               num_chunks=5, trace_writer=traces)
    assert len(traces.paths) == 1
    record = json.loads(open(traces.paths[0]).read())
    assert record["status"] == "FAILED" and record["failure_stage"] == "execute"
    assert json.loads(traces.run_path.read_text())["num_chunks"] == 0


@pytest.mark.parametrize("options, initialization", [
    ([], "current"), (["--home"], "home"),
    (["--initialization", "home"], "home"), (["--prepare-policy-first"], "policy"),
])
def test_execute_start_mode_accepts_continuous(monkeypatch, capsys, options, initialization):
    from deploy.fr3_sharpa_runtime import run
    captured = {}
    def execute(args):
        captured.update(vars(args))
        return {"status": "test"}
    monkeypatch.setattr(run, "execute_policy_prepare", execute)
    monkeypatch.setattr(sys, "argv", ["run", "--execute", "--enable-hardware",
        "--ros-overlay", "/test", "--num-chunks", "0", *options])
    run.main()
    assert captured["sharpa_backend"] == "sdk_direct"
    assert captured["prepare_policy_first"] is True
    assert captured["num_chunks"] == 0
    assert captured["execute_steps"] == 15
    assert captured["initialization"] == initialization


def test_initial_inference_failure_releases_sdk_worker(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from deploy.fr3_sharpa_runtime import run, motion_worker
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "setup.bash").write_text("")
    closed = []
    class Worker:
        def __init__(self, *args, **kwargs): pass
        def request(self, request): return {"operation": "ready", "status": "PASS"}
        def close(self): closed.append(True)
    class FailedPolicy:
        def __init__(self, *args): pass
        def infer_chunk(self, obs): raise RuntimeError("policy unavailable")
    monkeypatch.setattr(motion_worker, "MotionWorker", Worker)
    monkeypatch.setattr(run, "_start_observation_worker", lambda *args: None)
    monkeypatch.setattr(run, "live_observation", lambda *args: {})
    monkeypatch.setattr(run, "SharpaRuntime", FailedPolicy)
    monkeypatch.setattr(run.atexit, "register", lambda *args: None)
    args = SimpleNamespace(ros_overlay=overlay, sharpa_backend="sdk_direct", initialization="policy",
                           execution_trace_dir=tmp_path, policy_url="test", prompt="tomato")
    with pytest.raises(RuntimeError, match="policy unavailable"):
        run.execute_policy_prepare(args)
    assert closed == [True]
    report = json.loads(next(tmp_path.glob("*-run.json")).read_text())
    assert report["stage"] == "startup" and report["status"] == "FAILED"


def test_post_chunk_sensor_failure_does_not_hide_transmission_completion(tmp_path):
    robot = SimulatedRobot()
    traces = ExecutionTraceWriter(tmp_path)
    def observe(report):
        if robot.operations.count("chunk"):
            raise TimeoutError("camera stopped")
        return robot.observe(report)
    with pytest.raises(TimeoutError, match="camera stopped"):
        prepare_and_run_chunks(policy, observe, robot, robot.observe(),
                               num_chunks=5, trace_writer=traces)
    record = json.loads(open(traces.paths[0]).read())
    assert record["status"] == "COMPLETE"
    assert record["failure_stage"] == "post_chunk_observation"
    report = json.loads(traces.run_path.read_text())
    assert report["num_chunks"] == 1 and report["status"] == "FAILED"
