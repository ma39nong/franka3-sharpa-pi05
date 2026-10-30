"""No hardware: identity evidence, eight recipes, lifecycle, gates and HTTP controls."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import types
import urllib.error
import urllib.request

import numpy as np
import pytest

from deploy.fr3_sharpa_control import controller as control
from deploy.fr3_sharpa_control import plan as plans
from deploy.fr3_sharpa_control import recording
from deploy.fr3_sharpa_control.recorder_ros import BagCapture, TOPICS


@pytest.fixture
def checkpoint(tmp_path):
    root = tmp_path / "delta_named_but_not_evidence/5999"
    params = root / "params"
    params.mkdir(parents=True)
    entries = {}
    for name, kind, shape in (("action_in_proj", "kernel", [58, 128]), ("action_out_proj", "bias", [58])):
        entries[name] = {"key_metadata": [{"key": name}, {"key": kind}], "value_metadata": {"write_shape": shape}}
    (params / "_METADATA").write_text(json.dumps({"tree_metadata": entries}))
    (params / "manifest.ocdbt").write_bytes(b"synthetic identity, never executable weights")
    stats = root / "assets/stats/norm_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text(json.dumps({"norm_stats": {g: {"mean": [0.] * 58, "std": [1.] * 58,
        "q01": [-2.] * 58, "q99": [2.] * 58} for g in ("state", "actions")}}))
    return root


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repository with spaces"
    shared, full = root / "deploy/fr3_sharpa_runtime", root / "deploy/fr3_sharpa_full"
    shared.mkdir(parents=True)
    full.mkdir()
    for file in (shared / "serve.sh", shared / "runtime.sh", full / "serve_soft.sh",
                 full / "serve_async_soft.sh", full / "runtime_async_soft.sh"):
        file.write_text("# Not an executable deployment; recipe tests only\n")
    for file in (shared / "sdk_direct_motion_ros.py", shared / "async_soft_motion_ros.py"):
        file.write_text("# RecordingGate integration marker\n")
    (shared / "async_soft_stream.py").write_text("CAPABILITY = 'sharpa-async-soft-rolling-v3'\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fixture"], check=True)
    return root


def settings(repository, checkpoint, **kwargs):
    return plans.DeploymentSettings(str(repository), str(checkpoint), semantics="absolute", **kwargs)


def test_names_never_determine_semantics_and_binding_tracks_content(checkpoint, tmp_path):
    found = plans.inspect_checkpoint(checkpoint)
    assert found["semantics"] is None and found["profile"] is None
    confirmed = plans.inspect_checkpoint(checkpoint, semantics="absolute")
    cache = tmp_path / "bindings"
    cache.mkdir()
    (cache / f"{confirmed['identity']}.json").write_text('{"semantics":"absolute"}')
    assert plans.inspect_checkpoint(checkpoint, cache=cache)["semantics"] == "absolute"
    assert plans.inspect_checkpoint(checkpoint, cache=cache, semantics="delta")["profile"] == "full-delta"
    (checkpoint / "params/manifest.ocdbt").write_bytes(b"replaced weights")
    assert plans.inspect_checkpoint(checkpoint, cache=cache)["semantics"] is None


def test_training_record_digest_and_model_must_match(checkpoint, tmp_path):
    stats = next((checkpoint / "assets").rglob("norm_stats.json"))
    record = {"config": {"fields": {"model": {"fields": {"action_dim": 58, "action_horizon": 50,
        "max_token_len": 256, "paligemma_variant": "gemma_2b", "action_expert_variant": "gemma_300m"}},
        "data": {"fields": {"action_hz": 30, "extra_delta_transform": True}}}},
        "inputs": {"norm_stats_sha256": plans.fingerprint(stats)}}
    path = tmp_path / "training.json"
    path.write_text(json.dumps(record))
    found = plans.inspect_checkpoint(checkpoint, training_record=path)
    assert found["profile"] == "full-delta"
    with pytest.raises(ValueError, match="冲突"):
        plans.inspect_checkpoint(checkpoint, training_record=path, semantics="absolute")
    record["inputs"]["norm_stats_sha256"] = "wrong"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="摘要"):
        plans.inspect_checkpoint(checkpoint, training_record=path)


@pytest.mark.parametrize("semantics", ["absolute", "delta"])
@pytest.mark.parametrize("mode", ["serial", "rtc", "soft-serial", "soft-async"])
def test_all_eight_recipes_preserve_mode_meaning(repository, checkpoint, semantics, mode):
    steps = 26 if mode.startswith("soft-") else 20
    config = plans.DeploymentSettings(str(repository), str(checkpoint), semantics=semantics, mode=mode,
                                      speed="fast" if mode == "rtc" else "slow", steps=steps)
    plan = config.plan()
    assert plan["model"]["profile"] == ("full" if semantics == "absolute" else "full-delta")
    assert plan["execute"][plan["execute"].index("--execute-steps") + 1] == str(steps)
    assert plan["port"] == (18765 if semantics == "absolute" else 18767) + (100 if mode == "soft-async" else 0)
    assert "--enable-rtc" in plan["service"] if mode == "rtc" else "--enable-rtc" not in plan["service"]
    if mode == "soft-serial":
        assert "cubic" in plan["execute"] and "paper-rolling" in plan["execute"]
    if mode == "soft-async":
        assert "runtime_async_soft.sh" in plan["execute"][1] and "15" not in plan["execute"]


@pytest.mark.parametrize("options", [{"mode": "serial", "steps": 51}, {"mode": "rtc", "speed": "slow"},
    {"mode": "rtc", "speed": "fast", "rounds": 1}, {"mode": "soft-async", "steps": 20}])
def test_incompatible_configuration_never_generates_execution(repository, checkpoint, options):
    with pytest.raises(ValueError):
        settings(repository, checkpoint, **options).plan()


def test_old_async_fifteen_node_protocol_is_rejected(repository, checkpoint):
    (repository / "deploy/fr3_sharpa_runtime/async_soft_stream.py").write_text("sharpa-async-soft-v1")
    with pytest.raises(ValueError, match="rolling-v3"):
        settings(repository, checkpoint, mode="soft-async", steps=26).plan()


def test_atomic_numbering_does_not_overwrite_runs(tmp_path):
    now = datetime(2026, 10, 30)
    with ThreadPoolExecutor(8) as pool:
        paths = list(pool.map(lambda _: plans.allocate_run(tmp_path, now=now), range(16)))
    assert len(set(paths)) == 16
    assert (tmp_path / "2026-10-30/01").is_dir()
    assert (tmp_path / "2026-10-30/16").is_dir()


def test_record_gate_starts_after_bridge_before_model_frames(monkeypatch):
    order = []
    monkeypatch.setattr(recording, "rpc", lambda path, op, **kw: order.append(op))
    gate = recording.RecordingGate("fake-socket")
    def play(frames, send, feedback):
        order.append(list(frames))
    recording.play_task_frames([0, 1, 2, 3], 2, None, None, gate,
                               resume=lambda: order.append("resume"), play=play)
    recording.play_task_frames([4, 5], 0, None, None, gate,
                               resume=lambda: order.append("resume"), play=play)
    assert order == [[0, 1], "start", "resume", [2, 3], [4, 5]]
    gate.stop()
    assert order[-1] == "stop"


def test_disabled_gate_preserves_existing_single_play_call(monkeypatch):
    monkeypatch.delenv("SHARPA_RECORD_SOCKET", raising=False)
    calls = []
    recording.play_task_frames([0, 1, 2], 2, None, None, recording.RecordingGate(),
                              resume=lambda: pytest.fail("disabled gate must not reset stream"),
                              play=lambda frames, *_: calls.append(frames))
    assert calls == [[0, 1, 2]]


def test_recorded_target_publisher_uses_canonical_runtime_protocol(monkeypatch):
    from deploy.fr3_sharpa_runtime.protocol import SHARPA_JOINT_NAMES
    messages, publishers = [], []
    class JointState:
        def __init__(self):
            self.header = types.SimpleNamespace(stamp=types.SimpleNamespace(sec=0, nanosec=0))
    monkeypatch.setitem(sys.modules, "sensor_msgs", types.ModuleType("sensor_msgs"))
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", types.SimpleNamespace(JointState=JointState))
    def create(kind, topic, depth):
        publishers.append((kind, topic, depth))
        return types.SimpleNamespace(publish=messages.append)
    node = types.SimpleNamespace(create_publisher=create)
    values = np.arange(58, dtype=float) / 100
    recording.command_publisher(node, recording.RecordingGate("unused-socket"))(values, 1_234_567_890)
    assert publishers == [(JointState, "/openpi/executed_joint_targets", 30)]
    assert len(messages) == 1
    message = messages[0]
    expected = ([f"left_fr3_joint{i}" for i in range(1, 8)] + [f"left_{n}" for n in SHARPA_JOINT_NAMES]
                + [f"right_fr3_joint{i}" for i in range(1, 8)] + [f"right_{n}" for n in SHARPA_JOINT_NAMES])
    assert message.name == expected and len(message.name) == 58
    assert message.position == list(values)
    assert (message.header.stamp.sec, message.header.stamp.nanosec) == (1, 234_567_890)


def test_disabled_recording_does_not_construct_target_publisher(monkeypatch):
    monkeypatch.delenv("SHARPA_RECORD_SOCKET", raising=False)
    def create(*args):
        pytest.fail("Recording disabled must not create a publisher")
    publish = recording.command_publisher(types.SimpleNamespace(create_publisher=create), recording.RecordingGate())
    publish([0.] * 58, 1)


def test_bag_writer_is_not_opened_during_prepare_and_closes_cleanly(tmp_path, monkeypatch):
    writers = []
    class Writer:
        def __init__(self): self.rows = []; self.closed = False; writers.append(self)
        def open(self, *a): pass
        def create_topic(self, *a): pass
        def write(self, *args): self.rows.append(args)
        def close(self): self.closed = True
    module = types.SimpleNamespace(SequentialWriter=Writer, StorageOptions=lambda **kw: kw,
        ConverterOptions=lambda *a: a, TopicMetadata=lambda **kw: kw)
    monkeypatch.setitem(sys.modules, "rosbag2_py", module)
    bag = BagCapture(tmp_path)
    for topic in TOPICS:
        bag.accept(topic, b"prepare", 1)
    assert not writers and not any(bag.counts.values())
    bag.request("start")
    for topic in TOPICS:
        bag.accept(topic, b"task", 2)
    bag.request("stop")
    assert writers[0].closed and len(writers[0].rows) == len(TOPICS)
    assert all(row[1] == b"task" for row in writers[0].rows)
    assert json.loads((tmp_path / "recording.json").read_text())["status"] == "COMPLETE"


def test_stop_forwards_once_and_never_kills_cleanup(tmp_path, monkeypatch):
    events = []
    controller = control.DeploymentController(tmp_path)
    controller.state["running"] = True
    controller.runtime = object()
    monkeypatch.setattr(control, "interrupt", lambda child: events.append(child))
    controller.stop()
    controller.stop()
    assert events == [controller.runtime]
    assert controller.state["phase"] == "stopping" and controller.state["running"]


class FakeDeployment(control.DeploymentController):
    """No commands can escape this test: all launch operations are overridden."""
    def __init__(self, root): super().__init__(root); self.commands = []; self.abort_model = False
    def _command(self, command, name, env, **kwargs): self.commands.append(name)
    def _prepare_model(self, env):
        self.commands.append("model_ready")
        if self.abort_model: self.cancel.set()
    def _spawn(self, command, name, env):
        assert name == "runtime"  # Tests use recording=False.
        self.commands.append(name)
        trace = Path(env["SHARPA_SHUTDOWN_REPORT_DIR"])
        trace.mkdir(parents=True)
        (trace / "shutdown-test.json").write_text('{"status":"PASS"}')
        return types.SimpleNamespace(returncode=0, poll=lambda: 0)


def test_each_restart_includes_up_and_new_initialization(repository, checkpoint, tmp_path, monkeypatch):
    from deploy.fr3_sharpa_runtime import home_pose
    monkeypatch.setattr(home_pose, "load_home", lambda *a, **kw: ({}, {}))
    (repository / ".deployment").mkdir(exist_ok=True)
    for filename in ("operator_poses.yaml", "hand_home.json"):
        (repository / ".deployment" / filename).write_text("mock Home contents for lifecycle test")
    controller = FakeDeployment(repository)
    for initialization in ("home", "current"):
        values = settings(repository, checkpoint, output=str(tmp_path / "records"), initialization=initialization).__dict__
        controller.start(values)
        controller.thread.join(timeout=5)
        assert controller.state["phase"] == "finished"
        assert controller.plan["settings"]["initialization"] == initialization
    assert controller.commands.count("hardware") == 2
    assert controller.commands.count("runtime") == 2


def test_cancel_during_model_start_never_launches_motion(repository, checkpoint, tmp_path):
    controller = FakeDeployment(repository)
    controller.abort_model = True
    controller.start(settings(repository, checkpoint, output=str(tmp_path / "records")).__dict__)
    controller.thread.join(timeout=5)
    assert "runtime" not in controller.commands and "hardware_cleanup" in controller.commands
    assert controller.state["phase"] == "stopped"


def test_early_runtime_exit_without_disable_report_retries_hardware_cleanup(repository, checkpoint, tmp_path):
    class InterruptedBeforeWorker(FakeDeployment):
        def _spawn(self, *args):
            return types.SimpleNamespace(returncode=130, poll=lambda: 130)
    controller = InterruptedBeforeWorker(repository)
    controller.start(settings(repository, checkpoint, output=str(tmp_path / "records")).__dict__)
    controller.thread.join(timeout=5)
    assert controller.state["phase"] == "failed"
    assert "失能PASS" in controller.state["message"]
    assert "hardware_cleanup" in controller.commands


def test_changed_checkpoint_after_preview_is_rejected_before_start(repository, checkpoint):
    controller = FakeDeployment(repository)
    values = settings(repository, checkpoint).__dict__
    approved = controller.preview(values)["approval"]
    (checkpoint / "params/manifest.ocdbt").write_bytes(b"changed after preview")
    with pytest.raises(ValueError, match="核对后"):
        controller.start(values, expected_approval=approved)
    assert controller.thread is None and controller.commands == []


def test_current_does_not_need_home_but_home_is_checked_before_up(repository, checkpoint):
    config = settings(repository, checkpoint)
    assert config.plan()["settings"]["initialization"] == "current"
    with pytest.raises((ValueError, FileNotFoundError)):
        settings(repository, checkpoint, initialization="home").plan()


def test_stop_marker_prevents_more_writes_without_waiting_for_disable(tmp_path):
    marker = tmp_path / "stop-recording"
    capture = BagCapture(tmp_path, stop_marker=marker)
    calls = []
    capture.writer = types.SimpleNamespace(write=lambda *a: calls.append(a))
    capture.active = True
    capture.accept("/teleop/arm_commands", b"before stop", 1)
    marker.touch()
    capture.accept("/teleop/arm_commands", b"during hand disable", 2)
    assert len(calls) == 1 and capture.finished.is_set() and not capture.active


def test_separate_checkout_reuses_machine_home_and_python(repository, checkpoint, tmp_path, monkeypatch):
    machine = tmp_path / "workstation-main"
    home = machine / ".deployment"
    home.mkdir(parents=True)
    (home / "operator_poses.yaml").write_text("not loaded in current mode")
    (home / "hand_home.json").write_text("not loaded in current mode")
    (home / "sharpa_ros_messages/install").mkdir(parents=True)
    original = plans.fingerprint
    def fingerprint(path):
        assert Path(path).name not in ("operator_poses.yaml", "hand_home.json"), "Current must not read Home"
        return original(path)
    monkeypatch.setattr(plans, "fingerprint", fingerprint)
    monkeypatch.delenv("SHARPA_PYTHON", raising=False)
    monkeypatch.delenv("SHARPA_ROS_MESSAGES_OVERLAY", raising=False)
    plan = settings(repository, checkpoint).plan(machine_root=machine)
    env = plans.plan_environment(plan)
    assert env["SHARPA_PYTHON"] == sys.executable
    assert env["SHARPA_ROS_MESSAGES_OVERLAY"] == str(home / "sharpa_ros_messages/install")
    assert plan["home_files"]["--hand-home"] == str(home / "hand_home.json")


def test_old_video_export_keeps_its_directory_while_next_run_starts(repository, checkpoint, tmp_path):
    script = repository / "deploy/fr3_sharpa_control/record_session.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text('printf \'{"status":"COMPLETE"}\\n\' > "$2/video_export.json"\n')
    controller = FakeDeployment(repository)
    controller.plan = settings(repository, checkpoint).plan()
    old = tmp_path / "previous recording"
    old.mkdir()
    controller._queue_export(old, dict(os.environ))
    controller.start(settings(repository, checkpoint, output=str(tmp_path / "new runs")).__dict__)
    controller.thread.join(timeout=5)
    controller.wait_exports()
    assert controller.state["phase"] == "finished"
    assert json.loads((old / "video_export.json").read_text())["status"] == "COMPLETE"
    assert not (controller.directory / "video_export.json").exists()


def test_web_ui_get_does_not_launch_and_cross_origin_controls_are_rejected(tmp_path):
    from deploy.fr3_sharpa_control.ui import server_for
    server = server_for(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        info = json.load(urllib.request.urlopen(url + "/api/info"))
        state = json.load(urllib.request.urlopen(url + "/api/status"))
        assert not state["running"] and server.controller.thread is None
        request = urllib.request.Request(url + "/api/start", data=b"{}", headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 403
        request.add_header("X-Control-Token", info["token"])
        request.add_header("Origin", "http://unrelated.invalid")
        with pytest.raises(urllib.error.HTTPError): urllib.request.urlopen(request)
        assert server.controller.thread is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_video_time_axis_holds_frames_instead_of_zipping_frame_numbers():
    from deploy.fr3_sharpa_control.export_video import timed_samples
    events = [("head", 0, "h0"), ("left_wrist", 0, "l0"), ("right_wrist", 0, "r0"),
              ("head", 60_000_000, "h1"), (None, 110_000_000, None)]
    frames = list(timed_samples(events, fps=20))
    assert [round(t, 2) for t, _ in frames] == [0, 0.05, 0.1]
    assert frames[1][1]["head"] == "h0" and frames[2][1]["head"] == "h1"
    assert all(f["left_wrist"] == "l0" for _, f in frames)


def test_real_video_encoder_produces_three_synchronized_panels(tmp_path, monkeypatch):
    import cv2
    from deploy.fr3_sharpa_control.export_video import export
    events = []
    for index, topic in enumerate(("/cam0/color/image_raw", "/cam2/color/image_raw", "/cam1/color/image_raw")):
        image = np.zeros((24, 32, 3), np.uint8)
        image[:, :, index] = 220
        message = types.SimpleNamespace(encoding="bgr8", height=24, width=32, step=96, data=image.tobytes())
        events.append((topic, message, 0))
    events.append(("/teleop/arm_commands", b"ignored", 100_000_000))
    class Reader:
        def __init__(self): self.index = 0
        def open(self, *a): pass
        def has_next(self): return self.index < len(events)
        def read_next(self):
            event = events[self.index]
            self.index += 1
            return event
    monkeypatch.setitem(sys.modules, "rosbag2_py", types.SimpleNamespace(SequentialReader=Reader,
        StorageOptions=lambda **kw: kw, ConverterOptions=lambda *a: a))
    monkeypatch.setitem(sys.modules, "rclpy.serialization", types.SimpleNamespace(deserialize_message=lambda data, cls: data))
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", types.SimpleNamespace(Image=object))
    report = export(tmp_path, fps=20, width=160, height=120)
    assert report["status"] == "COMPLETE" and report["frames"] == 3
    video = cv2.VideoCapture(str(tmp_path / "video/three_cameras.mp4"))
    success, frame = video.read()
    video.release()
    assert success and frame.shape == (120, 480, 3)
    assert np.argmax(frame[70, 80]) == 0  # Head/left/right are distinct panels, not mislabeled duplicates.
    assert np.argmax(frame[70, 240]) == 1
    assert np.argmax(frame[70, 400]) == 2
