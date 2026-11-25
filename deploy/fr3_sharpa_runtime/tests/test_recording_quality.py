"""No hardware: faulted recordings, durable ratings, HTTP and lifecycle races."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import types
import urllib.error
import urllib.request

import pytest

from deploy.fr3_sharpa_control import quality
from deploy.fr3_sharpa_control.recording import write_json
from deploy.fr3_sharpa_control.ui import server_for
from deploy.fr3_sharpa_runtime.tests.test_deployment_center import (
    FakeDeployment, checkpoint, repository, settings,
)


@pytest.fixture
def recorded(tmp_path):
    directory = tmp_path / "2026-11-25/10.30"
    (directory / "rosbag").mkdir(parents=True)
    (directory / "rosbag/metadata.yaml").write_text("test metadata")
    (directory / "rosbag/test.db3").write_bytes(b"preserve original bytes")
    write_json(directory / "recording.json", {
        "status": "COMPLETE", "started_unix_ns": 1, "ended_unix_ns": 2,
        "counts": {"/cam0/color/image_raw": 2},
    })
    quality.begin(directory)
    return tmp_path, directory


def seal(directory, outcome="FAILED", closed=True):
    quality.finish(directory, outcome=outcome, reason="safety limit exceeded",
                   returncode=1, recorder_closed=closed)


def rate(root, directory, grade, **fields):
    current = quality.read(directory)
    return quality.rate(str(root), directory.relative_to(root).as_posix(),
                        run_id=current["run_id"], grade=grade,
                        expected_revision=current["revision"], **fields)


@pytest.mark.parametrize("grade", list(quality.GRADES))
def test_failed_execution_can_be_rated_without_changing_failure_or_bag(recorded, grade):
    root, directory = recorded
    seal(directory)
    original = (directory / "recording.json").read_bytes()
    result = rate(root, directory, grade, note="开始执行后安全超限")
    assert result["execution"]["outcome"] == "FAILED"
    assert result["grade"] == grade and result["label"] == quality.GRADES[grade]
    assert result["training_validation"] == "UNVERIFIED" and result["warnings"]
    assert (directory / "rosbag/test.db3").read_bytes() == b"preserve original bytes"
    assert (directory / "recording.json").read_bytes() == original
    assert quality.list_recordings(root)[0]["grade"] == grade  # Fresh disk read after restart.


def test_rating_waits_for_lifecycle_owner_but_not_for_video_export(recorded):
    root, directory = recorded
    with pytest.raises(ValueError, match="尚未收尾"):
        rate(root, directory, "D")
    seal(directory, closed=False)
    with pytest.raises(ValueError, match="尚未收尾"):
        rate(root, directory, "D")
    seal(directory)
    write_json(directory / "video_export.json", {"status": "EXPORTING"})
    assert rate(root, directory, "D")["grade"] == "D"
    assert json.loads((directory / "video_export.json").read_text())["status"] == "EXPORTING"


def test_crashed_recorder_with_missing_metadata_and_report_remains_rateable(recorded):
    root, directory = recorded
    (directory / "rosbag/metadata.yaml").unlink()
    (directory / "recording.json").write_text("partial JSON")
    seal(directory)
    result = rate(root, directory, "C")
    assert result["recording_status"] == "UNKNOWN" and len(result["warnings"]) >= 3


def test_not_started_is_diagnostic_only_and_cannot_be_graded(recorded):
    root, directory = recorded
    import shutil
    shutil.rmtree(directory / "rosbag")
    write_json(directory / "recording.json", {"status": "NOT_STARTED"})
    seal(directory)
    assert not quality.summary(directory)["has_recording"]
    with pytest.raises(ValueError, match="没有录制数据"):
        rate(root, directory, "A")


def test_regrade_keeps_history_and_rejects_stale_browser_or_wrong_run(recorded):
    root, directory = recorded
    seal(directory)
    before = quality.read(directory)
    rate(root, directory, "C")
    rate(root, directory, "D", note="放弃但保留诊断")
    result = quality.read(directory)
    assert [event["grade"] for event in result["history"]] == ["C", "D"]
    for run_id, revision in ((before["run_id"], 0), ("another-run", 2)):
        with pytest.raises(ValueError, match="版本已变化"):
            quality.rate(root, "2026-11-25/10.30", run_id=run_id, expected_revision=revision, grade="A")
    assert quality.read(directory) == result


def test_simultaneous_reviews_cannot_silently_overwrite_each_other(recorded):
    root, directory = recorded
    seal(directory)
    original = quality.read(directory)
    def save(grade):
        try:
            quality.rate(root, "2026-11-25/10.30", run_id=original["run_id"],
                         expected_revision=0, grade=grade)
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(save, ["C", "D"])) == [False, True]
    assert len(quality.read(directory)["history"]) == 1


@pytest.mark.parametrize("relative", ["../10.30", "/tmp/2026-11-25/10.30", "2026-11-25/../10.30"])
def test_run_paths_cannot_escape_recording_parent(tmp_path, relative):
    with pytest.raises(ValueError):
        quality.resolve_run(tmp_path, relative)


def test_symlink_outside_parent_is_not_listed_or_rateable(recorded, tmp_path):
    root, directory = recorded
    other = tmp_path / "outside-parent"
    (other / "2026-11-25").mkdir(parents=True)
    (other / "2026-11-25/10.30").symlink_to(directory, target_is_directory=True)
    assert quality.list_recordings(other) == []
    with pytest.raises(ValueError, match="不在"):
        quality.resolve_run(other, "2026-11-25/10.30")


def test_http_review_uses_selected_run_and_survives_new_controller(recorded, tmp_path):
    root, directory = recorded
    seal(directory)
    server = server_for(tmp_path / "ui-root", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        token = json.load(urllib.request.urlopen(base + "/api/info"))["token"]
        def post(endpoint, data, authenticated=True):
            headers = {"Content-Type": "application/json"}
            if authenticated:
                headers["X-Control-Token"] = token
            request = urllib.request.Request(base + "/api/" + endpoint,
                data=json.dumps(data).encode(), headers=headers)
            return json.load(urllib.request.urlopen(request))
        rows = post("recordings", {"output": str(root)})
        row = rows[0]
        data = dict(output=str(root), relative=row["relative"], run_id=row["run_id"],
                    expected_revision=row["revision"], grade="D", note="早期中断")
        with pytest.raises(urllib.error.HTTPError) as error:
            post("rate-recording", data, authenticated=False)
        assert error.value.code == 403
        assert post("rate-recording", data)["grade"] == "D"
        assert server.controller.thread is None  # Review never starts hardware.
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert quality.list_recordings(root)[0]["grade"] == "D"


def test_early_safety_failure_seals_review_after_recorder_cleanup(repository, checkpoint, tmp_path, monkeypatch):
    from deploy.fr3_sharpa_control import controller as control
    class FailedRecording(FakeDeployment):
        def _spawn(self, command, name, env):
            if name == "recorder":
                self.socket.touch()
                self.recorder_closed = False
                (self.directory / "rosbag").mkdir()
                write_json(self.directory / "recording.json", {
                    "status": "RECORDING", "started_unix_ns": 1, "counts": {}})
                return types.SimpleNamespace(poll=lambda: 0 if self.recorder_closed else None)
            assert name == "runtime"
            (self.directory / "runtime.log").write_text("ValueError: arm target exceeds safety limit\n")
            return types.SimpleNamespace(returncode=1, poll=lambda: 1)
        def _stop_recorder(self):
            assert not quality.summary(self.directory)["can_rate"]
            self.recorder_closed = True
            write_json(self.directory / "recording.json", {
                "status": "INCOMPLETE", "started_unix_ns": 1, "ended_unix_ns": 2, "counts": {}})
            self.recorder.wait = lambda **kwargs: 0
    monkeypatch.setattr(control, "rpc", lambda *a, **kw: {"status": "PASS"})
    owner = FailedRecording(repository)
    values = settings(repository, checkpoint, output=str(tmp_path / "records"), record=True).__dict__
    owner.ready(values)
    owner.start(values)
    owner.thread.join(timeout=5)
    assert owner.state["phase"] == "failed"
    result = quality.summary(owner.directory)
    assert result["can_rate"] and result["execution"]["outcome"] == "FAILED"
    assert "safety limit" in result["execution"]["reason"]
    assert "hardware_cleanup" in owner.commands


@pytest.mark.parametrize("status,expected", [(1, "FAILED"), (0, "FINISHED"), (130, "STOPPED")])
def test_terminal_recorder_also_persists_outcome_for_ui_rating(tmp_path, monkeypatch, status, expected):
    from deploy.fr3_sharpa_control import record_cli
    directory = tmp_path / "2026-11-25/11.20"
    directory.mkdir(parents=True)
    class Recorder:
        closed = False
        def __init__(self, *args, env, **kwargs):
            Path(env["SHARPA_RECORD_SOCKET"]).touch()
            (directory / "rosbag").mkdir()
            write_json(directory / "recording.json", {"status": "RECORDING", "started_unix_ns": 1})
        def poll(self):
            return 0 if self.closed else None
        def wait(self, **kwargs):
            self.closed = True
            write_json(directory / "recording.json", {
                "status": "INCOMPLETE", "started_unix_ns": 1, "ended_unix_ns": 2})
            return 0
    monkeypatch.setenv("SHARPA_RECORD_ENABLED", "1")
    monkeypatch.setattr(record_cli, "desktop", lambda: tmp_path)
    monkeypatch.setattr(record_cli, "allocate_run", lambda *args: directory)
    monkeypatch.setattr(record_cli.subprocess, "check_output", lambda *a, **kw: "test-commit")
    monkeypatch.setattr(record_cli.subprocess, "run", lambda *a, **kw: types.SimpleNamespace(returncode=0))
    monkeypatch.setattr(record_cli.subprocess, "Popen", Recorder)
    monkeypatch.setattr(record_cli, "rpc", lambda *a, **kw: {})
    monkeypatch.setattr(record_cli, "record_command", lambda *a, **kw: status)
    monkeypatch.setattr("sys.argv", ["record_cli", "--", "synthetic-runtime"])
    with pytest.raises(SystemExit) as error:
        record_cli.main()
    assert error.value.code == status
    result = quality.summary(directory)
    assert result["can_rate"] and result["execution"]["outcome"] == expected
    assert rate(tmp_path, directory, "D")["grade"] == "D"
