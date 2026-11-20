import json

import numpy as np
import pytest

from deploy.fr3_sharpa_control.qwen_memory import TaskMemory
from deploy.fr3_sharpa_control.qwen_observer import FrameBuffer
from deploy.fr3_sharpa_control.qwen_observer import read_frames


def history(tmp_path):
    path = tmp_path / "observation.npz"
    np.savez(path, rgb=np.zeros((3, 16, 256, 256, 3), dtype=np.uint8))
    stamps = [int((98.5 + i / 10) * 1e9) for i in range(16)]
    data = {
        "valid": True,
        "captured_unix": 100.0,
        "timestamp": 100.0,
        "ros_now": 100.0,
        "source_age_s": 0.0,
        "observation_npz": str(path),
        "history_source_ns": dict.fromkeys(("cam0", "cam1", "cam2"), stamps),
    }
    (tmp_path / "latest.json").write_text(json.dumps(data))
    return data


def test_source_identity_and_rgb_jpeg_without_retimestamping(tmp_path):
    history(tmp_path)
    frames = read_frames(tmp_path, now=100.1)
    assert len(frames) == 8
    assert {frame["camera"] for frame, _ in frames} == {"cam0", "cam2"}
    assert min(frame["timestamp"] for frame, _ in frames) == 98.5
    assert all(jpeg.startswith(b"\xff\xd8") for _, jpeg in frames)


@pytest.mark.parametrize("fault", ["stale", "clock", "old_images", "duplicate"])
def test_invalid_histories_are_not_sent(tmp_path, fault):
    data = history(tmp_path)
    if fault == "stale":
        data["captured_unix"] = 90
    elif fault == "clock":
        data["ros_now"] = 90
    elif fault == "old_images":
        data["history_source_ns"]["cam2"] = [t - 10_000_000_000 for t in data["history_source_ns"]["cam2"]]
    else:
        data["history_source_ns"]["cam2"][5] = data["history_source_ns"]["cam2"][4]
    (tmp_path / "latest.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match=".+"):
        read_frames(tmp_path, now=100.1)


def test_buffer_preserves_original_ids_and_filters_before_applied_stage(tmp_path):
    history(tmp_path)
    buffer = FrameBuffer()
    for frame, jpeg in read_frames(tmp_path, now=100.1):
        buffer.frames[frame["id"]] = (frame, jpeg)
    memory = TaskMemory("run")
    memory.acknowledge("harvest_1", run_id="run", expected_revision=0, applied_at=99)
    frames, images = buffer.window(memory)
    assert all(frame["timestamp"] >= 99 for frame in frames)
    assert all(frame["id"] in images for frame in frames)


def test_live_worker_reads_task_without_renewing_contact_or_sending_commands(tmp_path, monkeypatch):
    import sys
    import time

    from deploy.fr3_sharpa_control import qwen_observer

    calls = []
    now = time.time()

    def rpc(path, run_id, **kwargs):
        calls.append(kwargs)
        return {
            "task_profile": "tomato-in-hand119",
            "execution_state": "RUNNING",
            "active_stage": "pickup",
            "requested_stage": "pickup",
            "stage_version": 0,
            "active_applied_unix_s": now - 2,
        }

    def collect(self, directory, stop):
        with self.lock:
            for index in range(4):
                frame = {"id": str(index), "camera": "cam0", "timestamp": now - 1.5 + index * 0.5}
                self.frames[frame["id"]] = (frame, b"\xff\xd8fixture")
        stop.wait(5)

    monkeypatch.setattr(qwen_observer, "task_rpc", rpc)
    monkeypatch.setattr(FrameBuffer, "collect", collect)
    monkeypatch.setattr(
        qwen_observer,
        "recognize",
        lambda payload: {
            "answer": {
                "events": [],
                "ready": True,
                "ready_evidence": ["2", "3"],
                "left_empty": True,
                "counting_complete": True,
                "uncertainties": [],
            },
            "elapsed_s": 0.01,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "observer",
            "--history-directory",
            str(tmp_path),
            "--task-socket",
            "unused",
            "--run-id",
            "run",
            "--output",
            str(tmp_path / "out"),
            "--max-requests",
            "1",
        ],
    )
    qwen_observer.main()
    state = json.loads((tmp_path / "out/status.json").read_text())
    assert state["control_eligible"] is False
    assert state["evaluation"]["proposed_stage"] == "harvest_1"
    assert state["evaluation"]["memory"]["active_stage"] == "pickup"
    assert calls == [{"renew_contact": False}, {"renew_contact": False}]
