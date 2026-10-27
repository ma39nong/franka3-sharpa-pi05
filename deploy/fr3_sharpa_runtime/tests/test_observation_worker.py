import json
import sys
import time

import pytest

from deploy.fr3_sharpa_runtime.observation_worker import ObservationWorker


def test_worker_keeps_one_process_for_multiple_snapshots(tmp_path):
    code = """import sys,json,os
for line in sys.stdin:
 request=json.loads(line)
 if request.get('command')=='close':break
 print(json.dumps({'pid':os.getpid()}),flush=True)
"""
    worker = ObservationWorker([sys.executable, "-u", "-c", code], cwd=tmp_path)
    try:
        first = json.loads(worker.snapshot())
        second = json.loads(worker.snapshot())
        assert first["pid"] == second["pid"] == worker.process.pid
        assert worker.process.poll() is None
    finally:
        worker.close()
    assert worker.process.returncode == 0


def test_progress_messages_are_not_used_as_snapshot(tmp_path):
    code = """import sys,json
for line in sys.stdin:
 request=json.loads(line)
 if request.get('command')=='close':break
 print(json.dumps({'event':'observation_wait','topics':{'cam0':{'missing':True}}}),flush=True)
 print(json.dumps({'state':[1]}),flush=True)
"""
    worker = ObservationWorker([sys.executable, "-u", "-c", code], cwd=tmp_path)
    diagnostics, polls = [], []
    try:
        result = json.loads(worker.snapshot(timeout_s=1, on_diagnostic=diagnostics.append,
                                            on_poll=lambda: polls.append(1)))
        assert result == {"state": [1]}
        assert diagnostics[0]["topics"]["cam0"]["missing"] and polls
    finally:
        worker.close()


@pytest.mark.parametrize("motion_failure", [False, True])
def test_stuck_observer_stops_promptly_and_cannot_reuse_late_reply(tmp_path, motion_failure):
    code = """import sys,json,time
for line in sys.stdin:
 time.sleep(10)
 print(json.dumps({'state':[1]}),flush=True)
"""
    worker = ObservationWorker([sys.executable, "-u", "-c", code], cwd=tmp_path)
    started = time.monotonic()
    def poll():
        if motion_failure:
            raise RuntimeError("device failed first")
    try:
        with pytest.raises(RuntimeError if motion_failure else TimeoutError,
                           match="device failed first" if motion_failure else "timed out"):
            worker.snapshot(timeout_s=0.1, on_poll=poll)
        assert time.monotonic() - started < 1
        worker.process.wait(timeout=1)
        with pytest.raises(RuntimeError, match="exited"):
            worker.snapshot(timeout_s=0.1)
    finally:
        worker.close()


def test_motion_worker_reads_original_failure_after_child_exit(tmp_path):
    from deploy.fr3_sharpa_runtime.motion_worker import MotionWorker
    code = "import json; print(json.dumps({'status':'FAIL','reason':'RTC accepted chunk exhausted'}),flush=True)"
    worker = MotionWorker([sys.executable, "-u", "-c", code], cwd=tmp_path)
    try:
        worker.process.wait(timeout=2)
        with pytest.raises(RuntimeError, match="RTC accepted chunk exhausted"):
            worker.request({"operation": "stream_status"})
    finally:
        worker.close()
