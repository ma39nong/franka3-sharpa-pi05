import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]


def test_ctrl_c_records_final_cleanup_even_when_console_reader_is_gone(tmp_path):
    script = """
import signal,time
def stop(number, frame):
    print("disable/readback completed", flush=True)
    raise SystemExit(130)
signal.signal(signal.SIGINT, stop)
print("ready", flush=True)
while True: time.sleep(.01)
"""
    path = tmp_path / "console.log"
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    recorder = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "deploy.fr3_sharpa_runtime.record_console",
            "--log",
            str(path),
            "--",
            sys.executable,
            "-u",
            "-c",
            script,
        ],
        cwd=ROOT,
        stdout=write_fd,
        stderr=subprocess.DEVNULL,
    )
    os.close(write_fd)
    try:
        deadline = time.monotonic() + 5
        while not path.exists() or "ready" not in path.read_text():
            if time.monotonic() >= deadline:
                raise AssertionError("fake runtime failed to become ready")
            time.sleep(0.01)
        recorder.send_signal(signal.SIGINT)
        assert recorder.wait(timeout=5) == 130
        assert "disable/readback completed" in path.read_text()
    finally:
        if recorder.poll() is None:
            recorder.kill()
            recorder.wait()


def test_existing_recording_is_never_overwritten(tmp_path):
    path = tmp_path / "console.log"
    path.write_text("old recording")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deploy.fr3_sharpa_runtime.record_console",
            "--log",
            str(path),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(99)",
        ],
        cwd=ROOT,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert result.returncode != 0
    assert path.read_text() == "old recording"
