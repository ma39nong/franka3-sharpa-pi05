"""Read-only camera watchdog. Loss interrupts execution, never restarts a driver."""
# ruff: noqa: RUF001 -- Chinese operator diagnostics use Chinese punctuation.

from contextlib import suppress
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time


class CameraMonitor:
    def __init__(self, config):
        self.config = str(config)
        self.process = None
        self.last_seen = None
        self.pending = b""
        self.eof = False

    def _reports(self):
        reports = []
        while select.select([self.process.stdout], [], [], 0)[0]:
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data:
                self.eof = True
                break
            self.pending += data
            if len(self.pending) > 1048576:
                raise ValueError("相机监控应答超长")
            while b"\n" in self.pending:
                line, self.pending = self.pending.split(b"\n", 1)
                reports.append(json.loads(line))
        return reports

    def start(self):
        script = Path(__file__).with_name("camera_probe.sh")
        self.process = subprocess.Popen(
            ["bash", str(script), self.config, "--watch"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            start_new_session=True,
        )
        deadline = time.monotonic() + 55
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("执行前相机验证失败；不会连接SDK或Home，详见相机日志")
                if not select.select([self.process.stdout], [], [], 0.1)[0]:
                    continue
                for report in self._reports():
                    if report.get("status") == "PASS":
                        self.last_seen = report["checked_monotonic_ns"] / 1e9
                        if 0 <= time.monotonic() - self.last_seen < 0.75:
                            return self
                if self.eof:
                    raise RuntimeError("相机监控进程提前关闭输出")
            raise TimeoutError("执行前相机监控启动超时")
        except BaseException:
            self.close()
            raise

    def healthy(self):
        if self.process is None or self.process.poll() is not None:
            return False
        try:
            for report in self._reports():
                if report.get("status") != "PASS":
                    return False
                self.last_seen = report["checked_monotonic_ns"] / 1e9
        except (OSError, ValueError, TypeError, AttributeError, KeyError):
            return False
        return not self.eof and self.last_seen is not None and 0 <= time.monotonic() - self.last_seen < 0.75

    def close(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGINT)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # This process group contains only a read-only ROS camera monitor.
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
        self.process.stdout.close()
