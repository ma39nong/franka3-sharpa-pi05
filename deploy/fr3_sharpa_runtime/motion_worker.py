"""Explicit execute-only IPC to a persistent guarded ROS transport process."""
import json
import os
import subprocess
import select
import signal
import time
import threading

from .shutdown import DisableSettings, protect_disable_confirmation, stderr


class MotionWorker:
    def __init__(self, command, *, cwd, protocol_fd=False):
        self._protocol_read = None
        self._closed = False
        self._direct = protocol_fd
        self._close_grace_s = DisableSettings.from_env().worker_grace_s if protocol_fd else 5.0
        self._previous_sigterm = None
        pass_fds = ()
        env = None
        stdout = subprocess.PIPE
        if protocol_fd:
            read_fd, write_fd = os.pipe()
            os.set_inheritable(write_fd, True)
            self._protocol_read = os.fdopen(read_fd, "rb", buffering=0)
            env = os.environ.copy()
            env["SHARPA_PROTOCOL_FD"] = str(write_fd)
            pass_fds = (write_fd,)
            stdout = subprocess.DEVNULL
        self.process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.PIPE,
                                        stdout=stdout, text=True, bufsize=1,
                                        env=env, pass_fds=pass_fds, start_new_session=protocol_fd)
        if protocol_fd:
            os.close(write_fd)
            if threading.current_thread() is threading.main_thread():
                def terminate(_number, _frame):
                    raise KeyboardInterrupt("SIGTERM: await SDK disable confirmation")
                self._previous_sigterm = signal.signal(signal.SIGTERM, terminate)

    def request(self, payload):
        try:
            self.process.stdin.write(json.dumps(payload, allow_nan=False)+"\n")
            self.process.stdin.flush()
        except BrokenPipeError:
            # An exiting worker may already have queued the original FAIL.
            # Drain that report before falling back to the generic EOF error.
            pass
        while True:
            stream = self._protocol_read or self.process.stdout
            line = stream.readline()
            if not line:
                raise RuntimeError("Guarded motion worker exited; execute stopped")
            try:
                report = json.loads(line)
            except json.JSONDecodeError:
                continue
            if report.get("status") != "PASS":
                reason = report.get("reason", report)
                detail = report.get("detail")
                if detail:
                    reason = f"{reason}\n{detail.rstrip()}"
                raise RuntimeError(f"Guarded motion rejected: {reason}")
            return report

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._direct:
            try:
                with protect_disable_confirmation():
                    self._close_direct()
            finally:
                if self._previous_sigterm is not None:
                    signal.signal(signal.SIGTERM, self._previous_sigterm)
            return
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
        if self._protocol_read is not None:
            self._protocol_read.close()

    def _await_shutdown(self, deadline):
        pending = b""
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self._protocol_read], [], [], max(0, deadline - time.monotonic()))
            if not ready:
                break
            data = os.read(self._protocol_read.fileno(), 65536)
            if not data:
                break
            pending += data
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                try:
                    report = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(report, dict) and report.get("operation") == "shutdown":
                    if report.get("status") != "PASS":
                        stderr(f"[sharpa-shutdown] ERROR SDK 退出检查未完整通过: {report}")
                    return True
        stderr("[sharpa-shutdown] ERROR 未收到 SDK 失能读回确认；程序将退出，不能认定双手已失能。")
        return False

    def _close_direct(self):
        deadline = time.monotonic() + self._close_grace_s
        try:
            try:
                self.process.stdin.close()
            except (OSError, ValueError):
                pass
            # The SDK worker has its own process group. Interrupt its active
            # motion, but still wait for its final confirmation over the pipe.
            if self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
            acknowledged = self._await_shutdown(deadline)
            try:
                self.process.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                acknowledged = False
            if not acknowledged:
                stderr("[sharpa-shutdown] ERROR 退出确认未完成，清理 SDK 进程组；双手状态仍未确认。")
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                    # Also reap a surviving SDK child if its wrapper exited first.
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=5)
        finally:
            self._protocol_read.close()
