"""Runtime-owned persistent Pixi observation worker. No control interfaces."""
import json
import queue
import subprocess
import threading
import time


class ObservationWorker:
    def __init__(self, command, *, cwd):
        self.process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=None,
                                        text=True, bufsize=1)
        self.responses = queue.Queue()
        self.lock = threading.Lock()
        self.snapshot_ready = False
        def read():
            for line in self.process.stdout:
                self.responses.put(line)
            self.responses.put(None)
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def snapshot(self, timeout_s=None, *, on_poll=None, on_diagnostic=None):
        bounded = timeout_s is not None
        if timeout_s is None:
            timeout_s = 5.0 if self.snapshot_ready else 15.0
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError(f"ROS observation worker exited {self.process.returncode}")
            deadline = time.monotonic() + timeout_s + (0 if bounded else 10)
            self.process.stdin.write(json.dumps({"timeout_s": timeout_s,
                                                "diagnostics": on_diagnostic is not None}) + "\n")
            self.process.stdin.flush()
            try:
                while True:
                    if on_poll is not None:
                        on_poll()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("ROS observation worker response timed out")
                    try:
                        line = self.responses.get(timeout=min(0.01, remaining))
                    except queue.Empty:
                        continue
                    if line is None:
                        raise RuntimeError("ROS observation worker closed without a response")
                    payload = json.loads(line)
                    if payload.get("event") == "observation_wait":
                        if on_diagnostic is not None:
                            on_diagnostic(payload)
                        continue
                    if "error" not in payload:
                        self.snapshot_ready = True
                    return line
            except BaseException:
                # Do not accept a late reply as the next request's observation.
                # Reap in close(), after the caller has stopped motion. Waiting
                # for ROS shutdown here would consume the remaining old chunk.
                if self.process.poll() is None:
                    self.process.terminate()
                raise

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"command":"close"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.reader.join(timeout=1)
        self.process.stdin.close()
        self.process.stdout.close()
