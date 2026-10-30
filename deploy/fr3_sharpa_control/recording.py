"""Recording gate shared by control workers; never owns robot shutdown."""
from contextlib import suppress
import json
import os
from pathlib import Path
import socket
import threading


def rpc(path, operation, *, timeout=10):
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(timeout)
        client.connect(str(path))
        client.sendall(json.dumps({"operation": operation}).encode() + b"\n")
        data = b""
        while b"\n" not in data:
            block = client.recv(4096)
            if not block:
                raise RuntimeError("录制器提前关闭连接")
            data += block
            if len(data) > 65536:
                raise ValueError("录制器应答超长")
    result = json.loads(data.split(b"\n", 1)[0])
    if result.get("status") != "PASS":
        raise RuntimeError(result.get("error", "录制请求失败"))
    return result


def mark_stop(path):
    path = Path(path)
    if path.parent.is_dir():
        (path.parent / "stop-recording").touch(exist_ok=True)


class RecordingGate:
    def __init__(self, path=None):
        self.path = path if path is not None else os.environ.get("SHARPA_RECORD_SOCKET")
        self.started = False

    @property
    def enabled(self):
        return bool(self.path)

    def start(self):
        if self.enabled and not self.started:
            rpc(self.path, "start")
            self.started = True

    def stop(self, *, wait=True):
        if not self.enabled:
            return
        if Path(self.path).exists():
            with suppress(OSError):
                mark_stop(self.path)
        if not wait:
            threading.Thread(target=self.stop, daemon=True).start()
            return
        try:
            rpc(self.path, "stop", timeout=2)
        except (OSError, RuntimeError, ValueError) as error:
            # The recorder may already have been stopped by the UI. Persist
            # the warning but never prevent the existing hand disable path.
            from deploy.fr3_sharpa_runtime.shutdown import stderr
            stderr(f"[recording] stop: {error}")


def play_task_frames(frames, bridge_frames, send, feedback, gate, *, resume, play=None):
    if play is None:
        from deploy.fr3_sharpa_runtime.sdk_direct import play_command_frames
        play = play_command_frames
    if gate.enabled and not gate.started:
        if bridge_frames:
            play(frames[:bridge_frames], send, feedback)
        gate.start()  # Recording ready BEFORE the first task frame, not Home/bridge.
        resume()      # Recorder startup must not trip the inter-phase watchdog.
        play(frames[bridge_frames:], send, feedback)
    else:
        play(frames, send, feedback)


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def command_publisher(node, gate):
    if not gate.enabled:
        return lambda values, stamp: None
    from sensor_msgs.msg import JointState
    from deploy.fr3_sharpa_runtime.protocol import SHARPA_JOINT_NAMES
    publisher = node.create_publisher(JointState, "/openpi/executed_joint_targets", 30)
    names = ([f"left_fr3_joint{i}" for i in range(1, 8)] + [f"left_{n}" for n in SHARPA_JOINT_NAMES]
             + [f"right_fr3_joint{i}" for i in range(1, 8)] + [f"right_{n}" for n in SHARPA_JOINT_NAMES])
    def publish(values, stamp):
        message = JointState()
        message.header.stamp.sec = stamp // 1_000_000_000
        message.header.stamp.nanosec = stamp % 1_000_000_000
        message.name, message.position = names, [float(value) for value in values]
        publisher.publish(message)
    return publish
