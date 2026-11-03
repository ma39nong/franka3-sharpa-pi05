"""Paused ROS subscriptions; open a bag only at the first task chunk gate."""
import argparse
from collections import deque
import json
from pathlib import Path
import signal
import socketserver
import threading
import time

from .recording import write_json

TOPICS = {
    "/cam0/color/image_raw": "sensor_msgs/msg/Image",
    "/cam1/color/image_raw": "sensor_msgs/msg/Image",
    "/cam2/color/image_raw": "sensor_msgs/msg/Image",
    "/left/franka/joint_states": "sensor_msgs/msg/JointState",
    "/right/franka/joint_states": "sensor_msgs/msg/JointState",
    "/sharpa/left/joint_states": "sensor_msgs/msg/JointState",
    "/sharpa/right/joint_states": "sensor_msgs/msg/JointState",
    "/teleop/arm_commands": "teleop_interfaces/msg/ArmCommand",
    "/openpi/executed_joint_targets": "sensor_msgs/msg/JointState",
}


SENSORS = frozenset(TOPICS) - {"/teleop/arm_commands", "/openpi/executed_joint_targets"}
MAX_LAST_RECEIVE_AGE_NS = 200_000_000
PROGRESS_WINDOW_NS = 1_000_000_000


class BagCapture:
    """Serialize control requests and callbacks under one lock."""
    def __init__(self, directory, stop_marker=None, *, clock=time.monotonic_ns, sleep=time.sleep):
        self.directory = Path(directory)
        self.lock = threading.RLock()
        self.writer = None
        self.active = False
        self.finished = threading.Event()
        self.seen = set()
        self.clock, self.sleep = clock, sleep
        self.last_rx_monotonic_ns = {}
        self.recent_rx_monotonic_ns = {topic: deque(maxlen=512) for topic in TOPICS}
        self.recording_started_monotonic_ns = None
        self.counts = {name: 0 for name in TOPICS}
        self.report = {"status": "PREPARED", "topics": TOPICS, "started_unix_ns": None}
        self.error = None
        self.stop_marker = stop_marker

    def persist(self):
        write_json(self.directory / "recording.json", {**self.report, "counts": self.counts,
                    "missing_topics": [name for name, count in self.counts.items() if not count]})

    def _stopped(self):
        if self.stop_marker is not None and self.stop_marker.exists():
            self.active = False
            self.finished.set()
        return self.finished.is_set()

    def _fail(self, error):
        self.error = str(error)
        self.report.update(status="FAILED", error=self.error)
        self.active = False
        self.finished.set()

    def readiness(self, now_mono_ns=None):
        """Local receipt progress only; never cache sensor payloads."""
        with self.lock:
            now = self.clock() if now_mono_ns is None else now_mono_ns
            missing, stale, insufficient, ages = [], [], [], {}
            for topic in sorted(SENSORS):
                last = self.last_rx_monotonic_ns.get(topic)
                if last is None:
                    missing.append(topic)
                    ages[topic] = None
                else:
                    ages[topic] = (now - last) / 1_000_000
                    if now - last > MAX_LAST_RECEIVE_AGE_NS:
                        stale.append(topic)
                recent = self.recent_rx_monotonic_ns[topic]
                # Sorting handles callbacks that entered before another thread but
                # acquired the lock later. Distinct receipts must make progress.
                samples = sorted({stamp for stamp in recent
                                  if now - PROGRESS_WINDOW_NS <= stamp <= now})
                recent.clear()
                recent.extend(samples)
                if len(samples) < 2:
                    insufficient.append(topic)
            return {"ready": not (missing or stale or insufficient), "missing": missing,
                    "stale": stale, "insufficient_progress": insufficient, "ages_ms": ages}

    def accept(self, topic, message, rx_ros_ns, *, rx_mono_ns, serialize):
        with self.lock:
            stopped = self._stopped()
            self.seen.add(topic)
            self.last_rx_monotonic_ns[topic] = max(
                rx_mono_ns, self.last_rx_monotonic_ns.get(topic, rx_mono_ns))
            self.recent_rx_monotonic_ns[topic].append(rx_mono_ns)
            if (stopped or self.error or not self.active
                    or rx_mono_ns < self.recording_started_monotonic_ns):
                return
            try:
                serialized = serialize(message)
                self.writer.write(topic, serialized, rx_ros_ns)
                self.counts[topic] += 1
            except Exception as error:
                self._fail(error)

    def _start(self):
        deadline = self.clock() + 5_000_000_000
        while True:
            with self.lock:
                if self._stopped() or self.error:
                    raise RuntimeError(self.error or "录制已结束，不能复用旧录制器")
                if self.active:
                    return {"status": "PASS", "active": True}
                now = self.clock()
                health = self.readiness(now)
                if now >= deadline:
                    raise TimeoutError("录制传感器在5秒内未就绪：" + json.dumps(health, ensure_ascii=False))
                if health["ready"]:
                    # Hold the same lock from this fresh check through activation.
                    # Callbacks entering during writer setup remain PREPARED frames.
                    opened = self.clock()
                    try:
                        import rosbag2_py
                        self.writer = rosbag2_py.SequentialWriter()
                        self.writer.open(rosbag2_py.StorageOptions(uri=str(self.directory / "rosbag"), storage_id="sqlite3"),
                                         rosbag2_py.ConverterOptions("", ""))
                        for topic, kind in TOPICS.items():
                            self.writer.create_topic(rosbag2_py.TopicMetadata(name=topic, type=kind, serialization_format="cdr"))
                        activated = self.clock()
                        self.report["writer_initialization_ms"] = (activated - opened) / 1_000_000
                        health = self.readiness(activated)
                        if self._stopped() or activated >= deadline or not health["ready"]:
                            raise RuntimeError("录制初始化期间停止或传感器过期：" + json.dumps(health, ensure_ascii=False))
                        self.recording_started_monotonic_ns = activated
                        self.report.update(status="RECORDING", started_unix_ns=time.time_ns(),
                                           started_monotonic_ns=activated)
                        self.active = True
                        self.persist()
                    except Exception as error:
                        self._fail(error)
                        raise
                    return {"status": "PASS", "active": True}
            # Never wait while holding the callback lock.
            self.sleep(0.02)

    def request(self, operation):
        if operation == "start":
            return self._start()
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            if operation == "status":
                self._stopped()
                return {"status": "PASS", "active": self.active, "topics_seen": sorted(self.seen),
                        "sensor_readiness": self.readiness()}
            if operation == "stop":
                self.close()
                return {"status": "PASS", "active": False}
            raise ValueError("Unknown recorder operation")

    def close(self):
        with self.lock:
            self.active = False
            self.finished.set()
            writer, self.writer = self.writer, None
            if writer is not None:
                # Humble closes on destruction; newer versions expose close().
                if hasattr(writer, "close"):
                    writer.close()
                del writer
            if not self.error:
                self.report["status"] = ("INCOMPLETE" if any(not count for count in self.counts.values()) else "COMPLETE") if self.report["started_unix_ns"] else "NOT_STARTED"
            self.report.update(ended_unix_ns=time.time_ns(), ended_monotonic_ns=time.monotonic_ns())
            self.persist()
            self.finished.set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    import rclpy
    if args.check:
        import importlib
        import tempfile
        import cv2
        import numpy as np
        import rosbag2_py
        for module, names in (("teleop_interfaces.msg", ("ArmCommand",)), ("sensor_msgs.msg", ("Image", "JointState"))):
            for name in names:
                getattr(importlib.import_module(module), name)
        with tempfile.TemporaryDirectory(prefix="sharpa-record-check-") as temporary:
            writer = rosbag2_py.SequentialWriter()
            writer.open(rosbag2_py.StorageOptions(uri=str(Path(temporary) / "bag"), storage_id="sqlite3"),
                        rosbag2_py.ConverterOptions("", ""))
            writer.create_topic(rosbag2_py.TopicMetadata(name="/check", type="sensor_msgs/msg/JointState", serialization_format="cdr"))
            if hasattr(writer, "close"):
                writer.close()
            del writer
            video = cv2.VideoWriter(str(Path(temporary) / "check.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 64))
            try:
                if not video.isOpened():
                    raise RuntimeError("ROS导出环境的MP4编码器不可用")
                video.write(np.zeros((64, 64, 3), np.uint8))
            finally:
                video.release()
        # Exercise the worker's optional recording publisher before hardware up.
        # The sink is local; no ROS context, DDS publisher or SDK is constructed.
        from types import SimpleNamespace
        from .recording import RecordingGate, command_publisher
        messages = []
        sink = SimpleNamespace(create_publisher=lambda *args: SimpleNamespace(publish=messages.append))
        command_publisher(sink, RecordingGate("dependency-check"))([0.0] * 58, 1_234_567_890)
        if len(messages) != 1 or len(messages[0].name) != 58 or len(messages[0].position) != 58:
            raise RuntimeError("实际下发目标录制消息必须为58D")
        print("Recording/export dependencies and 58D target publisher OK; no ROS node or hardware started")
        return
    if args.directory is None or args.socket is None:
        parser.error("--directory and --socket required")
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from rclpy.serialization import serialize_message
    from rosidl_runtime_py.utilities import get_message
    rclpy.init()
    node = rclpy.create_node("sharpa_deployment_recorder")
    marker = args.socket.parent / "stop-recording"
    capture = BagCapture(args.directory, stop_marker=marker)
    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=30, reliability=ReliabilityPolicy.BEST_EFFORT)
    subscriptions = []
    for topic, kind in TOPICS.items():
        def callback(message, topic=topic):
            rx_mono_ns = time.monotonic_ns()
            rx_ros_ns = node.get_clock().now().nanoseconds
            capture.accept(topic, message, rx_ros_ns, rx_mono_ns=rx_mono_ns, serialize=serialize_message)
        subscriptions.append(node.create_subscription(get_message(kind), topic, callback, qos))

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                request = json.loads(self.rfile.readline(65536))
                reply = capture.request(request["operation"])
            except Exception as error:
                reply = {"status": "FAIL", "error": str(error)}
            self.wfile.write(json.dumps(reply).encode() + b"\n")
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    if args.socket.exists():
        raise FileExistsError(args.socket)
    server = socketserver.ThreadingUnixStreamServer(str(args.socket), Handler)
    args.socket.chmod(0o600)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    signal.signal(signal.SIGINT, lambda *_: capture.finished.set())
    signal.signal(signal.SIGTERM, lambda *_: capture.finished.set())
    capture.persist()
    try:
        while rclpy.ok() and not capture.finished.is_set():
            if marker.exists():
                break
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        try:
            capture.close()
        finally:
            server.shutdown()
            server.server_close()
            args.socket.unlink(missing_ok=True)
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
