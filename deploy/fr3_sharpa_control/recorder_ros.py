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
OPTIONAL_TOPICS = {
    '/dagger/frame': 'std_msgs/msg/String',
    '/dagger/events': 'std_msgs/msg/String',
    '/teleop/arm_command_status': 'teleop_interfaces/msg/ArmCommandStatus',
    '/teleop/validated_arm_commands': 'sensor_msgs/msg/JointState',
}
ALL_TOPICS = {**TOPICS, **OPTIONAL_TOPICS}
MAX_LAST_RECEIVE_AGE_NS = 200_000_000
PROGRESS_WINDOW_NS = 1_000_000_000


class BagCapture:
    """Serialize control requests and callbacks under one lock."""
    def __init__(self, directory, stop_marker=None, *, clock=time.monotonic_ns, sleep=time.sleep, static_extrinsics=None):
        self.static_extrinsics = static_extrinsics or []
        self.static_tf_edges = {}
        self.camera_info_snapshots = {}
        self.directory = Path(directory)
        self.lock = threading.RLock()
        self.writer = None
        self.active = False
        self.finished = threading.Event()
        self.seen = set()
        self.clock, self.sleep = clock, sleep
        self.last_rx_monotonic_ns = {}
        self.recent_rx_monotonic_ns = {topic: deque(maxlen=512) for topic in ALL_TOPICS}
        self.recording_started_monotonic_ns = None
        self.counts = {name: 0 for name in ALL_TOPICS}
        self.report = {"status": "PREPARED", "topics": ALL_TOPICS, "started_unix_ns": None}
        self.error = None
        self.stop_marker = stop_marker

    def persist(self):
        write_json(self.directory / "recording.json", {**self.report, "counts": self.counts,
                    "missing_topics": [name for name in TOPICS if not self.counts[name]]})

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
            if topic == "/tf_static" and self.static_extrinsics:
                for transform in message.transforms:
                    self.static_tf_edges[transform.child_frame_id] = transform
                message = self.calibrated_snapshot(rx_ros_ns)
            if topic.endswith('/camera_info'):
                self.camera_info_snapshots[topic]=message
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

    def calibrated_snapshot(self, stamp_ns):
        from tf2_msgs.msg import TFMessage
        from rosidl_runtime_py.set_message import set_message_fields
        from .calibrated_tf import merge_static
        transforms=[]
        for t in self.static_tf_edges.values():
            transforms.append(dict(header=dict(frame_id=t.header.frame_id,stamp=dict(sec=t.header.stamp.sec,nanosec=t.header.stamp.nanosec)),
                child_frame_id=t.child_frame_id,transform=dict(translation={k:getattr(t.transform.translation,k) for k in 'xyz'},
                rotation={k:getattr(t.transform.rotation,k) for k in ('x','y','z','w')})))
        message=TFMessage()
        set_message_fields(message,merge_static({'transforms':transforms},self.static_extrinsics,stamp_ns))
        return message

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
                if self.static_extrinsics:
                    missing_calibration=[f'/cam{i}/color/camera_info' for i in range(3)
                                         if f'/cam{i}/color/camera_info' not in self.camera_info_snapshots]
                    if not self.static_tf_edges:missing_calibration.append('/tf_static')
                    if missing_calibration:health.update(ready=False,missing_calibration=missing_calibration)
                if now >= deadline:
                    raise TimeoutError("录制传感器在5秒内未就绪：" + json.dumps(health, ensure_ascii=False))
                if health["ready"]:
                    # Hold the same lock from this fresh check through activation.
                    # Callbacks entering during writer setup remain PREPARED frames.
                    opened = self.clock()
                    try:
                        import rosbag2_py
                        self.writer = rosbag2_py.SequentialWriter()
                        self.writer.open(rosbag2_py.StorageOptions(uri=str((self.directory / "rosbag").resolve()), storage_id="sqlite3"),
                                         rosbag2_py.ConverterOptions("", ""))
                        for topic, kind in ALL_TOPICS.items():
                            self.writer.create_topic(rosbag2_py.TopicMetadata(name=topic, type=kind, serialization_format="cdr"))
                        activated = self.clock()
                        self.report["writer_initialization_ms"] = (activated - opened) / 1_000_000
                        health = self.readiness(activated)
                        if self._stopped() or activated >= deadline or not health["ready"]:
                            raise RuntimeError("录制初始化期间停止或传感器过期：" + json.dumps(health, ensure_ascii=False))
                        self.recording_started_monotonic_ns = activated
                        self.report.update(status="RECORDING", started_unix_ns=time.time_ns(),
                                           started_monotonic_ns=activated)
                        from rclpy.serialization import serialize_message
                        for topic,message in self.camera_info_snapshots.items():
                            self.writer.write(topic,serialize_message(message),self.report['started_unix_ns'])
                            self.counts[topic] += 1
                        if self.static_extrinsics:
                            from rclpy.serialization import serialize_message
                            stamp = self.report['started_unix_ns']
                            self.writer.write('/tf_static',serialize_message(self.calibrated_snapshot(stamp)),stamp)
                            self.counts['/tf_static'] += 1
                            write_json(self.directory/'rosbag/wrist_extrinsics.json',{'static_extrinsics':self.static_extrinsics})
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
                self.report["status"] = ("INCOMPLETE" if any(not self.counts[name] for name in TOPICS) else "COMPLETE") if self.report["started_unix_ns"] else "NOT_STARTED"
            self.report.update(ended_unix_ns=time.time_ns(), ended_monotonic_ns=time.monotonic_ns())
            self.persist()
            self.finished.set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    import os
    from .recording_options import enable_rosbag_channels
    enable_rosbag_channels()
    if os.environ.get("SHARPA_RECORD_FORMAT", "rosbag") == "mcap":
        from .mcap_capture import main as mcap_main
        return mcap_main(args)
    from .rosbag_contract import configure
    import sys
    configure(sys.modules[__name__])
    # The same real wrist calibration already used by MCAP, read only for rosbag.
    import yaml
    profile_path=Path(os.environ.get('SHARPA_MCAP_PROFILE',str(Path(__file__).resolve().parents[2]/'.deployment/mcap-recording.yaml')))
    if not profile_path.is_file():raise ValueError('Rosbag wrist calibration profile missing: '+str(profile_path))
    profile=yaml.safe_load(profile_path.read_text()) or {}
    static_extrinsics=profile.get('static_extrinsics',[])
    if not all(any(spec.get('child_frame','').startswith(camera+'_') for spec in static_extrinsics) for camera in ('cam1','cam2')):
        raise ValueError('Rosbag requires both real wrist extrinsics')
    from .calibrated_tf import validate_specs
    validate_specs(static_extrinsics)
    import rclpy
    if args.check:
        import importlib
        import tempfile
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
            if os.environ.get("SHARPA_COMPACT_RECORDS") == "1":
                import av
                with av.open(str(Path(temporary) / "check.mp4"), "w") as output:
                    video = output.add_stream("libx264", rate=30)
                    video.width = video.height = 64
                    video.pix_fmt = "yuv420p"
                    frame = av.VideoFrame.from_ndarray(np.zeros((64,64,3),np.uint8),format="rgb24")
                    for packet in video.encode(frame): output.mux(packet)
                    for packet in video.encode(): output.mux(packet)
            else:
                import cv2
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
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from rclpy.serialization import serialize_message
    from rosidl_runtime_py.utilities import get_message
    rclpy.init()
    node = rclpy.create_node("sharpa_deployment_recorder")
    marker = args.socket.parent / "stop-recording"
    capture = BagCapture(args.directory, stop_marker=marker, static_extrinsics=static_extrinsics)
    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=30, reliability=ReliabilityPolicy.BEST_EFFORT)
    subscriptions = []
    from .rosbag_contract import COMBINED_SOURCE, HAND_COMMANDS, hand_commands
    for topic, kind in ALL_TOPICS.items():
        if topic in HAND_COMMANDS.values():continue
        def callback(message, topic=topic):
            rx_mono_ns = time.monotonic_ns()
            rx_ros_ns = node.get_clock().now().nanoseconds
            capture.accept(topic, message, rx_ros_ns, rx_mono_ns=rx_mono_ns, serialize=serialize_message)
        subscriptions.append(node.create_subscription(get_message(kind), topic, callback,
            QoSProfile(depth=100,durability=DurabilityPolicy.TRANSIENT_LOCAL) if topic=='/tf_static' else
            1000 if topic in OPTIONAL_TOPICS or topic=='/teleop/validated_arm_commands' else qos))

    def submitted_hands(message):
        rx_mono_ns=time.monotonic_ns();rx_ros_ns=node.get_clock().now().nanoseconds
        try:
            for topic,msg in hand_commands(message).items():
                capture.accept(topic,msg,rx_ros_ns,rx_mono_ns=rx_mono_ns,serialize=serialize_message)
        except Exception as error:
            with capture.lock:capture._fail(error)
    subscriptions.append(node.create_subscription(get_message('sensor_msgs/msg/JointState'),COMBINED_SOURCE,submitted_hands,qos))

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
