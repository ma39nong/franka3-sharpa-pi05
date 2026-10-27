"""One fresh, causal ROS observation snapshot. No publishers or SDK calls."""

from __future__ import annotations

import base64
import json
import sys
import time
from collections import deque

import numpy as np

from .protocol import SHARPA_JOINT_NAMES

TOPICS = {
    "left_arm": "/left/franka/joint_states",
    "right_arm": "/right/franka/joint_states",
    "left_hand": "/sharpa/left/joint_states",
    "right_hand": "/sharpa/right/joint_states",
    "cam0": "/cam0/color/image_raw",
    "cam1": "/cam1/color/image_raw",
    "cam2": "/cam2/color/image_raw",
}
MAX_AGE_NS = 200_000_000
MAX_SPREAD_NS = 100_000_000


def timing_issues(entries, now_ros, now_mono):
    issues = []
    stamps = []
    for key, (message, received) in entries.items():
        stamp = stamp_ns(message)
        if not 0 <= now_mono - received <= MAX_AGE_NS:
            issues.append(f"Stale received observation: {key}")
        if stamp <= 0 or not 0 <= now_ros - stamp <= MAX_AGE_NS:
            issues.append(f"Stale/future ROS timestamp: {key} (age={(now_ros-stamp)/1e6:.3f} ms)")
        stamps.append(stamp)
    if stamps and max(stamps) - min(stamps) > MAX_SPREAD_NS:
        issues.append(f"Multimodal observation timestamps are not synchronized "
                      f"(span={(max(stamps)-min(stamps))/1e6:.3f} ms, limit={MAX_SPREAD_NS/1e6:.3f} ms)")
    return issues


def observation_diagnostics(latest, *, now_ros, now_mono, reason):
    stamps = [stamp_ns(value[0]) for value in latest.values()]
    return {
        "event": "observation_wait", "reason": reason,
        "captured_monotonic_ns": now_mono, "now_ros_ns": now_ros,
        "topics": {key: ({"missing": True} if key not in latest else {
            "missing": False,
            "source_age_ms": (now_ros - stamp_ns(latest[key][0])) / 1e6,
            "received_age_ms": (now_mono - latest[key][1]) / 1e6,
            "stamp_ns": stamp_ns(latest[key][0]),
        }) for key in TOPICS},
        "source_span_ms": (max(stamps) - min(stamps)) / 1e6 if stamps else None,
        "issues": timing_issues(latest, now_ros, now_mono),
    }


def select_aligned_observation(latest, histories, *, now_ros, now_mono):
    """Select original samples at/before a common source time; never retimestamp."""
    issues = timing_issues(latest, now_ros, now_mono)
    if not issues:
        return dict(latest), "latest"
    # A buffer cannot cure stale/future streams. Wait for genuine fresh updates.
    if any(not issue.startswith("Multimodal") for issue in issues):
        raise ValueError("; ".join(issues))
    anchor = min(stamp_ns(message) for message, _ in latest.values())
    selected = {}
    for key in latest:
        candidates = [entry for entry in histories[key]
                      if 0 < stamp_ns(entry[0]) <= anchor
                      and 0 <= now_ros - stamp_ns(entry[0]) <= MAX_AGE_NS
                      and 0 <= now_mono - entry[1] <= MAX_AGE_NS]
        if not candidates:
            raise ValueError("; ".join(issues) + f"; Waiting for time-aligned history: {key}")
        selected[key] = max(candidates, key=lambda entry: stamp_ns(entry[0]))
    issues = timing_issues(selected, now_ros, now_mono)
    if issues:
        raise ValueError("; ".join(issues))
    return selected, "source_time_buffer"


def ordered_state(message, side: str, hand: bool) -> list[float]:
    names = list(message.name)
    values = list(message.position)
    if len(names) != len(values) or len(names) != len(set(names)):
        raise ValueError("Invalid joint state names/values")
    data = dict(zip(names, values, strict=True))
    if hand:
        required = [f"{side}_{name}" for name in SHARPA_JOINT_NAMES]
    else:
        alternatives = [
            [f"{side}_fr3_joint{i}", f"{side}_fr3v2_joint{i}"]
            for i in range(1, 8)
        ]
        required = []
        for group in alternatives:
            present = [name for name in group if name in data]
            if len(present) != 1:
                raise ValueError(f"Missing/ambiguous FR3 state joint: {group}")
            required.append(present[0])
    if any(name not in data for name in required):
        raise ValueError(f"Missing {side} {'hand' if hand else 'arm'} state joint")
    result = np.asarray([data[name] for name in required], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite measured joint state")
    return result.tolist()


def rgb_image(message) -> dict:
    channels = {"rgb8": 3, "bgr8": 3}
    if message.encoding not in channels:
        raise ValueError(f"Unsupported non-RGB camera encoding: {message.encoding}")
    height, width, step = int(message.height), int(message.width), int(message.step)
    count = channels[message.encoding]
    if height <= 0 or width <= 0 or step < width * count:
        raise ValueError("Invalid camera image dimensions")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    if raw.size != height * step:
        raise ValueError("Invalid camera byte count")
    pixels = raw.reshape(height, step)[:, :width * count].reshape(height, width, count)
    if message.encoding.startswith("bgr"):
        pixels = pixels[:, :, [2, 1, 0]]
    else:
        pixels = pixels[:, :, :3]
    return {"shape": [height, width, 3], "rgb_b64": base64.b64encode(pixels.tobytes()).decode("ascii")}


def stamp_ns(message) -> int:
    stamp = message.header.stamp
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class ObservationProvider:
    """One ROS context/node and subscriptions for the runtime lifetime."""
    def __init__(self, *, startup_stable_s=0.0):
        import rclpy
        from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                               ReliabilityPolicy, qos_profile_sensor_data)
        from sensor_msgs.msg import Image, JointState
        self.rclpy = rclpy
        rclpy.init()
        self.node = rclpy.create_node("fr3_sharpa_observation")
        self.latest = {}
        self.histories = {key: deque(maxlen=16 if key.startswith("cam") else 256) for key in TOPICS}
        self.camera_windows = {}
        self.startup_stable_s = startup_stable_s
        self.startup_ready = startup_stable_s == 0
        self.subscriptions = []
        self.closed = False
        camera_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5,
                                reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.VOLATILE)
        try:
            for key, topic in TOPICS.items():
                self.subscriptions.append(self.node.create_subscription(
                    Image if key.startswith("cam") else JointState, topic,
                    lambda message, selected=key: self.remember(selected, message),
                    camera_qos if key.startswith("cam") else qos_profile_sensor_data))
        except BaseException:
            self.close()
            raise

    def remember(self, key, message):
        now = time.monotonic_ns()
        entry = (message, now)
        self.latest[key] = entry
        history = self.histories[key]
        history.append(entry)
        while history and now - history[0][1] > MAX_AGE_NS:
            history.popleft()
        if key.startswith("cam"):
            stamp = stamp_ns(message)
            window = self.camera_windows.get(key)
            # Match recording readiness: fresh advancing frames, <=150 ms gaps.
            if (window is None or stamp <= window[3] or stamp - window[3] > 150_000_000
                    or now - window[2] > 150_000_000):
                self.camera_windows[key] = (now, stamp, now, stamp, 1)
            else:
                self.camera_windows[key] = (window[0], window[1], now, stamp, window[4] + 1)

    def cameras_ready(self, now):
        if self.startup_ready:
            return True
        for key, minimum_hz in (("cam0", 18), ("cam1", 25), ("cam2", 25)):
            if key not in self.camera_windows:
                return False
            first, first_stamp, last, last_stamp, count = self.camera_windows[key]
            if (last - first < self.startup_stable_s * 1e9 or now - last > 150_000_000
                    or last_stamp <= first_stamp):
                return False
            if min((count-1)*1e9/(last-first), (count-1)*1e9/(last_stamp-first_stamp)) < minimum_hz:
                return False
        self.startup_ready = True
        return True

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self.node.destroy_node()
            finally:
                self.rclpy.shutdown()

    def snapshot(self, timeout_s: float = 5.0, *, on_diagnostic=None) -> dict:
        if self.closed:
            raise RuntimeError("Observation provider is closed")
        rclpy, node, latest = self.rclpy, self.node, self.latest
        deadline = time.monotonic() + timeout_s
        timing_failure = "No complete snapshot received"
        next_diagnostic = 0.0

        def diagnose(*, final=False):
            nonlocal next_diagnostic
            now = time.monotonic()
            if on_diagnostic is not None and (final or now >= next_diagnostic):
                on_diagnostic(observation_diagnostics(
                    latest, now_ros=node.get_clock().now().nanoseconds,
                    now_mono=time.monotonic_ns(), reason=timing_failure))
                next_diagnostic = now + 0.1

        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(0.05, max(0.0, deadline - time.monotonic())))
            if len(latest) != len(TOPICS):
                timing_failure = f"Missing ROS observations: {sorted(set(TOPICS) - set(latest))}"
                diagnose()
                continue
            # Keep receiving real updates until the existing timing checks pass.
            # No timestamps or frames are synthesized, shifted or duplicated.
            now_mono = time.monotonic_ns()
            now_ros = node.get_clock().now().nanoseconds
            try:
                selected, alignment_mode = select_aligned_observation(
                    latest, self.histories, now_ros=now_ros, now_mono=now_mono)
            except ValueError as error:
                timing_failure = str(error)
                diagnose()
                continue
            if not self.cameras_ready(now_mono):
                timing_failure = f"Waiting for three camera streams stable for {self.startup_stable_s:g}s"
                diagnose()
                continue
            break
        else:
            diagnose(final=True)
            missing = sorted(set(TOPICS) - set(latest))
            if missing:
                raise TimeoutError(f"Missing ROS observations: {missing}")
            raise TimeoutError(f"No valid ROS observation snapshot within {timeout_s:g}s: {timing_failure}")
        state = sum((ordered_state(selected[f"{side}_{kind}"][0], side, kind == "hand")
                     for side, kind in (("left", "arm"), ("left", "hand"), ("right", "arm"), ("right", "hand"))), [])
        serialization_start = time.monotonic_ns()
        result = {
            "state": state,
            "images": {key: rgb_image(selected[key][0]) for key in ("cam0", "cam1", "cam2")},
            "topic_stamp_ns": {key: stamp_ns(value[0]) for key, value in selected.items()},
            "captured_monotonic_ns": now_mono,
            "alignment": {"mode": alignment_mode,
                          "source_age_ms": {key: (now_ros-stamp_ns(value[0]))/1e6 for key, value in selected.items()},
                          "source_span_ms": (max(stamp_ns(v[0]) for v in selected.values())
                                             - min(stamp_ns(v[0]) for v in selected.values()))/1e6},
        }
        result["timing"] = {"snapshot_ready_monotonic_ns": now_mono,
                            "observation_encoding_ms": (time.monotonic_ns()-serialization_start)/1e6}
        return result


def snapshot(timeout_s: float = 5.0) -> dict:
    """Standalone diagnostic compatibility; runtime uses ObservationProvider."""
    provider = ObservationProvider()
    try:
        return provider.snapshot(timeout_s)
    finally:
        provider.close()


def write_snapshot(result):
    start = time.monotonic_ns()
    payload = json.dumps(result, allow_nan=False)
    timing = {"json_serialization_ms": (time.monotonic_ns()-start)/1e6,
              "ros_payload_ready_monotonic_ns": time.monotonic_ns()}
    print(payload[:-1] + ', "serialization_timing": ' + json.dumps(timing) + '}', flush=True)


def persistent_main():
    """Read-only request/reply worker; idle time continues to spin subscriptions."""
    import select
    provider = ObservationProvider(startup_stable_s=3.0)
    try:
        while True:
            if not select.select([sys.stdin], [], [], 0)[0]:
                provider.rclpy.spin_once(provider.node, timeout_sec=0.01)
                continue
            line = sys.stdin.readline()
            if not line:
                break
            try:
                request = json.loads(line)
                if request.get("command") == "close":
                    break
                diagnostic = (lambda data: print(json.dumps(data), flush=True)) if request.get("diagnostics") else None
                write_snapshot(provider.snapshot(float(request.get("timeout_s", 5.0)), on_diagnostic=diagnostic))
            except Exception as error:
                print(json.dumps({"error": str(error)}), flush=True)
    finally:
        provider.close()


if __name__ == "__main__":
    if "--persistent" in sys.argv:
        persistent_main()
    else:
        try:
            write_snapshot(snapshot())
        except Exception as error:
            print(f"observation snapshot failed: {error}", file=sys.stderr)
            raise
