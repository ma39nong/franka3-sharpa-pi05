"""Read-only, named ROS feedback for execute; never accepts episode state."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import threading
import time

import numpy as np

from .ros_observation import MAX_AGE_NS, MAX_SPREAD_NS, TOPICS, ordered_state, stamp_ns

FEEDBACK_KEYS = ("left_arm", "left_hand", "right_arm", "right_hand")
FEEDBACK_TOPICS = {key: TOPICS[key] for key in FEEDBACK_KEYS}


@dataclass(frozen=True)
class LiveFeedback:
    state: np.ndarray
    ros_stamps_ns: tuple[int, ...]
    received_monotonic_ns: tuple[int, ...]

    def validate_freshness(self, *, now_ros_ns: int, now_monotonic_ns: int) -> None:
        if self.state.shape != (58,) or not np.isfinite(self.state).all():
            raise ValueError("Live feedback must contain 58 finite measured positions")
        if len(self.ros_stamps_ns) != 4 or len(self.received_monotonic_ns) != 4:
            raise ValueError("All four live feedback topics are required")
        for key, stamp, received in zip(FEEDBACK_KEYS, self.ros_stamps_ns, self.received_monotonic_ns, strict=True):
            if stamp <= 0 or not 0 <= now_ros_ns - stamp <= MAX_AGE_NS:
                raise ValueError(f"Stale/future feedback timestamp: {FEEDBACK_TOPICS[key]}; "
                                 f"source_age_ms={(now_ros_ns-stamp)/1e6:.3f}, "
                                 f"receipt_age_ms={(now_monotonic_ns-received)/1e6:.3f}, "
                                 f"source_stamp_ns={stamp}, now_ros_ns={now_ros_ns}")
            if not 0 <= now_monotonic_ns - received <= MAX_AGE_NS:
                raise ValueError(f"Stale/future received feedback: {FEEDBACK_TOPICS[key]}; "
                                 f"receipt_age_ms={(now_monotonic_ns-received)/1e6:.3f}")
        if max(self.ros_stamps_ns) - min(self.ros_stamps_ns) > MAX_SPREAD_NS:
            span = (max(self.ros_stamps_ns) - min(self.ros_stamps_ns)) / 1e6
            details = ", ".join(
                f"{FEEDBACK_TOPICS[key]} source_age_ms={(now_ros_ns-stamp)/1e6:.3f}"
                f" receipt_age_ms={(now_monotonic_ns-received)/1e6:.3f}"
                for key, stamp, received in zip(
                    FEEDBACK_KEYS, self.ros_stamps_ns, self.received_monotonic_ns, strict=True
                )
            )
            raise ValueError(f"Live feedback timestamps are not synchronized: span_ms={span:.3f}, "
                             f"limit_ms={MAX_SPREAD_NS/1e6:.3f}; {details}")


class FeedbackCache:
    """Cache only observations received from the four state subscriptions."""

    def __init__(self):
        self.lock = threading.RLock()
        self.latest = {}
        self.errors = {}
        self.histories = {key: deque(maxlen=256) for key in FEEDBACK_KEYS}

    def update(self, key: str, message, received_monotonic_ns: int) -> None:
        if key not in FEEDBACK_KEYS:
            raise ValueError(f"Unexpected feedback topic: {key}")
        side, kind = key.split("_")
        try:
            values = ordered_state(message, side, kind == "hand")
            stamp = stamp_ns(message)
            if stamp <= 0:
                raise ValueError("Feedback has no valid ROS timestamp")
        except (ValueError, TypeError, AttributeError) as error:
            with self.lock:
                self.latest.pop(key, None)
                self.histories[key].clear()
                self.errors[key] = str(error)
            return
        with self.lock:
            entry = (values, stamp, received_monotonic_ns)
            self.latest[key] = entry
            self.histories[key].append(entry)
            self.errors.pop(key, None)

    def snapshot(self, *, now_ros_ns: int, now_monotonic_ns: int) -> LiveFeedback:
        with self.lock:
            missing = [FEEDBACK_TOPICS[key] for key in FEEDBACK_KEYS if key not in self.latest]
            if missing:
                raise ValueError(f"Missing/invalid live feedback: {missing}; errors={self.errors}")
            entries = [self.latest[key] for key in FEEDBACK_KEYS]
            histories = [tuple(self.histories[key]) for key in FEEDBACK_KEYS]

        def assemble(entries):
            return LiveFeedback(
                state=np.concatenate([entry[0] for entry in entries]).astype(np.float32),
                ros_stamps_ns=tuple(entry[1] for entry in entries),
                received_monotonic_ns=tuple(entry[2] for entry in entries),
            )

        result = assemble(entries)
        try:
            result.validate_freshness(now_ros_ns=now_ros_ns, now_monotonic_ns=now_monotonic_ns)
            return result
        except ValueError as error:
            if not str(error).startswith("Live feedback timestamps are not synchronized"):
                raise
            latest_error = error
        anchor = min(entry[1] for entry in entries)
        aligned = []
        for history in histories:
            candidates = [
                entry for entry in history
                if entry[1] <= anchor and 0 <= now_ros_ns - entry[1] <= MAX_AGE_NS
                and 0 <= now_monotonic_ns - entry[2] <= MAX_AGE_NS
            ]
            if not candidates:
                raise latest_error
            aligned.append(max(candidates, key=lambda entry: entry[1]))
        result = assemble(aligned)
        result.validate_freshness(now_ros_ns=now_ros_ns, now_monotonic_ns=now_monotonic_ns)
        return result


class ROSFeedbackProvider:
    """Keep live subscriptions active until the ROS transport node is closed."""

    def __init__(self, node):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState

        self.node = node
        self.cache = FeedbackCache()
        self.subscriptions = [
            node.create_subscription(
                JointState, topic,
                lambda message, selected=key: self.cache.update(selected, message, time.monotonic_ns()),
                qos_profile_sensor_data,
            )
            for key, topic in FEEDBACK_TOPICS.items()
        ]

    def snapshot(self) -> LiveFeedback:
        with self.cache.lock:
            return self.cache.snapshot(
                now_ros_ns=self.node.get_clock().now().nanoseconds,
                now_monotonic_ns=time.monotonic_ns(),
            )

    def wait(self, timeout_s: float = 5.0) -> LiveFeedback:
        import rclpy

        deadline = time.monotonic() + timeout_s
        last_error = "No live state received"
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.01)
            try:
                return self.snapshot()
            except ValueError as error:
                last_error = str(error)
        raise TimeoutError(f"EXECUTE PREFLIGHT: WAITING FOR LIVE ROS STATE; {last_error}")

    def close(self):
        for subscription in self.subscriptions:
            self.node.destroy_subscription(subscription)
