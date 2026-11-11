"""FLASH-only ROS provider. Ordinary observation selection stays unchanged."""

import time

from .flash_contract import GROUPS
from .flash_history import FlashHistoryBuffer
from .ros_observation import MAX_AGE_NS
from .ros_observation import ObservationProvider
from .ros_observation import TOPICS
from .ros_observation import ordered_state
from .ros_observation import rgb_image
from .ros_observation import stamp_ns


class FlashObservationProvider(ObservationProvider):
    def __init__(self, **kwargs):
        self.flash_history = FlashHistoryBuffer()
        self.publisher_ids = {}
        super().__init__(**kwargs)

    def subscription_callback(self, key):
        def callback(message, info=None):
            # Older workstation rclpy executors pass only the message. Resolve
            # a unique publisher from the DDS graph instead of discarding all
            # history or treating a missing identity as a stable publisher.
            if info is None:
                publishers = self.node.get_publishers_info_by_topic(TOPICS[key])
                if len(publishers) != 1:
                    self.flash_history.reset(f"{key} requires exactly one publisher; found {len(publishers)}")
                    return
                gid = getattr(publishers[0], "endpoint_gid", None)
            else:
                gid = getattr(info, "publisher_gid", None)
            if gid is None or not bytes(gid) or not any(bytes(gid)):
                self.flash_history.reset(f"{key} publisher identity unavailable")
                return
            gid = bytes(gid)
            previous = self.publisher_ids.get(key)
            if previous is not None and previous != gid:
                self.flash_history.reset(f"{key} publisher reconnect; new history generation")
            self.publisher_ids[key] = gid
            self.remember(key, message)

        return callback

    def remember(self, key, message):
        super().remember(key, message)
        if key in GROUPS:
            now = self.latest[key][1]
            side, kind = key.split("_")
            try:
                values = ordered_state(message, side, kind == "hand")
            except (ValueError, TypeError, AttributeError):
                self.flash_history.reset(f"invalid {key} feedback")
                return
            self.flash_history.observe(
                key,
                values,
                source_ns=stamp_ns(message),
                received_ns=now,
                now_ros_ns=self.node.get_clock().now().nanoseconds,
            )

    def snapshot(self, timeout_s=5.0, *, on_diagnostic=None):
        deadline = time.monotonic() + timeout_s
        last_error = "FLASH history not ready"
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.01)
            now_mono, now_ros = time.monotonic_ns(), self.node.get_clock().now().nanoseconds
            try:
                numeric = self.flash_history.snapshot(now_ros_ns=now_ros, now_mono_ns=now_mono)
                anchor = numeric["flash_context"]["observation_anchor_source_ns"]
                # Freeze all seven sources in this ROS spin thread. Cameras are
                # strictly causal to the same joint anchor, never swapped later.
                selected = {}
                for key in ("cam0", "cam1", "cam2"):
                    entry = next((item for item in reversed(self.histories[key]) if stamp_ns(item[0]) <= anchor), None)
                    if entry is None:
                        raise ValueError(f"FLASH missing causal {key} frame")
                    stamp = stamp_ns(entry[0])
                    if (
                        not 0 <= now_ros - stamp <= MAX_AGE_NS
                        or not 0 <= now_mono - entry[1] <= MAX_AGE_NS
                        or anchor - stamp > 100_000_000
                    ):
                        raise ValueError(f"FLASH stale/unaligned {key} frame")
                    selected[key] = entry
                stamps = {key: stamp_ns(entry[0]) for key, entry in selected.items()}
                stamps.update(dict(zip(GROUPS, numeric["flash_context"]["history_source_ns"][-1], strict=True)))
                if max(stamps.values()) - min(stamps.values()) > 100_000_000:
                    raise ValueError("FLASH seven-source span exceeds 100 ms")
                if not self.cameras_ready(now_mono):
                    raise ValueError("FLASH waiting for camera startup stability")
                return {
                    **numeric,
                    "images": {key: rgb_image(entry[0]) for key, entry in selected.items()},
                    "topic_stamp_ns": stamps,
                    "captured_monotonic_ns": now_mono,
                    "alignment": {
                        "mode": "flash_common_anchor_causal_zoh",
                        "source_age_ms": {key: (now_ros - stamp) / 1e6 for key, stamp in stamps.items()},
                        "source_span_ms": (max(stamps.values()) - min(stamps.values())) / 1e6,
                    },
                }
            except ValueError as error:
                last_error = str(error)
        if on_diagnostic:
            on_diagnostic({"event": "observation_wait", "reason": last_error})
        raise TimeoutError(last_error)
