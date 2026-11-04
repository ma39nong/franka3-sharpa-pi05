"""Bounded, causal measured-joint history, enabled only for FLASH observations."""

from collections import deque
import threading
import uuid

import numpy as np

from .flash_contract import GROUPS
from .flash_contract import RESAMPLER


class FlashHistoryBuffer:
    def __init__(self, *, retention_ns=1_000_000_000, capacity=8192):
        self.retention_ns, self.capacity = retention_ns, capacity
        self.lock = threading.RLock()
        self.reset("new observation session")

    def reset(self, reason):
        with self.lock:
            self.samples = {key: deque(maxlen=self.capacity) for key in GROUPS}
            self.session_id = uuid.uuid4().hex
            self.reason = reason
            self.clock_pair = None

    def observe(self, key, values, *, source_ns, received_ns, now_ros_ns):
        with self.lock:
            array = np.asarray(values, dtype=np.float32)
            width = 7 if key.endswith("arm") else 22
            delta_pair = (
                None
                if self.clock_pair is None
                else (now_ros_ns - self.clock_pair[0]) - (received_ns - self.clock_pair[1])
            )
            if delta_pair is not None and abs(delta_pair) > 200_000_000:
                self.reset("source/receive clock discontinuity")
            self.clock_pair = (now_ros_ns, received_ns)
            if (
                array.shape != (width,)
                or not np.isfinite(array).all()
                or source_ns <= 0
                or not 0 <= now_ros_ns - source_ns <= 200_000_000
            ):
                self.reset(f"invalid/stale {key} source measurement")
                return
            samples = self.samples[key]
            if samples and (source_ns <= samples[-1][0] or received_ns <= samples[-1][1]):
                self.reset(f"nonadvancing {key} source/receipt; history invalidated")
                return
            if samples and (source_ns - samples[-1][0] > 100_000_000 or received_ns - samples[-1][1] > 100_000_000):
                self.reset(f"{key} measurement gap; reacquiring history")
                samples = self.samples[key]
            samples.append((source_ns, received_ns, array.copy()))
            while samples and received_ns - samples[0][1] > self.retention_ns:
                samples.popleft()

    def snapshot(self, *, now_ros_ns, now_mono_ns, anchor=None):
        with self.lock:
            view = {key: tuple(items) for key, items in self.samples.items()}
            if any(not items for items in view.values()):
                raise ValueError(
                    f"FLASH history not ready: {self.reason}; missing={[k for k, v in view.items() if not v]}"
                )
            anchor = min(items[-1][0] for items in view.values()) if anchor is None else anchor
            grid = [anchor - round((7 - i) * 1e9 / 30) for i in range(8)]
            rows, sources = [], []
            for key, items in view.items():
                if (
                    not 0 <= now_ros_ns - items[-1][0] <= 200_000_000
                    or not 0 <= now_mono_ns - items[-1][1] <= 200_000_000
                ):
                    raise ValueError(f"FLASH latest feedback stale: {key}")
            for stamp in grid:
                parts, times = [], []
                for key in GROUPS:
                    item = next((entry for entry in reversed(view[key]) if entry[0] <= stamp), None)
                    if item is None or stamp - item[0] > 75_000_000:
                        raise ValueError(f"FLASH history coverage/causal age not ready: {key}, grid={stamp}")
                    if stamp == anchor and (
                        not 0 <= now_ros_ns - item[0] <= 200_000_000 or not 0 <= now_mono_ns - item[1] <= 200_000_000
                    ):
                        raise ValueError(f"FLASH selected current state stale: {key}")
                    parts.append(item[2])
                    times.append(item[0])
                rows.append(np.concatenate(parts))
                sources.append(times)
            history = np.asarray(rows, dtype=np.float32)
            return {
                "history": history.tolist(),
                "state": history[-1].tolist(),
                "flash_context": {
                    "protocol_version": 1,
                    "request_id": uuid.uuid4().hex,
                    "session_id": self.session_id,
                    "observation_anchor_source_ns": anchor,
                    "history_grid_source_ns": grid,
                    "history_source_ns": sources,
                    "captured_monotonic_ns": now_mono_ns,
                    "history_resampler_version": RESAMPLER,
                    "anchor_age_at_capture_ns": now_ros_ns - anchor,
                    "clock_pair": {"ros_ns": now_ros_ns, "monotonic_ns": now_mono_ns},
                },
            }
