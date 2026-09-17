"""Bounded raw SDK flight recorder; disk writes happen outside motion dispatch."""

from collections import deque
import json
from pathlib import Path
import time


class HandTrace:
    def __init__(self, capacity=20000):
        self.before = deque(maxlen=capacity)
        self.after = deque(maxlen=capacity)
        self.triggered = False
        self.dropped_before = self.dropped_after = 0
        self.coalesced_before = {"state": 0, "diagnostics": 0}
        self.coalesced_after = {"state": 0, "diagnostics": 0}

    def add(self, event, **values):
        target = self.after if self.triggered else self.before
        if len(target) == target.maxlen:
            if self.triggered:
                self.dropped_after += 1
            else:
                self.dropped_before += 1
        target.append({"event": event, "received_mono": time.monotonic(), "received_wall": time.time(), **values})

    def trigger(self, reason):
        if not self.triggered:
            self.add("fault_trigger", reason=str(reason))
            self.triggered = True

    def coalesce(self, *, states=0, diagnostics=0):
        """Account for healthy frames represented by the newest frame in a poll."""
        target = self.coalesced_after if self.triggered else self.coalesced_before
        target["state"] += int(states)
        target["diagnostics"] += int(diagnostics)

    def frame(self, frame, *, diagnostic=False):
        joints = []
        for j in frame.joints:
            item = {"nid": int(j.nid)}
            if diagnostic:
                status = j.status_word
                item.update(error_code=int(j.error_code_current), ext_state=int(status.ext_state))
                for field in ("position_limit_active", "velocity_limit_active", "current_limit_active"):
                    item[field] = bool(getattr(status, field, False))
                for field in ("current", "vbus_v_fb", "mcu_temp_c_fb"):
                    value = getattr(j, field, None)
                    item[field] = None if value is None else float(value)
            else:
                item.update(position=float(j.position), velocity=float(j.velocity))
                item["effort"] = None if not hasattr(j, "effort") else float(j.effort)
            joints.append(item)
        self.add(
            "diagnostics" if diagnostic else "state",
            device_timestamp_us=int(frame.header.timestamp_us),
            device_sequence=int(frame.header.seq) if hasattr(frame.header, "seq") else None,
            num_joints=int(frame.num_joints),
            joints=joints,
        )

    def write(self, path, identity):
        path = Path(path)
        with path.open("x") as stream:
            stream.write(
                json.dumps(
                    {
                        "event": "metadata",
                        "schema_version": 1,
                        "identity": identity,
                        "dropped_before": self.dropped_before,
                        "dropped_after": self.dropped_after,
                        "coalesced_before": self.coalesced_before,
                        "coalesced_after": self.coalesced_after,
                        "units": {"position": "rad", "velocity": "rad/s", "effort": "A"},
                        "capture": (
                            "newest state and diagnostic frame from each healthy poll; all frames from a "
                            "faulting poll; coalesced counters record omitted healthy frames; no requested "
                            "stream rate change"
                        ),
                    }
                )
                + "\n"
            )
            for event in (*self.before, *self.after):
                stream.write(json.dumps(event) + "\n")
