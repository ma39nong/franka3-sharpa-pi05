"""Wait for startup source clocks to settle; never retimestamp a frame."""

from .ros_observation import MAX_AGE_NS

STARTUP_SETTLE_SEC = 30.0
CLOCK_STABLE_SEC = 10.0
CAMERA_STARTUP_TIMEOUT_SEC = 90.0
CAMERA_PROCESS_TIMEOUT_SEC = 105.0


class SourceClockSettling:
    def __init__(self, keys, started_ns):
        self.started_ns = started_ns
        self.streams = {key: {"stamp": None, "received": None, "valid_since": None, "invalid_frames": 0,
                              "invalid_reasons": {}, "last_invalid": None}
                        for key in keys}

    def observe(self, key, stamp, *, now_ros, now_mono):
        state = self.streams[key]
        age = now_ros - stamp
        gap = now_mono - state["received"] if state["received"] is not None else None
        delta = stamp - state["stamp"] if state["stamp"] is not None else None
        reasons = []
        if stamp <= 0 or age > MAX_AGE_NS:
            reasons.append("stale_source")
        if age < 0:
            reasons.append("future_source")
        if delta is not None and delta <= 0:
            reasons.append("nonadvancing_source")
        if gap is not None and gap > 150_000_000:
            reasons.append("receive_gap")
        if reasons:
            state["valid_since"] = None
            state["invalid_frames"] += 1
            for reason in reasons:
                state["invalid_reasons"][reason] = state["invalid_reasons"].get(reason, 0) + 1
            state["last_invalid"] = {"reasons": reasons, "stamp_ns": stamp,
                                     "source_age_ms": age / 1e6,
                                     "source_delta_ms": delta / 1e6 if delta is not None else None,
                                     "receive_gap_ms": gap / 1e6 if gap is not None else None}
        elif state["valid_since"] is None:
            state["valid_since"] = now_mono
        state["stamp"], state["received"] = stamp, now_mono

    def ready(self, now_mono):
        return (now_mono - self.started_ns >= STARTUP_SETTLE_SEC * 1e9
                and all(state["valid_since"] is not None
                        and now_mono - state["valid_since"] >= CLOCK_STABLE_SEC * 1e9
                        and 0 <= now_mono - state["received"] <= MAX_AGE_NS
                        for state in self.streams.values()))

    def diagnostic(self, now_mono):
        return {"event": "camera_source_clock_settling",
                "elapsed_s": (now_mono - self.started_ns) / 1e9,
                "minimum_startup_s": STARTUP_SETTLE_SEC, "required_clock_stable_s": CLOCK_STABLE_SEC,
                "cameras": {key: {"valid_clock_s": (now_mono - state["valid_since"]) / 1e9
                                  if state["valid_since"] is not None else 0,
                                  "invalid_frames": state["invalid_frames"],
                                  "invalid_reasons": dict(state["invalid_reasons"]),
                                  "last_invalid": state["last_invalid"]}
                            for key, state in self.streams.items()}}
