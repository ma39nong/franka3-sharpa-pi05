"""Validate final-controller heartbeats and retain the first reported fault."""

import json
import math

STATUS_TOPIC = "/{side}/joint_impedance_controller/pi05_status"
STATUS_MAX_AGE_SECONDS = 0.15


class ControllerStatusUnavailable(ValueError):
    """Preparation may wait for status; an active session must stop."""


class ControllerStatus:
    def __init__(self):
        self.latest = {}
        self.first_fault = None

    def receive(self, side, data, now, wall):
        try:
            if side not in {"left", "right"} or len(data) > 4096:
                raise ValueError("invalid side or oversized status")
            value = json.loads(data)
            if value.get("version") != 1 or type(value.get("faulted")) is not bool:
                raise ValueError("unsupported controller status schema")
            if type(value.get("stamp_ns")) is not int or value["stamp_ns"] <= 0:
                raise ValueError("missing controller status timestamp")
            if not isinstance(value.get("reason"), str) or not value["reason"]:
                raise ValueError("missing controller fault reason")
            if (value["reason"] == "none") == value["faulted"]:
                raise ValueError("inconsistent controller fault flag")
            if not math.isfinite(now) or not math.isfinite(wall):
                raise ValueError("invalid receipt clock")
            age = wall - value["stamp_ns"] / 1e9
            if not -0.002 <= age <= STATUS_MAX_AGE_SECONDS:
                raise ValueError(f"stale/future controller status: age={age:.6f}s")
            record = {"side": side, "received_mono": now, "source_mono": now - age, "status": value}
            self.latest[side] = record
            pending = (
                self.first_fault is not None
                and self.first_fault["side"] == side
                and self.first_fault["status"]["reason"] == "fault_pending"
            )
            if value["faulted"] and (self.first_fault is None or pending):
                self.first_fault = record
        except (ValueError, TypeError, AttributeError) as error:
            if self.first_fault is None:
                self.first_fault = {
                    "side": side,
                    "received_mono": now,
                    "status": {"faulted": True, "reason": "invalid_status", "detail": str(error)},
                }

    def raise_fault(self):
        if self.first_fault is not None:
            raise RuntimeError("Final arm controller fault: " + json.dumps(self.first_fault, ensure_ascii=False))

    def check(self, now):
        self.raise_fault()
        for side in ("left", "right"):
            record = self.latest.get(side)
            if record is None:
                raise ControllerStatusUnavailable(
                    f"Missing {side} controller status; rebuild/start the diagnostic overlay"
                )
            if any(
                not -0.002 <= now - record[key] <= STATUS_MAX_AGE_SECONDS for key in ("received_mono", "source_mono")
            ):
                raise ControllerStatusUnavailable(f"Stale {side} controller status heartbeat")

    def report(self):
        return {"first_fault": self.first_fault, "latest": self.latest}
