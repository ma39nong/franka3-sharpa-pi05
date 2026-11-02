"""Bounded retries for cross-camera alignment, never for stale observations."""

import math

from .ros_observation import stamp_ns
from .ros_observation import timing_issues

DEFAULT_RECOVERY_SEC = 0.3
MAX_RECOVERY_SEC = 0.5


class AlignmentRecovery:
    def __init__(self, seconds=DEFAULT_RECOVERY_SEC):
        if not math.isfinite(seconds) or not 0 <= seconds <= MAX_RECOVERY_SEC:
            raise ValueError("camera alignment recovery must be finite and in [0, 0.5] seconds")
        self.seconds = seconds
        self.started = None

    def wait(self, error, latest, *, now_ros, now_mono):
        """Return retry diagnostics, or fail without resetting the episode clock."""
        issues = timing_issues(latest, now_ros, now_mono)
        # Only a source-time spread/history mismatch is recoverable. Receiving
        # fresh messages must not conceal a stale/future stream or malformed input.
        if not issues or any(not issue.startswith("Multimodal") for issue in issues):
            raise RuntimeError("Camera observation is not eligible for alignment recovery: " + str(error)) from error
        now = now_mono / 1e9
        if self.started is None:
            self.started = now
        elapsed = now - self.started
        if elapsed >= self.seconds:
            raise RuntimeError(
                f"Camera alignment did not recover within {self.seconds:.3f}s (elapsed={elapsed:.3f}s): {error}"
            ) from error
        return {
            "event": "camera_alignment_wait",
            "elapsed_s": elapsed,
            "recovery_window_s": self.seconds,
            "reason": str(error),
            "topics": {
                key: {
                    "stamp_ns": stamp_ns(message),
                    "source_age_ms": (now_ros - stamp_ns(message)) / 1e6,
                    "received_age_ms": (now_mono - received) / 1e6,
                }
                for key, (message, received) in latest.items()
            },
        }

    def recovered(self, now_mono):
        if self.started is None:
            return None
        elapsed = now_mono / 1e9 - self.started
        if elapsed >= self.seconds:
            raise RuntimeError(f"Camera alignment recovered too late ({elapsed:.3f}s >= {self.seconds:.3f}s)")
        report = {"event": "camera_alignment_recovered", "elapsed_s": elapsed}
        self.started = None
        return report
