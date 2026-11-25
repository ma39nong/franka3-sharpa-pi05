"""Separate offline diagnostics from the live first-target acquisition check."""

from __future__ import annotations

import numpy as np

from .feedback import LiveFeedback
from .protocol import (
    LEFT_FR3_SLICE, LEFT_SHARPA_SLICE, RIGHT_FR3_SLICE, RIGHT_SHARPA_SLICE,
    as_finite_action_chunk,
)

FIRST_TARGET_LIMITS_RAD = {"left_fr3": 0.05, "right_fr3": 0.05, "left_sharpa": 0.15, "right_sharpa": 0.15}
GROUP_SLICES = {
    "left_fr3": LEFT_FR3_SLICE, "left_sharpa": LEFT_SHARPA_SLICE,
    "right_fr3": RIGHT_FR3_SLICE, "right_sharpa": RIGHT_SHARPA_SLICE,
}


def execute_preflight(actions, feedback: LiveFeedback, *, now_ros_ns: int, now_monotonic_ns: int,
                      groups=None) -> dict:
    chunk = as_finite_action_chunk(actions)
    if not isinstance(feedback, LiveFeedback):
        raise ValueError("EXECUTE PREFLIGHT requires fresh ROS feedback, not an episode state array")
    feedback.validate_freshness(now_ros_ns=now_ros_ns, now_monotonic_ns=now_monotonic_ns)
    selected = tuple(GROUP_SLICES) if groups is None else tuple(groups)
    if not selected or any(name not in GROUP_SLICES for name in selected):
        raise ValueError("Unknown or empty preflight joint groups")
    deltas = {name: float(np.max(np.abs(chunk[0, GROUP_SLICES[name]] - feedback.state[GROUP_SLICES[name]])))
              for name in selected}
    for name, delta in deltas.items():
        if delta > FIRST_TARGET_LIMITS_RAD[name]:
            raise ValueError(f"EXECUTE PREFLIGHT: FAIL; {name} first-target delta {delta:.6f} rad "
                             f"> {FIRST_TARGET_LIMITS_RAD[name]:.6f} rad against live ROS feedback")
    return {"status": "PASS", "reference": "live_ros_feedback", "compared_action_index": 0,
            "max_delta_rad": deltas, "limits_rad": {name: FIRST_TARGET_LIMITS_RAD[name] for name in selected}}


def offline_sanity_check(actions, episode_state) -> dict:
    """Retain the old episode-relative checks as non-blocking diagnostics only."""
    issues = []
    try:
        chunk = as_finite_action_chunk(actions)
        reference = np.asarray(episode_state, dtype=np.float32)
        if reference.shape != (58,) or not np.isfinite(reference).all():
            raise ValueError("Offline reference must be a finite 58D episode state")
        deltas = {}
        for name, part in GROUP_SLICES.items():
            compared = chunk[:, part] if "sharpa" in name else chunk[:1, part]
            delta = float(np.max(np.abs(compared - reference[part])))
            deltas[name] = delta
            if delta > FIRST_TARGET_LIMITS_RAD[name]:
                issues.append(f"{name} episode-relative delta {delta:.6f} rad exceeds diagnostic bound")
            if "sharpa" in name and np.max(np.abs(np.diff(chunk[:, part], axis=0))) >= 0.10:
                issues.append(f"{name} raw adjacent action delta reaches 0.10 rad")
        return {"status": "FAIL" if issues else "PASS", "blocking": False,
                "reference": "offline_episode_state", "max_delta_rad": deltas, "issues": issues}
    except ValueError as error:
        return {"status": "FAIL", "blocking": False, "reference": "offline_episode_state", "issues": [str(error)]}
