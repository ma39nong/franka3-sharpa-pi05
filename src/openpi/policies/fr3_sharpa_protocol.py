"""Joint protocol for a dual FR3 + dual Sharpa Wave workcell.

This module is deliberately independent from the Wuji policy.  It records the
physical order used by the Sharpa dataset and makes the distinction between
measured state and commanded action explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FR3_JOINTS = tuple(f"{side}_fr3_joint{i}" for side in ("left", "right") for i in range(1, 8))
SHARPA_JOINTS = (
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
    "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
)
ORDER = (
    tuple(f"left_fr3_joint{i}" for i in range(1, 8))
    + tuple(f"left_{name}" for name in SHARPA_JOINTS)
    + tuple(f"right_fr3_joint{i}" for i in range(1, 8))
    + tuple(f"right_{name}" for name in SHARPA_JOINTS)
)
ACTION_DIM = len(ORDER)
STATE_DIM = ACTION_DIM * 2
UNIT = "rad"


@dataclass(frozen=True)
class JointSpec:
    index: int
    name: str
    side: str
    kind: str
    unit: str = UNIT
    source: str = ""


JOINT_SPECS = tuple(
    JointSpec(i, name, "left" if name.startswith("left_") else "right",
              "arm" if "fr3_joint" in name else "hand",
              source=("/left/franka/joint_states" if name.startswith("left_fr3_") else
                      "/right/franka/joint_states" if name.startswith("right_fr3_") else
                      "/sharpa/left/joint_states" if name.startswith("left_") else
                      "/sharpa/right/joint_states"))
    for i, name in enumerate(ORDER)
)


def reorder_named(values, names, expected=ORDER) -> np.ndarray:
    """Return values in protocol order; reject missing, duplicate, or non-finite data."""
    if len(values) != len(names):
        raise ValueError(f"name/value length mismatch: {len(names)} != {len(values)}")
    mapping = dict(zip(names, values, strict=True))
    missing = [name for name in expected if name not in mapping]
    if missing:
        raise ValueError(f"missing joints: {missing}")
    result = np.asarray([mapping[name] for name in expected], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("joint vector contains NaN or Inf")
    return result


def join_state(arm_left, hand_left, arm_right, hand_right) -> np.ndarray:
    """Build measured state [left arm, left hand, right arm, right hand]."""
    parts = (arm_left, hand_left, arm_right, hand_right)
    if any(np.asarray(part).shape != (7 if i in (0, 2) else 22,) for i, part in enumerate(parts)):
        raise ValueError("invalid measured state component shape")
    return np.concatenate(parts).astype(np.float32, copy=False)


def join_action(arm_left, hand_left, arm_right, hand_right) -> np.ndarray:
    """Build commanded action [left arm, left hand, right arm, right hand]."""
    return join_state(arm_left, hand_left, arm_right, hand_right)
