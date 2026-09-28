"""A bounded two-stage Home plan: arms first, then Wuji hands."""

import numpy as np

from deploy.fr3_wuji_slow.core import ARM
from deploy.fr3_wuji_slow.core import HAND
from deploy.fr3_wuji_slow.planner import Plan
from deploy.fr3_wuji_slow.planner import make_phase

ARM_HOME_SPEED_RAD_S = 0.20
ARM_HOME_ACCELERATION_RAD_S2 = 0.40
ARM_HOME_JERK_RAD_S3 = 3.20
HAND_HOME_SPEED_RAD_S = 0.50
SAMPLE_HZ = 100
HAND_KNOTS = 50
HAND_SOURCE_HZ = 30
SETTLE_SECONDS = 0.5
MAX_TOTAL_SECONDS = 60.0


def _config():
    arm_caps = {
        "velocity": ARM_HOME_SPEED_RAD_S,
        "acceleration": ARM_HOME_ACCELERATION_RAD_S2,
        "jerk": ARM_HOME_JERK_RAD_S3,
    }
    # Slider-mode hands apply their own measured-feedback limiter. These caps
    # retain the complete Plan schema; hand timing is set explicitly below.
    hand_caps = {"velocity": HAND_HOME_SPEED_RAD_S, "acceleration": 1.0, "jerk": 5.0}
    return {
        "schema_version": 1,
        "hardware_output": False,
        "source_hz": HAND_SOURCE_HZ,
        "sample_hz": SAMPLE_HZ,
        "execution_steps": HAND_KNOTS,
        "settle_seconds": SETTLE_SECONDS,
        "hand_control": "slider",
        "approach": {"arm": dict(arm_caps), "hand": dict(hand_caps)},
        "playback": {"arm": dict(arm_caps), "hand": dict(hand_caps)},
    }


def build_home_plan(start, target, limits, names):
    start = np.asarray(start, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if start.shape != (54,) or target.shape != (54,) or not np.isfinite(start).all() or not np.isfinite(target).all():
        raise ValueError("Home start and target must each contain 54 finite positions")
    lower = np.asarray(limits["lower"], dtype=np.float64)
    upper = np.asarray(limits["upper"], dtype=np.float64)
    speed = np.asarray(limits["speed"], dtype=np.float64)
    if any(value.shape != (54,) for value in (lower, upper, speed)) or len(names) != 54:
        raise ValueError("Home planning requires the complete 54-joint reference contract")
    for label, value in (("measured start", start), ("tomato Home", target)):
        bad = np.flatnonzero((value < lower) | (value > upper))
        if bad.size:
            index = int(bad[0])
            raise ValueError(f"{label} exceeds reference limit: {names[index]}={value[index]:.6f}")
    if float(np.min(speed[ARM])) < ARM_HOME_SPEED_RAD_S:
        raise ValueError("Home arm speed exceeds the reference speed contract")

    arm_home = start.copy()
    arm_home[ARM] = target[ARM]
    fractions = np.linspace(0.0, 1.0, HAND_KNOTS)[:, None]
    raw = np.repeat(arm_home[None, :], HAND_KNOTS, axis=0)
    raw[:, HAND] = arm_home[HAND] + fractions * (target[HAND] - arm_home[HAND])
    config = _config()
    approach = make_phase(
        "approach",
        np.array([0.0, 1.0]),
        np.stack((start, arm_home)),
        config,
        limits,
    )
    hand_distance = float(np.max(np.abs(target[HAND] - start[HAND])))
    base_hand_seconds = (HAND_KNOTS - 1) / HAND_SOURCE_HZ
    playback_scale = max(1.0, hand_distance / (HAND_HOME_SPEED_RAD_S * base_hand_seconds))
    playback = make_phase(
        "playback",
        np.arange(HAND_KNOTS) / HAND_SOURCE_HZ,
        raw,
        config,
        limits,
        minimum_scale=playback_scale,
    )
    total = approach.duration + playback.duration + 2 * SETTLE_SECONDS
    if total > MAX_TOTAL_SECONDS:
        raise ValueError(f"Home requires {total:.1f}s, exceeding the {MAX_TOTAL_SECONDS:.0f}s bounded-motion limit")
    report = {
        "schema_version": 1,
        "mode": "standalone_tomato_home",
        "hardware_output": False,
        "planning_checks_passed": True,
        "sequence": ["both_arms_to_home_and_settle", "both_wuji_hands_to_home_and_settle"],
        "arm_speed_rad_s": ARM_HOME_SPEED_RAD_S,
        "arm_acceleration_rad_s2": ARM_HOME_ACCELERATION_RAD_S2,
        "arm_jerk_rad_s3": ARM_HOME_JERK_RAD_S3,
        "hand_target_speed_rad_s": HAND_HOME_SPEED_RAD_S,
        "approach_seconds": approach.duration,
        "playback_seconds": playback.duration,
        "settle_seconds": SETTLE_SECONDS,
        "total_seconds": total,
        "joint_names": list(names),
    }
    return Plan(
        raw=raw,
        knots=raw.copy(),
        start=start.copy(),
        approach=approach,
        playback=playback,
        config=config,
        report=report,
    )
