"""Local, stateful 14-joint motion generation for the fast client only.

Keep the last commanded position, velocity and acceleration across *all* policy
nodes, chunk switches and settling phases. Ruckig runs locally with a single
target and no intermediate waypoints or cloud service. Hardware stop handling
never passes through this generator.
"""

import math

import numpy as np

from experiments.weight_motion_eval.oneshot.core import ARM, vector

from . import limits as fast_limits

RUCKIG_VERSION = "0.12.2"


def smoothing_settings():
    acceleration = fast_limits.ARM_ACCELERATION_RAD_S2
    jerk = fast_limits.ARM_JERK_RAD_S3
    for name, value in (("acceleration", acceleration), ("jerk", jerk)):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Fast arm {name} limit must be finite and positive")
    return {
        "algorithm": "ruckig_position_target",
        "version": RUCKIG_VERSION,
        "acceleration_rad_s2": float(acceleration),
        "jerk_rad_s3": float(jerk),
        "target_velocity_rad_s": 0.0,
        "scope": "fast_arms_only",
    }


def checked_backend():
    try:
        import ruckig
    except ImportError as error:
        raise RuntimeError("Fast smoothing requires deploy/fr3_wuji_fast/requirements.txt") from error
    if ruckig.__version__ != RUCKIG_VERSION:
        raise RuntimeError(f"Fast smoothing requires ruckig=={RUCKIG_VERSION}")
    return ruckig


class ArmSmoother:
    def __init__(self, start, limits, now, *, speed):
        self.settings = smoothing_settings()
        self.backend = checked_backend()
        self.position = vector(start)[ARM].copy()
        self.velocity = np.zeros(len(ARM))
        self.acceleration = np.zeros(len(ARM))
        self.lower = vector(limits["lower"])[ARM].copy()
        self.upper = vector(limits["upper"])[ARM].copy()
        if not math.isfinite(now) or not math.isfinite(speed) or speed <= 0:
            raise ValueError("Invalid fast smoothing time or speed")
        if np.any(self.lower >= self.upper) or np.any(self.position < self.lower) or np.any(self.position > self.upper):
            raise ValueError("Fast smoothing initial position outside joint limits")
        self.speed = speed
        self.last_time = now
        self.started = False
        self.max_deviation = 0.0
        self.generator = self.backend.Ruckig(len(ARM))
        self.input = self.backend.InputParameter(len(ARM))
        self.trajectory = self.backend.Trajectory(len(ARM))
        self.input.max_velocity = [speed] * len(ARM)
        self.input.max_acceleration = [self.settings["acceleration_rad_s2"]] * len(ARM)
        self.input.max_jerk = [self.settings["jerk_rad_s3"]] * len(ARM)
        self.input.target_velocity = [0.0] * len(ARM)
        self.input.target_acceleration = [0.0] * len(ARM)
        # Avoid a noisy target on one joint retiming the other thirteen joints.
        self.input.synchronization = self.backend.Synchronization.No

    def settled(self, target):
        return (
            np.max(np.abs(self.position - np.asarray(target)[ARM])) <= 1e-8
            and np.max(np.abs(self.velocity)) <= 1e-6
            and np.max(np.abs(self.acceleration)) <= 1e-5
        )

    def step(self, target, requested_velocity, now):
        q, dq = vector(target).copy(), vector(requested_velocity).copy()
        dt = now - self.last_time
        if not math.isfinite(dt) or dt < 0 or dt > 0.030000001 or (self.started and dt == 0):
            raise ValueError("Fast smoothing clock reversed or command stream stalled")
        if np.any(q[ARM] < self.lower) or np.any(q[ARM] > self.upper):
            raise ValueError("Fast smoothing target outside joint limits")
        self.input.current_position = self.position.tolist()
        self.input.current_velocity = self.velocity.tolist()
        self.input.current_acceleration = self.acceleration.tolist()
        self.input.target_position = q[ARM].tolist()
        # The current state is our previously checked output. Ruckig 0.12.2's
        # strict current-state viability test can reject its own result at the
        # speed boundary through floating-point roundoff. Validate the target
        # here; retain explicit finite/speed/acceleration/jerk checks below.
        self.generator.validate_input(self.input, False, True)
        result = self.generator.calculate(self.input, self.trajectory)
        if result not in (self.backend.Result.Working, self.backend.Result.Finished):
            raise RuntimeError(f"Fast smoothing calculation failed: {result}")
        # Check the complete braking/reversal trajectory, not just the next tick.
        extrema = self.trajectory.position_extrema
        if any(e.min < lo - 1e-10 or e.max > hi + 1e-10 for e, lo, hi in zip(extrema, self.lower, self.upper)):
            raise ValueError("Fast smoothing braking trajectory exceeds joint limits")
        position, velocity, acceleration = (np.asarray(x) for x in self.trajectory.at_time(dt))
        if not all(np.isfinite(x).all() for x in (position, velocity, acceleration)):
            raise RuntimeError("Nonfinite fast smoothing result")
        if (
            np.any(position < self.lower) or np.any(position > self.upper)
            or np.max(np.abs(velocity)) > self.speed + 1e-8
            or np.max(np.abs(acceleration)) > self.settings["acceleration_rad_s2"] + 1e-8
            or np.max(np.abs(acceleration - self.acceleration)) > self.settings["jerk_rad_s3"] * dt + 1e-8
        ):
            raise RuntimeError("Fast smoothing result exceeds motion limits")
        self.max_deviation = max(self.max_deviation, float(np.max(np.abs(q[ARM] - position))))
        self.position, self.velocity, self.acceleration = position, velocity, acceleration
        self.last_time, self.started = now, True
        q[ARM], dq[ARM] = position, velocity
        return q, dq
