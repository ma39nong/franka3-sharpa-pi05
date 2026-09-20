"""Limits specific to fast deployment; slow deployment keeps its own defaults."""

# Arm tracking error in radians, used by fast playback and its device send guard.
ARM_TRACKING_TOLERANCE_RAD = 0.2

# Fast arm command smoothing only. Restart the fast client after changing these.
# Speed remains the existing 1 rad/s limit in timeline.py.
ARM_ACCELERATION_RAD_S2 = 3.0
ARM_JERK_RAD_S3 = 30.0
