"""Bounded endpoint prefetch; motion and inference remain in separate processes."""
import numpy as np
from .player import ARM
from .runtime.limits import ARM_TRACKING_TOLERANCE_RAD

MAX_PREFETCH_AGE_SECONDS = 2.0


def endpoint_ready(player, feedback):
    # Player sets settled_since only after measured arms AND hands pass readiness.
    # Require tracking tolerance too, since that is tighter than endpoint tolerance.
    return (player.settled_since is not None
            and np.max(np.abs(feedback.positions[ARM] - player.plan.raw[-1, ARM])) <= ARM_TRACKING_TOLERANCE_RAD
            and np.max(np.abs(feedback.velocities[player.checked])) <= 0.02)


def accept_prefetch(candidate, now):
    candidate.admission.check()
    age = now - candidate.admission.observation_time
    if not 0 <= age <= MAX_PREFETCH_AGE_SECONDS:
        raise ValueError("Medium prefetched observation expired before activation")
