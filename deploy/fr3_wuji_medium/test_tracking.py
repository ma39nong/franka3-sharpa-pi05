"""One medium tracking threshold across the player, consumer and prefetch."""

import json

import numpy as np
import pytest

from .prefetch import endpoint_ready
from .runtime.core import ConsumerGuard
from .runtime.core import Feedback
from .runtime.core import Frame
from .runtime.limits import ARM_TRACKING_TOLERANCE_RAD
from .test_medium import make_player


def feedback(q, now):
    return Feedback(q, np.zeros(54), (now,) * 4, (now,) * 4, hand_control="slider")


@pytest.mark.parametrize("error", [0.081614, 0.199999, 0.200001])
def test_player_consumer_and_prefetch_agree_on_point_two(error):
    assert ARM_TRACKING_TOLERANCE_RAD == 0.2
    p = make_player()
    p.start(feedback(np.zeros(54), 10.0), 10.0)
    p.transition("playback", 10.0, "test")
    measured = p.plan.playback.sample(0.1)
    measured[6] -= error
    result = p.tick(feedback(measured, 10.1), 10.1)
    assert (result is not None) == (error <= 0.2)
    if error > 0.2:
        assert "exceeds 0.20 rad" in p.reason
    guard = ConsumerGuard(np.full(54, -2.0), np.full(54, 2.0), hand_control="slider")
    guard.arm("run", "hash", feedback(np.zeros(54), 10.0), 10.0)
    frame = Frame("run", "hash", 0, 10.0, 10.02, 0.0, np.zeros(54), np.zeros(54), "playback")
    measured = np.zeros(54)
    measured[6] = -error
    if error <= 0.2:
        guard.validate(frame, feedback(measured, 10.0), 10.0)
    else:
        with pytest.raises(ValueError, match="exceeds 0.20 rad"):
            guard.validate(frame, feedback(measured, 10.0), 10.0)
    p.settled_since = 10.0
    measured = p.plan.raw[-1].copy()
    measured[6] -= error
    assert endpoint_ready(p, feedback(measured, 10.0)) == (error <= 0.2)


def test_runtime_records_new_limit_and_slow_remains_unchanged(tmp_path):
    from experiments.weight_motion_eval.oneshot.limits import ARM_TRACKING_TOLERANCE_RAD as slow

    from .deploy import parse_args
    from .deploy import write_runtime

    assert slow == 0.08
    runtime, _, _ = write_runtime(parse_args(["--check", "--model", "19999"]), tmp_path)
    assert runtime["arm_tracking_rad"] == 0.2
    assert json.loads((tmp_path / "runtime.json").read_text())["arm_tracking_rad"] == 0.2
