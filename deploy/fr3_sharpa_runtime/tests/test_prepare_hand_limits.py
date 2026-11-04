"""Home interpolation limits and two-hand prevalidation; no ROS/SDK devices."""

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime import sdk_direct as sdk
from deploy.fr3_sharpa_runtime.feedback import LiveFeedback
from deploy.fr3_sharpa_runtime.policy_prepare import SlowApproach
from deploy.fr3_sharpa_runtime.protocol import DT_NS
from deploy.fr3_sharpa_runtime.protocol import LEFT_SHARPA_SLICE
from deploy.fr3_sharpa_runtime.protocol import RIGHT_SHARPA_SLICE


@pytest.fixture
def limits(monkeypatch):
    bounds = np.tile([-1.0, 1.0], (22, 1))
    bounds[17] = [0.0, 0.2618]  # Driver pinky_CMC interval.
    bounds[0, 1] = 0.1  # float32(.1) lies outside this float64 upper bound.
    monkeypatch.setattr(sdk, "hand_joint_limits", lambda: bounds.copy())
    return bounds


def test_prepare_valid_frame_and_feedback_are_unchanged(limits):
    previous = np.zeros(58, np.float32)
    command = np.full(58, 0.001, np.float32)
    result, changes = sdk.prepare_command_with_limits(command, previous, speed_rad_s=0.03)
    np.testing.assert_array_equal(result, command)
    np.testing.assert_array_equal(previous, np.zeros(58))
    assert not changes


@pytest.mark.parametrize(("side", "part"), [("left", LEFT_SHARPA_SLICE), ("right", RIGHT_SHARPA_SLICE)])
def test_tiny_feedback_excursion_can_reach_legal_boundary_within_existing_tick(limits, side, part):
    measured = np.zeros(58, np.float32)
    measured[part.start + 17] = -0.0005
    raw = measured.copy()
    result, changes = sdk.prepare_command_with_limits(raw, measured, speed_rad_s=0.03)
    assert result[part.start + 17] == 0
    assert changes[0]["side"] == side
    assert changes[0]["joint"] == "pinky_CMC"
    assert changes[0]["movement_rad"] <= 0.03 * DT_NS / 1e9
    assert measured[part.start + 17] < 0  # No fabricated feedback.
    np.testing.assert_array_equal(raw, measured)
    sdk.validate_hand_command(result[part], side)


@pytest.mark.parametrize("value", [-0.01, 0.28])
def test_large_feedback_excursion_fails_with_angle_and_limit_diagnostics(limits, value):
    measured = np.zeros(58, np.float32)
    measured[LEFT_SHARPA_SLICE.start + 17] = value
    with pytest.raises(
        ValueError, match="left Sharpa pinky_CMC.*previous_or_initial_feedback=.*limits=.*allowed prepare step"
    ):
        sdk.prepare_command_with_limits(measured, measured, speed_rad_s=0.03)


def test_float32_boundary_is_represented_inside_driver_interval(limits):
    measured = np.zeros(58, np.float32)
    measured[LEFT_SHARPA_SLICE.start] = np.float32(0.1)
    result, changes = sdk.prepare_command_with_limits(measured, measured, speed_rad_s=0.03)
    assert float(result[LEFT_SHARPA_SLICE.start]) <= 0.1
    assert changes
    sdk.validate_hand_command(result[LEFT_SHARPA_SLICE], "left")


def test_projection_cannot_hide_excessive_movement(limits):
    previous = np.zeros(58, np.float32)
    previous[LEFT_SHARPA_SLICE.start + 17] = -0.01
    command = previous.copy()
    command[LEFT_SHARPA_SLICE.start + 17] = -0.0001
    with pytest.raises(ValueError, match="allowed prepare step"):
        sdk.prepare_command_with_limits(command, previous, speed_rad_s=0.03)


def test_slow_approach_boundary_projection_preserves_target_and_continuity(limits):
    measured = np.zeros(58, np.float32)
    measured[LEFT_SHARPA_SLICE.start + 17] = -0.0005
    target = np.zeros(58, np.float32)
    target[0] = 0.5  # Global interpolation leaves the negative pinky near its start.
    target[LEFT_SHARPA_SLICE.start + 17] = 0.125
    planner = SlowApproach(target, measured, speed_rad_s=0.03)
    for index in range(10):
        now = 1_000_000_000 + index * DT_NS
        feedback = LiveFeedback(measured.copy(), (now,) * 4, (now,) * 4)
        previous = planner.previous.copy()
        raw, _, _ = planner.step(feedback, now_ros_ns=now, now_monotonic_ns=now, enforce_execute_delta=False)
        command, corrections = sdk.prepare_command_with_limits(raw, previous, speed_rad_s=0.03)
        if index == 0:
            assert corrections
        sdk.validate_hand_command(command[LEFT_SHARPA_SLICE], "left")
        sdk.validate_hand_command(command[RIGHT_SHARPA_SLICE], "right")
        assert np.max(abs(command.astype(float) - previous)) <= 0.00100001
        planner.previous = command.copy()
        measured = command.copy()
    np.testing.assert_array_equal(planner.target, target)


@pytest.mark.parametrize("bad_side", ["left", "right"])
def test_both_hands_prevalidated_before_either_sdk_write(limits, bad_side):
    writes = []

    class Hand:
        def command_joint_position(self, values):
            writes.append(values)

    executor = sdk.SharpaSdkDirectExecutor()
    executor.left = Hand()
    executor.right = Hand()
    left, right = np.zeros(22), np.zeros(22)
    (left if bad_side == "left" else right)[17] = -0.01
    with pytest.raises(ValueError, match=bad_side + " Sharpa pinky_CMC.*outside"):
        executor.send(left, right)
    assert not writes
    assert executor._last_send is None  # noqa: SLF001 -- Verify rejection did not start the watchdog epoch.


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_rejected_before_either_hand_write(limits, bad):
    class Hand:
        def command_joint_position(self, values):
            pytest.fail("invalid hand must not be sent")

    executor = sdk.SharpaSdkDirectExecutor()
    executor.left = Hand()
    executor.right = Hand()
    right = np.zeros(22)
    right[0] = bad
    with pytest.raises(ValueError, match="finite"):
        executor.send(np.zeros(22), right)
