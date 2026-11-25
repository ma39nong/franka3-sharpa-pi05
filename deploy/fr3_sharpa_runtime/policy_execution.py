"""Automatic command handling shared by ordinary deployment and DAgger."""

import numpy as np

from .policy_prepare import target_chunk
from .ros_transport import validate_action_ranges
from .sdk_direct import DEFAULT_PREDICTION_CLIP_RAD, clip_hand_predictions
from .stage_handoff import prepare_rtc_entry, validate_command_step


def prepare_policy_actions(actions, *, limiter=None,
                           max_correction_rad=DEFAULT_PREDICTION_CLIP_RAD):
    if limiter is not None:
        from .protocol import as_finite_action_chunk

        raw = as_finite_action_chunk(actions)
        projected = limiter.project(raw)
        adjustment = np.abs(projected - raw)
        return projected, {
            "joint_range_projected_values": int(np.count_nonzero(adjustment)),
            "max_joint_range_projection_rad": float(adjustment.max()),
        }
    actions, clipping = clip_hand_predictions(actions, execute_steps=50,
        max_correction_rad=max_correction_rad)
    validate_action_ranges(actions, check_slew=False)
    return actions, clipping


def policy_command(previous, target, *, limiter=None, arm_speed=None):
    command = limiter.step_toward(previous, target) if limiter is not None else target
    transition = target_chunk(command)
    transition[0] = previous
    validate_action_ranges(transition)
    if limiter is not None:
        validate_command_step(previous, command, arm_speed)
    return command


def install_policy_rtc(timeline, actions, receipt, now, previous, *,
                       task_epoch=None, arm_speed=None):
    def validate_window(window, *, check_slew=True):
        padded = target_chunk(window[-1])
        padded[:len(window)] = window
        validate_action_ranges(padded, check_slew=check_slew)

    def prepare_entry(candidate, skipped):
        return prepare_rtc_entry(candidate, skipped, previous, arm_speed,
            lambda window: validate_window(window, check_slew=False), validate_window)

    result = timeline.install(actions, receipt, now, task_epoch=task_epoch,
        prepare_entry=prepare_entry if arm_speed is not None else None)
    if not result.get("rejected") and arm_speed is None:
        validate_window(np.stack((previous, timeline.sample(now))))
    return result
