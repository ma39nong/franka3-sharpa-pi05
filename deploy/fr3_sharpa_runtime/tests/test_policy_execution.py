"""Shared automatic playback keeps the ordinary command and RTC contracts."""

import numpy as np
from openpi_client import rtc
import pytest

from deploy.fr3_sharpa_runtime import policy_execution as policy
from deploy.fr3_sharpa_runtime.rtc_stream import RTCTimeline


def test_policy_command_limits_target_and_validates_dispatched_transition(monkeypatch):
    previous = np.zeros(58, dtype=np.float32)
    target = np.full(58, .1, dtype=np.float32)
    checked = []
    monkeypatch.setattr(policy, "validate_action_ranges", lambda actions: checked.append(actions.copy()))

    class Limiter:
        def step_toward(self, previous, target):
            return previous + np.clip(target - previous, -.01, .01)

    command = policy.policy_command(previous, target, limiter=Limiter(), arm_speed=.3)
    np.testing.assert_allclose(command, .01)
    np.testing.assert_array_equal(checked[0][0], previous)
    np.testing.assert_array_equal(checked[0][1:], np.repeat(command[None, :], 49, axis=0))


def test_policy_command_preserves_unlimited_target_and_range_failure(monkeypatch):
    previous = np.zeros(58, dtype=np.float32)
    target = np.full(58, .02, dtype=np.float32)

    def validate(actions):
        if np.max(np.abs(actions)) > .03:
            raise ValueError("joint range exceeded")

    monkeypatch.setattr(policy, "validate_action_ranges", validate)
    np.testing.assert_array_equal(policy.policy_command(previous, target), target)
    with pytest.raises(ValueError, match="joint range"):
        policy.policy_command(previous, target + .1)


@pytest.mark.parametrize("bounded_entry", [False, True])
def test_rtc_replacement_keeps_receipts_and_shared_entry_rules(monkeypatch, bounded_entry):
    def validate(actions, *, check_slew=True):
        if check_slew and np.max(np.abs(np.diff(actions, axis=0))) > .011:
            raise ValueError("discontinuous command")

    monkeypatch.setattr(policy, "validate_action_ranges", validate)
    actions = np.zeros((50, 58), dtype=np.float32)
    timeline = RTCTimeline(actions, 1., 1.)
    context = timeline.snapshot(1., delay_steps=5, schedule="exp", weight=5.)
    candidate = np.full((50, 58), .02 if bounded_entry else .001, dtype=np.float32)
    result = policy.install_policy_rtc(timeline, candidate, rtc.make_receipt(context), 1.03,
        np.zeros(58, dtype=np.float32), arm_speed=.3 if bounded_entry else None)
    assert result["generation"] == 2 and result["skipped"] == 1
    if bounded_entry:
        assert result["entry_bridge_steps"] >= 6
        np.testing.assert_array_equal(timeline.sample(1.03), np.zeros(58))
    else:
        np.testing.assert_array_equal(timeline.sample(1.03), candidate[1])
    with pytest.raises(ValueError, match="outstanding RTC snapshot"):
        policy.install_policy_rtc(timeline, candidate, rtc.make_receipt(context), 1.04,
            np.zeros(58), arm_speed=.3 if bounded_entry else None)


def test_rejected_entry_preserves_old_timeline_and_consumes_receipt(monkeypatch):
    def reject(actions, **kwargs):
        raise ValueError("joint range exceeded")

    monkeypatch.setattr(policy, "validate_action_ranges", reject)
    actions = np.zeros((50, 58), dtype=np.float32)
    timeline = RTCTimeline(actions, 1., 1.)
    context = timeline.snapshot(1., delay_steps=5, schedule="exp", weight=5.)
    result = policy.install_policy_rtc(timeline, actions + .02, rtc.make_receipt(context),
        1.03, np.zeros(58), arm_speed=.3)
    assert result["rejected"] and timeline.generation == 1
    assert timeline.installed_at == 1. and timeline.pending is None
    np.testing.assert_array_equal(timeline.sample(1.04), actions[0])
