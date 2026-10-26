from deploy.fr3_sharpa_runtime.protocol import DT_NS, FPS
import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.protocol import LEFT_SHARPA_SLICE, RIGHT_SHARPA_SLICE
from deploy.fr3_sharpa_runtime.sdk_direct import (
    DirectSdkConfig, SharpaSdkDirectExecutor, validate_direct_hand_actions,
    scaled_action_frames,
)


def chunk(value=0.0):
    return np.full((50, 58), value, dtype=np.float32)


def test_direct_validator_rejects_slew(monkeypatch):
    # This is a slew unit test, independent of the workstation's vendor asset.
    from deploy.fr3_sharpa_runtime import sdk_direct
    monkeypatch.setattr(sdk_direct, "hand_joint_limits", lambda: np.tile([-1., 1.], (22, 1)))
    actions = chunk()
    actions[1, LEFT_SHARPA_SLICE.start] = 0.11
    with pytest.raises(ValueError, match="slew"):
        validate_direct_hand_actions(actions)


def test_scaled_action_frames_preserves_targets_and_stretches_time():
    actions = chunk()
    actions[1, 0] = 0.1
    actions[2, 0] = 0.2
    frames = np.asarray(list(scaled_action_frames(actions, 0.1)))
    assert frames.shape == (491, 58)
    assert np.allclose(frames[0], actions[0])
    assert np.allclose(frames[10], actions[1])
    assert np.allclose(frames[20], actions[2])
    assert np.allclose(frames[-1], actions[-1])


def test_scaled_action_frames_rejects_invalid_scale():
    with pytest.raises(ValueError, match="speed scale"):
        list(scaled_action_frames(chunk(), 0.0))


class FakeHand:
    def __init__(self):
        self.connected = False
        self.sent = []
        self.stopped = False

    def connect(self): self.connected = True
    def command_joint_position(self, values): self.sent.append(tuple(values))
    def stop(self): self.stopped = True
    def disconnect(self): self.connected = False


def test_direct_executor_uses_injected_hands_without_sdk_import():
    hands = []
    def factory(side, serial):
        hand = FakeHand(); hands.append(hand); return hand
    executor = SharpaSdkDirectExecutor(DirectSdkConfig(watchdog_timeout_s=1.0), factory)
    executor.connect()
    executor.send([0.0] * 22, [0.0] * 22)
    assert len(hands) == 2 and all(hand.sent for hand in hands)
    executor.stop()
    assert all(hand.stopped for hand in hands)


def test_resume_stream_starts_a_new_watchdog_epoch():
    executor = SharpaSdkDirectExecutor(DirectSdkConfig(watchdog_timeout_s=1.0),
                                       lambda side, serial: FakeHand())
    executor.connect()
    executor.send([0.0] * 22, [0.0] * 22)
    assert executor._last_send is not None
    executor.resume_stream()
    assert executor._last_send is None
    executor.send([0.0] * 22, [0.0] * 22)
    assert executor._last_send is not None
    executor.stop()


def test_stop_attempts_both_hands_when_one_stop_fails():
    class FailingStopHand(FakeHand):
        def stop(self):
            self.stopped = True
            raise RuntimeError("link lost")

    hands = []
    def factory(side, serial):
        hand = FailingStopHand() if side == "left" else FakeHand()
        hands.append(hand)
        return hand
    executor = SharpaSdkDirectExecutor(DirectSdkConfig(), factory)
    executor.connect()
    errors = executor.stop()
    assert any(error.startswith("left.stop:") for error in errors)
    assert all(hand.stopped for hand in hands)
    assert all(not hand.connected for hand in hands)


def test_chunk_bridge_covers_large_first_target_and_retains_full_horizon():
    from deploy.fr3_sharpa_runtime.sdk_direct import chunk_command_frames
    actions = chunk(0.12)
    actions[1:, 7] += 0.15  # Raw jump is too large; slow emitted frames are valid.
    frames, bridge = chunk_command_frames(actions, np.zeros(58), speed_scale=0.1)
    assert bridge == 120
    assert np.max(np.abs(frames[0])) <= 0.03 * (DT_NS/1e9) + 1e-7
    assert np.allclose(frames[bridge - 1], actions[0])
    assert np.allclose(frames[-1], actions[-1])
    assert len(frames) == bridge + 491
    assert np.max(np.abs(np.diff(frames[:, 7:29], axis=0))) < 0.10
    with pytest.raises(ValueError, match="Interpolated Sharpa stream"):
        chunk_command_frames(actions, np.zeros(58), speed_scale=1.0)


def test_nonreciprocal_scale_actually_slows_the_chunk():
    assert len(list(scaled_action_frames(chunk(), 0.7))) > 50


def test_clock_scheduler_does_not_accumulate_send_cost_or_burst_after_stall():
    from deploy.fr3_sharpa_runtime.sdk_direct import play_command_frames
    class Clock:
        now = 0.0
        def clock(self): return self.now
        def sleep(self, duration): self.now += duration
    timer = Clock()
    starts = []
    def send(frame):
        starts.append(timer.now)
        timer.now += 0.01
    play_command_frames(chunk()[:4], send, lambda: None,
                        clock=timer.clock, sleep=timer.sleep)
    assert starts == pytest.approx([i * DT_NS/1e9 for i in range(4)])
    assert timer.now == pytest.approx(4 * DT_NS/1e9)
    starts.clear()
    def stalled_send(frame):
        starts.append(timer.now)
        timer.now += 0.11
    with pytest.raises(RuntimeError, match="deadline missed"):
        play_command_frames(chunk()[:4], stalled_send, lambda: None,
                            clock=timer.clock, sleep=timer.sleep)
    assert len(starts) == 1


def test_sdk_direct_can_validate_raw_ranges_before_time_stretching(monkeypatch):
    from deploy.fr3_sharpa_runtime import sdk_direct
    monkeypatch.setattr(sdk_direct, "hand_joint_limits", lambda: np.tile([-1., 1.], (22, 1)))
    actions = chunk()
    actions[1:, 7] = 0.15
    validate_direct_hand_actions(actions, check_slew=False)
    with pytest.raises(ValueError, match="slew"):
        validate_direct_hand_actions(actions)
