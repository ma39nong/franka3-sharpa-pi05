from types import SimpleNamespace

import pytest

from deploy.fr3_sharpa_runtime.camera_alignment_recovery import AlignmentRecovery


def latest(now, *, oldest_age_ms=170):
    result = {}
    for key, age in (("cam0", oldest_age_ms), ("cam1", 10), ("cam2", 20)):
        stamp = now - int(age * 1e6)
        message = SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp // 1_000_000_000, nanosec=stamp % 1_000_000_000))
        )
        result[key] = message, now - 1_000_000
    return result


def wait(recovery, now, **kwargs):
    return recovery.wait(
        ValueError("Multimodal observation timestamps are not synchronized; Waiting for history"),
        latest(now, **kwargs),
        now_ros=now,
        now_mono=now,
    )


def test_transient_alignment_failure_recovers_and_a_later_episode_gets_its_own_window():
    recovery = AlignmentRecovery(0.3)
    diagnostic = wait(recovery, 10_000_000_000)
    assert diagnostic["event"] == "camera_alignment_wait"
    assert "status" not in diagnostic  # A retry is never a PASS heartbeat or valid observation.
    assert diagnostic["topics"]["cam0"]["source_age_ms"] == 170
    event = recovery.recovered(10_150_000_000)
    assert event["event"] == "camera_alignment_recovered"
    assert event["elapsed_s"] == pytest.approx(0.15)
    assert recovery.recovered(10_160_000_000) is None
    assert wait(recovery, 11_000_000_000)["elapsed_s"] == 0


def test_fresh_new_messages_do_not_extend_a_persistent_alignment_failure():
    recovery = AlignmentRecovery(0.3)
    wait(recovery, 10_000_000_000)
    assert wait(recovery, 10_200_000_000)["elapsed_s"] == pytest.approx(0.2)
    with pytest.raises(RuntimeError, match="did not recover"):
        wait(recovery, 10_310_000_000)


@pytest.mark.parametrize("age_ms", [201, -1, 10_000])
def test_stale_future_or_silent_stream_is_not_given_an_alignment_grace_period(age_ms):
    recovery = AlignmentRecovery(0.3)
    with pytest.raises(RuntimeError, match="not eligible"):
        wait(recovery, 10_000_000_000, oldest_age_ms=age_ms)


def test_healthy_observation_failure_is_not_mislabeled_as_alignment_jitter():
    with pytest.raises(RuntimeError, match="not eligible"):
        wait(AlignmentRecovery(0.3), 10_000_000_000, oldest_age_ms=25)


def test_recovery_after_deadline_cannot_reset_the_failure_timer():
    recovery = AlignmentRecovery(0.3)
    wait(recovery, 10_000_000_000)
    with pytest.raises(RuntimeError, match="too late"):
        recovery.recovered(10_310_000_000)


@pytest.mark.parametrize("seconds", [-1, 0.51, float("inf"), float("nan")])
def test_invalid_or_unbounded_recovery_setting_is_rejected(seconds):
    with pytest.raises(ValueError, match="finite"):
        AlignmentRecovery(seconds)


def test_zero_restores_immediate_failure_behavior():
    with pytest.raises(RuntimeError, match="did not recover"):
        wait(AlignmentRecovery(0), 10_000_000_000)
