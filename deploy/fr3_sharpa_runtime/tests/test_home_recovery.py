"""Fixed-Home native recovery only; fake SDK, feedback and clocks, no hardware."""

# ruff: noqa: SLF001 -- Fake and fault-inject the recovery-only SDK setting adapter.

from types import SimpleNamespace

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime import home_recovery as recovery
from deploy.fr3_sharpa_runtime import sdk_direct as sdk
from deploy.fr3_sharpa_runtime.policy_prepare import prepare_and_run_chunks
from deploy.fr3_sharpa_runtime.protocol import LEFT_SHARPA_SLICE
from deploy.fr3_sharpa_runtime.protocol import RIGHT_SHARPA_SLICE


@pytest.fixture
def rig(monkeypatch):
    bounds = np.tile([-1.0, 1.0], (22, 1))
    bounds[17] = [0, 0.2618]
    monkeypatch.setattr(sdk, "hand_joint_limits", lambda: bounds.copy())
    monkeypatch.setattr(recovery, "hand_joint_limits", lambda: bounds.copy())
    state = np.zeros(58, np.float64)
    state[24] = 0.2879344225

    class Hand:
        def __init__(self):
            self.config = SimpleNamespace(interpolate=True)
            self.speed = 0.3
            self.settings = []

        def _read_setting(self, name):
            assert name == "get_speed_coeff"
            return self.speed

        def _ensure_coefficient(self, name, observed, desired):
            assert name == "speed"
            self.settings.append(desired)
            self.speed = desired

    class Executor:
        def __init__(self):
            self.left, self.right = Hand(), Hand()
            self.rows = []
            self.resumes = 0

        def resume_stream(self):
            self.resumes += 1

        def send(self, left, right):
            sdk.validate_hand_command(left, "left", limits=bounds)
            sdk.validate_hand_command(right, "right", limits=bounds)
            assert self.left.speed <= 0.05
            assert self.right.speed <= 0.05
            self.rows.append((np.array(left).copy(), np.array(right).copy()))
            # Simulate firmware interpolation, not instantaneous target motion.
            for part, target in ((LEFT_SHARPA_SLICE, left), (RIGHT_SHARPA_SLICE, right)):
                state[part] += np.clip(np.asarray(target) - state[part], -0.001, 0.001)

    executor = Executor()
    clock = SimpleNamespace(now=0.0)

    def sleep(seconds):
        clock.now += seconds

    events = []

    def recover(**kw):
        return recovery.recover_before_home(
            executor,
            lambda: SimpleNamespace(state=state.copy(), received_monotonic_ns=(int(clock.now * 1e9),) * 4),
            clock=lambda: clock.now,
            sleep=sleep,
            emit=events.append,
            **kw,
        )

    return SimpleNamespace(executor=executor, state=state, bounds=bounds, clock=clock, events=events, recover=recover)


@pytest.mark.parametrize("angle", [0.2879344225, 0.3504213691, -0.02])
def test_logged_excursions_recover_with_legal_targets_then_restore_speed(rig, angle):
    rig.state[24] = angle
    original = rig.state.copy()
    report = rig.recover()
    assert report["status"] == "PASS"
    assert rig.executor.left.speed == 0.3
    assert rig.executor.right.speed == 0.3
    assert rig.executor.left.settings == [0.05, 0.3]
    assert rig.executor.right.settings == [0.05, 0.3]
    assert 0 <= rig.state[24] <= 0.2618
    assert rig.executor.rows
    assert abs(float(rig.executor.rows[0][0][17]) - angle) < sdk.SDK_SLEW_LIMIT_RAD
    np.testing.assert_array_equal(rig.state[:7], original[:7])
    np.testing.assert_array_equal(rig.state[29:36], original[29:36])
    assert rig.events[-1]["speed_restored"] is True
    # The ordinary task executor still rejects the original invalid position.
    with pytest.raises(ValueError, match="outside"):
        sdk.validate_hand_command(original[LEFT_SHARPA_SLICE], "left", limits=rig.bounds)


def test_legal_initial_state_does_not_touch_settings_or_send_commands(rig):
    rig.state[24] = 0.125
    assert rig.recover()["status"] == "SKIPPED"
    assert not rig.executor.rows
    assert not rig.executor.left.settings
    assert not rig.executor.right.settings


def test_large_excursion_rejected_before_speed_changes_or_motion(rig):
    rig.state[24] = 0.5
    with pytest.raises(ValueError, match="SDK slew"):
        rig.recover()
    assert not rig.executor.rows
    assert not rig.executor.left.settings


def test_both_hands_supported_before_any_write(rig):
    rig.executor.right.config.interpolate = False
    with pytest.raises(RuntimeError, match="native interpolation"):
        rig.recover()
    assert not rig.executor.rows
    assert not rig.executor.left.settings


@pytest.mark.parametrize("problem", ["send", "feedback", "interrupt", "stalled", "outward"])
def test_failures_restore_both_settings_and_block_home(rig, problem):
    send = rig.executor.send

    def fail_send(left, right):
        if problem == "send":
            raise RuntimeError("write failed")
        if problem == "interrupt":
            raise KeyboardInterrupt("stop")
        if problem == "feedback":
            raise ValueError("stale feedback")
        if problem == "outward":
            rig.state[24] += 0.02
        elif problem != "stalled":
            send(left, right)

    rig.executor.send = fail_send
    expected = KeyboardInterrupt if problem == "interrupt" else (RuntimeError, ValueError, TimeoutError)
    with pytest.raises(expected):
        rig.recover()
    assert rig.executor.left.speed == 0.3
    assert rig.executor.right.speed == 0.3
    assert not any(event.get("event") == "home_recovery_complete" for event in rig.events)


def test_partial_speed_setup_failure_restores_including_failed_hand(rig):
    ensure = rig.executor.right._ensure_coefficient

    def reject(name, observed, desired):
        ensure(name, observed, desired)
        if desired == 0.05:
            raise RuntimeError("readback failed after write")

    rig.executor.right._ensure_coefficient = reject
    with pytest.raises(RuntimeError, match="readback failed"):
        rig.recover()
    assert rig.executor.left.speed == 0.3
    assert rig.executor.right.speed == 0.3
    assert not rig.executor.rows


def test_restore_failure_blocks_completion_and_still_attempts_other_hand(rig):
    ensure = rig.executor.right._ensure_coefficient

    def reject(name, observed, desired):
        if desired == 0.3:
            raise RuntimeError("cannot restore")
        ensure(name, observed, desired)

    rig.executor.right._ensure_coefficient = reject
    with pytest.raises(RuntimeError, match="restoration failed"):
        rig.recover()
    assert rig.executor.left.speed == 0.3
    assert not any(event.get("event") == "home_recovery_complete" for event in rig.events)


def test_recovery_waits_for_actual_feedback_not_only_a_legal_command(rig):
    rig.executor.send = lambda *a: None
    with pytest.raises(TimeoutError, match="no progress"):
        rig.recover()
    assert rig.state[24] > 0.2618
    assert rig.executor.left.speed == 0.3


@pytest.mark.parametrize("initialization", ["home", "policy", "current"])
def test_serial_recovery_marker_is_only_on_fixed_home(initialization):
    requests = []

    class Worker:
        def request(self, request):
            requests.append(request)
            return {"status": "PASS", "published": True}

    def observe(report=None):
        return {"observation/state": np.zeros(58)}

    def infer(obs):
        return {"checkpoint": "fake", "norm_stats": "fake"}, np.zeros((50, 58)), None

    prepare_and_run_chunks(
        infer,
        observe,
        Worker(),
        observe(),
        num_chunks=1,
        initial_target=np.zeros(58) if initialization == "home" else None,
        prepare_initial=initialization != "current",
    )
    prepare = [r for r in requests if r["operation"] == "prepare"]
    if initialization == "home":
        assert prepare[0]["home_recovery"] is True
    elif initialization == "policy":
        assert "home_recovery" not in prepare[0]
    else:
        assert not prepare
    assert all("home_recovery" not in r for r in requests if r["operation"] == "chunk")


def test_fresh_feedback_failure_during_recovery_restores_speed(rig):
    def feedback():
        if rig.executor.rows:
            raise ValueError("stale feedback during recovery")
        return SimpleNamespace(state=rig.state.copy(), received_monotonic_ns=(int(rig.clock.now * 1e9),) * 4)

    with pytest.raises(ValueError, match="stale feedback"):
        recovery.recover_before_home(
            rig.executor,
            feedback,
            clock=lambda: rig.clock.now,
            sleep=lambda seconds: setattr(rig.clock, "now", rig.clock.now + seconds),
        )
    assert rig.executor.left.speed == 0.3
    assert rig.executor.right.speed == 0.3
    assert len(rig.executor.rows) == 1


def test_original_lower_speed_is_never_increased_and_restores_exactly(rig):
    rig.executor.left.speed = 0.02
    rig.executor.right.speed = 0.04
    assert rig.recover()["status"] == "PASS"
    assert rig.executor.left.speed == 0.02
    assert rig.executor.right.speed == 0.04
    assert rig.executor.left.settings == [0.02, 0.02]
    assert rig.executor.right.settings == [0.04, 0.04]


def test_silent_speed_restore_mismatch_blocks_task_even_if_adapter_returns(rig):
    ensure = rig.executor.right._ensure_coefficient

    def silently_ignore_restore(name, observed, desired):
        if desired != 0.3:
            ensure(name, observed, desired)

    rig.executor.right._ensure_coefficient = silently_ignore_restore
    with pytest.raises(RuntimeError, match="restoration failed"):
        rig.recover()
    assert rig.executor.left.speed == 0.3
    assert rig.executor.right.speed == 0.05
    assert not any(event.get("event") == "home_recovery_complete" for event in rig.events)


def test_speed_setup_stale_feedback_waits_without_motion_then_recovers(rig):
    calls = 0

    def feedback():
        nonlocal calls
        calls += 1
        if rig.executor.left.speed == 0.05 and rig.clock.now < 0.3:
            assert not rig.executor.rows
            raise ValueError("Stale/future feedback timestamp: /sharpa/left/joint_states; source_age_ms=350")
        return SimpleNamespace(state=rig.state.copy(), received_monotonic_ns=(int(rig.clock.now * 1e9),) * 4)

    result = recovery.recover_before_home(
        rig.executor,
        feedback,
        clock=lambda: rig.clock.now,
        sleep=lambda seconds: setattr(rig.clock, "now", rig.clock.now + seconds),
        emit=rig.events.append,
    )
    assert result["status"] == "PASS"
    assert calls > 3
    assert rig.executor.rows
    assert rig.executor.left.speed == 0.3
    assert [event["phase"] for event in rig.events if event["event"] == "home_recovery_feedback_ready"] == [
        "before_motion",
        "after_restore",
    ]


def test_cached_fresh_receipts_cannot_cross_settings_barrier(rig):
    def feedback():
        if not rig.executor.rows and rig.clock.now < 0.1:
            assert not rig.executor.rows
            return SimpleNamespace(state=rig.state.copy(), received_monotonic_ns=(0,) * 4)
        return SimpleNamespace(state=rig.state.copy(), received_monotonic_ns=(int(rig.clock.now * 1e9),) * 4)

    result = recovery.recover_before_home(
        rig.executor,
        feedback,
        clock=lambda: rig.clock.now,
        sleep=lambda seconds: setattr(rig.clock, "now", rig.clock.now + seconds),
    )
    assert result["status"] == "PASS"
    assert rig.clock.now >= 0.1


def test_setup_gate_timeout_restores_speeds_and_sends_nothing(rig):
    def feedback():
        if rig.executor.left.settings:
            raise ValueError("left source_age_ms=800")
        return SimpleNamespace(state=rig.state.copy())

    with pytest.raises(TimeoutError, match="before_motion.*5s.*source_age_ms=800"):
        recovery.recover_before_home(
            rig.executor,
            feedback,
            clock=lambda: rig.clock.now,
            sleep=lambda seconds: setattr(rig.clock, "now", rig.clock.now + seconds),
        )
    assert 5 <= rig.clock.now < 5.03
    assert rig.executor.left.speed == 0.3
    assert rig.executor.right.speed == 0.3
    assert not rig.executor.rows


def test_setup_gate_stop_restores_speeds_without_motion(rig):
    def feedback():
        if rig.executor.left.settings:
            raise KeyboardInterrupt("stop before motion")
        return SimpleNamespace(state=rig.state.copy())

    with pytest.raises(KeyboardInterrupt, match="stop before motion"):
        recovery.recover_before_home(rig.executor, feedback)
    assert rig.executor.left.speed == 0.3
    assert rig.executor.right.speed == 0.3
    assert not rig.executor.rows


@pytest.mark.parametrize("new_angle", [0.35, 0.5, 0.12])
def test_setup_gate_recomputes_target_and_rechecks_new_measurement(rig, new_angle):
    def feedback():
        if rig.executor.left.settings and not rig.executor.rows:
            rig.state[24] = new_angle
        return SimpleNamespace(state=rig.state.copy(), received_monotonic_ns=(int(rig.clock.now * 1e9),) * 4)

    kwargs = {"clock": lambda: rig.clock.now, "sleep": lambda seconds: setattr(rig.clock, "now", rig.clock.now + seconds)}
    if new_angle == 0.5:
        with pytest.raises(ValueError, match="SDK slew"):
            recovery.recover_before_home(rig.executor, feedback, **kwargs)
        assert not rig.executor.rows
    else:
        result = recovery.recover_before_home(rig.executor, feedback, **kwargs)
        if new_angle == 0.12:
            assert result["status"] == "SKIPPED"
            assert not rig.executor.rows
        else:
            assert result["joints"][0]["measured_rad"] == 0.35
            assert result["status"] == "PASS"
    assert rig.executor.left.speed == 0.3


def test_restoration_gate_waits_for_new_feedback_before_returning_home(rig):
    ready_after = None

    def feedback():
        nonlocal ready_after
        if rig.executor.rows and rig.executor.left.speed == 0.3:
            if ready_after is None:
                ready_after = rig.clock.now + 0.25
            if rig.clock.now < ready_after:
                raise ValueError("stale after restore")
        return SimpleNamespace(state=rig.state.copy(), received_monotonic_ns=(int(rig.clock.now * 1e9),) * 4)

    result = recovery.recover_before_home(
        rig.executor,
        feedback,
        clock=lambda: rig.clock.now,
        sleep=lambda seconds: setattr(rig.clock, "now", rig.clock.now + seconds),
    )
    assert result["status"] == "PASS"
    assert rig.clock.now >= ready_after


def test_feedback_age_diagnostics_keep_original_200ms_gate():
    from deploy.fr3_sharpa_runtime.feedback import LiveFeedback

    feedback = LiveFeedback(np.zeros(58), (1_000_000_000,) * 4, (2_000_000_000,) * 4)
    feedback.validate_freshness(now_ros_ns=1_200_000_000, now_monotonic_ns=2_200_000_000)
    with pytest.raises(ValueError, match="source_age_ms=201.000.*receipt_age_ms=201.000"):
        feedback.validate_freshness(now_ros_ns=1_201_000_000, now_monotonic_ns=2_201_000_000)
