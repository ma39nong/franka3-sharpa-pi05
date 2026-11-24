from deploy.fr3_sharpa_runtime.protocol import DT_NS
import sys
import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.feedback import LiveFeedback
from deploy.fr3_sharpa_runtime.policy_prepare import (
    HOME_SPEED_RAD_S, SlowApproach, prepare_and_run_chunks, check_validation_excursion,
)


def live(state, clock):
    return LiveFeedback(state.copy(), (clock,)*4, (clock,)*4)


def test_fixed_home_speed_limits_both_arms_and_hands():
    state = np.zeros(58, dtype=np.float32)
    target = np.full(58, 0.3, dtype=np.float32)
    planner = SlowApproach(target, state, speed_rad_s=HOME_SPEED_RAD_S)
    clock = 1_000_000_000
    command, _, _ = planner.step(live(state, clock), now_ros_ns=clock, now_monotonic_ns=clock,
                                 enforce_execute_delta=False)
    expected = 0.10 * DT_NS / 1e9
    np.testing.assert_allclose(command, expected, atol=1e-7)


def test_large_initial_mismatch_approaches_with_small_steps_and_measured_settling():
    state = np.zeros(58, dtype=np.float32)
    target = state.copy()
    target[15] = 0.6
    planner = SlowApproach(target, state, speed_rad_s=0.03)
    previous = state.copy()
    for i in range(1000):
        clock = 1_000_000_000+i*DT_NS
        command, settled, error = planner.step(live(state, clock), now_ros_ns=clock, now_monotonic_ns=clock)
        assert np.max(np.abs(command-previous)) <= 0.03*(DT_NS/1e9)+1e-7
        previous = command.copy()
        state = command.copy()  # Mock robot follows only guarded incremental commands.
        if settled:
            assert error <= 0.01
            assert np.allclose(command, target)
            break
    else:
        pytest.fail('did not settle')


def test_commanded_arrival_alone_is_not_measured_arrival():
    target = np.full(58, 0.005, dtype=np.float32)
    state = np.full(58, -0.02, dtype=np.float32)
    planner = SlowApproach(target, target, timeout_s=0.3, stable_duration_s=0.1)
    for i in range(10):
        clock = 1_000_000_000+i*DT_NS
        _, settled, _ = planner.step(live(state, clock), now_ros_ns=clock, now_monotonic_ns=clock)
        assert not settled
    clock += DT_NS
    with pytest.raises(TimeoutError):
        planner.step(live(state, clock), now_ros_ns=clock, now_monotonic_ns=clock)


def test_timeout_identifies_blocked_right_hand_joint():
    target = np.zeros(58, dtype=np.float32)
    state = target.copy()
    state[51] = 0.031  # right ring PIP; left hand is already at target
    planner = SlowApproach(target, target, tolerance_rad=0.02, timeout_s=0.1)
    planner.step(live(state, 1_000_000_000), now_ros_ns=1_000_000_000,
                 now_monotonic_ns=1_000_000_000, enforce_execute_delta=False)
    with pytest.raises(TimeoutError, match="worst_right_sharpa=.*ring_PIP"):
        planner.step(live(state, 1_200_000_000), now_ros_ns=1_200_000_000,
                     now_monotonic_ns=1_200_000_000, enforce_execute_delta=False)


def test_tracking_lag_stops_without_weakening_existing_delta_limit():
    state = np.zeros(58, dtype=np.float32)
    planner = SlowApproach(np.full(58, 0.6), state)
    with pytest.raises(ValueError, match='first-target delta'):
        for i in range(200):
            clock = 1_000_000_000+i*DT_NS
            planner.step(live(state, clock), now_ros_ns=clock, now_monotonic_ns=clock)


def test_sdk_direct_prepare_allows_large_delta_without_execute_gate():
    state = np.zeros(58, dtype=np.float32)
    target = np.full(58, 0.6, dtype=np.float32)
    planner = SlowApproach(target, state, speed_rad_s=0.03)
    command, settled, _ = planner.step(
        live(state, 1_000_000_000), now_ros_ns=1_000_000_000,
        now_monotonic_ns=1_000_000_000, enforce_execute_delta=False)
    assert not settled
    assert 0.0 < float(command[0]) <= 0.03 * (DT_NS/1e9) + 1e-7


def test_stale_feedback_and_nonfinite_targets_are_rejected():
    with pytest.raises(ValueError, match='finite'):
        SlowApproach(np.full(58, np.nan), np.zeros(58))
    planner = SlowApproach(np.ones(58), np.zeros(58))
    with pytest.raises(ValueError, match='Stale'):
        planner.step(live(np.zeros(58), 1_000_000_000), now_ros_ns=2_000_000_000,
                     now_monotonic_ns=2_000_000_000)


def test_first_prediction_is_discarded_and_every_chunk_gets_new_observation():
    events = []
    predictions = iter([np.ones((50,58)), np.full((50,58),2), np.full((50,58),3)])
    def infer(obs):
        events.append(('infer', obs))
        return {'checkpoint':'ckpt', 'norm_stats':'stats'}, next(predictions), None
    class Transport:
        def request(self, payload):
            events.append((payload['operation'], payload))
            return {'status':'PASS', 'published':False, 'completed_monotonic_ns':10, 'completed_ros_ns':20}
    def acquire(report):
        events.append(('new_observation', report))
        return 'after_prepare_or_chunk'
    report = prepare_and_run_chunks(infer, acquire, Transport(), 'initial', num_chunks=2)
    assert [e[0] for e in events] == ['infer','prepare','new_observation','infer','chunk','new_observation','infer','chunk']
    assert events[1][1]['target'] == [1]*58
    assert events[4][1]['actions'][0] == [2]*58
    assert events[7][1]['actions'][0] == [3]*58
    assert len(report['chunks']) == 2


def test_prepare_rejection_blocks_reinference_and_chunks():
    def infer(obs):
        return {}, np.zeros((50,58)), None
    class Transport:
        def request(self, payload):
            assert payload['operation'] == 'prepare'
            raise RuntimeError('guarded refusal')
    with pytest.raises(RuntimeError, match='guarded refusal'):
        prepare_and_run_chunks(infer, lambda _: pytest.fail('must not reacquire'), Transport(), {})


def test_cli_requires_hardware_authorization_before_policy_prepare(monkeypatch):
    from deploy.fr3_sharpa_runtime import run
    monkeypatch.setattr(sys, 'argv', ['run','--execute','--prepare-policy-first'])
    monkeypatch.setattr(run, 'execute_policy_prepare', lambda _: pytest.fail('cannot reach motion'))
    with pytest.raises(SystemExit):
        run.main()


def test_small_steps_do_not_circumvent_existing_supervised_total_excursion():
    target = np.zeros(58)
    target[15] = 0.6
    with pytest.raises(ValueError, match='total excursion'):
        check_validation_excursion(target, np.zeros(58),
            mode='supervised_hardware_validation', excursion_rad=0.15, source='litchibot')
    target[15] = 0.1
    check_validation_excursion(target, np.zeros(58),
        mode='supervised_hardware_validation', excursion_rad=0.15)


def test_openpi_prepare_matches_updated_native_baseline_exemption():
    target=np.zeros(58);target[15]=0.6
    check_validation_excursion(target,np.zeros(58),mode='supervised_hardware_validation',
                               excursion_rad=0.15,source='openpi')
