from types import SimpleNamespace
import json

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.feedback import FEEDBACK_KEYS, FeedbackCache
from deploy.fr3_sharpa_runtime.preflight import execute_preflight, offline_sanity_check
from deploy.fr3_sharpa_runtime.protocol import SHARPA_JOINT_NAMES

ROS_NOW = 10_000_000_000
MONO_NOW = 20_000_000_000


def mock_realtime_feedback(state, *, age_ns=10_000_000):
    """Feed four named JointState messages through the production cache."""
    cache = FeedbackCache()
    parts = (state[:7], state[7:29], state[29:36], state[36:58])
    for key, positions in zip(FEEDBACK_KEYS, parts, strict=True):
        side, kind = key.split("_")
        names = ([f"{side}_fr3v2_joint{i}" for i in range(1, 8)] if kind == "arm"
                 else [f"{side}_{joint}" for joint in SHARPA_JOINT_NAMES])
        stamp = ROS_NOW - age_ns
        message = SimpleNamespace(
            name=names[::-1], position=list(positions)[::-1],
            header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp // 1_000_000_000, nanosec=stamp % 1_000_000_000)),
        )
        cache.update(key, message, MONO_NOW - age_ns)
    return cache.snapshot(now_ros_ns=ROS_NOW, now_monotonic_ns=MONO_NOW)


def measured_state():
    arm = [-0.1, -0.5, 0.0, -1.8, 0.0, 1.5, 0.0]
    return np.array(arm + [0.0] * 22 + arm + [0.0] * 22, dtype=np.float32)


def test_offline_mismatch_does_not_reject_live_first_target():
    live_state = measured_state()
    actions = np.tile(live_state, (50, 1))
    # Later task movement exceeds 0.15 from the first state, but first target is close.
    actions[:, 7] += np.linspace(0.01, 0.25, 50)
    offline_state = live_state.copy()
    offline_state[36] = 0.6
    offline = offline_sanity_check(actions, offline_state)
    assert offline["status"] == "FAIL" and offline["blocking"] is False
    result = execute_preflight(
        actions, mock_realtime_feedback(live_state), now_ros_ns=ROS_NOW, now_monotonic_ns=MONO_NOW,
    )
    assert result["status"] == "PASS"
    assert result["reference"] == "live_ros_feedback"
    assert result["compared_action_index"] == 0


@pytest.mark.parametrize("joint,delta", [(0, 0.06), (7, 0.16), (29, 0.06), (36, 0.16)])
def test_live_initial_delta_exceeding_limit_rejected(joint, delta):
    state = measured_state()
    actions = np.tile(state, (50, 1))
    changed_feedback = state.copy()
    changed_feedback[joint] += delta
    with pytest.raises(ValueError, match="first-target delta"):
        execute_preflight(actions, mock_realtime_feedback(changed_feedback),
                          now_ros_ns=ROS_NOW, now_monotonic_ns=MONO_NOW)


def test_offline_array_cannot_be_used_as_live_feedback():
    state = measured_state()
    with pytest.raises(ValueError, match="not an episode state"):
        execute_preflight(np.tile(state, (50, 1)), state, now_ros_ns=ROS_NOW, now_monotonic_ns=MONO_NOW)


def test_feedback_that_expires_before_dispatch_is_rejected():
    state = measured_state()
    feedback = mock_realtime_feedback(state, age_ns=199_000_000)
    with pytest.raises(ValueError, match="Stale/future feedback timestamp"):
        execute_preflight(np.tile(state, (50, 1)), feedback,
                          now_ros_ns=ROS_NOW + 2_000_000, now_monotonic_ns=MONO_NOW + 2_000_000)


@pytest.mark.parametrize("age_ns", [201_000_000, -1])
def test_stale_or_future_live_stamp_rejected(age_ns):
    with pytest.raises(ValueError, match="Stale/future feedback timestamp"):
        mock_realtime_feedback(measured_state(), age_ns=age_ns)


def test_missing_live_topic_rejected():
    with pytest.raises(ValueError, match="Missing/invalid live feedback"):
        FeedbackCache().snapshot(now_ros_ns=ROS_NOW, now_monotonic_ns=MONO_NOW)


def test_feedback_aligns_original_recent_samples_and_diagnoses_missing_history():
    cache = FeedbackCache()

    def update(key, age_ms, position):
        side, kind = key.split("_")
        names = ([f"{side}_fr3v2_joint{i}" for i in range(1, 8)] if kind == "arm"
                 else [f"{side}_{joint}" for joint in SHARPA_JOINT_NAMES])
        stamp = ROS_NOW - age_ms * 1_000_000
        message = SimpleNamespace(
            name=names, position=[position] * len(names),
            header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp // 1_000_000_000,
                                                        nanosec=stamp % 1_000_000_000)),
        )
        cache.update(key, message, MONO_NOW - age_ms * 1_000_000)

    for key in FEEDBACK_KEYS:
        update(key, 130, 1)
    for key in FEEDBACK_KEYS[:-1]:
        update(key, 10, 2)
    update(FEEDBACK_KEYS[-1], 120, 3)
    result = cache.snapshot(now_ros_ns=ROS_NOW, now_monotonic_ns=MONO_NOW)
    assert result.ros_stamps_ns == (ROS_NOW-130_000_000,)*3 + (ROS_NOW-120_000_000,)
    assert result.state[0] == 1 and result.state[-1] == 3

    cache.histories["left_arm"].clear()
    cache.histories["left_arm"].append(cache.latest["left_arm"])
    with pytest.raises(ValueError, match=r"span_ms=110\.000.*left/franka/joint_states.*source_age_ms"):
        cache.snapshot(now_ros_ns=ROS_NOW, now_monotonic_ns=MONO_NOW)


def test_action_shape_and_finite_are_still_enforced():
    feedback = mock_realtime_feedback(measured_state())
    with pytest.raises(ValueError, match="Expected action shape"):
        execute_preflight(np.zeros((49, 58)), feedback, now_ros_ns=ROS_NOW, now_monotonic_ns=MONO_NOW)
    actions = np.tile(measured_state(), (50, 1))
    actions[2, 3] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        execute_preflight(actions, feedback, now_ros_ns=ROS_NOW, now_monotonic_ns=MONO_NOW)


def test_transport_payload_does_not_contain_episode_or_inference_state(monkeypatch, tmp_path):
    from deploy.fr3_sharpa_runtime.run import ros_boundary

    (tmp_path / "setup.bash").write_text("")
    captured = {}

    def mock_subprocess(*args, **kwargs):
        captured.update(json.loads(kwargs["input"]))
        return SimpleNamespace(returncode=0, stderr="", stdout='{"published": false}')

    monkeypatch.setattr("deploy.fr3_sharpa_runtime.run.subprocess.run", mock_subprocess)
    ros_boundary(np.tile(measured_state(), (50, 1)), "session", "execute", tmp_path, True)
    assert set(captured) == {"actions", "session_id"}
