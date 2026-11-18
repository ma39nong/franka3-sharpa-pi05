import json

import numpy as np
import pytest
import yaml

from deploy.fr3_sharpa_runtime.feedback import LiveFeedback
from deploy.fr3_sharpa_runtime.home_pose import home_target
from deploy.fr3_sharpa_runtime.home_pose import load_home
from deploy.fr3_sharpa_runtime.policy_prepare import HOME_TOLERANCE_RAD
from deploy.fr3_sharpa_runtime.policy_prepare import SlowApproach
from deploy.fr3_sharpa_runtime.policy_prepare import prepare_and_run_chunks
from deploy.fr3_sharpa_runtime.protocol import DT_NS
from deploy.fr3_sharpa_runtime.protocol import SHARPA_JOINT_NAMES
from deploy.fr3_sharpa_runtime.sdk_direct import action_prefix
from deploy.fr3_sharpa_runtime.sdk_direct import chunk_command_frames


@pytest.mark.parametrize(("error", "expected"), [(0.019, True), (0.024, True), (0.026, False)])
def test_home_tolerance_still_requires_stable_fresh_feedback(error, expected):
    planner = SlowApproach(np.zeros(58), np.zeros(58), tolerance_rad=HOME_TOLERANCE_RAD)
    settled = False
    for index in range(25):
        stamp = 1_000_000_000 + index * DT_NS
        state = np.full(58, error, dtype=np.float32)
        feedback = LiveFeedback(state, (stamp,) * 4, (stamp,) * 4)
        _, settled, _ = planner.step(feedback, now_ros_ns=stamp, now_monotonic_ns=stamp, enforce_execute_delta=False)
        if index < 15:
            assert not settled
    assert settled is expected


def test_fixed_home_dispatches_twenty_five_milliradian_tolerance():
    requests = []

    class Transport:
        def request(self, payload):
            requests.append(payload)
            return {"status": "FAIL"}

    def infer(observation):
        pytest.fail("Home must settle before inference")

    with pytest.raises(RuntimeError, match="Prepare"):
        prepare_and_run_chunks(infer, lambda report: None, Transport(), {}, initial_target=np.zeros(58))
    assert requests[0]["tolerance_rad"] == 0.025


def test_configured_hand_home_maps_same_named_angles_to_both_sides(tmp_path):
    pose = tmp_path / "poses.yaml"
    pose.write_text(
        yaml.safe_dump(
            {
                "tasks": {
                    "powder_weighing": {
                        "home": {
                            side: {"joint_names": [f"{side}_fr3_joint{i}" for i in range(1, 8)], "positions": [0.0] * 7}
                            for side in ("left", "right")
                        }
                    }
                }
            }
        )
    )
    values = (np.arange(22) / 100).tolist()
    config = {
        "unit": "rad",
        "source": {"kind": "left_hand_live_feedback"},
        "tasks": {
            "powder_weighing": {
                side: {
                    "joint_names": [f"{side}_{name}" for name in reversed(SHARPA_JOINT_NAMES)],
                    "positions": list(reversed(values)),
                }
                for side in ("left", "right")
            }
        },
    }
    path = tmp_path / "hands.json"
    path.write_text(json.dumps(config))
    home, metadata = load_home(pose, "powder_weighing", hand_path=path)
    target = home_target(home, np.full(58, -0.2))
    np.testing.assert_allclose(target[7:29], values)
    np.testing.assert_array_equal(target[7:29], target[36:58])
    assert metadata["hands"] == "fixed_named_hand_home"
    assert metadata["hand_home_source"] == config["source"]
    config["unit"] = "degree"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="radians"):
        load_home(pose, "powder_weighing", hand_path=path)
    config["unit"] = "rad"
    config["tasks"]["powder_weighing"]["right"]["joint_names"][0] = "unknown"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="right hand"):
        load_home(pose, "powder_weighing", hand_path=path)


def test_home_uses_named_arms_and_holds_both_measured_hands(tmp_path):
    source = {
        "tasks": {
            "powder_weighing": {
                "home": {
                    side: {
                        "joint_names": [f"{side}_fr3_joint{i}" for i in range(7, 0, -1)],
                        "positions": list(range(7, 0, -1)),
                    }
                    for side in ("left", "right")
                }
            }
        }
    }
    path = tmp_path / "poses.yaml"
    path.write_text(yaml.safe_dump(source))
    arms, metadata = load_home(path, "powder_weighing")
    measured = np.linspace(-0.2, 0.2, 58, dtype=np.float32)
    original = measured.copy()
    target = home_target(arms, measured)
    np.testing.assert_array_equal(target[:7], np.arange(1, 8))
    np.testing.assert_array_equal(target[29:36], np.arange(1, 8))
    np.testing.assert_array_equal(target[7:29], measured[7:29])
    np.testing.assert_array_equal(target[36:], measured[36:])
    np.testing.assert_array_equal(measured, original)
    assert metadata["hands"] == "hold_initial_measured_positions"
    path.with_suffix(".source.json").write_text(json.dumps({"sha256": "wrong", "source": "/source"}))
    with pytest.raises(ValueError, match="resync"):
        load_home(path, "powder_weighing")


@pytest.mark.parametrize("steps", [0, 51, 1.5, True])
def test_invalid_execution_prefix_is_rejected(steps):
    with pytest.raises(ValueError, match="execute_steps"):
        action_prefix(np.zeros((50, 58)), steps)


@pytest.mark.parametrize(("scale", "count"), [(1.0, 15), (0.1, 141)])
def test_prefix_never_sends_discarded_future_and_retains_model_horizon(scale, count):
    actions = np.repeat(np.linspace(0, 0.02, 50, dtype=np.float32)[:, None], 58, axis=1)
    actions[15:] = 1000  # Unexecuted future must never enter the command stream.
    frames, bridge = chunk_command_frames(actions, np.zeros(58), speed_scale=scale, execute_steps=15)
    assert bridge == 0
    assert len(frames) == count
    np.testing.assert_allclose(frames[-1], actions[14])
    assert actions.shape == (50, 58)
