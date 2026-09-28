import numpy as np
import pytest

from deploy.fr3_wuji_home import deploy
from deploy.fr3_wuji_home.deploy import parse_args
from deploy.fr3_wuji_home.planner import ARM_HOME_SPEED_RAD_S
from deploy.fr3_wuji_home.planner import HAND_HOME_SPEED_RAD_S
from deploy.fr3_wuji_home.planner import build_home_plan
from deploy.fr3_wuji_home.poses import load_home
from deploy.fr3_wuji_slow.core import ARM
from deploy.fr3_wuji_slow.core import HAND
from deploy.fr3_wuji_slow.runner import simulate


def limits():
    return {
        "lower": np.full(54, -10.0),
        "upper": np.full(54, 10.0),
        "speed": np.ones(54),
    }


def test_tomato_pose_is_a_pi05_owned_copy_of_the_assembly_home():
    home = load_home()
    assert home.name == "tomato"
    assert home.target.shape == (54,)
    assert home.target.flags.writeable is False
    assert home.provenance["source_home"] == "assembly"
    np.testing.assert_allclose(
        home.target[0:7],
        [0.0909309983, 0.9387366176, -1.0184751749, -1.9697190523, -1.7120615244, 3.3553025723, 0.5689566135],
    )
    np.testing.assert_allclose(
        home.target[27:34],
        [-0.6066353321, 0.7345613837, 1.1655430794, -2.1794548035, 0.6614411473, 3.2823071480, 2.4422383308],
    )


def test_home_plan_moves_arms_before_hands_and_respects_caps():
    home = load_home()
    start = np.zeros(54)
    plan = build_home_plan(start, home.target, limits(), [f"joint-{index}" for index in range(54)])

    np.testing.assert_allclose(plan.approach.sample(0.0), start, atol=1e-9)
    arm_endpoint = plan.approach.sample(plan.approach.duration)
    np.testing.assert_allclose(arm_endpoint[ARM], home.target[ARM], atol=1e-9)
    np.testing.assert_allclose(arm_endpoint[HAND], start[HAND], atol=1e-9)
    np.testing.assert_allclose(plan.raw[0], arm_endpoint, atol=1e-9)
    np.testing.assert_allclose(plan.raw[-1], home.target, atol=1e-9)

    arm_peak = np.max(np.abs(plan.approach.sample(np.linspace(0, plan.approach.duration, 1001), 1)[..., ARM]))
    assert arm_peak <= ARM_HOME_SPEED_RAD_S + 1e-8
    hand_distance = np.max(np.abs(home.target[HAND] - start[HAND]))
    assert hand_distance / plan.playback.duration <= HAND_HOME_SPEED_RAD_S + 1e-8
    assert plan.report["sequence"] == [
        "both_arms_to_home_and_settle",
        "both_wuji_hands_to_home_and_settle",
    ]


def test_home_plan_runs_through_the_existing_bounded_simulator():
    home = load_home()
    plan = build_home_plan(np.zeros(54), home.target, limits(), [f"joint-{index}" for index in range(54)])
    report, trace = simulate(plan, limits())
    assert report["state"] == "complete"
    assert report["frames"] > 0
    assert set(trace["state"]) >= {"approach", "playback", "settle_start", "settle_end"}


def test_unknown_home_and_hardware_flags_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown Home pose"):
        load_home("missing")
    with pytest.raises(SystemExit):
        parse_args(["--execute", "--output", str(tmp_path / "out")])
    args = parse_args([])
    assert args.execute is False
    assert args.home is None
    assert not hasattr(args, "task")


def test_standalone_home_refuses_a_running_teleop_backend(monkeypatch):
    monkeypatch.setattr(deploy.hardware, "preflight_owners", lambda **_kwargs: None)
    monkeypatch.setattr(deploy.subprocess, "check_output", lambda *_args, **_kwargs: "python -m teleop_runtime.cli")
    with pytest.raises(RuntimeError, match="Teleop backend/UI is running"):
        deploy.reject_other_owners()
