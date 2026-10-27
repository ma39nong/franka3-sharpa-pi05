"""No robot/ROS: RTC timing, prefix identity, failures and serial compatibility."""

from types import SimpleNamespace
import json
import sys

import numpy as np
from openpi_client import rtc
import pytest

from deploy.fr3_sharpa_runtime.rtc_stream import RTCTimeline, RTCSettings, canonical_prefix, require_capabilities


def actions(base=0):
    return np.repeat((base + np.arange(50, dtype=np.float32) * 0.001)[:, None], 58, axis=1)


@pytest.mark.parametrize("scale, expected", [(1.0, 3), (0.5, 2), (0.1, 1)])
def test_latency_skip_uses_playback_model_clock_and_receipt(scale, expected):
    plan = RTCTimeline(actions(), 1.0, scale)
    context = plan.snapshot(1.0, delay_steps=5, schedule="exp", weight=5.0)
    installed = plan.install(actions(0.1), rtc.make_receipt(context), 1.1)
    assert installed["skipped"] == expected
    np.testing.assert_allclose(plan.sample(1.1), actions(0.1)[expected])
    assert plan.status(1.1)["consumed_since_install"] == 0


def test_snapshot_uses_accepted_clipped_targets_and_is_independent_of_original_arrays():
    original = actions(0.2)
    plan = RTCTimeline(original, 0, 0.5)
    original[:] = 100
    context = plan.snapshot(0.1, delay_steps=5, schedule="exp", weight=5)
    np.testing.assert_allclose(context["previous_actions"][0], 0.2015)
    assert context["source_sha256"] == plan.digest
    adapted = canonical_prefix(context["previous_actions"])
    adapted[:] = -100
    assert np.all(context["previous_actions"] > 0)


def test_stale_receipt_or_exhaustion_never_revives_old_plan():
    plan = RTCTimeline(actions(), 0, 1)
    context = plan.snapshot(0, delay_steps=5, schedule="exp", weight=5)
    receipt = rtc.make_receipt(context)
    receipt["token"] = "wrong"
    with pytest.raises(ValueError, match="does not match"):
        plan.install(actions(), receipt, 0.1)
    assert plan.generation == 1
    with pytest.raises(TimeoutError, match="guided prefix"):
        plan.install(actions(), rtc.make_receipt(context), 0.9)
    with pytest.raises(TimeoutError, match="exhausted"):
        plan.sample(50 / 30)


@pytest.mark.parametrize(
    "field,value",
    [
        ("initial_delay_steps", 26),
        ("delay_buffer_size", 0),
        ("max_guidance_weight", float("nan")),
        ("prefix_attention_schedule", "invalid"),
    ],
)
def test_invalid_settings_are_rejected(field, value):
    with pytest.raises(ValueError):
        RTCSettings(**{field: value})


def test_missing_guided_service_is_rejected():
    with pytest.raises(ValueError, match="--enable-rtc"):
        require_capabilities({"rtc_supported": True})


@pytest.mark.parametrize("speed", ["slow", "medium", "fast"])
def test_cli_keeps_speed_separate_from_broker(monkeypatch, capsys, speed):
    from deploy.fr3_sharpa_runtime import run

    monkeypatch.setattr(sys, "argv", ["run", "--check", "--broker-mode", "rtc", "--speed", speed, "--num-chunks", "2"])
    run.main()
    report = json.loads(capsys.readouterr().out)
    assert report["broker_mode"] == "rtc" and report["speed"] == speed and not report["hardware_output"]


@pytest.mark.parametrize(
    "extra",
    [
        ["--num-chunks", "1"],
        ["--execute-steps", "25"],
        ["--initialization", "policy"],
        ["--rtc-initial-delay-steps", "26"],
    ],
)
def test_invalid_rtc_modes_rejected_without_devices(monkeypatch, extra):
    from deploy.fr3_sharpa_runtime import run

    monkeypatch.setattr(sys, "argv", ["run", "--check", "--broker-mode", "rtc", "--num-chunks", "2", *extra])
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2


def test_delta_prefix_reanchors_to_new_state_through_real_policy(monkeypatch):
    from openpi import transforms
    from openpi.policies.policy import Policy
    from openpi.policies.fr3_sharpa_policy import Fr3SharpaInputs, Fr3SharpaOutputs
    from openpi.models.model import ModelType
    from openpi.shared.normalize import NormStats
    from openpi.shared import nnx_utils

    monkeypatch.setattr(nnx_utils, "module_jit", lambda f: f)
    arms = np.r_[0:7, 29:36]
    mask = transforms.make_bool_mask(7, -22, 7, -22)
    stats = {
        k: NormStats(mean=np.zeros(58), std=np.ones(58), q01=np.full(58, -2.0), q99=np.full(58, 3.0))
        for k in ("state", "actions")
    }
    recorded = {}

    class Model:
        action_dim = 58
        action_horizon = 50

        def sample_actions(self, rng, observation, **kwargs):
            recorded.update(kwargs)
            return kwargs["rtc_actions"]

    policy = Policy(
        Model(),
        rtc_action_repack=canonical_prefix,
        transforms=[
            Fr3SharpaInputs(ModelType.PI05),
            transforms.DeltaActions(mask),
            transforms.Normalize(stats, use_quantiles=True),
        ],
        output_transforms=[
            transforms.Unnormalize(stats, use_quantiles=True),
            transforms.AbsoluteActions(mask),
            Fr3SharpaOutputs(),
        ],
    )
    old = actions(0.3)
    context = RTCTimeline(old, 0, 1).snapshot(0, delay_steps=5, schedule="exp", weight=5)
    for current_state in (np.linspace(-0.4, 0.2, 58, dtype=np.float32), np.linspace(0.1, 0.6, 58, dtype=np.float32)):
        obs = {"observation/state": current_state}
        for k in ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image"):
            obs[k] = np.zeros((8, 8, 3), np.uint8)
        result = policy.infer(obs, rtc_context=context)
        np.testing.assert_allclose(result["actions"], old, atol=2e-6)
        target = np.array(recorded["rtc_actions"])[0]
        expected = old.copy()
        expected[:, arms] -= current_state[arms]
        normalized = (expected + 2.0) / (5.0 + 1e-6) * 2.0 - 1.0
        np.testing.assert_allclose(target, normalized, atol=2e-6)
        rtc.validate_receipt(result["rtc"], context)
    np.testing.assert_array_equal(context["previous_actions"], old)


def test_rtc_capability_check_happens_before_sdk_worker(monkeypatch, tmp_path):
    from deploy.fr3_sharpa_runtime import run, motion_worker

    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "setup.bash").write_text("")

    class Runtime:
        def __init__(self, *args):
            pass

        def inspect_metadata(self, **kwargs):
            raise ValueError("no RTC support")

    monkeypatch.setattr(run, "SharpaRuntime", Runtime)
    monkeypatch.setattr(motion_worker, "MotionWorker", lambda *a, **k: pytest.fail("SDK must not start"))
    args = SimpleNamespace(
        ros_overlay=overlay, sharpa_backend="sdk_direct", initialization="current", broker_mode="rtc", policy_url="test"
    )
    with pytest.raises(ValueError, match="no RTC support"):
        run.execute_policy_prepare(args)


def test_websocket_envelope_and_timed_policy_forward_guidance():
    import asyncio
    import websockets
    from openpi_client import msgpack_numpy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from deploy.fr3_sharpa_runtime.timing import TimedPolicy

    context = RTCTimeline(actions(), 0, 1).snapshot(0, delay_steps=5, schedule="exp", weight=5)
    calls = []

    class Policy:
        _input_transform = _sample_actions = _output_transform = lambda *a, **k: None

        def infer(self, obs, **kwargs):
            calls.append(kwargs)
            result = {"actions": actions()}
            if kwargs:
                result["rtc"] = rtc.make_receipt(kwargs["rtc_context"])
            return result

    class Socket:
        remote_address = "fake"

        def __init__(self):
            self.messages = [
                msgpack_numpy.packb({"state": "ordinary"}),
                msgpack_numpy.packb(rtc.make_request({"state": "guided"}, context)),
            ]
            self.sent = []

        async def recv(self):
            if self.messages:
                return self.messages.pop(0)
            raise websockets.ConnectionClosedOK(None, None)

        async def send(self, value):
            self.sent.append(msgpack_numpy.unpackb(value))

    socket = Socket()
    asyncio.run(WebsocketPolicyServer(TimedPolicy(Policy()))._handler(socket))
    assert calls[0] == {} and calls[1]["rtc_context"].previous_actions.shape == (50, 58)
    rtc.validate_receipt(socket.sent[-1]["rtc"], context)
