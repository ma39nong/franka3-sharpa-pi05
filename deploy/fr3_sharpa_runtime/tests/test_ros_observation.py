from types import SimpleNamespace

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.ros_observation import ordered_state, rgb_image
from deploy.fr3_sharpa_runtime.protocol import SHARPA_JOINT_NAMES


def test_state_order_matches_training_protocol():
    arm = SimpleNamespace(
        name=[f"left_fr3v2_joint{i}" for i in range(7, 0, -1)],
        position=list(range(7, 0, -1)),
    )
    hand = SimpleNamespace(
        name=[f"left_{joint}" for joint in reversed(SHARPA_JOINT_NAMES)],
        position=list(range(21, -1, -1)),
    )
    assert ordered_state(arm, "left", False) == list(range(1, 8))
    assert ordered_state(hand, "left", True) == list(range(22))
    with pytest.raises(ValueError, match="Missing"):
        ordered_state(SimpleNamespace(name=arm.name[:-1], position=arm.position[:-1]), "left", False)


def test_camera_bgr_to_rgb_and_invalid_shape():
    message = SimpleNamespace(encoding="bgr8", height=1, width=1, step=3, data=bytes([1, 2, 3]))
    import base64

    result = rgb_image(message)
    np.testing.assert_array_equal(np.frombuffer(base64.b64decode(result["rgb_b64"]), dtype=np.uint8), [3, 2, 1])
    with pytest.raises(ValueError, match="Invalid camera byte count"):
        rgb_image(SimpleNamespace(encoding="rgb8", height=2, width=1, step=3, data=b"123"))


def test_only_camera_subscriptions_use_reliable_qos(monkeypatch):
    import sys
    from deploy.fr3_sharpa_runtime.ros_observation import snapshot, TOPICS

    sensor_qos = object()
    subscribed = {}
    node = SimpleNamespace(
        create_subscription=lambda kind, topic, callback, qos: subscribed.setdefault(topic, qos),
        destroy_node=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "rclpy", SimpleNamespace(
        init=lambda: None, create_node=lambda _: node, shutdown=lambda: None,
    ))
    monkeypatch.setitem(sys.modules, "rclpy.qos", SimpleNamespace(
        qos_profile_sensor_data=sensor_qos, QoSProfile=lambda **kwargs: kwargs,
        HistoryPolicy=SimpleNamespace(KEEP_LAST="KEEP_LAST"),
        ReliabilityPolicy=SimpleNamespace(RELIABLE="RELIABLE"),
        DurabilityPolicy=SimpleNamespace(VOLATILE="VOLATILE"),
    ))
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", SimpleNamespace(Image=object, JointState=object))
    with pytest.raises(TimeoutError):
        snapshot(timeout_s=0)
    for key, topic in TOPICS.items():
        if key.startswith("cam"):
            assert subscribed[topic] == dict(history="KEEP_LAST", depth=5,
                                             reliability="RELIABLE", durability="VOLATILE")
        else:
            assert subscribed[topic] is sensor_qos


@pytest.mark.parametrize("first_age_ms", [150, 250, -20])
def test_snapshot_waits_for_synchronized_fresh_updates(monkeypatch, first_age_ms):
    probe, messages, spins = _timing_probe(monkeypatch, first_age_ms, recover=True)
    result = probe.snapshot(timeout_s=0.1)
    assert len(spins) == 2
    assert result["topic_stamp_ns"]["cam0"] == probe.stamp_ns(messages["cam0"])
    assert result["state"] == [0.0] * 58


def test_snapshot_timeout_reports_last_timing_failure(monkeypatch):
    probe, _, _ = _timing_probe(monkeypatch, 250, recover=False)
    with pytest.raises(TimeoutError, match="Stale/future ROS timestamp: cam0.*span="):
        probe.snapshot(timeout_s=0.05)


def test_wait_diagnostics_retain_actual_topic_ages_and_span(monkeypatch):
    probe, _, _ = _timing_probe(monkeypatch, 250, recover=False)
    diagnostics = []
    provider = probe.ObservationProvider()
    try:
        with pytest.raises(TimeoutError):
            provider.snapshot(timeout_s=0.05, on_diagnostic=diagnostics.append)
    finally:
        provider.close()
    assert diagnostics
    last = diagnostics[-1]
    assert last["topics"]["cam0"]["source_age_ms"] == 250
    assert last["topics"]["cam0"]["received_age_ms"] == 0
    assert last["source_span_ms"] == 250
    assert "cam0" in last["reason"]
    missing = probe.observation_diagnostics({}, now_ros=1, now_mono=1, reason="missing")
    assert all(item["missing"] for item in missing["topics"].values())


def test_persistent_provider_reuses_node_and_closes_only_at_exit(monkeypatch):
    probe, _, _ = _timing_probe(monkeypatch, 90, recover=True)
    import sys
    ros = sys.modules["rclpy"]
    calls = []
    create = ros.create_node
    def create_once(name):
        calls.append("create")
        node = create(name)
        node.destroy_node = lambda: calls.append("destroy")
        return node
    monkeypatch.setattr(ros, "init", lambda: calls.append("init"))
    monkeypatch.setattr(ros, "create_node", create_once)
    monkeypatch.setattr(ros, "shutdown", lambda: calls.append("shutdown"))
    provider = probe.ObservationProvider()
    first = provider.snapshot(timeout_s=0.1)
    second = provider.snapshot(timeout_s=0.1)
    assert calls == ["init", "create"]
    assert first["captured_monotonic_ns"] < second["captured_monotonic_ns"]
    provider.close()
    provider.close()
    assert calls == ["init", "create", "destroy", "shutdown"]
    with pytest.raises(RuntimeError, match="closed"):
        provider.snapshot()


def _timing_probe(monkeypatch, first_age_ms, *, recover):
    import sys
    from deploy.fr3_sharpa_runtime import ros_observation as probe

    clock = [1000.0]
    callbacks, messages, spins = {}, {}, []
    def msg(age_ms):
        stamp = int(clock[0] * 1e9) - int(age_ms * 1e6)
        return SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(
            sec=stamp // 10**9, nanosec=stamp % 10**9)))
    node = SimpleNamespace(
        create_subscription=lambda kind, topic, callback, qos: callbacks.setdefault(topic, callback),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(clock[0]*1e9))),
        destroy_node=lambda: None,
    )
    def spin(node, **kwargs):
        clock[0] += 0.01
        spins.append(clock[0])
        for key, topic in probe.TOPICS.items():
            age = (90 if recover and len(spins) > 1 else first_age_ms) if key == "cam0" else 0
            messages[key] = msg(age)
            callbacks[topic](messages[key])
    monkeypatch.setattr(probe.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(probe.time, "monotonic_ns", lambda: int(clock[0]*1e9))
    monkeypatch.setitem(sys.modules, "rclpy", SimpleNamespace(
        init=lambda: None, create_node=lambda _: node, shutdown=lambda: None, spin_once=spin))
    monkeypatch.setitem(sys.modules, "rclpy.qos", SimpleNamespace(
        qos_profile_sensor_data=object(), QoSProfile=lambda **kwargs: kwargs,
        HistoryPolicy=SimpleNamespace(KEEP_LAST=1), ReliabilityPolicy=SimpleNamespace(RELIABLE=1),
        DurabilityPolicy=SimpleNamespace(VOLATILE=1)))
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", SimpleNamespace(Image=object, JointState=object))
    monkeypatch.setattr(probe, "ordered_state", lambda message, side, hand: [0.0] * (22 if hand else 7))
    monkeypatch.setattr(probe, "rgb_image", lambda message: {"shape": [1, 1, 3]})
    return probe, messages, spins
