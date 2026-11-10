"""ROS message boundary. ros-test creates messages only; execute is explicit."""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from . import config
from .action_adapter import SharpaActionAdapter
from .feedback import ROSFeedbackProvider
from .preflight import execute_preflight
from .ros_observation import MAX_AGE_NS
from .protocol import (
    DT_NS, LEFT_FR3_SLICE, RIGHT_FR3_SLICE,
    SHARPA_JOINT_NAMES, as_finite_action_chunk,
)
from .sdk_direct import SharpaSdkDirectExecutor, validate_direct_hand_actions


def _stamp(ns):
    from builtin_interfaces.msg import Time

    sec, nanosec = divmod(ns, 1_000_000_000)
    return Time(sec=sec, nanosec=nanosec)


def construct_messages(frame):
    from sensor_msgs.msg import JointState
    from teleop_interfaces.msg import ArmCommand

    arm = ArmCommand()
    arm.header.stamp = _stamp(frame.timestamp_ns)
    arm.source = frame.fr3.source
    arm.session_id = frame.fr3.session_id
    arm.sequence = frame.sequence
    arm.active_sides = list(frame.fr3.active_sides)
    arm.joint_names = list(frame.fr3.joint_names)
    arm.positions = [float(x) for x in frame.fr3.positions]
    hands = []
    for side, record in (("left", frame.left_sharpa), ("right", frame.right_sharpa)):
        msg = JointState()
        msg.header.stamp = _stamp(frame.timestamp_ns)
        msg.header.frame_id = f"openpi:{side}:{record.session_id}:{frame.sequence}"
        msg.name = list(SHARPA_JOINT_NAMES)
        msg.position = [float(x) for x in record.position]
        msg.velocity = []
        msg.effort = []
        hands.append(msg)
    return arm, hands[0], hands[1]


def _require_parameter(node, remote: str, name: str, expected):
    import rclpy
    from rcl_interfaces.srv import GetParameters

    # Pixi's rclpy omits parameter_client; query the same standard read-only
    # service directly, retaining the existing deadline and required values.
    client = node.create_client(GetParameters, f"/{remote.strip('/')}/get_parameters")
    try:
        if not client.wait_for_service(timeout_sec=1.0):
            raise RuntimeError(f"Cannot inspect {remote} parameter {name}")
        request = GetParameters.Request()
        request.names = [name]
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=1.0)
        if not future.done() or future.exception() is not None:
            raise RuntimeError(f"Cannot inspect {remote} parameter {name}")
        values = future.result().values
        if len(values) != 1:
            raise RuntimeError(f"Cannot inspect {remote} parameter {name}")
        value = values[0]
    finally:
        node.destroy_client(client)
    actual = value.string_array_value if isinstance(expected, (list, tuple)) else value.bool_value
    if isinstance(expected, (list, tuple)):
        if not set(expected).issubset(set(actual)):
            raise RuntimeError(f"{remote} does not allow required source {expected}")
    elif actual is not expected:
        raise RuntimeError(f"{remote} parameter {name} is not enabled")


def validate_action_ranges(actions: np.ndarray, *, check_slew: bool = True) -> None:
    """Intrinsic chunk checks, independent of any offline or live state."""
    from teleop_core.safety import LOWER_LIMITS, UPPER_LIMITS

    actions = as_finite_action_chunk(actions)
    arm = np.concatenate((actions[:, LEFT_FR3_SLICE], actions[:, RIGHT_FR3_SLICE]), axis=1)
    if np.any(arm < LOWER_LIMITS) or np.any(arm > UPPER_LIMITS):
        raise ValueError("FR3 chunk exceeds existing gateway joint limits")
    validate_direct_hand_actions(actions, check_slew=check_slew)


def read_gateway_joint_speed(node):
    """Read the live arm speed bound before starting an online-task worker."""
    import rclpy
    import math
    from rcl_interfaces.srv import GetParameters
    client = node.create_client(GetParameters, "/teleop_safety_gateway/get_parameters")
    try:
        if not client.wait_for_service(timeout_sec=1.):
            raise RuntimeError("Cannot read FR3 gateway speed; online handover blocked")
        request = GetParameters.Request()
        request.names = ["max_joint_speed"]
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=1.)
        if not future.done() or future.exception() is not None:
            raise RuntimeError("FR3 gateway speed query timed out")
        values = future.result().values
        if len(values) != 1 or values[0].type != 3:
            raise ValueError("FR3 gateway max_joint_speed must be a double")
        speed = values[0].double_value
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("Invalid FR3 gateway max_joint_speed")
        return speed
    finally:
        node.destroy_client(client)


def validate_gateway_graph(node) -> None:
    if node.get_node_names().count("teleop_safety_gateway") != 1:
        raise RuntimeError("Missing/duplicate teleop_safety_gateway; commands stopped")
    gateway = node.get_subscriptions_info_by_topic(config.FR3_TOPIC)
    if not any(item.node_name == "teleop_safety_gateway" for item in gateway):
        raise RuntimeError("Safety Gateway is not subscribed to ArmCommand")


def validate_gateway_admission(node) -> None:
    """Fail before SDK connection or command publishers if OpenPI is not admitted."""
    validate_gateway_graph(node)
    _require_parameter(node, "teleop_safety_gateway", "allowed_sources", [config.SOURCE])


def validate_guarded_graph(node) -> None:
    if node.get_node_names().count("sharpa_driver") != 1:
        raise RuntimeError("Missing/duplicate sharpa_driver; commands stopped")
    for topic in (config.SHARPA_LEFT_TOPIC, config.SHARPA_RIGHT_TOPIC):
        sinks = node.get_subscriptions_info_by_topic(topic)
        if len(sinks) != 1 or sinks[0].node_name != "sharpa_driver":
            raise RuntimeError(f"Guarded OpenPI input ownership invalid: {topic}")
    for side in ("left", "right"):
        public_topic = f"/sharpa/{side}/command"
        if any(item.node_name == "sharpa_driver" for item in node.get_subscriptions_info_by_topic(public_topic)):
            raise RuntimeError("Ordinary Sharpa command subscriber detected; execute blocked")
        owners = node.get_publishers_info_by_topic(public_topic)
        if len(owners) != 1 or owners[0].node_name != "sharpa_driver":
            raise RuntimeError("Guarded Sharpa public command owner invalid")


def validate_execution_graph(node, backend: str = "litchibot") -> None:
    """The same existing endpoint/ownership requirements used in both modes."""
    validate_gateway_graph(node)
    if backend == "litchibot":
        validate_guarded_graph(node)


def validate_no_existing_hand_source(node) -> None:
    for topic in (config.SHARPA_LEFT_TOPIC,config.SHARPA_RIGHT_TOPIC):
        if node.get_publishers_info_by_topic(topic):
            raise RuntimeError('Existing hand source owns the topics; use persistent --prepare-policy-first handoff')


def prepare_messages(actions, session_id: str, start_ns: int):
    chunk = SharpaActionAdapter().adapt_chunk(
        actions, start_timestamp_ns=start_ns, session_id=session_id, source=config.SOURCE,
    )
    messages = [construct_messages(frame) for frame in chunk.frames]
    for index, (arm, left, right) in enumerate(messages):
        frame = chunk.frames[index]
        if arm.sequence != index or arm.header.stamp != _stamp(frame.timestamp_ns):
            raise ValueError("FR3 chunk sequence/timestamp mismatch")
        for side, hand in (("left", left), ("right", right)):
            if (hand.header.stamp != arm.header.stamp
                    or hand.header.frame_id != f"openpi:{side}:{chunk.session_id}:{index}"
                    or tuple(hand.name) != SHARPA_JOINT_NAMES
                    or len(hand.position) != 22):
                raise ValueError(f"Guarded {side} hand message contract mismatch")
    return chunk, messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("ros-test", "preflight-only", "execute"), required=True)
    parser.add_argument("--enable-hardware", action="store_true")
    parser.add_argument("--sharpa-backend", choices=("litchibot", "sdk_direct"), default="litchibot")
    args = parser.parse_args()
    if args.mode == "execute" and not args.enable_hardware:
        parser.error("execute requires --enable-hardware")
    if args.mode != "execute" and args.enable_hardware:
        parser.error("--enable-hardware is only valid for execute")
    payload = json.load(sys.stdin)
    actions = payload.get("actions") if args.mode == "preflight-only" else as_finite_action_chunk(payload["actions"])
    if args.mode == "execute":
        validate_action_ranges(actions)
    import rclpy
    from sensor_msgs.msg import JointState
    from teleop_interfaces.msg import ArmCommand

    rclpy.init()
    node = rclpy.create_node("fr3_sharpa_preflight" if args.mode == "preflight-only" else "fr3_sharpa_runtime")
    feedback_provider = None
    preflight_result = {"status": "WAITING FOR LIVE ROS STATE"}
    try:
        if args.mode == "preflight-only":
            # This branch returns before any command publisher creation path.
            from .live_preflight import inspect_live_preflight

            result = inspect_live_preflight(node, actions, payload["session_id"])
            print(json.dumps(result))
            if result["status"] == "FAIL":
                raise SystemExit(1)
            return
        if args.mode == "execute":
            # Check the exact endpoint names before any publisher is created.
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and "teleop_safety_gateway" not in node.get_node_names():
                time.sleep(0.1)
            validate_execution_graph(node, args.sharpa_backend)
            if args.sharpa_backend == "litchibot":
                validate_no_existing_hand_source(node)
            _require_parameter(node, "teleop_safety_gateway", "allowed_sources", [config.SOURCE])
            if args.sharpa_backend == "litchibot":
                _require_parameter(node, "sharpa_driver", "supervised_validation_guard", True)
            targets = (config.FR3_TOPIC, config.SHARPA_LEFT_TOPIC, config.SHARPA_RIGHT_TOPIC)
            print("EXECUTE PREFLIGHT: WAITING FOR LIVE ROS STATE", file=sys.stderr)
            feedback_provider = ROSFeedbackProvider(node)
            feedback = feedback_provider.wait()
            if args.sharpa_backend == "litchibot":
                preflight_result = execute_preflight(
                    actions, feedback, now_ros_ns=node.get_clock().now().nanoseconds,
                    now_monotonic_ns=time.monotonic_ns(),
                )
            else:
                feedback.validate_freshness(now_ros_ns=node.get_clock().now().nanoseconds,
                                            now_monotonic_ns=time.monotonic_ns())
                preflight_result = {"status": "PASS", "reference": "live_ros_feedback",
                                    "acquisition": "disabled", "backend": "sdk_direct"}
            direct = SharpaSdkDirectExecutor() if args.sharpa_backend == "sdk_direct" else None
            if direct is not None:
                direct.connect()
            # Do not create publishers if the live first-target check fails.
            arm_publisher = node.create_publisher(ArmCommand, targets[0], 10)
            hand_publishers = None if args.sharpa_backend == "sdk_direct" else (
                node.create_publisher(JointState, targets[1], 10),
                node.create_publisher(JointState, targets[2], 10),
            )
            publishers = (arm_publisher,)
            start_mono = time.monotonic_ns() + 500_000_000
            start_ns = node.get_clock().now().nanoseconds + 500_000_000
        else:
            publishers = None
            start_ns = 1_000_000_000
        chunk, messages = prepare_messages(actions, payload["session_id"], start_ns)
        if args.mode == "execute":
            for index, triple in enumerate(messages):
                target = start_mono + index * DT_NS
                while time.monotonic_ns() < target:
                    rclpy.spin_once(node, timeout_sec=max(0.0, min(0.01, (target - time.monotonic_ns()) / 1e9)))
                # Drain feedback queued at the frame deadline, then revalidate age.
                for _ in range(16):
                    rclpy.spin_once(node, timeout_sec=0.0)
                feedback = feedback_provider.snapshot()
                validate_execution_graph(node, args.sharpa_backend)
                if args.sharpa_backend == "litchibot" and index == 0:
                    preflight_result = execute_preflight(
                        actions, feedback, now_ros_ns=node.get_clock().now().nanoseconds,
                        now_monotonic_ns=time.monotonic_ns(),
                    )
                    print("EXECUTE PREFLIGHT: PASS (live ROS feedback, action_chunk[0])", file=sys.stderr)
                now_ros = node.get_clock().now().nanoseconds
                if not 0 <= now_ros - chunk.frames[index].timestamp_ns <= MAX_AGE_NS:
                    raise RuntimeError("Stale/future command timestamp; stopped publishing")
                if time.monotonic_ns() - target > DT_NS:
                    raise RuntimeError("30Hz deadline missed; stopped publishing")
                if direct is not None:
                    direct.send(triple[1].position, triple[2].position)
                    publishers[0].publish(triple[0])
                else:
                    publishers[0].publish(triple[0])
                    for publisher, message in zip(hand_publishers, triple[1:], strict=True):
                        publisher.publish(message)
        result = {
            "mode": args.mode,
            "frames": len(messages),
            "fr3_positions": len(messages[0][0].positions),
            "left_sharpa_positions": len(messages[0][1].position),
            "right_sharpa_positions": len(messages[0][2].position),
            "sequence": [messages[0][0].sequence, messages[-1][0].sequence],
            "timestamp_step_ns": DT_NS,
            "topics": [config.FR3_TOPIC] if args.sharpa_backend == "sdk_direct" else [config.FR3_TOPIC, config.SHARPA_LEFT_TOPIC, config.SHARPA_RIGHT_TOPIC],
            "sharpa_backend": args.sharpa_backend,
            "published": args.mode == "execute",
            "execute_preflight": preflight_result,
        }
        print(json.dumps(result))
    finally:
        if 'direct' in locals() and direct is not None:
            direct.stop()
        if feedback_provider is not None:
            feedback_provider.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
