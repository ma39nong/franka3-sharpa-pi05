"""ROS bridge for the direct SDK prepare/execute path.

Only FR3 commands cross ROS (through the Safety Gateway). Sharpa positions
are sent directly through ``SharpaSdkDirectExecutor``.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import threading
import time
import uuid
import traceback

import numpy as np

from . import config
from .action_adapter import SharpaActionAdapter
from .feedback import ROSFeedbackProvider
from .policy_prepare import SlowApproach, target_chunk
from .protocol import (DT_NS, LEFT_SHARPA_SLICE, RIGHT_SHARPA_SLICE,
                       as_finite_action_chunk)
from .ros_transport import construct_messages, validate_action_ranges, validate_gateway_admission
from .sdk_direct import (DEFAULT_PREDICTION_CLIP_RAD, SharpaSdkDirectExecutor, action_prefix,
                         chunk_command_frames, clip_hand_predictions, play_command_frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-hardware", action="store_true", required=True)
    parser.add_argument("--speed-rad-s", type=float, default=0.03)
    parser.add_argument("--execute-speed-scale", type=float, default=1.0)
    args = parser.parse_args()
    from .shutdown import DisableSettings
    DisableSettings.from_env()  # Reject invalid shutdown bounds before touching devices.

    import rclpy
    from sensor_msgs.msg import JointState
    from teleop_interfaces.msg import ArmCommand

    rclpy.init()
    node = rclpy.create_node("fr3_sharpa_runtime_sdk_direct")
    feedback = ROSFeedbackProvider(node)
    executor = SharpaSdkDirectExecutor()
    protocol = (os.fdopen(int(os.environ["SHARPA_PROTOCOL_FD"]), "w", buffering=1)
                if os.environ.get("SHARPA_PROTOCOL_FD") else sys.stdout)
    arm_pub = None
    hand_pubs = None
    feedback_stop = threading.Event()
    feedback_thread = None
    ros_executor = None
    ros_spin_thread = None
    session_id = f"openpi-sdk-direct-{uuid.uuid4().hex}"
    sequence = 0
    clipping_report = None
    experimental_audit = None

    def publish_frame(values: np.ndarray) -> None:
        nonlocal sequence
        stamp = node.get_clock().now().nanoseconds
        if stamp <= 0:
            raise RuntimeError("Invalid ROS timestamp; stopped")
        frame = SharpaActionAdapter().adapt_chunk(
            target_chunk(values), start_timestamp_ns=stamp,
            session_id=session_id, source=config.SOURCE).frames[0]
        arm, _, _ = construct_messages(frame)
        arm.sequence = sequence
        arm_pub.publish(arm)
        sequence += 1

    def fresh_feedback():
        value = feedback.snapshot()
        value.validate_freshness(now_ros_ns=node.get_clock().now().nanoseconds,
                                 now_monotonic_ns=time.monotonic_ns())
        return value

    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and "teleop_safety_gateway" not in node.get_node_names():
            rclpy.spin_once(node, timeout_sec=0.05)
        validate_gateway_admission(node)
        existing = {
            topic: node.get_publishers_info_by_topic(topic)
            for topic in ("/sharpa/left/joint_states", "/sharpa/right/joint_states")
        }
        if any(existing.values()):
            raise RuntimeError(
                "sdk_direct requires exclusive Sharpa SDK ownership; stop existing "
                "sharpa_driver before starting direct execution")
        connect_error = None
        for attempt in range(3):
            try:
                executor.connect()
                connect_error = None
                break
            except Exception as error:
                connect_error = error
                executor.stop()
                if attempt < 2:
                    time.sleep(1.0)
        if connect_error is not None:
            raise connect_error
        arm_pub = node.create_publisher(ArmCommand, config.FR3_TOPIC, 10)
        hand_pubs = (
            node.create_publisher(JointState, "/sharpa/left/joint_states", 10),
            node.create_publisher(JointState, "/sharpa/right/joint_states", 10),
        )

        def publish_feedback():
            while not feedback_stop.is_set():
                try:
                    for side, publisher, hand in (
                        ("left", hand_pubs[0], executor.left),
                        ("right", hand_pubs[1], executor.right),
                    ):
                        state = hand.read_joint_state()
                        message = JointState()
                        message.header.stamp = node.get_clock().now().to_msg()
                        message.name = [f"{side}_{name}" for name in state.names]
                        message.position = list(state.position)
                        message.velocity = list(state.velocity)
                        message.effort = list(state.effort)
                        publisher.publish(message)
                except Exception:
                    pass
                feedback_stop.wait(0.05)

        feedback_thread = threading.Thread(target=publish_feedback, daemon=True)
        feedback_thread.start()
        from rclpy.executors import MultiThreadedExecutor
        ros_executor = MultiThreadedExecutor(num_threads=2)
        ros_executor.add_node(node)
        ros_spin_thread = threading.Thread(target=ros_executor.spin, daemon=True)
        ros_spin_thread.start()
        while rclpy.ok():
            # Keep ROS feedback subscriptions alive while the IPC client is
            # idle between ready/prepare/chunk requests.
            readable, _, _ = select.select([sys.stdin], [], [], 0.01)
            if not readable:
                continue
            line = sys.stdin.readline()
            if not line:
                break
            request = json.loads(line)
            clipping_report = None
            experimental_audit = None
            operation = request.get("operation")
            if operation == "ready":
                print(json.dumps({"status": "PASS", "operation": "ready",
                                  "feedback_topics": ["/sharpa/left/joint_states",
                                                       "/sharpa/right/joint_states"]}),
                      file=protocol, flush=True)
            elif operation == "prepare":
                target = target_chunk(request["target"])[0]
                validate_action_ranges(target_chunk(target))
                current = fresh_feedback()
                approach = SlowApproach(target, current.state,
                                        speed_rad_s=float(request.get("speed_rad_s", args.speed_rad_s)),
                                        # Keep prepare completion strictly inside
                                        # the execute FR3 first-target gate
                                        # (0.05 rad) without changing that gate.
                                        tolerance_rad=float(request.get("tolerance_rad", 0.049)))
                next_tick = time.monotonic_ns()
                settled = False
                last_error = None
                while not settled:
                    now = time.monotonic_ns()
                    if now < next_tick:
                        time.sleep(min(0.01, (next_tick-now)/1e9))
                        continue
                    if now - next_tick > DT_NS:
                        raise RuntimeError("Prepare 30Hz deadline missed; stopped")
                    current = fresh_feedback()
                    command, settled, last_error = approach.step(
                        current, now_ros_ns=node.get_clock().now().nanoseconds,
                        now_monotonic_ns=now, enforce_execute_delta=False)
                    executor.send(command[LEFT_SHARPA_SLICE], command[RIGHT_SHARPA_SLICE])
                    publish_frame(command)
                    next_tick += DT_NS
                print(json.dumps({"status": "PASS", "operation": "prepare",
                                  "published": True, "completed_ros_ns": node.get_clock().now().nanoseconds,
                                  "completed_monotonic_ns": time.monotonic_ns(),
                                  "maximum_pose_error_rad": last_error,
                                  "policy_first_target": target.tolist(),
                                  "frames": sequence}), file=protocol, flush=True)
            elif operation == "chunk":
                actions = as_finite_action_chunk(np.asarray(request["actions"], dtype=np.float32))
                execute_steps = request.get("execute_steps", 50)
                experiment_report = None
                original_actions = actions.copy()
                if "execution_targets" in request:
                    from .compression import plan_execution, recheck_compression
                    if execute_steps != 26:
                        raise ValueError("Rolling execution requires 26 consumed model steps")
                    speed_scale = float(request.get("execute_speed_scale", args.execute_speed_scale))
                    current = fresh_feedback()
                    validate_action_ranges(target_chunk(current.state), check_slew=False)
                    compression = dict(request["compression"])
                    candidate = request["execution_targets"]
                    # Recheck hand variation against feedback obtained immediately before planning.
                    candidate, compression = recheck_compression(actions, candidate, compression, current.state)
                    frames, bridge_frames, selected, clipping_report, experiment_report = plan_execution(
                        actions, candidate, compression, current.state, speed_scale=speed_scale,
                        approach_speed_rad_s=float(request.get("approach_speed_rad_s", args.speed_rad_s)),
                        range_validator=validate_action_ranges,
                        prediction_clip_rad=float(request.get("prediction_clip_rad", DEFAULT_PREDICTION_CLIP_RAD)))
                else:
                    actions, clipping_report = clip_hand_predictions(
                        actions, execute_steps=execute_steps,
                        max_correction_rad=float(request.get("prediction_clip_rad", DEFAULT_PREDICTION_CLIP_RAD)))
                    if clipping_report["corrected_values"]:
                        print(f"[prediction-clip] {clipping_report['corrected_values']} values corrected; "
                              f"max={clipping_report['max_applied_correction_rad']:.6f} rad", file=sys.stderr, flush=True)
                    selected = action_prefix(actions, execute_steps)
                    validation_actions = actions.copy()
                    validation_actions[execute_steps:] = selected[-1]
                    validate_action_ranges(validation_actions, check_slew=False)
                    speed_scale = float(request.get("execute_speed_scale", args.execute_speed_scale))
                    if not np.isfinite(speed_scale) or not 0.0 < speed_scale <= 1.0:
                        raise ValueError("execute_speed_scale must be finite and in (0, 1]")
                    current = fresh_feedback()
                    validate_action_ranges(target_chunk(current.state), check_slew=False)
                    frames, bridge_frames = chunk_command_frames(
                        actions, current.state, speed_scale=speed_scale,
                        execute_steps=execute_steps,
                        approach_speed_rad_s=float(request.get("approach_speed_rad_s", args.speed_rad_s)))

                # A policy inference/observation boundary may exceed the
                # hand driver's 0.25 s continuous-stream watchdog.  This is
                # an explicit new command phase; continuity protection resumes
                # from the first frame below and remains active within it.
                if experiment_report is not None:
                    experimental_audit = {"compression": experiment_report, "execution_targets": selected.tolist(),
                        "command_frames": frames.tolist(), "sent_command_count": 0,
                        "command_timestamps_monotonic_ns": [], "consumed_model_steps": 26,
                        "bridge_frames": bridge_frames, "planning_complete_monotonic_ns": time.monotonic_ns()}
                executor.resume_stream()
                def send_chunk_frame(frame):
                    executor.send(frame[LEFT_SHARPA_SLICE], frame[RIGHT_SHARPA_SLICE])
                    publish_frame(frame)
                    if experimental_audit is not None:
                        experimental_audit["sent_command_count"] += 1
                        experimental_audit["command_timestamps_monotonic_ns"].append(time.monotonic_ns())

                play_command_frames(frames, send_chunk_frame, fresh_feedback)
                end_feedback = fresh_feedback()
                error_by_group = {
                    name: float(np.max(np.abs(end_feedback.state[part] - selected[-1, part])))
                    for name, part in (("left_arm", slice(0, 7)), ("left_hand", LEFT_SHARPA_SLICE),
                                       ("right_arm", slice(29, 36)), ("right_hand", RIGHT_SHARPA_SLICE))
                }
                print(json.dumps({"status": "PASS", "operation": "chunk", "published": True,
                                  "frames": len(frames), "policy_frames": execute_steps,
                                  "predicted_policy_frames": len(actions), "execute_steps": execute_steps,
                                  "prediction_clipping": clipping_report,
                                  "executed_action_indices": [0, execute_steps - 1],
                                  "bridge_frames": bridge_frames,
                                  "completion": "commands_sent", "end_error_rad": error_by_group,
                                  "execute_speed_scale": speed_scale,
                                  "completed_ros_ns": node.get_clock().now().nanoseconds,
                                  "completed_monotonic_ns": time.monotonic_ns(),
                                  **({"compression": experiment_report,
                                      "execution_target_count": len(selected),
                                      "execution_targets": selected.tolist(),
                                      "original_prediction": original_actions.tolist(),
                                      "execution_plan_audit": experimental_audit,
                                      "command_frames": frames.tolist(),
                                      "alignment": "trajectory_phase_not_wall_clock"}
                                     if experiment_report is not None else {})}),
                      file=protocol, flush=True)
            elif operation == "close":
                break
            else:
                raise ValueError(f"Unknown direct SDK operation: {operation}")
    except Exception as error:
        # Keep the one-line IPC contract, but include the Python traceback in
        # the report.  Direct SDK failures otherwise collapse into messages
        # such as ``'str' object has no attribute 'value'`` and cannot be
        # distinguished from ROS or gateway failures by the parent runtime.
        detail = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        if clipping_report is not None:
            detail += "\nPrediction clipping audit: " + json.dumps(clipping_report, allow_nan=False)
        if experimental_audit is not None:
            detail += "\nExperimental execution audit: " + json.dumps(experimental_audit, allow_nan=False)
        print(json.dumps({"status": "FAIL", "published": sequence > 0,
                          "reason": str(error), "detail": detail}), file=protocol, flush=True)
        raise SystemExit(1)
    finally:
        from .shutdown import protect_disable_confirmation
        with protect_disable_confirmation():
            feedback_stop.set()
            try:
                if ros_executor is not None:
                    ros_executor.shutdown(timeout_sec=1.0)
                if ros_spin_thread is not None:
                    ros_spin_thread.join(timeout=1.0)
                if feedback_thread is not None:
                    feedback_thread.join(timeout=1.0)
            except Exception as error:
                print(f"WARNING: ROS cleanup failed before hand disable: {error}", file=sys.stderr, flush=True)
            try:
                shutdown_report = executor.shutdown_confirmed()
            except Exception as error:
                shutdown_report = {"status": "FAIL", "error": str(error)}
                print(f"[sharpa-shutdown] ERROR 失能确认异常，双手状态未确认: {error}", file=sys.stderr, flush=True)
            # Dedicated IPC acknowledgement: a pixi wrapper exiting does not
            # prove its SDK child has completed disable/readback.
            try:
                print(json.dumps({"operation": "shutdown", **shutdown_report}), file=protocol, flush=True)
            except (OSError, ValueError) as error:
                print(f"[sharpa-shutdown] WARNING cannot deliver shutdown acknowledgement: {error}",
                      file=sys.stderr, flush=True)
            feedback.close()
            node.destroy_node()
            rclpy.shutdown()
            if protocol is not sys.stdout:
                protocol.close()


if __name__ == "__main__":
    main()
