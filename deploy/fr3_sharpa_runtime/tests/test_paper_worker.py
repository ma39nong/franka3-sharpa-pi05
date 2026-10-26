"""Run the real worker request loop with entirely simulated ROS and SDK objects."""

# ruff: noqa: PLC0415
import io
import json
import sys
import time
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.compression import compression_candidate
from deploy.fr3_sharpa_runtime.feedback import LiveFeedback
from deploy.fr3_sharpa_runtime.rolling import RollingOptions


@pytest.mark.parametrize("fail_after", [None, 2])
def test_real_worker_compressed_stream_and_partial_failure_audit(monkeypatch, capsys, fail_after):
    from deploy.fr3_sharpa_runtime import sdk_direct
    from deploy.fr3_sharpa_runtime import sdk_direct_motion_ros as worker

    state = np.zeros(58, np.float32)
    sent, stopped = [], []

    class Stamp:
        @property
        def nanoseconds(self):
            return time.monotonic_ns()

        def to_msg(self):
            return None

    class Node:
        def get_clock(self):
            return SimpleNamespace(now=Stamp)

        def get_node_names(self):
            return ["teleop_safety_gateway"]

        def get_publishers_info_by_topic(self, topic):
            return []

        def destroy_node(self):
            pass

        def create_publisher(self, message, topic, depth):
            def publish(value):
                if topic == worker.config.FR3_TOPIC:
                    state[:7], state[29:36] = value.positions[:7], value.positions[7:]

            return SimpleNamespace(publish=publish)

    class Message:
        def __init__(self):
            self.header = SimpleNamespace(stamp=None)

    class Feedback:
        def __init__(self, node):
            pass

        def snapshot(self):
            now = time.monotonic_ns()
            return LiveFeedback(state.copy(), (now,) * 4, (now,) * 4)

        def close(self):
            pass

    class SDK:
        def __init__(self):
            hand = SimpleNamespace(
                read_joint_state=lambda: SimpleNamespace(names=[], position=[], velocity=[], effort=[])
            )
            self.left = self.right = hand

        def connect(self):
            pass

        def resume_stream(self):
            pass

        def send(self, left, right):
            if fail_after is not None and len(sent) >= fail_after:
                raise RuntimeError("simulated interrupted stream")
            state[7:29], state[36:58] = left, right
            sent.append((left.copy(), right.copy()))

        def shutdown_confirmed(self):
            stopped.append(True)
            return {"status": "PASS", "simulation": True}

    class ROSExecutor:
        def __init__(self, **kwargs):
            pass

        def add_node(self, node):
            pass

        def spin(self):
            pass

        def shutdown(self, **kwargs):
            pass

    ros = ModuleType("rclpy")
    ros.init = ros.shutdown = lambda: None
    ros.ok = lambda: True
    ros.create_node = lambda name: Node()
    executor_module = ModuleType("rclpy.executors")
    executor_module.MultiThreadedExecutor = ROSExecutor
    monkeypatch.setitem(sys.modules, "rclpy", ros)
    monkeypatch.setitem(sys.modules, "rclpy.executors", executor_module)
    for name, symbol in [("sensor_msgs", "JointState"), ("teleop_interfaces", "ArmCommand")]:
        parent, child = ModuleType(name), ModuleType(name + ".msg")
        setattr(child, symbol, Message)
        parent.msg = child
        monkeypatch.setitem(sys.modules, name, parent)
        monkeypatch.setitem(sys.modules, name + ".msg", child)
    monkeypatch.setattr(worker, "ROSFeedbackProvider", Feedback)
    monkeypatch.setattr(worker, "SharpaSdkDirectExecutor", SDK)
    monkeypatch.setattr(worker, "validate_gateway_admission", lambda node: None)

    def ranges(values, **kwargs):
        if np.max(np.abs(values)) > 1:
            raise ValueError("simulated native bounds")

    monkeypatch.setattr(worker, "validate_action_ranges", ranges)
    monkeypatch.setattr(sdk_direct, "hand_joint_limits", lambda: np.tile([-1.0, 1.0], (22, 1)))
    monkeypatch.setattr(
        worker, "construct_messages", lambda frame: (SimpleNamespace(positions=frame.fr3.positions), None, None)
    )

    def play(frames, send, feedback):
        for frame in frames:
            feedback()
            send(frame)

    monkeypatch.setattr(worker, "play_command_frames", play)
    monkeypatch.setattr(worker.select, "select", lambda *args: ([sys.stdin], [], []))
    monkeypatch.delenv("SHARPA_PROTOCOL_FD", raising=False)
    predictions = np.broadcast_to(np.linspace(0, 0.02, 50)[:, None], (50, 58)).astype(np.float32).copy()
    candidate, compression = compression_candidate(predictions, state, RollingOptions(compression="cubic"))
    payload = {
        "operation": "chunk",
        "actions": predictions.tolist(),
        "execute_steps": 26,
        "execution_targets": candidate.tolist(),
        "compression": compression,
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload) + '\n{"operation":"close"}\n'))
    monkeypatch.setattr(sys, "argv", ["worker", "--enable-hardware"])
    if fail_after is None:
        worker.main()
    else:
        with pytest.raises(SystemExit):
            worker.main()
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows[-1]["operation"] == "shutdown"
    assert stopped == [True]
    result = rows[0]
    if fail_after is None:
        assert result["policy_frames"] == 26
        assert result["execution_target_count"] == 20
        assert len(result["command_frames"]) == len(sent) == 20
        assert result["execution_plan_audit"]["sent_command_count"] == 20
        np.testing.assert_allclose(state, predictions[25])
    else:
        assert result["status"] == "FAIL"
        detail = result["detail"].split("Experimental execution audit: ")[-1]
        audit = json.loads(detail)
        assert audit["sent_command_count"] == 2
        assert len(audit["command_frames"]) == 20
