"""Real ROS status propagation with fake sensors/hands in a network-none container."""

import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def main():
    if (
        not Path("/.dockerenv").exists()
        or set(os.listdir("/sys/class/net")) != {"lo"}
        or os.environ.get("ROS_DOMAIN_ID") != "197"
    ):
        raise RuntimeError("This test requires Docker --network none and ROS_DOMAIN_ID=197")
    sys.path.insert(0, "/workspace/franka_upper_body_teleop/ros_ws/src/teleop_core")
    import rclpy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from teleop_core import contract

    from .controller_status import STATUS_TOPIC
    from .devices import DeviceSession
    from .ros_devices import RosArms

    rclpy.init()
    robot = rclpy.create_node("pi05_fake_controller_status")
    statuses = {s: robot.create_publisher(String, STATUS_TOPIC.format(side=s), 1) for s in ("left", "right")}
    states = {s: robot.create_publisher(JointState, contract.ARM_STATE_TOPIC.format(side=s), 1) for s in statuses}

    class FakeHand:
        def __init__(self):
            self.last, self.stops = None, 0
            self.latest_received = time.monotonic()

        def poll(self, now, wall):
            self.latest_received = now
            return np.zeros(20), np.zeros(20), now

        def emergency_stop(self):
            self.stops += 1

    def publish(fault_side=None):
        for side in statuses:
            fault = side == fault_side
            statuses[side].publish(
                String(
                    data=json.dumps(
                        {
                            "version": 1,
                            "stamp_ns": time.time_ns(),
                            "faulted": fault,
                            "reason": "previous_command_expired" if fault else "none",
                            "expected_sequence": 504,
                            "received_sequence": 503,
                        }
                    )
                )
            )
            msg = JointState()
            msg.header.stamp = robot.get_clock().now().to_msg()
            msg.name = [f"{side}_fr3_joint{i}" for i in range(1, 8)]
            msg.position, msg.velocity = [0.0] * 7, [0.0] * 7
            states[side].publish(msg)

    try:
        for side in ("left", "right"):
            arms = RosArms()
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    publish()
                    arms.spin(0.003)
                    if len(arms.controller_status.latest) == 2 and len(arms.samples) == 2:
                        break
                    time.sleep(0.002)
                arms.check_controller_status()
                hands = {s: FakeHand() for s in statuses}
                session = DeviceSession(arms, hands, {"lower": np.full(54, -2), "upper": np.full(54, 2)})
                session.feedback()
                # Simulate an acquired session; no motion/enable requests are sent.
                session.state = "armed"
                began = time.monotonic()
                while time.monotonic() - began < 1:
                    publish(side)
                    # Keep sender watchdog healthy, isolating controller-fault handling.
                    session.last_activity = time.monotonic()
                    session.last_health_check = 0
                    session.service()
                    if session.state == "stopped":
                        break
                    time.sleep(0.001)
                assert session.state == "stopped", "controller fault did not stop the session"
                assert "previous_command_expired" in session.fault, session.fault
                assert arms.stopped and all(h.stops == 1 for h in hands.values())
                # Faulted controller feedback remains readable after stopping.
                publish(side)
                arms.spin(0.005)
                session.feedback()
                print(
                    json.dumps(
                        {
                            "case": side,
                            "passed": True,
                            "hardware_output": False,
                            "fault": session.fault,
                            "detection_ms": (time.monotonic() - began) * 1000,
                        }
                    )
                )
            finally:
                arms.stop()
                arms.close()
    finally:
        robot.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
