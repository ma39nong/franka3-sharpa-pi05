"""Idle-only, arm-only Home through the existing FR3 safety gateway."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import uuid

import numpy as np

from .home_pose import load_home
from .policy_prepare import HOME_SPEED_RAD_S, HOME_TOLERANCE_RAD
from .protocol import DT_NS, FR3_COMMAND_NAMES


def home_command(previous, target, measured, lower, upper, *, speed_rad_s=HOME_SPEED_RAD_S):
    values = [np.asarray(value, dtype=np.float64) for value in (previous, target, measured, lower, upper)]
    if any(value.shape != (14,) or not np.isfinite(value).all() for value in values):
        raise ValueError('Arm Home requires fourteen finite joint positions and limits')
    previous, target, measured, lower, upper = values
    if np.any(target < lower) or np.any(target > upper):
        raise ValueError('Arm Home target exceeds robot joint limits')
    if np.any(measured < lower) or np.any(measured > upper):
        raise ValueError('Measured arm positions exceed robot joint limits')
    if not np.isfinite(speed_rad_s) or speed_rad_s <= 0:
        raise ValueError('Arm Home speed must be finite and positive')
    step = speed_rad_s * DT_NS / 1e9
    command = previous + np.clip(target - previous, -step, step)
    if np.max(np.abs(command - measured)) > .05:
        raise ValueError('Arm Home tracking error exceeds 0.05 rad')
    return command


def ros_home(path, task):
    import rclpy
    from sensor_msgs.msg import JointState
    from teleop_interfaces.msg import ArmCommand, ArmCommandStatus
    from teleop_core.safety import LOWER_LIMITS, UPPER_LIMITS

    from . import config
    from .ros_observation import MAX_AGE_NS, MAX_SPREAD_NS, TOPICS, ordered_state, stamp_ns
    from .ros_transport import read_gateway_joint_speed, validate_gateway_admission

    arms, metadata = load_home(path, task)
    metadata.pop('hands', None)
    target = np.concatenate((arms['left'], arms['right']))
    rclpy.init()
    node = rclpy.create_node('openpi_idle_arm_home')
    session = 'openpi-arm-home-' + uuid.uuid4().hex
    samples, acknowledgement = {}, {}

    def receive(side, message):
        samples[side] = (ordered_state(message, side, False), stamp_ns(message), time.monotonic_ns())

    def acknowledge(message):
        if message.session_id == session and message.source == config.SOURCE:
            acknowledgement.update(sequence=int(message.sequence), accepted=set(message.accepted_sides),
                                   faults=list(message.faults), received=time.monotonic())

    def measured():
        now_ros = node.get_clock().now().nanoseconds
        now_mono = time.monotonic_ns()
        if set(samples) != {'left', 'right'}:
            raise ValueError('Waiting for both arm feedback topics')
        stamps = []
        for side in ('left', 'right'):
            positions, stamp, received = samples[side]
            if not 0 <= now_ros - stamp <= MAX_AGE_NS or not 0 <= now_mono - received <= MAX_AGE_NS:
                raise ValueError('Stale/future arm feedback: ' + side)
            stamps.append(stamp)
        if max(stamps) - min(stamps) > MAX_SPREAD_NS:
            raise ValueError('Arm Home feedback is not synchronized')
        return np.concatenate((samples['left'][0], samples['right'][0]))

    try:
        for side in ('left', 'right'):
            node.create_subscription(JointState, TOPICS[side + '_arm'],
                                     lambda message, side=side: receive(side, message), 1)
        node.create_subscription(ArmCommandStatus, '/teleop/arm_command_status', acknowledge, 30)
        deadline = time.monotonic() + 15
        while True:
            rclpy.spin_once(node, timeout_sec=.01)
            try:
                initial = measured()
                if 'teleop_safety_gateway' in node.get_node_names():
                    break
            except ValueError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError('Arm Home gateway/feedback startup timed out')
        validate_gateway_admission(node)
        speed = min(HOME_SPEED_RAD_S, read_gateway_joint_speed(node))
        if node.get_publishers_info_by_topic(config.FR3_TOPIC):
            raise RuntimeError('Another arm command publisher is active; idle Home refused')
        publisher = node.create_publisher(ArmCommand, config.FR3_TOPIC, 10)
        previous = initial.copy()
        last_feedback = initial.copy()
        started = next_tick = time.monotonic()
        timeout = max(120., np.max(np.abs(target - initial)) / speed * 2 + 10)
        stable_since = None
        sequence = 0
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.001)
            now = time.monotonic()
            if now < next_tick:
                continue
            if now - next_tick > DT_NS / 1e9:
                raise RuntimeError('Arm Home 30Hz dispatch deadline missed')
            current = measured()
            if any(info.node_name != 'openpi_idle_arm_home'
                   for info in node.get_publishers_info_by_topic(config.FR3_TOPIC)):
                raise RuntimeError('Another arm command publisher appeared during Home')
            if now - started > timeout:
                raise TimeoutError('Arms did not reach and settle at Home')
            if now - started > .2 and (not acknowledgement
                    or now - acknowledgement['received'] > .2
                    or acknowledgement['accepted'] != {'left', 'right'}
                    or acknowledgement['faults']
                    or acknowledgement['sequence'] < max(0, sequence - 4)):
                raise RuntimeError('Arm Home gateway acknowledgement missing/stale/rejected')
            command = home_command(previous, target, current, LOWER_LIMITS, UPPER_LIMITS, speed_rad_s=speed)
            message = ArmCommand()
            message.header.stamp = node.get_clock().now().to_msg()
            message.source, message.session_id, message.sequence = config.SOURCE, session, sequence
            message.active_sides = ['left', 'right']
            message.joint_names, message.positions = list(FR3_COMMAND_NAMES), command.tolist()
            publisher.publish(message)
            sequence += 1
            stable = (np.max(np.abs(command - target)) <= 1e-6
                      and np.max(np.abs(current - target)) <= HOME_TOLERANCE_RAD
                      and np.max(np.abs(command - target)) <= 1e-6
                      and np.max(np.abs(current - last_feedback)) <= .005)
            stable_since = (now if stable_since is None else stable_since) if stable else None
            previous, last_feedback = command, current
            if stable_since is not None and now - stable_since >= .5:
                print(json.dumps(dict(status='PASS', operation='arms_home', home=metadata,
                                      maximum_pose_error_rad=float(np.max(np.abs(current-target))))), flush=True)
                return
            next_tick += DT_NS / 1e9
        raise RuntimeError('ROS stopped before Arm Home completed')
    finally:
        node.destroy_node()
        rclpy.shutdown()


def run_home(path, task):
    from .camera_lease import CameraLease
    from .hardware_session import HardwareSession, run_session

    load_home(path, task)
    lease = CameraLease().acquire()
    hardware = None
    try:
        candidate = HardwareSession()
        gateway = candidate.owned.get('teleop-control')
        if gateway and gateway['State']['Running'] and not candidate.is_openpi(gateway, 'teleop-control'):
            raise RuntimeError('Teleoperation owns the gateway; idle Home refused')
        hardware = candidate
        try:
            subprocess.run([*hardware.compose, 'up', '-d', 'franka-control', 'teleop-control'], check=True, timeout=120)
        finally:
            hardware.owned = hardware.inventory()
            hardware.owned.pop('openpi-cameras', None)
        root = Path(__file__).resolve().parents[2]
        gello = Path(os.environ['SHARPA_GELLO_REPO'])
        values = {}
        for line in (gello/'docker/.env').read_text().splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                key, value = line.split('=', 1)
                values[key.strip()] = value.strip().strip('"\'')
        domain = values.get('TELEOP_ROS_DOMAIN_ID', '')
        if not domain.isdecimal():
            raise ValueError('Missing TELEOP_ROS_DOMAIN_ID in hardware docker/.env')
        os.environ.update(ROS_DOMAIN_ID=domain, ROS_LOCALHOST_ONLY='1',
                          RMW_IMPLEMENTATION='rmw_cyclonedds_cpp',
                          CYCLONEDDS_URI='file://' + str(gello/'config/cyclonedds.xml'))
        overlay = Path(os.environ.get('SHARPA_ROS_MESSAGES_OVERLAY', root/'.deployment/sharpa_ros_messages/install'))
        if not (overlay/'setup.bash').is_file():
            raise FileNotFoundError('Build the teleop_interfaces ROS overlay before using Arm Home')
        script = (f'source {shlex.quote(str(overlay/"setup.bash"))}; '
                  f'export PYTHONPATH={shlex.quote(str(root))}:{shlex.quote(str(gello/"ros_ws/src/teleop_core"))}:"${{PYTHONPATH:-}}"; '
                  f'exec python -m deploy.fr3_sharpa_runtime.home_arms --ros '
                  f'--home-poses {shlex.quote(str(path))} --task {shlex.quote(task)}')
        command = ['pixi', 'run', '--locked', '-e', 'ros2', 'bash', '-lc', script]
        original = Path.cwd()
        try:
            os.chdir(gello/'portable_deps/litchi_hardware')
            return run_session(command, hardware)
        finally:
            os.chdir(original)
            hardware = None  # run_session already released its exact container IDs.
    finally:
        try:
            if hardware is not None:
                hardware.cleanup()
        finally:
            lease.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home-poses', type=Path, required=True)
    parser.add_argument('--task', default='powder_weighing')
    parser.add_argument('--ros', action='store_true')
    args = parser.parse_args()
    if args.ros:
        ros_home(args.home_poses, args.task)
    else:
        raise SystemExit(run_home(args.home_poses, args.task))


if __name__ == '__main__':
    main()
