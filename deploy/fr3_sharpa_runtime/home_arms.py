"""Idle-only, arm-only Home through the existing FR3 safety gateway."""

import argparse
import importlib
import inspect
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


HOME_TRACKING_LIMIT_RAD = .05


def home_command(previous, target, measured, lower, upper, *, speed_rad_s=HOME_SPEED_RAD_S,
                 sequence=None, elapsed_s=None):
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
    previous_error = np.abs(previous - measured)
    # Hold the last command until both arms catch up, leaving margin below the
    # tracking fault limit. Never rebase the command onto moving feedback.
    command = (previous.copy() if np.max(previous_error) > HOME_TOLERANCE_RAD
               else previous + np.clip(target - previous, -step, step))
    checked = previous if np.max(previous_error) > HOME_TRACKING_LIMIT_RAD else command
    errors = np.abs(checked - measured)
    worst = int(np.argmax(errors))
    if errors[worst] > HOME_TRACKING_LIMIT_RAD:
        detail = dict(side='left' if worst < 7 else 'right', joint=FR3_COMMAND_NAMES[worst],
                      joint_index=worst % 7 + 1, command_rad=float(checked[worst]),
                      previous_command_rad=float(previous[worst]), measured_rad=float(measured[worst]),
                      target_rad=float(target[worst]), error_rad=float(checked[worst] - measured[worst]),
                      maximum_error_rad=float(errors[worst]), limit_rad=HOME_TRACKING_LIMIT_RAD,
                      sequence=sequence, elapsed_s=elapsed_s)
        raise ValueError('Arm Home tracking error exceeds 0.05 rad (' + json.dumps(detail) + ')')
    return command


def wait_for_arm_controllers(node, *, timeout_s=15.):
    """Use local ROS bindings, or the bindings in this session's FR3 container."""
    try:
        importlib.import_module('controller_manager_msgs.srv')
    except ModuleNotFoundError as error:
        if error.name not in ('controller_manager_msgs', 'controller_manager_msgs.srv'):
            raise
    else:
        return _wait_for_arm_controllers_ros(node, timeout_s=timeout_s)

    container = os.environ.get('SHARPA_HOME_CONTROLLER_CONTAINER')
    if not container:
        raise RuntimeError('Arm Home controller messages unavailable; use the Home launcher to select its FR3 container')
    # The Pixi observation environment omits ros2_control messages. Query the
    # existing controller container with the same helper, without installing or
    # mixing Python-version-specific message bindings into the host process.
    script = ('import json, time, rclpy\n' + inspect.getsource(_wait_for_arm_controllers_ros)
              + '\nrclpy.init()\nnode = rclpy.create_node("openpi_home_controller_check")\n'
              + f'try:\n    states = _wait_for_arm_controllers_ros(node, timeout_s={timeout_s!r})\n'
              + 'finally:\n    node.destroy_node()\n    rclpy.shutdown()\n'
              + 'print(json.dumps(states), flush=True)\n')
    try:
        result = subprocess.run(
            ['docker', 'exec', '-i', '-e', 'PYTHONDONTWRITEBYTECODE=1', container, 'bash', '-lc',
             'set -e; source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"; exec python3 -'],
            input=script, text=True, capture_output=True, timeout=timeout_s+5.)
    except subprocess.TimeoutExpired as error:
        raise TimeoutError('Arm Home controller query timed out in the FR3 container') from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError('Arm Home controller readiness check failed: ' + detail)
    try:
        states = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError('Arm Home controller readiness returned invalid status') from error
    if states != dict(left='active', right='active'):
        raise RuntimeError('Arm Home controllers not active (' + json.dumps(states) + ')')
    return states


def _wait_for_arm_controllers_ros(node, *, timeout_s=15.):
    """Require both named ROS controllers to be active before creating motion."""
    import rclpy
    from controller_manager_msgs.srv import ListControllers

    clients = {}
    states = dict(left='service unavailable', right='service unavailable')
    deadline = time.monotonic() + timeout_s
    try:
        for side in states:
            clients[side] = node.create_client(ListControllers, f'/{side}/controller_manager/list_controllers')
        while rclpy.ok():
            for side, client in clients.items():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                if not client.wait_for_service(timeout_sec=min(.1, remaining)):
                    states[side] = 'service unavailable'
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                future = client.call_async(ListControllers.Request())
                rclpy.spin_until_future_complete(node, future, timeout_sec=min(.2, remaining))
                if not future.done():
                    client.remove_pending_request(future)
                    future.cancel()
                    states[side] = 'query timed out'
                    continue
                if future.exception() is not None:
                    raise RuntimeError(f'Arm Home controller query failed for {side}') from future.exception()
                states[side] = next((item.state for item in future.result().controller
                                     if item.name == 'joint_impedance_controller'), 'controller missing')
            if time.monotonic() >= deadline:
                raise TimeoutError('Arm Home controllers not active (' + json.dumps(states) + ')')
            if all(state == 'active' for state in states.values()):
                return states
            rclpy.spin_once(node, timeout_sec=.05)
        raise RuntimeError('ROS stopped while waiting for Arm Home controllers')
    finally:
        for client in clients.values():
            node.destroy_client(client)


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
                measured()
                if 'teleop_safety_gateway' in node.get_node_names():
                    break
            except ValueError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError('Arm Home gateway/feedback startup timed out')
        validate_gateway_admission(node)
        speed = min(HOME_SPEED_RAD_S, read_gateway_joint_speed(node))
        controllers = wait_for_arm_controllers(node)
        if node.get_publishers_info_by_topic(config.FR3_TOPIC):
            raise RuntimeError('Another arm command publisher is active; idle Home refused')
        # Discard pre-activation samples, including samples received while
        # checking gateway parameters. The trajectory starts at this new state.
        feedback_after_ros_ns = node.get_clock().now().nanoseconds
        feedback_after_mono_ns = time.monotonic_ns()
        deadline = time.monotonic() + 5.
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.01)
            try:
                initial = measured()
                if all(stamp > feedback_after_ros_ns and received > feedback_after_mono_ns
                       for _, stamp, received in samples.values()):
                    break
            except ValueError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError('Arm Home requires fresh dual-arm feedback after controller activation')
        else:
            raise RuntimeError('ROS stopped before Arm Home acquired its initial state')
        publisher = node.create_publisher(ArmCommand, config.FR3_TOPIC, 10)
        previous = initial.copy()
        last_feedback = initial.copy()
        started = next_tick = time.monotonic()
        timeout = max(120., np.max(np.abs(target - initial)) / speed * 2 + 10)
        stable_since = None
        sequence = 0
        next_progress = started
        print(json.dumps(dict(event='home_initial_feedback', controllers=controllers,
                              measured_rad=initial.tolist(), target_rad=target.tolist(),
                              speed_rad_s=speed)), flush=True)
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
            command = home_command(previous, target, current, LOWER_LIMITS, UPPER_LIMITS,
                                   speed_rad_s=speed, sequence=sequence, elapsed_s=now-started)
            if now >= next_progress:
                worst = int(np.argmax(np.abs(command-current)))
                print(json.dumps(dict(event='home_progress', sequence=sequence, elapsed_s=now-started,
                                      tracking_joint=FR3_COMMAND_NAMES[worst],
                                      command_rad=float(command[worst]), measured_rad=float(current[worst]),
                                      target_rad=float(target[worst]),
                                      tracking_error_rad=float(command[worst]-current[worst]),
                                      maximum_pose_error_rad=float(np.max(np.abs(current-target))),
                                      waiting_for_tracking=bool(np.max(np.abs(previous-current))
                                                                > HOME_TOLERANCE_RAD))), flush=True)
                next_progress = now + 1.
            message = ArmCommand()
            message.header.stamp = node.get_clock().now().to_msg()
            message.source, message.session_id, message.sequence = config.SOURCE, session, sequence
            message.active_sides = ['left', 'right']
            message.joint_names, message.positions = list(FR3_COMMAND_NAMES), command.tolist()
            publisher.publish(message)
            sequence += 1
            stable = (np.max(np.abs(command - target)) <= 1e-6
                      and np.max(np.abs(current - target)) <= HOME_TOLERANCE_RAD
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
        container = hardware.owned.get('franka-control', {}).get('Id')
        if not container:
            raise RuntimeError('Arm Home could not identify its FR3 controller container')
        os.environ['SHARPA_HOME_CONTROLLER_CONTAINER'] = container
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
