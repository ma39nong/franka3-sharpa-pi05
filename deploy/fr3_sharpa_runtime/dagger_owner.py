"""DAgger dispatch loop inside the existing SDK/ROS owner; no service restarts."""
import json
import os
from pathlib import Path
import select
import sys
import threading
import time

import numpy as np

from .dagger import DaggerAuthority, vector
from .dagger_inputs import InputCache, ARM_TOPIC, HAND_TOPIC, configuration
from .dagger_io import AuditWriter, ControlChannel, FRAME_TOPIC, EVENT_TOPIC
from .policy_prepare import target_chunk
from .policy_execution import install_policy_rtc, policy_command, prepare_policy_actions
from .preflight import FIRST_TARGET_LIMITS_RAD, execute_preflight
from .protocol import DT_NS, LEFT_SHARPA_SLICE, RIGHT_SHARPA_SLICE
from .ros_transport import RTCCommandLimiter, validate_action_ranges
from .rtc_stream import RTCTimeline
from .sdk_direct import chunk_command_frames


HUMAN_HAND_SPEED_RAD_S = 0.8


def smooth_follow_target(target, previous, measured, dt, *, size, speed_rad_s, group):
    """Bound human targets by both last dispatch and current measured feedback."""
    target = vector(target, size)
    previous = vector(previous, size)
    measured = vector(measured, size)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('Human smoothing requires a positive finite tick duration')
    if not np.isfinite(speed_rad_s) or speed_rad_s <= 0:
        raise ValueError('Human smoothing requires a positive finite joint speed')
    step = speed_rad_s * dt
    # Leave float32 rounding room inside the existing acquisition bound.
    lead = FIRST_TARGET_LIMITS_RAD[group] - 1e-5
    lower = np.maximum(previous - step, measured - lead)
    upper = np.minimum(previous + step, measured + lead)
    if np.any(lower > upper):
        raise ValueError(f'{group} tracking error prevents a continuous human-follow command')
    return np.clip(target, lower, upper)


def smooth_left_hand_target(target, previous, measured, dt):
    return smooth_follow_target(target, previous, measured, dt,
        size=22, speed_rad_s=HUMAN_HAND_SPEED_RAD_S, group='left_sharpa')


def smooth_left_arm_target(target, previous, measured, dt, speed_rad_s):
    return smooth_follow_target(target, previous, measured, dt,
        size=7, speed_rad_s=speed_rad_s, group='left_fr3')


def run_owner(node, executor, feedback, recording, publish, protocol, *, gateway_speed, initial_request, session_id):
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from teleop_interfaces.msg import ArmCommandStatus
    dt = DT_NS/1e9
    directory = Path(os.environ['SHARPA_DAGGER_DIRECTORY'])
    audit = AuditWriter(directory/'dagger')
    frame_pub = node.create_publisher(String, FRAME_TOPIC, 30)
    event_pub = node.create_publisher(String, EVENT_TOPIC, 30)
    def emit(value):
        audit.put(value)
        message = String()
        message.data = json.dumps(value, ensure_ascii=False, allow_nan=False)
        (event_pub if 'event' in value else frame_pub).publish(message)
    def fresh():
        value = feedback.snapshot()
        value.validate_freshness(now_ros_ns=node.get_clock().now().nanoseconds,
                                 now_monotonic_ns=time.monotonic_ns())
        return value
    current = fresh()
    validate_action_ranges(target_chunk(current.state), check_slew=False)
    core = DaggerAuthority(current.state, run_id=os.environ['SHARPA_DAGGER_RUN_ID'], emit=emit)
    core.model_version = json.loads((directory/'dagger_manifest.json').read_text())['model_version']
    mapping = configuration(os.environ['SHARPA_GELLO_REPO'])
    inputs = InputCache(mapping)
    ack = {}
    ack_lock = threading.Lock()
    def acknowledge(message):
        if message.session_id != session_id or message.source != 'openpi':
            return
        emit(dict(schema='sharpa.dagger.gateway.v1', event='gateway_status',
                  sequence=int(message.sequence), source=message.source, session_id=message.session_id,
                  accepted_sides=list(message.accepted_sides), faults=list(message.faults),
                  monotonic_ns=time.monotonic_ns()))
        with ack_lock:
            ack.update(sequence=int(message.sequence), received=time.monotonic(),
                       accepted=list(message.accepted_sides), faults=list(message.faults))
    subscriptions = [
        node.create_subscription(JointState, ARM_TOPIC,
            lambda m: inputs.receive('arm', m, node.get_clock().now().nanoseconds), 1),
        node.create_subscription(JointState, HAND_TOPIC,
            lambda m: inputs.receive('hand', m, node.get_clock().now().nanoseconds), 1),
        node.create_subscription(ArmCommandStatus, '/teleop/arm_command_status', acknowledge, 30),
    ]
    limiter = RTCCommandLimiter(gateway_speed)
    control = ControlChannel(os.environ['SHARPA_DAGGER_SOCKET'], core)
    frames = None
    frame_index = bridge_steps = 0
    timeline = None
    pending_rtc_actions = None
    plan_done = True
    mode = initial_request['broker_mode']
    policy_limiter = limiter if mode == 'rtc' and initial_request.get('online_task_handoff') else None
    entry_bridge_until = None
    first_tick = next_tick = time.monotonic()
    latest_report = dict(policy_done=True, consumed_since_install=0, phase=0)
    def reply(value):
        print(json.dumps(value, allow_nan=False), file=protocol, flush=True)
    recording.start()  # Ready before the first DAgger command/label.
    executor.resume_stream()
    first_tick = next_tick = time.monotonic()
    core.last_contact = first_tick
    reply(dict(status='PASS', operation='dagger_start', dagger=core.snapshot()))
    reason = 'owner closed'
    try:
        while True:
            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(.003, next_tick-now))
                continue
            if now-next_tick > dt:
                raise RuntimeError('DAgger 30Hz dispatch deadline missed')
            audit.check()
            current = fresh()
            human = inputs.snapshot()
            if any(len(node.get_publishers_info_by_topic(topic)) != 1 for topic in (ARM_TOPIC,HAND_TOPIC)):
                human = None
            with core.lock:
                if now-core.last_contact > 5:
                    raise RuntimeError('Local DAgger controller lease expired')
                previous_generation = core.generation
                pending = control.apply(current.state, human)
                acknowledgements = []
                with ack_lock:
                    gateway = dict(ack)
                now = time.monotonic()
                gateway_ok = (bool(gateway) and 0 <= now-gateway['received'] <= .2
                    and set(gateway['accepted']) == {'left','right'} and not gateway['faults']
                    and gateway['sequence'] >= max(0, core.sequence-4))
                if now-first_tick > .2 and not gateway_ok:
                    raise RuntimeError('Missing/stale/rejected FR3 gateway acknowledgement: '+str(gateway))
                target = core.step(current.state, human, gateway_ok=gateway_ok, now=now)
                if core.generation != previous_generation:
                    if timeline is not None:
                        timeline.invalidate()
                    timeline = frames = None
                    pending_rtc_actions = None
                    entry_bridge_until = None
                    plan_done = True
                if core.mode != 'POLICY':
                    timeline = frames = None
                    pending_rtc_actions = None
                    entry_bridge_until = None
                for _ in range(4):
                    readable, _, _ = select.select([sys.stdin], [], [], 0)
                    if not readable:
                        break
                    line = sys.stdin.readline()
                    if not line:
                        return
                    request = json.loads(line)
                    operation = request.get('operation')
                    if operation == 'close':
                        return
                    if operation == 'dagger_status':
                        acknowledgements.append(dict(status='PASS', dagger=core.snapshot(), **latest_report))
                    elif operation == 'dagger_request_policy':
                        if request.get('generation') != core.generation:
                            acknowledgements.append(dict(status='PASS',rejected=True))
                            continue
                        core.begin_policy()
                        timeline = frames = None
                        pending_rtc_actions = None
                        entry_bridge_until = None
                        plan_done = True
                        target = core.hold.copy()
                        acknowledgements.append(dict(status='PASS', dagger=core.snapshot()))
                    elif operation == 'dagger_snapshot':
                        if request['generation'] != core.generation or core.mode != 'POLICY' or timeline is None or frame_index < bridge_steps:
                            acknowledgements.append(dict(status='PASS', rejected=True))
                        else:
                            context = timeline.snapshot(now, delay_steps=request['delay_steps'],
                                schedule=request['schedule'], weight=request['weight'],
                                hard_prefix=request.get('hard_prefix', False))
                            context['previous_actions'] = context['previous_actions'].tolist()
                            acknowledgements.append(dict(status='PASS', context=context))
                    elif operation == 'dagger_install':
                        if request['generation'] != core.generation or core.mode not in ('POLICY','RESUMING'):
                            acknowledgements.append(dict(status='PASS', rejected=True, reason='obsolete authority'))
                            continue
                        actions, clipping = prepare_policy_actions(request['actions'],
                            limiter=policy_limiter, max_correction_rad=request['prediction_clip_rad'])
                        receipt = request.get('receipt')
                        if receipt is not None:
                            if timeline is None or core.mode != 'POLICY':
                                acknowledgements.append(dict(status='PASS', rejected=True, reason='obsolete prefix'))
                                continue
                            result = install_policy_rtc(timeline, actions, receipt, now, core.last,
                                arm_speed=gateway_speed if policy_limiter is not None else None)
                            if result.get('rejected'):
                                acknowledgements.append(dict(status='PASS', **result))
                                continue
                            steps = result.get('entry_bridge_steps', 0)
                            entry_bridge_until = timeline.installed_at + steps / timeline.node_hz if steps else None
                            frame_index = bridge_steps
                        else:
                            age = time.monotonic_ns()-request['captured_monotonic_ns']
                            if not 0 <= age <= 200_000_000:
                                raise ValueError('DAgger fresh prediction exceeded original 200ms admission')
                            count = initial_request['execute_steps'] if mode == 'serial' else 50
                            frames, bridge_steps = chunk_command_frames(actions, current.state,
                                execute_steps=count, speed_scale=initial_request['speed_scale'],
                                approach_speed_rad_s=initial_request['approach_speed_rad_s'])
                            frame_index = 0
                            timeline = None
                            pending_rtc_actions = actions if mode == 'rtc' else None
                            entry_bridge_until = None
                            result = dict(skipped=0)
                        if not core.accept_policy(request['generation'], stage=request.get('stage'), prompt=request['prompt']):
                            raise RuntimeError('DAgger authority changed during install')
                        plan_done = False
                        acknowledgements.append(dict(status='PASS', **result, prediction_clipping=clipping))
                    else:
                        raise ValueError('Unknown DAgger owner operation: '+str(operation))
                phase = 'hold'
                if core.mode == 'POLICY':
                    if frames is not None and not plan_done and (mode == 'serial' or frame_index < bridge_steps):
                        target = frames[min(frame_index, len(frames)-1)]
                        phase = 'approach' if frame_index < bridge_steps else 'policy'
                        frame_index += 1
                        plan_done = timeline is None and frame_index >= len(frames)
                        latest_report = dict(policy_done=plan_done, consumed_since_install=max(0, frame_index-bridge_steps), phase=0)
                    elif timeline is not None or pending_rtc_actions is not None:
                        if pending_rtc_actions is not None:
                            # Anchor to the actual bridge completion tick. Dispatch
                            # counts and monotonic time can differ by scheduler jitter.
                            timeline = RTCTimeline(pending_rtc_actions, now,
                                initial_request['speed_scale'])
                            pending_rtc_actions = None
                        sample_time = (max(timeline.installed_at, next_tick)
                            if entry_bridge_until is not None and next_tick <= entry_bridge_until else now)
                        target = timeline.sample(sample_time)
                        phase = 'policy'
                        status = timeline.status(now)
                        latest_report = dict(policy_done=False, **status)
                    elif plan_done:
                        target = core.last.copy()
                        phase = 'waiting_prediction'
                elif core.mode == 'HUMAN_LEFT':
                    phase = 'human_follow'
                    target[:7] = np.clip(target[:7], limiter.lower[:7]+mapping['margin'], limiter.upper[:7]-mapping['margin'])
                    target[:7] = smooth_left_arm_target(target[:7], core.last[:7], current.state[:7], dt,
                        min(mapping['arm_speed'], gateway_speed))
                    target[LEFT_SHARPA_SLICE] = smooth_left_hand_target(
                        target[LEFT_SHARPA_SLICE], core.last[LEFT_SHARPA_SLICE],
                        current.state[LEFT_SHARPA_SLICE], dt)
                if core.mode == 'POLICY':
                    command = policy_command(core.last, target,
                        limiter=policy_limiter, arm_speed=gateway_speed)
                else:
                    command = policy_command(core.last, target,
                        limiter=limiter, arm_speed=gateway_speed)
                    if core.mode == 'HUMAN_LEFT':
                        execute_preflight(target_chunk(command), current,
                            now_ros_ns=node.get_clock().now().nanoseconds, now_monotonic_ns=time.monotonic_ns(),
                            groups=('left_fr3', 'left_sharpa'))
                executor.send(command[LEFT_SHARPA_SLICE], command[RIGHT_SHARPA_SLICE])
                sequence, stamp = publish(command)
                core.commit(command, current.state, sequence=sequence, stamp_ns=stamp, phase=phase,
                            gateway_sequence=gateway.get('sequence'))
                control.confirm(pending)
                for report in acknowledgements:
                    if 'dagger' in report:
                        report['dagger'] = core.snapshot()
                    reply(report)
            next_tick += dt
    except BaseException as error:
        reason = str(error) or type(error).__name__
        raise
    finally:
        # Reporting failures must never skip the caller's existing disable path.
        try:
            core.end(reason)
        finally:
            try:
                control.close()
            finally:
                for subscription in subscriptions:
                    node.destroy_subscription(subscription)
                audit.close()
