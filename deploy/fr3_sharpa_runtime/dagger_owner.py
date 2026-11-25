"""DAgger dispatch loop inside the existing SDK/ROS owner; no service restarts."""
import json
import os
from pathlib import Path
import select
import sys
import threading
import time

import numpy as np

from .dagger import DaggerAuthority
from .dagger_inputs import InputCache, ARM_TOPIC, HAND_TOPIC, configuration
from .dagger_io import AuditWriter, ControlChannel, FRAME_TOPIC, EVENT_TOPIC
from .policy_prepare import target_chunk
from .preflight import execute_preflight
from .protocol import DT_NS, LEFT_SHARPA_SLICE, RIGHT_SHARPA_SLICE
from .ros_transport import RTCCommandLimiter, validate_action_ranges
from .rtc_stream import RTCTimeline
from .sdk_direct import chunk_command_frames, clip_hand_predictions


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
    plan_done = True
    mode = initial_request['broker_mode']
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
                    plan_done = True
                if core.mode != 'POLICY':
                    timeline = frames = None
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
                        actions, clipping = clip_hand_predictions(request['actions'], execute_steps=50,
                            max_correction_rad=request['prediction_clip_rad'])
                        validate_action_ranges(actions, check_slew=False)
                        receipt = request.get('receipt')
                        if receipt is not None:
                            if timeline is None or core.mode != 'POLICY':
                                acknowledgements.append(dict(status='PASS', rejected=True, reason='obsolete prefix'))
                                continue
                            result = timeline.install(actions, receipt, now)
                            frame_index = bridge_steps
                        else:
                            age = time.monotonic_ns()-request['captured_monotonic_ns']
                            if not 0 <= age <= 200_000_000:
                                raise ValueError('DAgger fresh prediction exceeded original 200ms admission')
                            count = initial_request['execute_steps'] if mode == 'serial' else 50
                            frames, bridge_steps = chunk_command_frames(actions, core.last,
                                execute_steps=count, speed_scale=initial_request['speed_scale'],
                                approach_speed_rad_s=initial_request['approach_speed_rad_s'])
                            frame_index = 0
                            timeline = (RTCTimeline(actions, now+bridge_steps*dt,
                                initial_request['speed_scale']) if mode == 'rtc' else None)
                            result = dict(skipped=0)
                        if not core.accept_policy(request['generation'], stage=request.get('stage'), prompt=request['prompt']):
                            raise RuntimeError('DAgger authority changed during install')
                        plan_done = False
                        acknowledgements.append(dict(status='PASS', **result, prediction_clipping=clipping))
                    else:
                        raise ValueError('Unknown DAgger owner operation: '+str(operation))
                phase = 'hold'
                if core.mode == 'POLICY':
                    if frames is not None and not plan_done and (timeline is None or frame_index < bridge_steps):
                        target = frames[min(frame_index, len(frames)-1)]
                        phase = 'approach' if frame_index < bridge_steps else 'policy'
                        frame_index += 1
                        plan_done = timeline is None and frame_index >= len(frames)
                        latest_report = dict(policy_done=plan_done, consumed_since_install=max(0, frame_index-bridge_steps), phase=0)
                    elif timeline is not None:
                        target = timeline.sample(now)
                        phase = 'policy'
                        status = timeline.status(now)
                        latest_report = dict(policy_done=False, **status)
                    elif plan_done:
                        target = core.last.copy()
                        phase = 'waiting_prediction'
                elif core.mode == 'HUMAN_LEFT':
                    phase = 'human_follow'
                    target[:7] = np.clip(target[:7], limiter.lower[:7]+mapping['margin'], limiter.upper[:7]-mapping['margin'])
                    step = mapping['arm_speed']*dt
                    target[:7] = core.last[:7]+np.clip(target[:7]-core.last[:7], -step, step)
                command = limiter.step_toward(core.last, target)
                if core.mode == 'POLICY' and np.max(np.abs(command-target)) > 1e-6:
                    raise ValueError('Policy timeline exceeds installed dispatch speed; refused inconsistent RTC prefix')
                validate_action_ranges(np.vstack((core.last, np.repeat(command[None,:],49,axis=0))))
                execute_preflight(target_chunk(command), current,
                    now_ros_ns=node.get_clock().now().nanoseconds, now_monotonic_ns=time.monotonic_ns())
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
