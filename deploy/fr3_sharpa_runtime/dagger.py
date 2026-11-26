"""Hardware-free DAgger authority and event contract; one owner calls step/commit.

Labels describe dispatched software targets. Gateway admission and measured
motion remain separate evidence, never fabricated hardware acknowledgements.
"""
from collections import OrderedDict
import hashlib
import json
import threading
import time

import numpy as np

GROUPS = ('left_arm', 'left_hand', 'right_arm', 'right_hand')
SLICES = (slice(0, 7), slice(7, 29), slice(29, 36), slice(36, 58))
ACTIVE = ('POLICY', 'HOLDING', 'HOLD', 'HUMAN_LEFT', 'RESUMING')
FRESH_PREDICTION_MAX_AGE_NS = 240_000_000


def fresh_prediction_budget_s(captured_monotonic_ns, now_monotonic_ns):
    age = now_monotonic_ns - captured_monotonic_ns
    if not 0 <= age <= FRESH_PREDICTION_MAX_AGE_NS:
        raise ValueError(
            f'DAgger fresh prediction age {age / 1e6:.3f}ms outside '
            f'0..{FRESH_PREDICTION_MAX_AGE_NS / 1e6:.0f}ms admission')
    return (FRESH_PREDICTION_MAX_AGE_NS - age) / 1e9


def vector(value, size=58):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'Expected {size} finite joint positions')
    return result.copy()


class DaggerAuthority:
    def __init__(self, initial, *, run_id, emit, clock=time.monotonic):
        self.lock = threading.RLock()
        self.run_id, self.emit, self.clock = run_id, emit, clock
        self.last = vector(initial)
        self.hold = self.last.copy()
        self.mode = 'IDLE'
        self.generation = 0
        self.intervention_id = 0
        self.intervening = False
        self.hold_since = None
        self.previous_feedback = None
        self.anchor = self.input_anchor = None
        self.input_session = None
        self.reason = ''
        self.stage = self.prompt = None
        self.last_contact = clock()
        self.seen = OrderedDict()
        self.events = []
        self.sequence = -1
        self.frame_index = -1
        self.pending_intervention_end = False
        self.recorded_intervention = False
        self.last_frame = None
        self.input_ready = False
        self.input_error = 'waiting for left GELLO and MANUS'
        self.source_lost_at = None

    def snapshot(self):
        with self.lock:
            return dict(run_id=self.run_id, mode=self.mode, generation=self.generation,
                        intervention_id=self.intervention_id if self.intervening else None,
                        hold_confirmed=self.mode == 'HOLD', sequence=self.sequence, frame_index=self.frame_index,
                        stage=self.stage, prompt=self.prompt, reason=self.reason,
                        input_ready=self.input_ready, input_error=self.input_error)

    def event(self, name, **fields):
        # Buffered until both hand sends and ArmCommand.publish have succeeded.
        self.events.append(dict(event=name, **fields))

    def invalidate(self, reason):
        self.generation += 1
        self.hold = self.last.copy()
        self.hold_since = None
        self.previous_feedback = None
        self.reason = reason

    def check_input(self, human, now):
        if human is None:
            raise ValueError('Left GELLO/MANUS input missing')
        for key in ('arm_time', 'hand_time'):
            age = now - human[key]
            if not 0 <= age <= human.get('timeout_s', .2):
                raise ValueError('Left input stale/future: ' + key)
        vector(human['arm'], 7)
        vector(human['hand'], 22)
        if not human.get('session'):
            raise ValueError('Missing input session')
        return human

    def command(self, command, request_id, expected_generation, *, measured, human, now=None, **unused):
        now = self.clock() if now is None else now
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
            raise ValueError('request_id required')
        fingerprint = hashlib.sha256(json.dumps([command, expected_generation]).encode()).hexdigest()
        if request_id in self.seen:
            if self.seen[request_id] != fingerprint:
                raise ValueError('request_id reused for different command')
            return
        if type(expected_generation) is not int or expected_generation != self.generation:
            raise ValueError('Control generation changed; refresh status')
        if command == 'takeover':
            if self.mode not in ('POLICY', 'RESUMING'):
                raise ValueError('Takeover requires policy execution')
            self.invalidate('operator takeover')
            if not self.intervening:
                self.intervention_id += 1
                self.intervening = True
                self.event('human_start', intervention_id=self.intervention_id)
            self.mode = 'HOLDING'
            self.event('takeover_applied')
        elif command == 'hold':
            if self.mode != 'HUMAN_LEFT':
                raise ValueError('Hold requires human following')
            self.invalidate('operator hold')
            self.mode = 'HOLDING'
            self.event('human_hold')
        elif command == 'follow':
            if self.mode != 'HOLD':
                raise ValueError('Wait for measured hold and gateway admission')
            if self.previous_feedback is None or np.max(np.abs(vector(measured)-self.previous_feedback)) > .005:
                raise ValueError('Measured hold changed; wait for stability again')
            human = self.check_input(human, now)
            self.anchor = vector(measured)[:7]
            self.input_anchor = vector(human['arm'], 7)
            self.input_session = human['session']
            self.mode = 'HUMAN_LEFT'
            self.source_lost_at = None
            self.event('human_follow')
        elif command == 'resume':
            if self.mode != 'HOLD':
                raise ValueError('Hold and confirm stability before resuming policy')
            self.invalidate('explicit resume; fresh observation required')
            self.mode = 'RESUMING'
            self.source_lost_at = None
            self.event('resume_requested')
        else:
            raise ValueError('Unknown DAgger command')
        self.seen[request_id] = fingerprint
        if len(self.seen) > 256:
            self.seen.popitem(last=False)

    def begin_policy(self):
        # A newer stage may supersede a fresh prediction still awaiting admission.
        if self.mode not in ('IDLE', 'HOLD', 'POLICY', 'RESUMING') or self.intervening:
            raise ValueError('Policy start not authorized')
        self.invalidate('fresh policy requested')
        self.mode = 'RESUMING'
        self.event('policy_requested')

    def accept_policy(self, generation, *, stage, prompt):
        if generation != self.generation or self.mode not in ('POLICY', 'RESUMING'):
            return False
        if self.intervening:
            self.event('human_end', intervention_id=self.intervention_id, reason='policy restored')
            self.pending_intervention_end = True
        self.mode, self.stage, self.prompt = 'POLICY', stage, prompt
        self.event('policy_applied')
        return True

    def step(self, measured, human, *, gateway_ok, now=None):
        now = self.clock() if now is None else now
        measured = vector(measured)
        try:
            self.check_input(human, now)
            self.input_ready, self.input_error = True, ''
        except (ValueError, KeyError, TypeError) as error:
            self.input_ready, self.input_error = False, str(error)
        if self.mode == 'HUMAN_LEFT':
            if not self.input_ready or human['session'] != self.input_session:
                self.invalidate('human source lost/restarted; explicit follow required')
                self.mode = 'HOLDING'
                self.source_lost_at = now
                self.event('input_lost', reason=self.input_error or 'input session restarted')
            else:
                sensitivity = vector(human['sensitivity'], 7)
                caps = vector(human['max_delta'], 7)
                if np.any(sensitivity <= 0) or np.any(sensitivity > 2.2) or np.any(caps <= 0):
                    raise ValueError('Invalid GELLO mapping')
                target = self.hold.copy()
                target[:7] = self.anchor + np.clip((vector(human['arm'], 7)-self.input_anchor)*sensitivity, -caps, caps)
                target[7:29] = vector(human['hand'], 22)
                return target
        if self.source_lost_at is not None:
            if self.input_ready:
                # Recovery grants no following authority. Operator must follow again.
                self.source_lost_at = None
            elif now - self.source_lost_at > 1.5:
                raise RuntimeError('Human input unavailable for 1.5s; stopped')
        if self.mode in ('HOLDING', 'HOLD'):
            stable = (gateway_ok and np.max(np.abs((measured-self.hold)[np.r_[0:7,29:36]])) <= .05
                      and self.previous_feedback is not None
                      and np.max(np.abs(measured-self.previous_feedback)) <= .005)
            self.previous_feedback = measured.copy()
            if stable:
                if self.hold_since is None:
                    self.hold_since = now
                if self.mode == 'HOLDING' and now-self.hold_since >= .5:
                    self.mode = 'HOLD'
                    self.event('hold_confirmed', basis='gateway_admitted_and_measured_stable')
            else:
                self.hold_since = None
                if self.mode == 'HOLD':
                    self.mode = 'HOLDING'
        return self.hold.copy()

    def commit(self, command, measured, *, sequence, stamp_ns, phase, gateway_sequence, dispatch_timing=None):
        self.last = vector(command)
        self.sequence = sequence
        self.frame_index += 1
        if self.pending_intervention_end:
            self.intervening = False
            self.pending_intervention_end = False
        if any(e['event'] == 'human_start' for e in self.events):
            self.recorded_intervention = True
        if not self.intervening:
            self.recorded_intervention = False
        owners = (['policy']*4 if self.mode == 'POLICY' and phase != 'waiting_prediction' else
                  ['human', 'human', 'hold', 'hold'] if self.mode == 'HUMAN_LEFT' else ['hold']*4)
        frame = dict(schema='sharpa.dagger.frame.v1', run_id=self.run_id, sequence=sequence, frame_index=self.frame_index,
                     source_timestamp_ns=stamp_ns, monotonic_ns=time.monotonic_ns(),
                     generation=self.generation, mode=self.mode, phase=phase,
                     intervention_id=self.intervention_id if self.intervening else None,
                     action_owner=owners, stage=self.stage, prompt=self.prompt,
                     model_version=getattr(self, 'model_version', None),
                     action=self.last.tolist(), state=vector(measured).tolist(),
                     action_basis='software_dispatched_target', gateway_sequence=gateway_sequence)
        if dispatch_timing is not None:
            frame['dispatch_timing'] = dispatch_timing
        self.last_frame = frame
        for event in self.events:
            self.emit({**frame, **event, 'schema': 'sharpa.dagger.event.v1'})
        self.events.clear()
        self.emit(frame)
        return frame

    def end(self, reason):
        if self.intervening and self.recorded_intervention:
            self.emit(dict(schema='sharpa.dagger.event.v1', event='human_end',
                           run_id=self.run_id, intervention_id=self.intervention_id,
                           generation=self.generation, sequence=self.sequence,
                           monotonic_ns=time.monotonic_ns(), reason=reason, incomplete=True))
        self.emit(dict(schema='sharpa.dagger.event.v1',event='run_end',run_id=self.run_id,
                       sequence=self.sequence,frame_index=self.frame_index,monotonic_ns=time.monotonic_ns(),reason=reason))
        self.mode = 'STOPPED'
