"""Drive the real persistent-owner loop and Unix RPC using simulated devices."""
import json
import os
import queue
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime import dagger_owner as owner
from deploy.fr3_sharpa_runtime import policy_execution
from deploy.fr3_sharpa_runtime.dagger_io import rpc


class Protocol:
    def __init__(self):
        self.buffer='';self.lines=queue.Queue()
    def write(self, value):
        self.buffer+=value
        while '\n' in self.buffer:
            line,self.buffer=self.buffer.split('\n',1)
            self.lines.put(json.loads(line))
    def flush(self):
        pass


@pytest.mark.parametrize('target_value',[.8,-.8])
def test_hand_follow_ramps_large_glove_offset_and_respects_measured_lead(target_value):
    previous=np.zeros(22,dtype=np.float32)
    measured=previous.copy()
    target=np.full(22,target_value,dtype=np.float32)
    for _ in range(180):
        command=owner.smooth_left_hand_target(target,previous,measured,1/30)
        assert np.max(np.abs(command-previous))<=owner.HUMAN_HAND_SPEED_RAD_S/30+1e-6
        assert np.max(np.abs(command-measured))<.15
        measured+=.2*(command-measured)
        previous=command
    np.testing.assert_allclose(command,target,atol=1e-4)


def test_hand_follow_waits_for_feedback_and_handles_new_glove_target():
    measured=np.zeros(22,dtype=np.float32)
    previous=measured.copy()
    for _ in range(30):
        previous=owner.smooth_left_hand_target(np.ones(22),previous,measured,1/30)
    assert .14<float(previous.max())<.15
    changed=owner.smooth_left_hand_target(-np.ones(22),previous,measured,1/30)
    assert np.all(changed<previous)
    assert np.max(np.abs(changed-previous))<=owner.HUMAN_HAND_SPEED_RAD_S/30+1e-6


def test_hand_follow_rejects_tracking_jump_that_cannot_be_joined_continuously():
    with pytest.raises(ValueError,match='tracking error'):
        owner.smooth_left_hand_target(np.ones(22),np.full(22,.4),np.zeros(22),1/30)


@pytest.mark.parametrize('target_value',[.8,-.8])
def test_arm_follow_limits_accumulated_tracking_error_with_slow_feedback(target_value):
    previous=np.zeros(7,dtype=np.float32)
    measured=previous.copy()
    target=np.full(7,target_value,dtype=np.float32)
    for _ in range(300):
        command=owner.smooth_left_arm_target(target,previous,measured,1/30,.7)
        assert np.max(np.abs(command-previous))<=.7/30+1e-6
        assert np.max(np.abs(command-measured))<.05
        measured+=.1*(command-measured)
        previous=command
    np.testing.assert_allclose(command,target,atol=1e-4)


def test_arm_follow_caps_1647_failure_and_waits_for_feedback():
    previous=np.zeros(7,dtype=np.float32)
    previous[6]=.044283
    measured=np.zeros(7,dtype=np.float32)
    target=previous.copy()
    target[6]=.053082
    for _ in range(20):
        command=owner.smooth_left_arm_target(target,previous,measured,1/30,.7)
        assert np.max(np.abs(command-measured))<.05
        previous=command
    assert .049<float(command[6])<.05
    measured[:]=command
    next_command=owner.smooth_left_arm_target(target,command,measured,1/30,.7)
    np.testing.assert_allclose(next_command,target)


def test_arm_follow_rejects_feedback_jump_without_a_continuous_safe_command():
    with pytest.raises(ValueError,match='left_fr3 tracking error'):
        owner.smooth_left_arm_target(np.ones(7),np.full(7,.1),np.zeros(7),1/30,.7)


@pytest.mark.parametrize('broker_mode',['serial','rtc'])
@pytest.mark.parametrize('bridge_distance',[0., .03])
@pytest.mark.parametrize('staged',[False, True])
def test_live_loop_takeover_old_reply_rejection_left_only_resume_and_labels(tmp_path,monkeypatch,broker_mode,bridge_distance,staged):
    run_id='simulated-run'
    sock=tmp_path/'dagger.sock'
    for key,value in dict(SHARPA_DAGGER_DIRECTORY=str(tmp_path),SHARPA_DAGGER_RUN_ID=run_id,
                          SHARPA_DAGGER_SOCKET=str(sock),SHARPA_GELLO_REPO=str(tmp_path)).items():
        monkeypatch.setenv(key,value)
    (tmp_path/'dagger_manifest.json').write_text('{"model_version":"simulated-weights"}')
    class Message:
        pass
    for module,fields in [('sensor_msgs.msg',dict(JointState=Message)),('std_msgs.msg',dict(String=Message)),
                          ('teleop_interfaces.msg',dict(ArmCommandStatus=Message))]:
        monkeypatch.setitem(sys.modules,module,SimpleNamespace(**fields))
    mapping=dict(timeout_s=.2,sensitivity=[1]*7,max_delta=[1.5]*7,margin=[0]*7,arm_speed=.8)
    monkeypatch.setattr(owner,'configuration',lambda _:mapping)
    source=dict(arm=0.,hand=0.)
    class Inputs:
        def __init__(self,*args): pass
        def receive(self,*args): pass
        def snapshot(self):
            return dict(mapping,arm=np.full(7,source['arm']),hand=np.full(22,source['hand']),
                        arm_time=time.monotonic(),hand_time=time.monotonic(),session='stable-input')
    monkeypatch.setattr(owner,'InputCache',Inputs)
    class Limiter:
        def __init__(self,*args):self.lower=np.full(58,-2.);self.upper=np.full(58,2.)
        def step_toward(self,previous,target):
            step=np.full(58,.03)
            step[np.r_[7:29,36:58]]=.249
            return previous+np.clip(target-previous,-step,step)
        def project(self,actions):
            return np.clip(actions,self.lower,self.upper)
    monkeypatch.setattr(owner,'RTCCommandLimiter',Limiter)
    monkeypatch.setattr(owner,'validate_action_ranges',lambda q,**kw: None if np.isfinite(q).all() else (_ for _ in ()).throw(ValueError('nonfinite')))
    monkeypatch.setattr(policy_execution,'validate_action_ranges',owner.validate_action_ranges)
    def preflight(q,feedback,**kwargs):
        delta=np.abs(q[0]-feedback.state)
        groups=kwargs.get('groups',('left_fr3','right_fr3','left_sharpa','right_sharpa'))
        slices={'left_fr3':slice(0,7),'right_fr3':slice(29,36),
                'left_sharpa':slice(7,29),'right_sharpa':slice(36,58)}
        for name in groups:
            np.testing.assert_array_less(delta[slices[name]],.05 if 'fr3' in name else .15)
    monkeypatch.setattr(owner,'execute_preflight',preflight)
    first_rtc_phases=[]
    class Timeline(owner.RTCTimeline):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.first_sample=True
        def sample(self,now):
            if self.first_sample:
                first_rtc_phases.append(self.phase(now))
                self.first_sample=False
            return super().sample(now)
    monkeypatch.setattr(owner,'RTCTimeline',Timeline)
    class Node:
        def __init__(self):self.callbacks={};self.messages=[]
        def create_publisher(self,kind,topic,depth):
            return SimpleNamespace(publish=lambda msg:self.messages.append((topic,json.loads(msg.data))))
        def create_subscription(self,kind,topic,callback,depth):
            self.callbacks[topic]=callback;return topic
        def destroy_subscription(self,*args):pass
        def get_publishers_info_by_topic(self,*args):return [object()]
        def get_clock(self):return SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=time.time_ns()))
    node=Node()
    class SDK:
        def __init__(self):self.state=np.zeros(58,dtype=np.float32);self.sends=[];self.resume_count=0
        def send(self,left,right):
            self.state[7:29]=left;self.state[36:]=right;self.sends.append((left.copy(),right.copy()))
        def resume_stream(self):self.resume_count+=1
    sdk=SDK()
    feedback_offset=[0.]
    held_hand_offset=[0.]
    def feedback_snapshot():
        measured=sdk.state.copy()
        measured[35]+=feedback_offset[0]
        measured[36]+=held_hand_offset[0]
        return SimpleNamespace(state=measured,validate_freshness=lambda **kw:None)
    feedback=SimpleNamespace(snapshot=feedback_snapshot)
    sequence=[0]
    dispatched=[]
    def publish(command):
        seq=sequence[0];sequence[0]+=1
        sdk.state[:7]+=.15*(command[:7]-sdk.state[:7])
        sdk.state[7:]=command[7:]
        dispatched.append(command.copy())
        node.callbacks['/teleop/arm_command_status'](SimpleNamespace(
            session_id='same-sdk-session',source='openpi',sequence=seq,accepted_sides=['left','right'],faults=[]))
        return seq,time.time_ns()
    read_fd,write_fd=os.pipe()
    reader=os.fdopen(read_fd,'r');writer=os.fdopen(write_fd,'w',buffering=1)
    monkeypatch.setattr(owner,'sys',SimpleNamespace(stdin=reader))
    protocol=Protocol();errors=[];started=[]
    def work():
        try:
            owner.run_owner(node,sdk,feedback,SimpleNamespace(start=lambda:started.append(True)),publish,protocol,
                gateway_speed=1.,initial_request=dict(broker_mode=broker_mode,execute_steps=15,
                    speed_scale=1.,approach_speed_rad_s=.3,
                    online_task_handoff=staged),session_id='same-sdk-session')
        except BaseException as error:
            errors.append(error)
    thread=threading.Thread(target=work,daemon=True);thread.start()
    def request(data):
        writer.write(json.dumps(data)+'\n');writer.flush()
        try:
            reply=protocol.lines.get(timeout=3)
        except queue.Empty:
            if errors:raise errors[0]
            raise
        if errors:raise errors[0]
        assert reply['status']=='PASS'
        return reply
    def wait_hold():
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            status=rpc(sock,run_id,renew_contact=True)
            if status['mode']=='HOLD':return status
            if errors:raise errors[0]
            time.sleep(.04)
        raise AssertionError('Measured hold was not confirmed')
    def install(generation,receipt=None):
        actions=np.zeros((50,58))
        actions[:,0]=bridge_distance
        return request(dict(operation='dagger_install',generation=generation,
            actions=actions.tolist(),captured_monotonic_ns=time.monotonic_ns(),
            receipt=receipt,prediction_clip_rad=.05,prompt='pick this tomato',stage='harvest_1'))
    def wait_policy_frame():
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            status=request(dict(operation='dagger_status'))
            if status['dagger']['mode']=='POLICY' and status.get('consumed_since_install',0)>0:
                return status
        raise AssertionError('Policy playback did not start after the bridge')
    try:
        initial=protocol.lines.get(timeout=3)
        first=request(dict(operation='dagger_request_policy',generation=initial['dagger']['generation']))
        generation=first['dagger']['generation']
        # Keep all fake hands in legal native bounds by replacing only prediction
        # clipping, which is tested by the existing SDK/prediction suites.
        monkeypatch.setattr(policy_execution,'clip_hand_predictions',lambda actions,**kw:(np.asarray(actions,dtype=np.float32),{}))
        assert not install(generation).get('rejected')
        interventions=1
        if staged and bridge_distance:
            # Revoke an unfinished entry bridge before any RTC policy frame.
            rpc(sock,run_id,'takeover',request_id='bridge-takeover',expected_generation=generation)
            hold=wait_hold()
            resumed=rpc(sock,run_id,'resume',request_id='bridge-resume',expected_generation=hold['generation'])
            generation=resumed['generation']
            assert not install(generation).get('rejected')
            interventions+=1
        wait_policy_frame()
        # Ordinary automatic playback allows tracking lag past the acquisition
        # gate. DAgger must use that same policy while no human is following.
        feedback_offset[0]=.06
        for _ in range(3):
            assert request(dict(operation='dagger_status'))['dagger']['mode']=='POLICY'
        feedback_offset[0]=0.
        receipt=None
        if broker_mode=='rtc':
            from openpi_client import rtc
            context=request(dict(operation='dagger_snapshot',generation=generation,
                delay_steps=1,schedule='exp',weight=5))['context']
            receipt=rtc.make_receipt(context)
            assert not install(generation,receipt).get('rejected')
            context=request(dict(operation='dagger_snapshot',generation=generation,
                delay_steps=1,schedule='exp',weight=5))['context']
            receipt=rtc.make_receipt(context)
        feedback_offset[0]=.06
        held_hand_offset[0]=.23
        takeover=rpc(sock,run_id,'takeover',request_id='takeover',expected_generation=generation)
        assert takeover['mode']=='HOLDING'
        feedback_offset[0]=0.
        rejected=install(generation,receipt)
        assert rejected['rejected']
        hold=wait_hold()
        assert hold['intervention_id']==interventions
        source['hand']=.8
        rpc(sock,run_id,'follow',request_id='follow',expected_generation=hold['generation'])
        source['arm']=.5
        for _ in range(4):request(dict(operation='dagger_status'))
        assert np.max(sdk.state[:29])>0 and np.max(np.abs(sdk.state[29:]))==0, [(t,v.get('mode'),v.get('phase'),v.get('reason'),v.get('action',[None])[0]) for t,v in node.messages[-12:]]
        rpc(sock,run_id,'hold',request_id='hold',expected_generation=hold['generation'])
        hold=wait_hold()
        source['hand']=-.4
        rpc(sock,run_id,'follow',request_id='follow-again',expected_generation=hold['generation'])
        for _ in range(4):request(dict(operation='dagger_status'))
        rpc(sock,run_id,'hold',request_id='hold-again',expected_generation=hold['generation'])
        hold=wait_hold()
        held_hand_offset[0]=0.
        resumed=rpc(sock,run_id,'resume',request_id='resume',expected_generation=hold['generation'])
        assert resumed['mode']=='RESUMING'
        assert not install(resumed['generation']).get('rejected')
        status=request(dict(operation='dagger_status'))
        assert status['dagger']['mode']=='POLICY'
        wait_policy_frame()
        writer.write('{"operation":"close"}\n');writer.flush();thread.join(timeout=3)
        assert not thread.is_alive() and not errors
        assert started==[True] and sdk.resume_count==1  # No reconnect/restart on any transition.
        assert first_rtc_phases==([0.,0.] if broker_mode=='rtc' else [])
        frames=[v for topic,v in node.messages if topic=='/dagger/frame']
        human=[f for f in frames if f['mode']=='HUMAN_LEFT']
        assert human and all(f['action_owner']==['human','human','hold','hold'] for f in human)
        assert len([f for f in human if f['action'][7]>0])>1
        for previous,frame in zip(frames,frames[1:]):
            if frame['mode']=='HUMAN_LEFT':
                arm_delta=np.abs(np.asarray(frame['action'][:7])-frame['state'][:7])
                assert np.max(arm_delta)<.05
                hand_delta=np.abs(np.asarray(frame['action'][7:29])-previous['action'][7:29])
                assert np.max(hand_delta)<=owner.HUMAN_HAND_SPEED_RAD_S/30+1e-6
        events=[v for topic,v in node.messages if topic=='/dagger/events']
        assert len([e for e in events if e.get('event')=='human_start'])==interventions
        ends=[e for e in events if e.get('event')=='human_end']
        assert len(ends)==interventions and all(e['reason']=='policy restored' for e in ends)
        assert (tmp_path/'dagger/frames.jsonl').is_file()
    finally:
        if thread.is_alive():
            writer.write('{"operation":"close"}\n');writer.flush();thread.join(timeout=3)
        writer.close();reader.close()
