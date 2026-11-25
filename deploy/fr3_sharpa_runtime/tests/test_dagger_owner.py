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


@pytest.mark.parametrize('broker_mode',['serial','rtc'])
def test_live_loop_takeover_old_reply_rejection_left_only_resume_and_labels(tmp_path,monkeypatch,broker_mode):
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
            return previous+np.clip(target-previous,-.03,.03)
    monkeypatch.setattr(owner,'RTCCommandLimiter',Limiter)
    monkeypatch.setattr(owner,'validate_action_ranges',lambda q,**kw: None if np.isfinite(q).all() else (_ for _ in ()).throw(ValueError('nonfinite')))
    monkeypatch.setattr(owner,'execute_preflight',lambda q,feedback,**kw: np.testing.assert_array_less(np.abs(q[0]-feedback.state),.151))
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
    def feedback_snapshot():
        return SimpleNamespace(state=sdk.state.copy(),validate_freshness=lambda **kw:None)
    feedback=SimpleNamespace(snapshot=feedback_snapshot)
    sequence=[0]
    dispatched=[]
    def publish(command):
        seq=sequence[0];sequence[0]+=1
        sdk.state[:]=command;dispatched.append(command.copy())
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
                    speed_scale=1.,approach_speed_rad_s=.03),session_id='same-sdk-session')
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
        return request(dict(operation='dagger_install',generation=generation,
            actions=np.zeros((50,58)).tolist(),captured_monotonic_ns=time.monotonic_ns(),
            receipt=receipt,prediction_clip_rad=.05,prompt='pick this tomato',stage='harvest_1'))
    try:
        initial=protocol.lines.get(timeout=3)
        first=request(dict(operation='dagger_request_policy',generation=initial['dagger']['generation']))
        generation=first['dagger']['generation']
        # Keep all fake hands in legal native bounds by replacing only prediction
        # clipping, which is tested by the existing SDK/prediction suites.
        monkeypatch.setattr(owner,'clip_hand_predictions',lambda actions,**kw:(np.asarray(actions,dtype=np.float32),{}))
        assert not install(generation).get('rejected')
        receipt=None
        if broker_mode=='rtc':
            from openpi_client import rtc
            context=request(dict(operation='dagger_snapshot',generation=generation,
                delay_steps=1,schedule='exp',weight=5))['context']
            receipt=rtc.make_receipt(context)
        rpc(sock,run_id,'takeover',request_id='takeover',expected_generation=generation)
        rejected=install(generation,receipt)
        assert rejected['rejected']
        hold=wait_hold()
        assert hold['intervention_id']==1
        rpc(sock,run_id,'follow',request_id='follow',expected_generation=hold['generation'])
        source.update(arm=.1,hand=.1)
        for _ in range(4):request(dict(operation='dagger_status'))
        assert np.max(sdk.state[:29])>0 and np.max(np.abs(sdk.state[29:]))==0, [(t,v.get('mode'),v.get('phase'),v.get('reason'),v.get('action',[None])[0]) for t,v in node.messages[-12:]]
        rpc(sock,run_id,'hold',request_id='hold',expected_generation=hold['generation'])
        hold=wait_hold()
        resumed=rpc(sock,run_id,'resume',request_id='resume',expected_generation=hold['generation'])
        assert resumed['mode']=='RESUMING'
        assert not install(resumed['generation']).get('rejected')
        status=request(dict(operation='dagger_status'))
        assert status['dagger']['mode']=='POLICY'
        writer.write('{"operation":"close"}\n');writer.flush();thread.join(timeout=3)
        assert not thread.is_alive() and not errors
        assert started==[True] and sdk.resume_count==1  # No reconnect/restart on any transition.
        frames=[v for topic,v in node.messages if topic=='/dagger/frame']
        human=[f for f in frames if f['mode']=='HUMAN_LEFT']
        assert human and all(f['action_owner']==['human','human','hold','hold'] for f in human)
        events=[v for topic,v in node.messages if topic=='/dagger/events']
        assert len([e for e in events if e.get('event')=='human_start'])==1
        ends=[e for e in events if e.get('event')=='human_end']
        assert len(ends)==1 and ends[0]['reason']=='policy restored'
        assert (tmp_path/'dagger/frames.jsonl').is_file()
    finally:
        if thread.is_alive():
            writer.write('{"operation":"close"}\n');writer.flush();thread.join(timeout=3)
        writer.close();reader.close()
