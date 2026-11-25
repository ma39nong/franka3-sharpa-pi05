"""Exercise the real inference coordinator with delayed, obsolete predictions."""
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime import dagger_execution as engine
from deploy.fr3_sharpa_runtime import run, startup_gate
from deploy.fr3_sharpa_control import controller as control


@pytest.mark.parametrize('profile',['none','tomato-in-hand119','tomato-b'])
def test_takeover_revokes_inflight_prediction_and_resume_uses_fresh_observation(tmp_path,monkeypatch,profile):
    (tmp_path/'overlay').mkdir();(tmp_path/'overlay/setup.bash').touch()
    (tmp_path/'config/modes').mkdir(parents=True);(tmp_path/'config/modes/gello.yaml').write_text('test config')
    for key,value in dict(SHARPA_DAGGER_DIRECTORY=str(tmp_path),SHARPA_TASK_SOCKET=str(tmp_path/'task.sock'),
                          SHARPA_TASK_RUN_ID='task-run',SHARPA_RECORD_SOCKET=str(tmp_path/'record.sock'),
                          SHARPA_DAGGER_SOCKET=str(tmp_path/'dagger.sock'),SHARPA_DAGGER_RUN_ID='dagger-run').items():monkeypatch.setenv(key,value)
    delayed=threading.Event();release=threading.Event();task_holder=[];inferences=[];installed=[];requests=[]
    class Inputs:
        pid=12345
        def poll(self):return None
        def wait(self,**kw):return 0
    monkeypatch.setattr(engine.subprocess,'Popen',lambda *a,**kw:Inputs())
    monkeypatch.setattr(engine.os,'killpg',lambda *a:None)
    monkeypatch.setattr(run,'GELLO_REPO',tmp_path)
    monkeypatch.setattr(run,'ROS_PROJECT',tmp_path)
    monkeypatch.setattr(run,'_start_observation_worker',lambda *a:None)
    monkeypatch.setattr(run,'live_observation',lambda overlay,prompt,**kw:dict(
        _captured_monotonic_ns=time.monotonic_ns(),**{'observation/state':np.zeros(58)},prompt=prompt))
    monkeypatch.setattr(startup_gate,'wait_for_cameras',lambda:None)
    monkeypatch.setattr(engine,'require_capabilities',lambda *a:None)
    class Server:
        def __init__(self,path,task):task_holder.append(task)
        def close(self):pass
    monkeypatch.setattr(engine,'TaskServer',Server)
    monkeypatch.setattr(engine,'TaskLease',lambda *a:SimpleNamespace(start=lambda:None,close=lambda:None))
    class Runtime:
        last_receipt=None
        def __init__(self,*a,**kw):pass
        def inspect_metadata(self,**kw):return {'checkpoint':'simulated'}
        def infer_chunk(self,observation,**kw):
            number=len(inferences)+1
            inferences.append(dict(number=number,prompt=observation['prompt'],context=kw.get('rtc_context')))
            if number==2:
                delayed.set()
                assert release.wait(3)
            self.last_receipt={'fake':'obsolete receipt'} if kw.get('rtc_context') else None
            return {'checkpoint':'simulated'},np.full((50,58),number*.001),None
    monkeypatch.setattr(engine,'SharpaRuntime',Runtime)
    class Worker:
        def __init__(self,*a,**kw):
            self.mode='IDLE';self.generation=0;self.takeover_ticks=0;self.shutdown_report={'status':'PASS'}
        def request(self,request,**kw):
            requests.append(dict(request))
            operation=request['operation']
            if operation=='ready':return {'status':'PASS'}
            if operation=='dagger_start':return {'dagger':dict(mode='IDLE',generation=0)}
            if operation=='dagger_request_policy':
                self.generation+=1;self.mode='RESUMING';return {'status':'PASS'}
            if operation=='dagger_install':
                assert request['generation']==self.generation
                installed.append(dict(request));self.mode='POLICY';return {'skipped':0}
            if operation=='dagger_snapshot':return {'context':{'previous_actions':[[0.]*58]*25,'prefix_attention_horizon':25}}
            assert operation=='dagger_status'
            if task_holder and task_holder[0].data['execution_state']=='READY':
                task=task_holder[0];task.command('start','start',expected_version=task.data['stage_version'])
            if delayed.is_set() and self.mode=='POLICY' and len(installed)==1:
                self.mode='HOLDING';self.generation+=1
            if self.mode in ('HOLDING','HOLD'):
                self.takeover_ticks+=1
                if self.takeover_ticks==2:self.mode='HOLD'
                if self.takeover_ticks>=4:
                    self.mode='RESUMING';self.generation+=1;release.set()
            return dict(dagger=dict(mode=self.mode,generation=self.generation,
                intervention_id=1 if self.mode in ('HOLDING','HOLD','RESUMING') and delayed.is_set() else None),
                policy_done=False,consumed_since_install=15,phase=15)
        def close(self):self.mode='STOPPED';release.set()
    monkeypatch.setattr(engine,'MotionWorker',Worker)
    args=SimpleNamespace(execution_trace_dir=tmp_path/'logs',task_mode=profile,
        initial_stage='pickup' if profile=='tomato-in-hand119' else 'pickup_1',ros_overlay=tmp_path/'overlay',
        initialization='current',policy_url='ws://fake',broker_mode='rtc',rtc_initial_delay_steps=5,
        rtc_delay_buffer_size=3,rtc_prefix_attention_schedule='exp',rtc_max_guidance_weight=5.,
        execute_steps=15,execute_speed_scale=1.,prepare_speed_rad_s=.03,prediction_clip_rad=.05,
        num_chunks=2,prompt='same tomato',train_rtc=False,paper_rtc=False)
    result=engine.execute_dagger(args)
    assert result['accepted_chunks']==2 and len(inferences)==3
    assert inferences[1]['context'] is not None and inferences[2]['context'] is None
    assert installed[0]['generation']<installed[1]['generation']
    assert [round(r['actions'][0][0],3) for r in installed]==[.001,.003]  # .002 is never dispatched.
    assert installed[0]['prompt']==installed[1]['prompt']
    assert len([r for r in requests if r['operation']=='dagger_start'])==1
    assert all(r['operation'] not in ('close','prepare') for r in requests)
    if profile!='none':
        assert task_holder[0].data['active_stage']==args.initial_stage
        assert task_holder[0].data['stage_version']>0


def test_controller_rejects_stale_browser_run_before_any_control_change(tmp_path):
    controller=control.DeploymentController(tmp_path)
    controller.dagger_endpoint=tmp_path/'socket';controller.dagger_run_id='new-run'
    controller.state.update(running=True);controller.operation='execute'
    with pytest.raises(ValueError,match='身份已变化'):
        controller.dagger_command(dict(run_id='old-run',command='takeover',expected_generation=1,request_id='id'))


def test_missing_recording_identity_is_rejected_before_any_subprocess(monkeypatch):
    monkeypatch.delenv('SHARPA_RECORD_SOCKET',raising=False)
    monkeypatch.setattr(engine.subprocess,'Popen',lambda *a,**kw: pytest.fail('must reject before launch'))
    with pytest.raises(ValueError,match='独立运行身份'):
        engine.execute_dagger(SimpleNamespace())
