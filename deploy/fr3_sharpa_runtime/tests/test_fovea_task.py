"""No hardware: real five-stage IPC/Agent gate and serial action/box ownership."""
from contextlib import contextmanager
from types import SimpleNamespace
import sys
import time
import uuid

import numpy as np
import pytest

from deploy.fr3_sharpa_control.agent import AgentSupervisor, InHandGate, LABELS
from deploy.fr3_sharpa_runtime.fovea_runtime import FoveaRuntime
from deploy.fr3_sharpa_runtime.fovea_task import FoveaTask
from deploy.fr3_sharpa_runtime.motion_worker import MotionWorker
from deploy.fr3_sharpa_runtime.task_control import TaskControl, TaskServer
from deploy.fr3_sharpa_runtime.task_stages import IN_HAND_AGENT_SOURCE, IN_HAND_STAGES
from openpi.subtasks import fovea


def command(task, op, **options):
    return task.command(op, uuid.uuid4().hex, expected_version=task.snapshot()['stage_version'], **options)


class Runtime(FoveaRuntime):
    def __init__(self):
        super().__init__('ws://localhost:8010')
        self.calls = []
        self.during_inference = None

    def infer_chunk(self, obs, **kwargs):
        self.calls.append((obs['prompt'], self.query_index, self.box.copy(), self.session))
        if self.during_inference:
            callback, self.during_inference = self.during_inference, None
            callback()
        result = {'actions': np.zeros((50,58),np.float32), 'foveation': {
            'session_id': self.session, 'query_index': self.query_index, 'used_box': self.box.tolist(),
            'next_box': [.3,.3,.7,.8], 'prediction_valid': True, 'execute_steps': 15, 'source_size': [640,400]}}
        return self.accept_result({'checkpoint':'simulation','norm_stats':'simulation'}, result)


class Backend:
    def __init__(self):
        self.dispatched = []
        self.during_motion = None
        self.partial = False
        self.stopped = False

    def execute(self, actions, *, admission, on_poll):
        on_poll()
        with admission():
            self.dispatched.append(actions.copy())
        if self.during_motion:
            callback, self.during_motion = self.during_motion, None
            callback()
        return {'status':'PASS','execute_steps':15,'policy_frames':14 if self.partial else 15,
                'completed_ros_ns':time.time_ns(),'completed_monotonic_ns':time.monotonic_ns()}

    def stop(self):
        self.stopped = True
        return {'status':'PASS','hands_disabled':True,'arms_safe':True,'old_stream_invalidated':True,
                'simulation_only':True}


def setup(tmp_path):
    task = TaskControl('fovea-test','pickup',profile='tomato-in-hand119')
    runtime, backend = Runtime(), Backend()
    def observe(**options):
        options['on_poll']()
        return {'observation/state':np.zeros(58,np.float32), '_captured_monotonic_ns':time.monotonic_ns(),
                '_topic_stamp_ns':{str(i):time.time_ns() for i in range(7)}}
    args = SimpleNamespace(execute_steps=15,execute_speed_scale=1.0,num_chunks=0)
    engine = FoveaTask(task,runtime,backend,observe,args,tmp_path)
    task.ready();command(task,'start')
    return task,runtime,backend,engine


def test_same_stage_commits_after_15_and_stage_switch_resets_session(tmp_path):
    task,runtime,backend,engine = setup(tmp_path)
    engine.tick()
    assert task.snapshot()['active_stage']=='pickup' and runtime.query_index==1
    np.testing.assert_array_equal(runtime.box,[.3,.3,.7,.8])
    old_session = runtime.session
    backend.during_motion = lambda:command(task,'stage',target='harvest_1',source=IN_HAND_AGENT_SOURCE)
    engine.tick()
    assert len(backend.dispatched)==2 and runtime.query_index==2
    assert task.snapshot()['active_stage']=='pickup'
    assert task.snapshot()['execution_state']=='SWITCH_REQUESTED'
    engine.tick()
    assert runtime.calls[-1][0]==IN_HAND_STAGES[1].prompt
    assert runtime.calls[-1][1]==0 and runtime.session!=old_session
    np.testing.assert_array_equal(runtime.calls[-1][2],fovea.INITIAL_BOX)
    assert task.snapshot()['active_stage']=='harvest_1'
    assert not backend.stopped


def test_superseded_inference_never_dispatches_or_commits_box(tmp_path):
    task,runtime,backend,engine = setup(tmp_path)
    engine.tick()
    runtime.during_inference = lambda:command(task,'next')
    engine.tick()
    assert len(backend.dispatched)==1 and runtime.query_index==1 and runtime.pending is None
    engine.tick()
    assert task.snapshot()['active_stage']=='harvest_1' and runtime.calls[-1][1]==0


def test_stage_change_at_dispatch_race_is_rejected_atomically(tmp_path):
    task,runtime,backend,engine = setup(tmp_path)
    engine.tick()
    def execute(actions, *, admission, on_poll):
        command(task,'next')
        with admission():
            pytest.fail('superseded actions gained dispatch authority')
    backend.execute = execute
    engine.tick()
    assert len(backend.dispatched)==1 and runtime.query_index==1


def test_partial_execution_and_expired_observation_never_commit(tmp_path):
    task,runtime,backend,engine = setup(tmp_path)
    backend.partial = True
    with pytest.raises(ValueError,match='complete fifteen-step'):
        engine.tick()
    assert runtime.query_index==0
    backend.partial = False
    runtime.during_inference = lambda:time.sleep(.21)
    with pytest.raises(TimeoutError,match='expired'):
        engine.tick()
    assert len(backend.dispatched)==1 and runtime.query_index==0


def test_real_agent_gate_and_task_socket_advance_all_five_stages(tmp_path):
    task,runtime,backend,engine = setup(tmp_path)
    server = TaskServer(tmp_path/'task.sock',task)
    owner = SimpleNamespace(root=tmp_path,task_endpoint=server.path,task_run_id=task.run_id)
    agent = AgentSupervisor(owner)
    agent.settings.update(provider='tairan',profile='workstation')
    agent.state.update(status='ready',received_unix=time.time())
    agent.gate = InHandGate()
    agent._task = task.snapshot
    try:
        engine.tick()
        agent.set_mode('auto')
        for following in IN_HAND_STAGES[1:]:
            now=time.time()
            p=[.99 if label==following.action else .01/3 for label in LABELS]
            packet=dict(protocol='tomato-stage-prediction-v1',model_version='test',control_eligible=False,
                        labels=list(LABELS),future_horizon_s=1.0,input_timestamp=now,valid=True,
                        current_probabilities=p,future_probabilities=p,latency_ms=1,roi_fallback_frames=0)
            meta=dict(timestamp=now,captured_unix=now,ros_now=now,observation_id=uuid.uuid4().hex,source_age_s=0)
            previous = task.snapshot()['active_stage']
            agent._display_inhand(packet,meta,'test',agent.epoch)
            assert task.snapshot()['requested_stage']==following.id
            assert task.snapshot()['active_stage']==previous
            engine.tick()
            assert task.snapshot()['active_stage']==following.id
            assert runtime.calls[-1][0]==following.prompt and runtime.calls[-1][1]==0
        assert len(backend.dispatched)==5 and not backend.stopped
        assert task.snapshot()['execution_state']=='RUNNING'  # Final completion stays manual.
        command(task,'complete');assert not engine.tick()
        assert task.snapshot()['execution_state']=='COMPLETED'
    finally:
        server.close()


def test_stop_during_prediction_prevents_dispatch_and_pause_cannot_auto_resume(tmp_path):
    task,runtime,backend,engine=setup(tmp_path)
    runtime.during_inference=lambda:command(task,'stop')
    with pytest.raises(InterruptedError):engine.tick()
    assert not backend.dispatched
    assert not engine.tick() and backend.stopped


def test_motion_worker_admission_prevents_real_pipe_dispatch(tmp_path):
    source='import sys,json\nfor line in sys.stdin: print(json.dumps(dict(status="PASS",echo=json.loads(line))),flush=True)'
    worker=MotionWorker([sys.executable,'-u','-c',source],cwd=tmp_path)
    @contextmanager
    def blocked():
        raise RuntimeError('revoked')
        yield
    try:
        with pytest.raises(RuntimeError,match='revoked'):worker.request({'id':'old'},admission=blocked)
        assert worker.request({'id':'new'},timeout=2)['echo']=={'id':'new'}
    finally:
        worker.close()


def test_pause_without_arm_stop_proof_never_grants_resume(tmp_path):
    from deploy.fr3_sharpa_runtime.fovea_task import FoveaTaskBackend
    task,runtime,backend,engine=setup(tmp_path)
    engine.tick()
    worker=SimpleNamespace(close=lambda:None,shutdown_report={'status':'PASS'})
    engine.backend=FoveaTaskBackend(worker,engine.args)
    command(task,'pause')
    with pytest.raises(RuntimeError,match='禁止自动恢复'):
        engine.tick()
    assert not task.snapshot()['safe_to_resume']
    with pytest.raises(ValueError):command(task,'resume',confirmed=True)


def test_new_stage_requires_post_completion_camera_stamps(tmp_path):
    task,runtime,backend,engine=setup(tmp_path)
    engine.tick()
    fresh=engine.observe
    count=0
    def observe(**options):
        nonlocal count
        count+=1
        obs=fresh(**options)
        if count==1:obs['_topic_stamp_ns']={str(i):1 for i in range(7)}
        return obs
    engine.observe=observe
    command(task,'next')
    engine.tick()
    assert count==2 and len(backend.dispatched)==2


def test_owner_loop_waits_for_start_and_cleans_up_without_another_chunk(tmp_path):
    import threading
    task,runtime,backend,engine=setup(tmp_path)
    # New owner entry starts from IDLE; unlike a test tick, run owns ready().
    task=TaskControl('loop','pickup',profile='tomato-in-hand119')
    engine.control=task
    backend.during_motion=lambda:command(task,'stop')
    errors=[]
    def run():
        try:engine.run()
        except BaseException as error:errors.append(error)
    thread=threading.Thread(target=run)
    thread.start()
    try:
        deadline=time.monotonic()+2
        while task.snapshot()['execution_state']!='READY' and time.monotonic()<deadline:
            time.sleep(.005)
        assert not backend.dispatched
        command(task,'start')
        thread.join(timeout=3)
        assert not thread.is_alive() and not errors
        assert len(backend.dispatched)==1 and backend.stopped
        assert task.snapshot()['execution_state']=='STOPPED'
        assert (tmp_path/'task-run.json').is_file()
    finally:
        if thread.is_alive():
            command(task,'stop');thread.join(timeout=2)
