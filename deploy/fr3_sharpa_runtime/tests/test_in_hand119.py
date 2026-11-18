"""New five-stage authority and verbatim training contract; old seven stages stay frozen."""
import ast
import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.task_stages import IN_HAND_PROMPTS, catalog, phase_ids, stage, PHASES
from deploy.fr3_sharpa_runtime.task_control import TaskControl, TaskServer, task_rpc
from deploy.fr3_sharpa_runtime.in_hand_crop import InHandInteractionSelector, InHandCropWindow
from deploy.fr3_sharpa_runtime.interaction_crop import InteractionCropper
from deploy.fr3_sharpa_control.agent import AgentSupervisor
from deploy.fr3_sharpa_control.plan import DeploymentSettings, inspect_checkpoint

PROFILE = 'tomato-in-hand119'
IDS = ('pickup', 'harvest_1', 'adjust_in_hand', 'harvest_2', 'return_stem')


def command(task, op, **kw):
    return task.command(op, uuid.uuid4().hex, expected_version=task.snapshot()['stage_version'], **kw)


def test_five_stages_match_training_source():
    assert phase_ids(PROFILE) == IDS
    assert len(PHASES) == 7
    assert stage('adjust').prompt == 'Right fingers reposition the truss on the right plate'
    assert stage('adjust_in_hand', PROFILE).prompt == IN_HAND_PROMPTS['adjust']
    assert [s['action'] for s in catalog(PROFILE)] == ['pickup','harvest','adjust','harvest','return']
    source = Path('/home/user/tomato-openpi/src/openpi/subtasks/tomato_in_hand.py')
    if source.exists():
        definitions = {n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(source.read_text()).body
                       if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
                       and n.targets[0].id in ('ACTION_PROMPTS','PHASES')}
        assert definitions == {'ACTION_PROMPTS':IN_HAND_PROMPTS,'PHASES':IDS}


def test_real_ipc_five_stage_sequence_no_premature_activation(tmp_path):
    task = TaskControl('new119', 'pickup', profile=PROFILE)
    server = TaskServer(tmp_path/'task.sock', task)
    try:
        task.ready()
        command(task,'start')
        first = task.ticket(); task.activate(first,'first')
        assert first['prompt'] == IN_HAND_PROMPTS['pickup']
        for old, target in zip(IDS, IDS[1:]):
            result=task_rpc(server.path, task.run_id, 'next', request_id=uuid.uuid4().hex,
                            expected_version=task.snapshot()['stage_version'])
            assert result['requested_stage']==target and result['active_stage']==old
            assert result['task_profile']==PROFILE and len(result['stages'])==5
            ticket=task.ticket()
            assert ticket['prompt']==stage(target,PROFILE).prompt
            task.transition('SWITCHING','simulated execution owner')
            task.activate(ticket, target)
        with pytest.raises(ValueError,match='最后阶段'):
            command(task,'next')
        assert task.snapshot()['execution_state']=='RUNNING'
        command(task,'complete')
        assert task.snapshot()['execution_state']=='STOPPING'
    finally:
        server.close()


def test_new_task_rejects_old_stage_and_agent_even_direct_ipc(tmp_path):
    task=TaskControl('119','pickup',profile=PROFILE);task.ready()
    with pytest.raises(ValueError,match='未知任务阶段'):
        command(task,'stage',target='return_truss')
    with pytest.raises(ValueError,match='旧Agent'):
        command(task,'stage',target='harvest_1',source='agent')
    assert task.snapshot()['requested_stage']=='pickup'
    owner=SimpleNamespace(root=tmp_path)
    supervisor=AgentSupervisor(owner)
    supervisor.settings['profile']='workstation'
    supervisor.state.update(status='ready',received_unix=time.time())
    supervisor._task=lambda:task.snapshot()
    with pytest.raises(ValueError,match='119条'):
        supervisor.set_mode('auto')
    assert supervisor.mode=='manual'


def test_pause_and_explicit_recovery_keep_new_prompt():
    task=TaskControl(initial_stage='pickup',profile=PROFILE);task.ready();command(task,'start')
    ticket=task.ticket();task.activate(ticket,'first');command(task,'pause')
    task.transition('PAUSED','simulation only')
    with pytest.raises(ValueError,match='明确确认'):
        command(task,'recover',target='adjust_in_hand')
    command(task,'recover',target='adjust_in_hand',confirmed=True)
    assert task.ticket()['prompt']==IN_HAND_PROMPTS['adjust']
    assert not task.valid(ticket)


def det(role,box):
    return dict(label=role,score=.99,box=box)


def test_roi_matches_offline_current_extent_rule_and_ignores_left_transport():
    import importlib.util
    path=Path('/home/user/tomato-openpi/scripts/tomato_full_block_roi.py')
    rgb=np.zeros((400,640,3),np.uint8);rgb[135:150,330:345]=[255,0,0]
    detections=[det(1,[300,150,350,210]),det(2,[20,220,75,270])]
    selector=InHandInteractionSelector();q=np.ones(58)*2
    a=selector.select(detections,q,640,400,rgb)
    b=selector.select(detections[:1],q,640,400,rgb)
    assert a==b and a['roles']==[1]
    if path.exists():
        import sys
        original=list(sys.path);sys.path.insert(0,str(path.parents[1]))
        try:
            spec=importlib.util.spec_from_file_location('reference_full_block_roi',path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            assert np.array_equal(a['interaction_box'],module.interaction(rgb,detections)['box'])
        finally:
            sys.path[:]=original
    assert not selector.select(detections[1:],q,640,400,rgb)['valid']


def test_roi_window_bounds_containment_and_current_full_fallback():
    window=InHandCropWindow()
    for i,b in enumerate([[290,132,365,228],[320,140,420,240],[100,50,350,300]]):
        box=np.array(window.select(b,640,400,(i+1)*500_000_000))
        assert 180<=box[2]-box[0]<=300
        assert np.all(box[:2]<=b[:2]) and np.all(box[2:]>=b[2:])
    assert window.select([0,0,400,350],640,400,3_000_000_000) is None
    class Locator:
        def predict(self,images):return [[] for _ in images]
    cropper=InteractionCropper(Locator(),np.ones(58),roi_profile='in_hand119_right_truss_v1')
    image=np.full((400,640,3),170,np.uint8)
    result=cropper.apply({'observation/image':image,'observation/state':np.zeros(58),
                         '_crop_context':{'source_monotonic_ns':1_000_000_000,'session':'test'}})
    assert result['observation/interaction_image'].shape==(224,224,3)
    assert cropper.last['mode']=='current_full_fallback' and cropper.last['image_available']
    assert not cropper.last['uses_future_frames'] and cropper.last['roi_profile']=='in_hand119_right_truss_v1'


@pytest.mark.parametrize('step',[10000,15000,20000])
def test_real_checkpoint_registered_and_wrong_task_rejected(step,tmp_path):
    root=Path('/media/user/8C0F29AFFD6CD874/新的权重')/str(step)
    if not root.exists():pytest.skip('external checkpoint not mounted')
    metadata=inspect_checkpoint(root,model_type='full-delta')
    assert len(metadata['tomato_model']['image_keys'])==4
    assert metadata['tomato_model']['task_profile']==PROFILE
    settings=dict(repository=str(Path.cwd()),checkpoint=str(root),mode='rtc',speed='fast',
                  steps=15,rounds=0,model_type='full-delta',crop_locator=str(Path.cwd()/'.deployment/crop_locator'))
    with pytest.raises(ValueError,match='流程不匹配'):
        DeploymentSettings(**settings,task_mode='tomato-b').plan()
    plan=DeploymentSettings(**settings,task_mode=PROFILE,initial_stage='pickup').plan()
    assert plan['model']['profile']=='full-delta'
    assert '--task-mode' in plan['execute'] and PROFILE in plan['execute']
    assert plan['model']['crop_locator']


def test_five_stage_rtc_handoff_uses_each_new_prompt_and_single_owner(tmp_path):
    from deploy.fr3_sharpa_runtime.tests.test_tomato_task import FakeRuntime, FakeBackend, until
    from deploy.fr3_sharpa_runtime.task_rtc import TaskRTC
    task=TaskControl(initial_stage='pickup',profile=PROFILE)
    runtime, backend=FakeRuntime(), FakeBackend()
    def observe(**options):
        if options.get('on_poll'):options['on_poll']()
        return {'observation/state':np.zeros(58,np.float32),'_captured_monotonic_ns':time.monotonic_ns()}
    args=SimpleNamespace(rtc_initial_delay_steps=5,rtc_delay_buffer_size=10,
                         rtc_prefix_attention_schedule='exp',rtc_max_guidance_weight=5,
                         execute_steps=15,num_chunks=0)
    engine=TaskRTC(task,runtime,backend,observe,args,tmp_path)
    try:
        task.ready();command(task,'start');until(engine,lambda:task.snapshot()['execution_state']=='RUNNING')
        for target in IDS[1:]:
            command(task,'next');engine.tick()
            assert task.snapshot()['active_stage']!=target
            until(engine,lambda:task.snapshot()['execution_state']=='RUNNING')
            assert task.snapshot()['active_stage']==target
            assert runtime.calls[-1][0]['prompt']==stage(target,PROFILE).prompt
            assert runtime.calls[-1][1] is None  # No previous-stage RTC prefix.
        assert len([c for c in backend.calls if isinstance(c,tuple) and c[0]=='start'])==1
        assert 'stop' not in backend.calls
        command(task,'complete');assert engine.tick() is False
        assert task.snapshot()['execution_state']=='COMPLETED'
    finally:
        engine.pool.shutdown(wait=True,cancel_futures=True)
        engine.observation_pool.shutdown(wait=True,cancel_futures=True)


def test_crop_handshake_matches_actual_selected_profile():
    from deploy.fr3_sharpa_runtime.interaction_crop import crop_capabilities
    from deploy.fr3_sharpa_runtime.tomato_model import FOUR_IMAGE_KEYS
    old=crop_capabilities({'image_keys':list(FOUR_IMAGE_KEYS)},'sha')
    new=crop_capabilities({'image_keys':list(FOUR_IMAGE_KEYS),'task_profile':PROFILE},'sha')
    assert old['roi_revision']=='causal_hand_recovery_v2' and 'roi_profile' not in old
    assert new['roi_revision']=='in_hand119_right_truss_causal_v1'
    assert new['roi_profile']=='in_hand119_right_truss_v1'
