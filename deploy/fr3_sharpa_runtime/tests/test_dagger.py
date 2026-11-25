"""DAgger authority, intervals and masks; all tests are hardware-free."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.dagger import DaggerAuthority
from deploy.fr3_sharpa_runtime.dagger_inputs import InputCache
from deploy.fr3_sharpa_runtime.dagger_io import AuditWriter
from deploy.fr3_sharpa_runtime.dagger_export import export, horizon_mask
from deploy.fr3_sharpa_runtime.protocol import SHARPA_JOINT_NAMES
from deploy.fr3_sharpa_runtime.task_control import TaskControl
from deploy.fr3_sharpa_runtime.tests.test_deployment_center import checkpoint, repository, settings


def human(now=0, session='input-1'):
    return dict(arm=np.zeros(7),hand=np.zeros(22),arm_time=now,hand_time=now,session=session,
                sensitivity=[1]*7,max_delta=[1.5]*7,timeout_s=.2)


def commit(core, target=None, phase='hold', seq=None):
    target = core.last if target is None else target
    sequence = core.sequence+1 if seq is None else seq
    return core.commit(target, target, sequence=sequence,stamp_ns=1_000_000_000+sequence*33_333_333,
                       phase=phase,gateway_sequence=sequence-1)


def policy(core):
    core.begin_policy()
    assert core.accept_policy(core.generation,stage='harvest_1',prompt='pick this tomato')
    commit(core,phase='policy')


def command(core, name, now=0, source=None, request_id=None):
    return core.command(name,request_id or f'{name}-{core.generation}-{core.sequence}',core.generation,
                        measured=core.last,human=source if source is not None else human(now),now=now)


def held(core):
    for now in (0,.1,.3,.61):
        core.step(core.last,human(now),gateway_ok=True,now=now)
        commit(core)
    assert core.mode=='HOLD'


def test_left_only_follow_and_explicit_recovery_have_exact_committed_labels():
    events=[]
    core=DaggerAuthority(np.zeros(58),run_id='run',emit=events.append)
    policy(core)
    old_generation=core.generation
    command(core,'takeover');frame=commit(core)
    assert core.generation>old_generation and frame['action_owner']==['hold']*4
    assert [e['event'] for e in events if 'event' in e].count('human_start')==1
    with pytest.raises(ValueError,match='Wait for measured'):
        command(core,'follow')
    held(core)
    command(core,'follow',now=.61,source=human(.61));commit(core)
    source=human(.7);source['arm'][:]=.1;source['hand'][:]=.2
    target=core.step(core.last,source,gateway_ok=True,now=.7)
    np.testing.assert_array_equal(target[29:],0)
    np.testing.assert_allclose(target[:7],.1)
    frame=commit(core,target,phase='human_follow')
    assert frame['action_owner']==['human','human','hold','hold'] and frame['intervention_id']==1
    with pytest.raises(ValueError,match='Hold and confirm'):
        command(core,'resume',now=.7)
    command(core,'hold',now=.7);commit(core);held(core)
    command(core,'resume',now=.7);commit(core)
    assert core.mode=='RESUMING' and core.intervening
    assert not core.accept_policy(old_generation,stage='harvest_2',prompt='wrong')
    assert core.accept_policy(core.generation,stage='harvest_1',prompt='pick this tomato')
    assert core.intervening  # Until the first successful automatic dispatch.
    frame=commit(core,phase='approach')
    assert frame['action_owner']==['policy']*4 and frame['intervention_id'] is None
    end=[e for e in events if e.get('event')=='human_end']
    assert len(end)==1 and end[0]['frame_index']==frame['frame_index']


def test_no_automatic_follow_after_input_recovers_or_source_restarts():
    core=DaggerAuthority(np.zeros(58),run_id='run',emit=lambda _:None)
    policy(core);command(core,'takeover');commit(core);held(core)
    command(core,'follow',now=.61);commit(core)
    generation=core.generation
    target=core.step(core.last,human(0),gateway_ok=True,now=1)
    np.testing.assert_array_equal(target,core.last)
    assert core.mode=='HOLDING' and core.generation>generation
    for now in (1.1,1.2,1.4,1.8):
        core.step(core.last,human(now,session='new-session'),gateway_ok=True,now=now)
        commit(core)
    assert core.mode=='HOLD'
    command(core,'follow',now=1.8,source=human(1.8,session='new-session'));commit(core)
    core.step(core.last,human(1.9,session='another'),gateway_ok=True,now=1.9)
    assert core.mode=='HOLDING'
    # Healthy restarted input remains held; no implicit reauthorization.
    assert core.mode=='HOLDING'
    core.source_lost_at=1.9
    with pytest.raises(RuntimeError,match='unavailable'):
        core.step(core.last,None,gateway_ok=True,now=3.5)


def test_hold_needs_fresh_gateway_admission_and_measured_stability():
    core=DaggerAuthority(np.zeros(58),run_id='run',emit=lambda _:None)
    policy(core);command(core,'takeover');commit(core)
    for now in (0,1,2):
        core.step(core.last,None,gateway_ok=False,now=now)
    assert core.mode=='HOLDING'
    moving=np.full(58,.1)
    core.step(moving,None,gateway_ok=True,now=3)
    core.step(moving,None,gateway_ok=True,now=4)
    assert core.mode=='HOLDING'
    held(core)


def test_requests_are_idempotent_and_bound_to_generation():
    core=DaggerAuthority(np.zeros(58),run_id='run',emit=lambda _:None)
    policy(core)
    generation=core.generation
    core.command('takeover','same',generation,measured=core.last,human=human(),now=0)
    commit(core)
    core.command('takeover','same',generation,measured=core.last,human=human(),now=0)
    assert core.intervention_id==1
    with pytest.raises(ValueError,match='reused'):
        core.command('resume','same',generation,measured=core.last,human=human(),now=0)
    with pytest.raises(ValueError,match='generation'):
        core.command('follow','other',generation,measured=core.last,human=human(),now=0)


def test_failed_policy_dispatch_cannot_fabricate_successful_human_end():
    events=[];core=DaggerAuthority(np.zeros(58),run_id='run',emit=events.append)
    policy(core);command(core,'takeover');commit(core);held(core)
    command(core,'resume',now=.7);commit(core)
    core.accept_policy(core.generation,stage='harvest_1',prompt='same')
    core.end('SDK send failed before dispatch commit')
    ends=[e for e in events if e.get('event')=='human_end']
    assert len(ends)==1 and ends[0]['incomplete']
    assert not any(e.get('event')=='policy_applied' and e.get('sequence',0)>0 for e in events)


def test_repeated_interventions_have_increasing_ids_and_zero_based_frame_indices():
    events=[];core=DaggerAuthority(np.zeros(58),run_id='run',emit=events.append)
    policy(core)
    for expected in range(1,6):
        command(core,'takeover');commit(core);held(core)
        assert core.intervention_id==expected
        command(core,'resume',now=.7);commit(core)
        core.accept_policy(core.generation,stage='harvest_1',prompt='same');commit(core,phase='policy')
    frames=[e for e in events if 'event' not in e]
    assert [f['frame_index'] for f in frames]==list(range(len(frames)))
    assert len([e for e in events if e.get('event')=='human_start'])==5


def test_input_cache_preserves_original_age_and_reorders_exact_named_joints():
    cache=InputCache(dict(timeout_s=.2))
    def message(names,ns):
        return SimpleNamespace(name=list(reversed(names)),position=list(reversed(range(len(names)))),
            header=SimpleNamespace(stamp=SimpleNamespace(sec=ns//10**9,nanosec=ns%10**9),frame_id='session'))
    names=[f'left_fr3_joint{i}' for i in range(1,8)]
    cache.receive('arm',message(names,990_000_000),1_000_000_000,mono=10)
    cache.receive('hand',message(SHARPA_JOINT_NAMES,990_000_000),1_000_000_000,mono=10)
    value=cache.snapshot()
    assert value['arm_time']==pytest.approx(9.99)
    np.testing.assert_array_equal(value['hand'],np.arange(22))
    cache.receive('hand',message(SHARPA_JOINT_NAMES,990_000_000),1_000_000_000,mono=10.1)
    assert cache.snapshot() is None  # Repeated source cannot refresh its age.


def test_task_stage_commands_cannot_escape_dagger_takeover():
    task=TaskControl(profile='tomato-in-hand119',initial_stage='pickup')
    task.dagger_mode=True;task.dagger_suspended=True;task.ready()
    with pytest.raises(ValueError,match='人工接管'):
        task.command('start','id',expected_version=0)
    with pytest.raises(ValueError,match='接管区'):
        task.command('pause','pause',expected_version=0)
    task.command('stop','stop',expected_version=0)
    assert task.data['execution_state']=='STOPPING'


def test_dagger_is_opt_in_and_requires_recording(repository,checkpoint):
    with pytest.raises(ValueError,match='同时录制'):
        settings(repository,checkpoint,dagger=True,record=False).plan()
    from deploy.fr3_sharpa_control.plan import plan_environment
    with pytest.raises(ValueError,match='缺少DAgger执行器'):
        settings(repository,checkpoint,dagger=True,record=True).plan()
    for name in ('dagger_execution.py','dagger_owner.py'):
        (repository/'deploy/fr3_sharpa_runtime'/name).write_text('# simulated executor presence')
    plan=settings(repository,checkpoint,dagger=True,record=True).plan()
    assert plan_environment(plan)['SHARPA_DAGGER_ENABLED']=='1'
    assert plan_environment(settings(repository,checkpoint).plan())['SHARPA_DAGGER_ENABLED']=='0'


def test_auto_export_masks_only_human_and_matches_same_gateway_sequence(tmp_path):
    audit=AuditWriter(tmp_path/'dagger')
    core=DaggerAuthority(np.zeros(58),run_id='run',emit=audit.put)
    policy(core);command(core,'takeover');commit(core);held(core)
    command(core,'follow',now=.61);frame=commit(core,phase='human_follow')
    audit.put(dict(event='gateway_status',sequence=frame['sequence'],accepted_sides=['left','right'],faults=[]))
    missing_ack_frame=commit(core,phase='human_follow')
    command(core,'hold',now=.7);commit(core);core.end('operator stop');audit.close()
    report=export(tmp_path)
    assert report['raw_log_finalized'] and report['interventions']==1
    arrays=np.load(tmp_path/'dagger/supervision.npz')
    mask=arrays['frame_mask'];index=frame['frame_index']
    assert mask[index,:29].all() and not mask[index,29:].any()
    assert not mask[missing_ack_frame['frame_index'],:7].any()
    assert mask[missing_ack_frame['frame_index'],7:29].all()
    window=horizon_mask(mask,arrays['source_timestamp_ns'],arrays['intervention_id'],index)
    assert window[0,:29].all() and not window[2:].any() and not window[:,29:].any()
    intervals=json.loads((tmp_path/'dagger/interventions.json').read_text())
    assert intervals[0]['start']['frame_index'] < intervals[0]['end']['frame_index']
    assert report['training_validation']=='UNVERIFIED'


def test_input_launch_has_only_input_producers_and_no_robot_output():
    import ast
    path=Path(__file__).resolve().parents[1]/'dagger_inputs.launch.py'
    tree=ast.parse(path.read_text())
    packages=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='Node':
            packages.extend(ast.literal_eval(kw.value) for kw in node.keywords if kw.arg=='package')
    assert packages==['manus_driver','manus_sharpa_teleop']
    for name in ('dagger_inputs.py','dagger_gello_reader.py'):
        source=(path.parent/name).read_text()
        assert 'SharpaSdkDirectExecutor' not in source and 'ArmCommand' not in source


def test_gello_python_can_be_explicitly_bound_to_existing_environment(tmp_path,monkeypatch):
    from deploy.fr3_sharpa_runtime.dagger_inputs import teleop_python
    executable=tmp_path/'python'
    executable.write_text('test only');executable.chmod(0o700)
    monkeypatch.setenv('SHARPA_GELLO_INPUT_PYTHON',str(executable))
    assert teleop_python()==executable
    executable.unlink()
    with pytest.raises(FileNotFoundError,match='SHARPA_GELLO_INPUT_PYTHON'):
        teleop_python()


def test_gello_child_does_not_inherit_ros_python_or_native_library_paths(monkeypatch):
    from deploy.fr3_sharpa_runtime.dagger_inputs import reader_environment
    monkeypatch.setenv('PYTHONPATH','/ros/lib/python3.12/site-packages')
    monkeypatch.setenv('PYTHONHOME','/ros/python')
    monkeypatch.setenv('LD_LIBRARY_PATH','/ros/lib')
    env=reader_environment('/existing/teleop/bin/python')
    assert 'python3.12' not in env['PYTHONPATH']
    assert 'PYTHONHOME' not in env and 'LD_LIBRARY_PATH' not in env
    assert env['CONDA_PREFIX']=='/existing/teleop'
