"""Real authority/timeline regression: armed stage handoff -> human -> fresh policy.

No ROS nodes, SDKs, inference subprocesses or hardware connections are created.
"""
import numpy as np
from openpi_client import rtc
from deploy.fr3_sharpa_runtime.dagger import DaggerAuthority
from deploy.fr3_sharpa_runtime.stage_handoff import StageTimeline


def human(now):
    return dict(arm=np.zeros(7),hand=np.zeros(22),arm_time=now,hand_time=now,
                session='simulated-input',sensitivity=[1]*7,max_delta=[1.5]*7,timeout_s=.2)


def expect_rejected(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Obsolete/unauthorized request was accepted')


def main():
    events=[]
    initial=np.zeros(58,dtype=np.float32);initial[55]=.23
    core=DaggerAuthority(initial,run_id='offline-stage-takeover',emit=events.append)
    def commit(target,phase='hold'):
        return core.commit(target,target,sequence=core.sequence+1,
            stamp_ns=1_000_000_000+(core.sequence+2)*33_333_333,
            phase=phase,gateway_sequence=core.sequence)
    def command(name,now):
        core.command(name,f'{name}-{core.generation}-{core.sequence}',core.generation,
                     measured=core.last,human=human(now),now=now)
    def hold(start):
        for offset in (0.,.1,.3,.61):
            now=start+offset
            target=core.step(core.last,human(now),gateway_ok=True,now=now)
            commit(target)
        assert core.mode=='HOLD'

    core.begin_policy()
    assert core.accept_policy(core.generation,stage='harvest_1',prompt='old prompt')
    commit(initial,'policy')
    timeline=StageTimeline(np.repeat(initial[None],50,axis=0),1.,1.,task_epoch=0)
    old_context=timeline.snapshot(1.1,delay_steps=1,schedule='exp',weight=5)
    boundary=timeline.begin_switch(1,'manual-stage',1.2)
    candidate=np.repeat(initial[None],50,axis=0);candidate[:,0]=.001;candidate[:,55]=.25
    installed=timeline.prepare_handoff(candidate,{**boundary,'captured_monotonic_s':1.21},
        1.22,1.233333333,initial,.7,lambda window:None)
    assert installed['accepted']
    # A real dispatch part way through the bridge precedes takeover.
    partial=timeline.sample(installed['bridge_start']+.05)
    commit(partial,'stage_bridge')
    assert timeline.applied_token is None and core.stage=='harvest_1'
    held_right=core.last[29:].copy()
    assert held_right[26]>.23 and held_right[26]<.25

    command('takeover',1.3);takeover_generation=core.generation
    timeline.invalidate()  # The owner performs this on changed authority.
    assert timeline.switch is None and timeline.bridge is None and timeline.pending is None
    expect_rejected(lambda:timeline.install(candidate,rtc.make_receipt(old_context),1.31,task_epoch=1))
    assert not core.accept_policy(takeover_generation-1,stage='harvest_2',prompt='obsolete stage')
    assert core.stage=='harvest_1'
    commit(core.last)
    hold(1.3);command('follow',1.91)
    source=human(2.);source['arm'][:]=.1;source['hand'][:]=.2
    target=core.step(core.last,source,gateway_ok=True,now=2.)
    np.testing.assert_array_equal(target[29:],held_right)
    assert np.max(np.abs(target[:29]-core.last[:29]))>0
    frame=commit(target,'human_follow')
    assert frame['action_owner']==['human','human','hold','hold']
    expect_rejected(core.begin_policy)
    assert not core.accept_policy(takeover_generation-1,stage='return_stem',prompt='late result')
    expect_rejected(lambda:command('resume',2.))

    command('hold',2.);commit(core.last);hold(2.)
    np.testing.assert_array_equal(core.last[29:],held_right)
    command('resume',2.61)
    assert core.mode=='RESUMING' and core.intervening
    assert not core.accept_policy(takeover_generation,stage='harvest_2',prompt='pre-resume result')
    assert core.accept_policy(core.generation,stage='harvest_2',prompt='fresh observation prompt')
    frame=commit(core.last,'policy')
    assert frame['action_owner']==['policy']*4 and frame['intervention_id'] is None
    assert core.stage=='harvest_2'
    assert len([event for event in events if event.get('event')=='human_end'])==1
    print('Stage bridge cancellation, nonzero right hold, human authority and fresh recovery passed; no hardware started')


if __name__=='__main__':main()
