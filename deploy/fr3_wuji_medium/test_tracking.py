"""One medium tracking threshold across the player, consumer and prefetch."""
import json
import numpy as np
import pytest
from .runtime.core import ConsumerGuard, Feedback, Frame
from .runtime.limits import ARM_TRACKING_TOLERANCE_RAD
from .prefetch import endpoint_ready
from .test_medium import make_player


def feedback(q, now):
    return Feedback(q,np.zeros(54),(now,)*4,(now,)*4,hand_control='slider')


@pytest.mark.parametrize('error',[.081614,.199999,.200001])
def test_player_consumer_and_prefetch_agree_on_point_two(error):
    assert ARM_TRACKING_TOLERANCE_RAD == .2
    p=make_player()
    p.start(feedback(np.zeros(54),10.),10.)
    p.transition('playback',10.,'test')
    measured=p.plan.playback.sample(.1);measured[6]-=error
    result=p.tick(feedback(measured,10.1),10.1)
    assert (result is not None)==(error<=.2)
    if error>.2:
        assert 'exceeds 0.20 rad' in p.reason
    guard=ConsumerGuard(np.full(54,-2.),np.full(54,2.),hand_control='slider')
    guard.arm('run','hash',feedback(np.zeros(54),10.),10.)
    frame=Frame('run','hash',0,10.,10.02,0.,np.zeros(54),np.zeros(54),'playback')
    measured=np.zeros(54);measured[6]=-error
    if error<=.2:
        guard.validate(frame,feedback(measured,10.),10.)
    else:
        with pytest.raises(ValueError,match='exceeds 0.20 rad'):
            guard.validate(frame,feedback(measured,10.),10.)
    p.settled_since=10.
    measured=p.plan.raw[-1].copy();measured[6]-=error
    assert endpoint_ready(p,feedback(measured,10.))==(error<=.2)


def test_runtime_records_new_limit_and_slow_remains_unchanged(tmp_path):
    from .deploy import parse_args,write_runtime
    from experiments.weight_motion_eval.oneshot.limits import ARM_TRACKING_TOLERANCE_RAD as slow
    assert slow==.08
    runtime,_,_=write_runtime(parse_args(['--check','--model','19999']),tmp_path)
    assert runtime['arm_tracking_rad']==.2
    assert json.loads((tmp_path/'runtime.json').read_text())['arm_tracking_rad']==.2
