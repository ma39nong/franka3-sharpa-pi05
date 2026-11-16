import json
import numpy as np
import pytest
from deploy.fr3_sharpa_runtime.shared_roi import SharedROIWriter,read_history


def publish(w,i,session='s'):
    stamp=1_000_000_000+i*100_000_000;image=np.full((32,32,3),i,np.uint8)
    obs={'observation/image':image,'_crop_context':dict(session=session,topic_stamp_ns={'cam0':stamp})}
    status=dict(roi_revision='test',local_roi_valid=True,crop_box=[1,1,20,20],source_size=[32,32])
    assert w.submit(obs,np.full((224,224,3),i,np.uint8),status)
    w.pending.join()
    return {'cam0':stamp},image


def test_shared_pixels_are_exact_and_history_never_fills_gaps(tmp_path):
    w=SharedROIWriter(tmp_path);samples=[publish(w,i) for i in range(16)];w.close();rows,images=zip(*samples)
    rgb,m,identity=read_history(tmp_path,list(rows),native_rgb=list(images),now_unix=2.54)
    assert rgb.shape==(16,224,224,3) and m.shape==(116,) and identity['uses_future_frames'] is False
    assert (rgb[-1]==15).all() and m[112]==0
    with pytest.raises(ValueError,match='stale'):read_history(tmp_path,list(rows),native_rgb=list(images),now_unix=9)
    bad=list(images);bad[0]=images[-1]
    with pytest.raises(ValueError,match='mismatch'):read_history(tmp_path,list(rows),native_rgb=bad,now_unix=2.54)
    (next(p for p in tmp_path.glob('*.npz') if 'none' not in p.name)).unlink()
    with pytest.raises(ValueError,match='incomplete'):read_history(tmp_path,list(rows),native_rgb=list(images),now_unix=2.54)


def test_new_session_cannot_reuse_previous_history(tmp_path):
    w=SharedROIWriter(tmp_path);samples=[publish(w,i) for i in range(16)];publish(w,16,session='new');w.close();rows,images=zip(*samples)
    with pytest.raises(ValueError,match='incomplete'):read_history(tmp_path,list(rows),native_rgb=list(images),now_unix=2.6)


def test_low_rate_roi_is_not_presented_as_ten_hz(tmp_path):
    w=SharedROIWriter(tmp_path);samples=[publish(w,i) for i in range(0,80,5)];w.close();rows,images=zip(*samples)
    with pytest.raises(ValueError,match='irregular'):read_history(tmp_path,list(rows),native_rgb=list(images),now_unix=8.54)


def test_cropper_publishes_same_pixels_as_policy_input(tmp_path):
    from deploy.fr3_sharpa_runtime.interaction_crop import InteractionCropper
    from deploy.fr3_sharpa_runtime.tests.test_interaction_crop import Locator,observation
    w=SharedROIWriter(tmp_path);cropper=InteractionCropper(Locator(),np.ones(58));cropper.shared_writer=w
    data=observation();data['_crop_context']['topic_stamp_ns']={'cam0':1_000_000_000}
    result=cropper.apply(data);w.close();p=next(tmp_path.glob('*.npz'))
    with np.load(p,allow_pickle=False) as z:np.testing.assert_array_equal(z['roi'],result['observation/interaction_image'])


def test_shadow_agent_can_report_without_becoming_control_evidence():
    from deploy.fr3_sharpa_control.agent import DecisionGate
    from deploy.fr3_sharpa_runtime.tests.test_agent_integration import packet,task
    gate=DecisionGate()
    for t in [100.,100.5]:
        value=packet(t);value['control_eligible']=False
        assert gate.update(value,task(),now=t+.1,model_version='b0') is None
        assert gate.count==0
    assert gate.update(packet(101.),task(),now=101.1,model_version='b0') is None
    assert gate.update(packet(101.5),task(),now=101.6,model_version='b0')=='harvest_1'
