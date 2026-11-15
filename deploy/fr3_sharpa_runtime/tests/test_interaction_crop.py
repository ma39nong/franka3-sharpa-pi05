"""Causal pixels, session resets, four-view encoding, output delta and identity."""
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.interaction_crop import InteractionCropper
from deploy.fr3_sharpa_runtime.tomato_model import identify, R11_STATS_SHA256, R11_MANIFEST_SHA256, METADATA_SHA256, FOUR_IMAGE_KEYS


class Locator:
    def __init__(self):
        self.detections = [{'label':1,'score':.99,'box':[100,100,160,170]}]
    def predict(self, images):
        return [self.detections for _ in images]


def observation(stamp=1_000_000_000, value=50, session='one'):
    return {'observation/image':np.full((400,640,3),value,np.uint8),
            'observation/state':np.zeros(58,np.float32),
            '_crop_context':{'source_monotonic_ns':stamp,'session':session}}


def test_current_pixels_and_source_geometry():
    cropper = InteractionCropper(Locator(),np.ones(58))
    first = cropper.apply(observation())
    second = cropper.apply(observation(1_500_000_000,150))
    assert first['observation/interaction_image'].shape == (224,224,3)
    assert (second['observation/interaction_image']==150).all()
    assert second['observation/interaction_valid']
    assert cropper.last['local_roi_valid'] and not cropper.last['uses_future_frames']
    assert cropper.last['source_size'] == [640,400]


def test_lost_roi_uses_current_full_frame_and_separate_validity():
    locator = Locator(); locator.detections=[]
    cropper = InteractionCropper(locator,np.ones(58))
    data = observation(value=71)
    result = cropper.apply(data)
    assert (result['observation/interaction_image'][43:181]==71).all()
    assert not result['observation/interaction_image'][:42].any()
    assert result['observation/interaction_valid']
    assert not cropper.last['local_roi_valid']
    assert cropper.last['mode']=='current_full_fallback'


@pytest.mark.parametrize('stamp,session',[(500_000_000,'one'),(1_000_000_000,'one'),(2_000_000_000,'one'),(1_500_000_000,'two')])
def test_time_or_session_discontinuity_clears_tracking(stamp,session):
    locator = Locator(); cropper=InteractionCropper(locator,np.ones(58))
    cropper.apply(observation())
    locator.detections=[]
    cropper.apply(observation(stamp=stamp,session=session))
    assert cropper.last['reset']
    assert cropper.track is None
    assert cropper.last['mode']=='current_full_fallback'


def test_track_cannot_refresh_its_own_deadline(monkeypatch):
    locator = Locator(); cropper=InteractionCropper(locator,np.ones(58))
    cropper.apply(observation())
    locator.detections=[]
    cropper.track=SimpleNamespace(update=lambda *a:dict(valid=True,box=[100,100,160,170]))
    cropper.apply(observation(stamp=1_500_000_000))
    assert cropper.last['mode']=='tracked_local'
    cropper.apply(observation(stamp=1_800_000_000,value=190))
    assert cropper.last['mode']=='current_full_fallback'


def test_recorded_preview_matches_current_crop(tmp_path):
    cropper=InteractionCropper(Locator(),np.ones(58),status_dir=tmp_path)
    cropper.apply(observation(value=99))
    status=json.loads((tmp_path/'status.json').read_text())
    assert status['source_monotonic_ns']==1_000_000_000
    preview=cv2.imread(str(tmp_path/'preview.jpg'))
    assert preview.shape==(252,672,3)
    assert abs(float(preview[50:200,480:640].mean())-99)<2


def test_only_verified_four_view_weights_enable_crop():
    found=identify(R11_STATS_SHA256,R11_MANIFEST_SHA256,METADATA_SHA256)
    assert found['online_roi_required'] and tuple(found['image_keys'])==FOUR_IMAGE_KEYS
    with pytest.raises(ValueError,match='Four-view'):
        identify(R11_STATS_SHA256,'different',METADATA_SHA256)


def test_fourth_view_reaches_embedding_in_training_order():
    import jax.numpy as jnp
    from openpi.models.pi0 import Pi0
    from openpi.models.model import Observation,preprocess_observation
    calls=[]
    def encode(image,train=False):
        calls.append(float(image.mean()))
        return image.mean(axis=(1,2,3))[:,None,None],None
    fake=SimpleNamespace(image_keys=FOUR_IMAGE_KEYS,PaliGemma=SimpleNamespace(img=encode))
    obs=Observation(images={key:jnp.full((1,224,224,3),i/4) for i,key in enumerate(FOUR_IMAGE_KEYS)},
                    image_masks={key:jnp.ones((1,),bool) for key in FOUR_IMAGE_KEYS},state=jnp.zeros((1,58)))
    obs=preprocess_observation(None,obs,train=False,image_keys=FOUR_IMAGE_KEYS)
    tokens,_,_=Pi0.embed_prefix(fake,obs)
    assert calls==[0,.25,.5,.75] and tokens.shape==(1,4,1)
    assert float(tokens[0,-1,0])==.75


def test_crop_adapter_preserves_arm_only_delta_and_canonical_state():
    from deploy.fr3_sharpa_runtime.crop_input import CropRuntimeData
    from openpi.models.pi0_config import Pi0Config
    from openpi.transforms import compose
    model=Pi0Config(pi05=True,action_dim=58,image_keys=FOUR_IMAGE_KEYS)
    data=CropRuntimeData(extra_delta_transform=True,action_hz=30).create(Path('/tmp/no-assets'),model)
    state=np.arange(58,dtype=np.float32)
    raw={'observation/state':state,'observation/image':np.zeros((224,224,3),np.uint8),
         'observation/left_wrist_image':np.zeros((224,224,3),np.uint8),
         'observation/right_wrist_image':np.zeros((224,224,3),np.uint8),
         'observation/interaction_image':np.full((224,224,3),170,np.uint8),
         'observation/interaction_valid':np.bool_(True), 'actions':np.ones((50,58),np.float32)}
    encoded=compose(data.data_transforms.inputs)(raw)
    assert tuple(encoded['image'])==FOUR_IMAGE_KEYS
    assert (encoded['image']['interaction_0_rgb']==170).all()
    np.testing.assert_array_equal(encoded['state'],state)
    restored=compose(data.data_transforms.outputs)(encoded)
    np.testing.assert_allclose(restored['actions'],np.ones((50,58)))


def test_track_cannot_cross_hand_role_switch():
    locator=Locator();cropper=InteractionCropper(locator,np.ones(58))
    cropper.apply(observation())
    locator.detections=[]
    cropper.track_roles=(2,)  # Old crop followed left transport; current target is right.
    cropper.track=SimpleNamespace(update=lambda *a:dict(valid=True,box=[100,100,160,170]))
    cropper.apply(observation(stamp=1_500_000_000))
    assert cropper.last['mode']=='current_full_fallback'


def test_empty_fast_left_arm_does_not_retarget_right_truss():
    from deploy.fr3_sharpa_runtime.crop_selector import CurrentInteractionSelector
    selector=CurrentInteractionSelector()
    rgb=np.full((400,640,3),100,np.uint8)
    detections=[{'label':1,'score':.99,'box':[450,160,510,230]},
                {'label':2,'score':.99,'box':[100,160,160,230]}]
    selector.select(detections,np.zeros(58),640,400,rgb=rgb)
    state=np.zeros(58);state[:7]=1.5
    result=selector.select(detections,state,640,400,rgb=rgb)
    assert result['roles']==[1]


def test_near_left_hand_joins_truss_interaction():
    from deploy.fr3_sharpa_runtime.crop_selector import CurrentInteractionSelector
    selector=CurrentInteractionSelector()
    rgb=np.full((400,640,3),100,np.uint8)
    result=selector.select([{'label':1,'score':.99,'box':[300,160,360,230]},
                            {'label':2,'score':.99,'box':[240,160,295,230]}],np.zeros(58),640,400,rgb=rgb)
    assert result['roles']==[1,2]
    b=result['crop_box'];assert b[0]<=240 and b[2]>=360


def test_live_probe_reads_and_infers_without_motion(monkeypatch,tmp_path):
    from deploy.fr3_sharpa_runtime import crop_live_check,run
    import sys,time
    obs=observation();obs['_captured_monotonic_ns']=time.monotonic_ns()
    runtime=SimpleNamespace(inspect_metadata=lambda **kw:{'interaction_crop':{'enabled':True}},
                            infer_chunk=lambda observation:(None,np.zeros((50,58)),None),last_timing={})
    monkeypatch.setattr(crop_live_check,'SharpaRuntime',lambda **kw:runtime)
    monkeypatch.setattr(run,'live_observation',lambda *a,**kw:obs)
    closed=[]
    monkeypatch.setattr(run,'_observation_worker',SimpleNamespace(close=lambda:closed.append(True)))
    monkeypatch.setattr(crop_live_check.time,'sleep',lambda x:None)
    out=tmp_path/'live.json'
    monkeypatch.setattr(sys,'argv',['crop_live_check','--ros-overlay',str(tmp_path),'--out',str(out),'--samples','1'])
    crop_live_check.main()
    report=json.loads(out.read_text())
    assert report['no_hardware_commands'] and report['samples']==1 and closed==[True]
