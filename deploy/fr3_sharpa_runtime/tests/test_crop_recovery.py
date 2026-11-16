"""Causal recovery must use fresh evidence and preserve hand identity/containment."""
from types import SimpleNamespace

import cv2
import numpy as np

from deploy.fr3_sharpa_runtime.crop_recovery import CausalHandRecovery, CausalCropWindow
from deploy.fr3_sharpa_runtime.interaction_crop import InteractionCropper


def test_current_detail_reacquires_without_history_and_without_lowering_threshold():
    image=np.full((400,640,3),123,np.uint8)
    seen=[]
    def predict(images):
        seen.extend(images)
        return [[dict(label=1,score=.9,box=[100,100,160,170])]]
    recovery=CausalHandRecovery()
    hands,sources,attempts=recovery.recover(image,image[:,:,0],[],1_000_000_000,1,
                                          SimpleNamespace(predict_detail=predict),[1])
    assert sources=={1:'reacquired'} and hands[0]['box']==[100,100,160,170]
    assert seen[0] is image
    assert attempts[-1]['method']=='current_detail_480'
    hands,_,_=CausalHandRecovery().recover(image,image[:,:,0],[],1_000_000_000,1,
        SimpleNamespace(predict_detail=lambda images:[[dict(label=1,score=.74,box=[100,100,160,170])]]),[1])
    assert not hands


def test_local_reacquisition_maps_coordinates_back_to_native_frame():
    from deploy.fr3_sharpa_runtime.crop_vendor.geometry import square_box
    image=np.full((400,640,3),123,np.uint8)
    recovery=CausalHandRecovery()
    history=[340,100,400,170]
    recovery.hands={1:dict(box=history,stamp=1_000_000_000,track=None)}
    region=square_box(history,640,400,scale=2.1)
    hands,_,_=recovery.recover(image,image[:,:,0],[],1_100_000_000,2,
        SimpleNamespace(predict=lambda images:[[dict(label=1,score=.9,box=[40,40,100,110])]]),[1])
    assert hands[0]['box']==[region[0]+40,region[1]+40,region[0]+100,region[1]+110]


def test_disjoint_same_role_reacquisition_is_rejected():
    image=np.zeros((400,640,3),np.uint8)
    recovery=CausalHandRecovery()
    hands,_,attempts=recovery.recover(image,image[:,:,0],[],1_000_000_000,1,
        SimpleNamespace(predict_detail=lambda images:[[dict(label=1,score=.9,box=[100,100,160,170]),
                                                       dict(label=1,score=.9,box=[340,100,400,170])]]),[1])
    assert not hands and attempts[-1]['ambiguous']


def test_real_optical_flow_bridges_dropout_using_current_pixels():
    rng=np.random.default_rng(21)
    first=np.zeros((400,640,3),np.uint8)
    first[110:180,110:180]=rng.integers(30,200,(70,70,3),dtype=np.uint8)
    second=cv2.warpAffine(first,np.float32([[1,0,7],[0,1,4]]),(640,400))
    class Locator:
        def predict(self,images):
            return [[dict(label=1,score=.99,box=[100,100,190,190])]] if images[0] is first else [[] for _ in images]
    cropper=InteractionCropper(Locator(),np.ones(58))
    def obs(image,stamp):
        return {'observation/image':image,'observation/state':np.zeros(58),
                '_crop_context':{'source_monotonic_ns':stamp,'session':'test'}}
    cropper.apply(obs(first,1_000_000_000))
    result=cropper.apply(obs(second,1_100_000_000))
    assert cropper.last['mode']=='tracked_local'
    np.testing.assert_allclose(cropper.last['interaction_box'],[107,104,197,194],atol=1)
    from deploy.fr3_sharpa_runtime.crop_vendor.geometry import native_crop
    expected=cv2.resize(native_crop(second,cropper.last['crop_box']),(224,224),interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal(result['observation/interaction_image'],expected)


def test_windows_contain_abrupt_motion_and_shrink_gradually():
    window=CausalCropWindow()
    old=window.select([250,140,400,280],640,400,1_000_000_000)
    moved=[450,220,580,340]
    new=window.select(moved,640,400,1_100_000_000)
    assert new[0]<=moved[0] and new[1]<=moved[1] and new[2]>=moved[2] and new[3]>=moved[3]
    assert new[2]-new[0] >= (old[2]-old[0])*.97


def test_windows_at_edges_and_wide_interactions_preserve_containment():
    rng=np.random.default_rng(7)
    window=CausalCropWindow()
    for index in range(500):
        x=sorted(rng.uniform(0,640,2));y=sorted(rng.uniform(0,400,2))
        box=window.select([*map(float,[x[0],y[0],x[1],y[1]])],640,400,(index+1)*100_000_000)
        assert box[0]<=x[0] and box[1]<=y[0] and box[2]>=x[1] and box[3]>=y[1]
        assert box[2]-box[0]==box[3]-box[1]


def test_individual_hand_track_survives_interaction_role_combination():
    image=np.full((400,640,3),100,np.uint8)
    right=dict(label=1,score=.99,box=[300,160,360,230])
    left=dict(label=2,score=.99,box=[240,160,295,230])
    class Locator:
        detections=[right,left]
        def predict(self,images):
            return [self.detections for _ in images]
    locator=Locator();cropper=InteractionCropper(locator,np.ones(58))
    def obs(stamp):
        return {'observation/image':image,'observation/state':np.zeros(58),
                '_crop_context':{'source_monotonic_ns':stamp,'session':'test'}}
    cropper.apply(obs(1_000_000_000))
    assert cropper.last['roles']==[1,2]
    cropper.recovery.hands[1]['track']=SimpleNamespace(update=lambda *a:dict(valid=True,box=right['box']))
    locator.detections=[left]
    cropper.apply(obs(1_100_000_000))
    assert cropper.last['mode']=='tracked_local' and cropper.last['roles']==[1,2]
    assert cropper.last['hand_sources']=={'2':'detected','1':'tracked'}


def test_detail_detection_restores_global_resize_after_error():
    from threading import RLock
    import pytest
    from deploy.fr3_sharpa_runtime.crop_vendor.locator import RgbLocator
    locator=RgbLocator.__new__(RgbLocator)
    locator.model=SimpleNamespace(transform=SimpleNamespace(min_size=(320,),max_size=640))
    locator._prediction_lock=RLock()
    seen=[]
    def fail(images):
        seen.append((locator.model.transform.min_size,locator.model.transform.max_size))
        raise ValueError('detector failure')
    locator._predict=fail
    with pytest.raises(ValueError,match='detector failure'):
        locator.predict([],min_size=480)
    assert seen==[((480,),960)]
    assert locator.model.transform.min_size==(320,) and locator.model.transform.max_size==640


def test_return_from_left_transport_reacquires_default_right_anchor():
    # Previous selection was left transport; current state/RGB can return to
    # right anchoring before the next query. Search must include that anchor.
    right=dict(label=1,score=.99,box=[400,160,460,230])
    left=dict(label=2,score=.99,box=[100,160,160,230])
    locator=SimpleNamespace(predict=lambda images:[[left] for _ in images],
                            predict_detail=lambda images:[[right,left] for _ in images])
    cropper=InteractionCropper(locator,np.ones(58))
    cropper.selector.roles=[2]
    cropper.previous_shape=(400,640,3)
    cropper.previous_session='test'
    cropper.previous_stamp=1_000_000_000
    import time
    cropper.previous_arrival=time.monotonic()
    cropper.apply({'observation/image':np.full((400,640,3),100,np.uint8),
                   'observation/state':np.zeros(58),
                   '_crop_context':{'source_monotonic_ns':1_100_000_000,'session':'test'}})
    assert cropper.last['roles']==[1] and cropper.last['mode']=='reacquired_local'
    assert cropper.last['hand_sources']['1']=='reacquired'


def test_crop_diagnostics_round_trip_strict_runtime_msgpack():
    from openpi_client import msgpack_numpy
    image=np.full((400,640,3),100,np.uint8)
    locator=SimpleNamespace(predict=lambda images:[[dict(label=1,score=.99,box=[100,100,160,170])]])
    cropper=InteractionCropper(locator,np.ones(58))
    cropper.apply({'observation/image':image,'observation/state':np.zeros(58),
                   '_crop_context':{'source_monotonic_ns':1_000_000_000,'session':'test'}})
    received=msgpack_numpy.unpackb(msgpack_numpy.Packer().pack({'interaction_crop':cropper.last}))
    assert received['interaction_crop']['hand_sources']=={'1':'detected'}
