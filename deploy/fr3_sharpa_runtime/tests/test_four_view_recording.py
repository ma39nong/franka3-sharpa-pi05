"""Actual ROI bytes, source alignment, recording flush and four-panel export."""
import json
from pathlib import Path
import numpy as np
import pytest
from deploy.fr3_sharpa_runtime.four_view_recording import FourViewRecording, KEYS, pixel_sha
from deploy.fr3_sharpa_control.export_four_view import samples, export_four_views


def input_frame(value=40,stamp=10):
    obs={k:np.full((48,64,3),value,np.uint8) for k in KEYS}
    obs.update(prompt='test prompt',_captured_monotonic_ns=stamp,_topic_stamp_ns={'cam0':stamp})
    pixels=np.full((224,224,3),value+20,np.uint8)
    payload=dict(pixels=pixels,metadata=dict(source_monotonic_ns=stamp,roi_pixel_sha256=pixel_sha(pixels),mode='detected_local'))
    return obs,payload


def test_record_exact_immutable_four_inputs_and_source(tmp_path):
    recorder=FourViewRecording(tmp_path/'policy_views')
    obs,payload=input_frame()
    recorder.submit(obs,payload,query_id='one',task={'stage_id':'adjust_in_hand'})
    for k in KEYS:obs[k][:]=0
    payload['pixels'][:]=0
    recorder.close()
    data=list(samples(tmp_path/'policy_views'))
    assert len(data)==1
    meta,frames=data[0]
    assert meta['task']['stage_id']=='adjust_in_hand'
    assert (frames['head']==40).all() and (frames['interaction']==60).all()
    assert json.loads((tmp_path/'policy_views/status.json').read_text())['status']=='COMPLETE'


def test_mismatched_roi_refused_and_reported(tmp_path):
    recorder=FourViewRecording(tmp_path/'policy_views')
    obs,payload=input_frame();payload['metadata']['source_monotonic_ns']=11
    recorder.submit(obs,payload,query_id='bad');recorder.close()
    status=json.loads((tmp_path/'policy_views/status.json').read_text())
    assert status['status']=='FAILED' and not status['written']
    assert not list((tmp_path/'policy_views').glob('*.npz'))


def test_real_mp4_four_panels_and_gate_trim(tmp_path):
    import cv2
    recorder=FourViewRecording(tmp_path/'policy_views')
    obs,payload=input_frame();recorder.submit(obs,payload,query_id='seed');recorder.close()
    meta,frames=next(samples(tmp_path/'policy_views'))
    stamp=meta['completed_monotonic_ns']
    (tmp_path/'recording.json').write_text(json.dumps(dict(started_monotonic_ns=stamp+10_000_000,ended_monotonic_ns=stamp+210_000_000)))
    report=export_four_views(tmp_path,fps=10,width=160,height=90)
    assert report['frames']==3 and report['pre_gate_samples']==1 and not report['roi_recomputed']
    for name in ['head','left_wrist','right_wrist','interaction','four_cameras']:
        video=cv2.VideoCapture(str(tmp_path/'video/four_view'/f'{name}.mp4'))
        try:
            ok,image=video.read();assert ok
            assert image.shape[:2]==((180,320) if name=='four_cameras' else (90,160))
            assert int(video.get(cv2.CAP_PROP_FRAME_COUNT))==3
        finally:video.release()
    with pytest.raises(FileExistsError):export_four_views(tmp_path,fps=10,width=160,height=90)


def test_tampered_pixels_fail_export(tmp_path):
    recorder=FourViewRecording(tmp_path/'policy_views');obs,payload=input_frame()
    recorder.submit(obs,payload,query_id='bad');recorder.close()
    path=next((tmp_path/'policy_views').glob('*.npz'))
    with np.load(path,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    data['interaction'][:]=0;np.savez(path,**data)
    with pytest.raises(ValueError,match='Changed'):list(samples(tmp_path/'policy_views'))
