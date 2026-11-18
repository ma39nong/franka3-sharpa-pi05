"""Four synchronized policy-input videos and 2x2 playback, no ROI re-inference.

Output 30fps is a presentation timeline: all panels hold the actual previous
policy sample, with held age shown. It is not a claim of 30Hz ROI production.
"""
import json
from pathlib import Path

import numpy as np

from deploy.fr3_sharpa_runtime.four_view_recording import NAMES, pixel_sha


def samples(directory):
    for path in sorted(Path(directory).glob('*.npz')):
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']))
            frames = {name:z[name].copy() for name in NAMES}
        for name, image in frames.items():
            if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3
                    or pixel_sha(image) != meta['pixel_sha256'][name]):
                raise ValueError(f'Changed policy input: {path.name}/{name}')
        if frames['interaction'].shape != (224,224,3):
            raise ValueError('Invalid actual ROI shape')
        yield meta, frames


def export_four_views(directory, *, fps=30, width=640, height=360):
    import cv2
    directory = Path(directory)
    journal = directory/'policy_views'
    status = json.loads((journal/'status.json').read_text())
    if status['status'] != 'COMPLETE' or not status['written']:
        raise ValueError('Four-view journal incomplete: '+json.dumps(status))
    gate_path = directory/'recording.json'
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}
    start = gate.get('started_monotonic_ns')
    end = gate.get('ended_monotonic_ns')
    if not start or not end or end < start:
        raise ValueError('Recording gate monotonic start/end required for four-view export')
    output = directory/'video/four_view'
    output.mkdir(parents=True,exist_ok=True)
    writers = {}
    count = used = pre_gate = 0
    try:
        for name in (*NAMES,'four_cameras'):
            path = output/(name+'.mp4')
            if path.exists():raise FileExistsError(f'已有视频，不覆盖：{path}')
            size = (width*2,height*2) if name == 'four_cameras' else (width,height)
            w = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
            if not w.isOpened():raise RuntimeError(f'Video encoder unavailable: {path}')
            writers[name] = w
        iterator = iter(samples(journal))
        pending = next(iterator,None)
        latest = None
        while pending is not None and pending[0]['completed_monotonic_ns'] <= start:
            latest = pending
            pre_gate += 1
            pending = next(iterator,None)
        tick = 0
        while start + round(tick*1e9/fps) <= end:
            stamp = start + round(tick*1e9/fps)
            while pending is not None and pending[0]['completed_monotonic_ns'] <= stamp:
                latest = pending; used += 1
                pending = next(iterator,None)
            panels = []
            for name in NAMES:
                panel = np.zeros((height,width,3),np.uint8)
                if latest is None:
                    label = f'{name} NO POLICY INPUT'
                else:
                    meta, frames = latest
                    image = cv2.cvtColor(frames[name],cv2.COLOR_RGB2BGR)
                    h,w = image.shape[:2];scale = min(width/w,height/h)
                    image = cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))))
                    rh,rw = image.shape[:2]
                    panel[(height-rh)//2:(height-rh)//2+rh,(width-rw)//2:(width-rw)//2+rw] = image
                    held = (stamp-meta['completed_monotonic_ns'])/1e9
                    label = f'{name} policy input HELD {held:.2f}s'
                    if name == 'interaction':
                        label += ' '+meta['roi'].get('mode','')
                cv2.putText(panel,label,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
                cv2.putText(panel,f'record +{(stamp-start)/1e9:.2f}s',(8,height-10),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
                writers[name].write(panel);panels.append(panel)
            grid = np.concatenate([np.concatenate(panels[:2],axis=1),np.concatenate(panels[2:],axis=1)],axis=0)
            writers['four_cameras'].write(grid)
            count += 1;tick += 1
        if latest is None:raise ValueError('No actual policy input within recording window')
        return dict(status='COMPLETE',directory=str(output),frames=count,fps=fps,
                    input_samples_in_window=used,pre_gate_samples=pre_gate,
                    initial_sample='latest_completed_before_gate' if pre_gate else 'blank_until_first_response',
                    time_axis='recording_gate_monotonic',cadence='actual_inference_samples_held_for_playback',
                    synthetic_new_roi_frames=False,roi_recomputed=False,
                    panel_order=list(NAMES),layout='2x2',journal=status)
    finally:
        for writer in writers.values():writer.release()
