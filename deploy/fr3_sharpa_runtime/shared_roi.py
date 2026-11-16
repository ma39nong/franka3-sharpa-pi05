"""Bounded nonblocking ROI publication; consumers require exact source stamps.

This is an observation channel, never a stage/robot command channel. Missing10Hz
samples are rejected rather than synthesized from a lower-rate policy stream.
"""
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time
import uuid
import numpy as np

PROTOCOL='tomato-shared-roi-v1'


def rgb_sha(image):return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


class SharedROIWriter:
    def __init__(self,directory,*,keep=64):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True);self.keep=keep;self.pending=queue.Queue(maxsize=1);self.closed=False;self.dropped=0;self.error=None;self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()

    def submit(self,observation,crop,status):
        context=observation.get('_crop_context',{});stamps=context.get('topic_stamp_ns',{})
        if type(stamps.get('cam0')) is not int or not context.get('session'):return False
        # Copy pixels now: later preprocessing must not mutate this observation.
        item=(dict(protocol=PROTOCOL,session_id=context['session'],source_ros_ns=stamps['cam0'],topic_stamp_ns=dict(stamps),source_rgb_sha256=rgb_sha(observation['observation/image']),roi_revision=status['roi_revision'],roi_contract='live-causal-mixed-hand-v2',uses_future_frames=False,local_roi_valid=status['local_roi_valid'],crop_box=status['crop_box'],source_size=status['source_size'],created_unix=time.time()),np.asarray(crop).copy())
        if self.closed:return False
        try:self.pending.put_nowait(item)
        except queue.Full:
            try:self.pending.get_nowait();self.pending.task_done();self.dropped+=1
            except queue.Empty:pass
            try:self.pending.put_nowait(item)
            except queue.Full:self.dropped+=1;return False
        return True

    def _run(self):
        while True:
            item=self.pending.get()
            try:
                if item is None:return
                meta,crop=item;meta['roi_pixel_sha256']=rgb_sha(crop);key=hashlib.sha256(f"{meta['session_id']}:{meta['source_ros_ns']}".encode()).hexdigest();path=self.directory/(key+'.npz');tmp=self.directory/(uuid.uuid4().hex+'.tmp.npz')
                np.savez(tmp,roi=crop,metadata=np.array(json.dumps(meta)));os.replace(tmp,path)
                latest=self.directory/(uuid.uuid4().hex+'.tmp');latest.write_text(json.dumps(dict(path=path.name,session_id=meta['session_id'],protocol=PROTOCOL)));os.replace(latest,self.directory/'latest.json')
                files=sorted(self.directory.glob('*.npz'),key=lambda p:p.stat().st_mtime_ns)
                for old in files[:-self.keep]:old.unlink(missing_ok=True)
            except Exception as e:self.error=str(e)
            finally:self.pending.task_done()

    def close(self):
        self.closed=True;self.pending.join();self.pending.put(None);self.thread.join(timeout=2)


def read_history(directory,rows,*,native_rgb,now_unix,max_age_s=2.,anchor_ns=None):
    """Return16 ROI frames +116 metadata only for exactly matching ROS samples."""
    directory=Path(directory);latest=json.loads((directory/'latest.json').read_text());session=latest['session_id'];bank={}
    if latest.get('protocol')!=PROTOCOL:raise ValueError('Shared ROI protocol mismatch')
    for p in directory.glob('*.npz'):
        with np.load(p,allow_pickle=False) as z:
            m=json.loads(str(z['metadata']));image=z['roi'].copy()
        if m['session_id']==session:bank[m['source_ros_ns']]=(m,image)
    if len(rows)!=16 or len(native_rgb)!=16:raise ValueError('Need16 real aligned ROI history samples')
    meta=[];images=[];times=[r['cam0'] for r in rows];anchor=times[-1] if anchor_ns is None else anchor_ns
    if not 0<=now_unix-anchor/1e9<=max_age_s:raise ValueError('ROI history stale or future')
    for stamps,rgb in zip(rows,native_rgb):
        if stamps['cam0'] not in bank:raise ValueError('Shared ROI history incomplete; need fresh10Hz producer, never repeat2Hz ROI')
        m,im=bank[stamps['cam0']]
        if m['topic_stamp_ns']!=stamps or m['source_rgb_sha256']!=rgb_sha(rgb):raise ValueError('ROI/raw observation mismatch')
        if m.get('protocol')!=PROTOCOL or m.get('uses_future_frames') or m['roi_contract']!='live-causal-mixed-hand-v2' or im.shape!=(224,224,3) or im.dtype!=np.uint8 or rgb_sha(im)!=m['roi_pixel_sha256']:raise ValueError('Invalid ROI source contract/pixels')
        images.append(im);meta.append(m)
    if any(t>anchor for t in times) or any(abs((b-a)-100_000_000)>40_000_000 for a,b in zip(times,times[1:])):raise ValueError('Noncausal or irregular ROI history')
    if len({m['roi_revision'] for m in meta})!=1:raise ValueError('Mixed ROI revisions')
    boxes=[]
    for m in meta:
        w,h=m['source_size'];boxes.extend(np.asarray(m['crop_box'] if m['local_roi_valid'] else [0,0,w,h],float)/[w,h,w,h])
    offsets=(np.array(times)-anchor)/1e9
    state_offsets=np.array([(max([v for k,v in r.items() if not k.startswith('cam')] or [r['cam0']])-r['cam0'])/1e9 for r in rows])
    if np.any(offsets+state_offsets>1e-6):raise ValueError('Future joint history')
    features=np.r_[boxes,[m['local_roi_valid'] for m in meta],offsets,state_offsets,0.,1.,0.,0.].astype('float32')
    return np.stack(images),features,dict(protocol=PROTOCOL,session_id=session,roi_contract='live-causal-mixed-hand-v2',source_ros_ns=times,uses_future_frames=False)
