"""Real recorded RGB/state + real causal locator; no ROS or robot publishers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
from .dataset_sample import canonical_positions
from .interaction_crop import InteractionCropper
from .shared_roi import SharedROIWriter,read_history


def run(dataset,episode,out,norm_stats):
    import av
    import pyarrow.parquet as pq
    from . import config
    if out.exists():raise FileExistsError('Preserve previous replay')
    out.mkdir(parents=True);info=json.loads((dataset/'meta/info.json').read_text());fmt=dict(episode_index=episode,episode_chunk=episode//1000);table=pq.read_table(dataset/info['data_path'].format(**fmt));names=info['features']['observation.state']['names'];states=table['observation.state'].to_pylist();video=dataset/info['video_path'].format(**fmt,video_key='observation.images.cam0');cropper=InteractionCropper.from_environment(norm_stats);writer=SharedROIWriter(out/'shared');cropper.shared_writer=writer;samples=[];latency=[];expected=[]
    try:
        with av.open(str(video)) as stream:
            for i,frame in enumerate(stream.decode(video=0)):
                if i<300 or (i-300)%3:continue
                rgb=frame.to_ndarray(format='rgb24');stamp=round((i+1)/30*1e9);stamps={'cam0':stamp};obs={'observation/image':rgb,'observation/state':np.asarray(canonical_positions(states[i],names),np.float32),'_crop_context':dict(source_monotonic_ns=stamp,session='recorded-replay-only',topic_stamp_ns=stamps)}
                t=time.perf_counter();result=cropper.apply(obs);latency.append((time.perf_counter()-t)*1000);writer.pending.join();samples.append((stamps,rgb));expected.append(result['observation/interaction_image'].copy())
                if len(samples)==16:break
    finally:writer.close()
    rows,images=zip(*samples);args=dict(native_rgb=list(images),now_unix=rows[-1]['cam0']/1e9+.04)
    # Two consumers receive the exact same pixels. VLA computation is NOT mocked
    # into an actual inference timing result; this checks shared observation I/O.
    with ThreadPoolExecutor(max_workers=2) as pool:
        got=list(pool.map(lambda _:read_history(out/'shared',list(rows),**args),range(2)))
    for pixels,meta,identity in got:np.testing.assert_array_equal(pixels,np.stack(expected));assert meta.shape==(116,)
    report=dict(scope='Recorded training episode, real locator and measured state; offline replay clock; two concurrent observation consumers; NO VLA/Memory model compute, ROS, hardware or accuracy evaluation',episode=episode,dataset=str(dataset),source_frames=list(range(300,346,3)),samples=16,pixel_parity=True,source_timestamp_parity=True,locator_sha256=cropper.locator_sha256,roi_revision=cropper.revision,counts=dict(cropper.counts),producer_compute_ms=dict(first=latency[0],warm_p50=float(np.median(latency[1:])),warm_p95=float(np.percentile(latency[1:],95))),hardware_commands=0)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--episode',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--norm-stats',type=Path,required=True);a=p.parse_args();run(a.dataset,a.episode,a.out,a.norm_stats)
