"""Replay measured states and native cam0 RGB in source order; never drives hardware.

Example: python -m deploy.fr3_sharpa_runtime.crop_replay --dataset DATASET \
  --episode 4 --out .deployment/crop_validation/episode48 --stride 15
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .interaction_crop import InteractionCropper
from .dataset_sample import canonical_positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=15)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int)
    args = parser.parse_args()
    if args.stride < 1 or args.start < 0:
        parser.error('positive stride and nonnegative start required')
    from . import config
    import av
    import pyarrow.parquet as pq
    info = json.loads((args.dataset/'meta/info.json').read_text())
    episode = args.episode
    data = args.dataset/info['data_path'].format(episode_index=episode, episode_chunk=episode//info['chunks_size'])
    table = pq.read_table(data)
    names = info['features']['observation.state']['names']
    states = [canonical_positions(row, names) for row in table['observation.state'].to_pylist()]
    video = args.dataset/info['video_path'].format(episode_index=episode, episode_chunk=episode//info['chunks_size'],
                                                  video_key='observation.images.cam0')
    cropper = InteractionCropper.from_environment(config.NORM_STATS)
    cropper.status_dir = args.out
    args.out.mkdir(parents=True, exist_ok=True)
    latencies, rows = [], []
    output = cv2.VideoWriter(str(args.out/'preview.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30/args.stride,(672,252))
    try:
        with av.open(str(video)) as src:
            for frame, item in enumerate(src.decode(video=0)):
                if args.end is not None and frame >= args.end:
                    break
                if frame < args.start or (frame-args.start)%args.stride:
                    continue
                obs = {'observation/image':item.to_ndarray(format='rgb24'), 'observation/state':states[frame],
                       '_crop_context':{'source_monotonic_ns':int((frame+1)/30*1e9),'session':'replay'}}
                start = time.perf_counter()
                result = cropper.apply(obs)
                elapsed = (time.perf_counter()-start)*1000
                row = {**cropper.last,'source_frame':frame,'end_to_end_ms':elapsed}
                rows.append(row)
                if len(rows)>5:
                    latencies.append(elapsed)
                preview = cv2.imread(str(args.out/'preview.jpg'))
                cv2.putText(preview,f'f{frame} {row["mode"]} {elapsed:.1f}ms',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                output.write(preview)
                if len(rows)%50==0:
                    print(f'{len(rows)} queries; frame={frame}; {elapsed:.1f}ms',flush=True)
    finally:
        output.release()
    if not latencies:
        raise ValueError('At least six replay samples required')
    summary = dict(dataset=str(args.dataset),episode=episode,frames=len(rows),stride=args.stride,
                   source_hz=30,query_hz=30/args.stride,counts=dict(Counter(r['mode'] for r in rows)),
                   warmed_total_ms={key:float(np.percentile(latencies,value)) for key,value in [('p50',50),('p95',95),('p99',99),('max',100)]},
                   cold_first_ms=rows[0]['end_to_end_ms'],locator_sha256=cropper.locator_sha256,
                   no_hardware=True,no_action_inference=True,no_future_frames=True,
                   independent_semantic_accuracy_measured=False)
    (args.out/'replay.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (args.out/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
