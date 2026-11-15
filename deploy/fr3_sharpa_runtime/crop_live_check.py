"""Read existing live camera/state streams and infer without sending robot actions.

Cameras and measured feedback must already be running on the workstation.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .runtime import SharpaRuntime
from .task_stages import PHASES,stage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ros-overlay',type=Path,required=True)
    parser.add_argument('--stage',choices=PHASES,default='pickup_1')
    parser.add_argument('--samples',type=int,default=20)
    parser.add_argument('--hz',type=float,default=2.)
    parser.add_argument('--policy-url')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if not 1<=args.samples<=1000 or not 0<args.hz<=10:
        parser.error('samples1..1000 and Hz0..10 required')
    from . import run
    runtime=SharpaRuntime(**({'policy_url':args.policy_url} if args.policy_url else {}))
    metadata=runtime.inspect_metadata(require_rtc=True)
    if not metadata.get('interaction_crop',{}).get('enabled'):
        raise ValueError('Start the four-view crop service first')
    rows=[]
    try:
        for i in range(args.samples):
            start=time.monotonic()
            obs=run.live_observation(args.ros_overlay,stage(args.stage).prompt)
            _,actions,_=runtime.infer_chunk(obs)
            age_ms=(time.monotonic_ns()-obs['_captured_monotonic_ns'])/1e6
            row=dict(sample=i,snapshot_to_action_ms=age_ms,fresh_within_200ms=0<=age_ms<=200,
                     actions_finite=bool(np.isfinite(actions).all()),timing=runtime.last_timing)
            rows.append(row)
            print(json.dumps(row),flush=True)
            remaining=1/args.hz-(time.monotonic()-start)
            if remaining>0:time.sleep(remaining)
    finally:
        if run._observation_worker is not None:
            run._observation_worker.close()
    report=dict(no_hardware_commands=True,stage=args.stage,samples=len(rows),
                fresh_samples=sum(r['fresh_within_200ms'] for r in rows),
                snapshot_to_action_ms={name:float(np.percentile([r['snapshot_to_action_ms'] for r in rows],q))
                                       for name,q in [('p50',50),('p95',95),('max',100)]},rows=rows)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
