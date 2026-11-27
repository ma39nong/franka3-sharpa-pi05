"""Prepare and run the supplied read-only S1 checks on one closed real episode."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml
from .mcap_storage import digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode',type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--tools',type=Path,default=Path.home()/'.local/share/sharpa-conformance/recorder-conformance')
    args=p.parse_args();episode=args.episode.resolve();out=args.out.resolve()
    if out.is_relative_to(episode):p.error('Report directory must be outside the episode')
    out.mkdir(parents=True,exist_ok=False)
    info=json.loads((episode/'episode.json').read_text());shared=(episode/info['assets']['devices_ref']['path']).resolve().parent
    target=yaml.safe_load((args.tools/'templates/certification_target.yaml').read_text())
    target['target']['recorder']=info['recorder']
    target['target']['config_files']=[dict(path='devices.yaml',sha256=digest(shared/'devices.yaml'))]
    target['event_types']=dict(unavailable=['unavailable'],forced_keyframe=['forced_idr'],clock=['clock'],interrupted=['interrupted'],recovery=['recovery'])
    (out/'target.yaml').write_text(yaml.safe_dump(target))
    (out/'runs.yaml').write_text(yaml.safe_dump(dict(runs=[dict(run_id=info['episode_uid'],scenario='S1',episode_dir=str(episode),shared_dir=str(shared))])))
    raise SystemExit(subprocess.call([sys.executable,str(args.tools/'scripts/run_checks.py'),'check','--target',str(out/'target.yaml'),
        '--runs',str(out/'runs.yaml'),'--out',str(out/'report')],env=os.environ.copy()))


if __name__=='__main__':main()
