"""Run only the left GELLO serial reader in the existing teleop Python env."""
import json
import os
from pathlib import Path
import signal
import sys
import time
import uuid

import numpy as np


def main():
    def interrupt(_number,_frame):
        raise KeyboardInterrupt('GELLO input shutdown')
    signal.signal(signal.SIGTERM,interrupt)
    protocol=sys.stdout
    sys.stdout=sys.stderr  # Vendor diagnostics must never corrupt the JSON pipe.
    repo=Path(os.environ['SHARPA_GELLO_REPO'])
    for path in ('adapters/pico/src','portable_deps/gello_software',
                 'portable_deps/gello_software/third_party/DynamixelSDK/python/src'):
        sys.path.insert(0,str(repo/path))
    from pico_bimanual_franka_teleop.gello_input import _SideReader, DynamixelJointReader, load_gello_config
    config=load_gello_config(repo/'config/modes/gello.yaml')
    side=config.left
    reader=_SideReader('left',side,baudrate=config.baudrate,joint_ids=config.joint_ids,
        signs=np.asarray(config.standard_signs)*np.asarray(side.direction_correction),
        max_joint_jump=config.max_joint_jump,stale_timeout=config.stale_timeout,
        reader_factory=DynamixelJointReader)
    session='dagger-gello-'+uuid.uuid4().hex
    previous=None
    sequence=0
    try:
        while True:
            values,stamp,error,_=reader.snapshot()
            if values is not None and not error and stamp!=previous and 0<=time.monotonic()-stamp<=config.stale_timeout:
                previous=stamp
                print(json.dumps(dict(session=session,sequence=sequence,source_monotonic_s=stamp,
                                      positions=values.tolist()),allow_nan=False),file=protocol,flush=True)
                sequence+=1
            time.sleep(.01)
    finally:
        reader.close()


if __name__=='__main__':
    main()
