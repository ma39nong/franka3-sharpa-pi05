"""Bounded asynchronous, lossless journal of actual four-view policy inputs.

One sample per successful inference response, not a synthetic 30Hz camera.
Recording failure is reported without blocking the robot control deadline.
"""
import atexit
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time
import uuid

import numpy as np

KEYS = ('observation/image', 'observation/left_wrist_image', 'observation/right_wrist_image')
NAMES = ('head', 'left_wrist', 'right_wrist', 'interaction')


def pixel_sha(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


_recordings = {}
_recording_lock = threading.Lock()


def recording_for(directory):
    # Metadata probes and serial-mode clients may create multiple Runtime
    # objects; one execution directory must have exactly one journal owner.
    key = str(Path(directory).resolve())
    with _recording_lock:
        if key not in _recordings:
            _recordings[key] = FourViewRecording(key)
        return _recordings[key]


class FourViewRecording:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.pending = queue.Queue(maxsize=16)
        self.accepted = self.written = self.dropped = 0
        self.error = None
        self.closed = False
        self.thread = threading.Thread(target=self._run, daemon=True, name='four-view-input-recorder')
        self.thread.start()
        atexit.register(self.close)

    def persist(self):
        data = dict(status='FAILED' if self.error else 'INCOMPLETE' if self.dropped else 'COMPLETE' if self.closed else 'RECORDING',
                    accepted=self.accepted, written=self.written, dropped=self.dropped, error=self.error,
                    stream='actual_policy_inputs', cadence='one_per_inference_response', pixel_encoding='lossless_uint8_rgb_npz')
        tmp = self.directory / 'status.tmp'
        tmp.write_text(json.dumps(data)); os.replace(tmp, self.directory/'status.json')

    def submit(self, observation, payload, *, query_id, task=None):
        try:
            pixels = np.asarray(payload['pixels'])
            metadata = payload['metadata']
            if (pixels.shape != (224,224,3) or pixels.dtype != np.uint8
                    or pixel_sha(pixels) != metadata['roi_pixel_sha256']
                    or metadata['source_monotonic_ns'] != observation.get('_captured_monotonic_ns')):
                raise ValueError('ROI recording pixels/source do not match this policy request')
            frames = [np.asarray(observation[k]) for k in KEYS] + [pixels]
            if any(x.ndim != 3 or x.shape[-1] != 3 or x.dtype != np.uint8 for x in frames):
                raise ValueError('Four-view recording requires RGB uint8')
            meta = dict(query_id=query_id, prompt=observation['prompt'], task=task,
                        captured_monotonic_ns=observation.get('_captured_monotonic_ns'),
                        completed_monotonic_ns=time.monotonic_ns(), topic_stamp_ns=observation.get('_topic_stamp_ns', {}),
                        roi=metadata, pixel_sha256={n:pixel_sha(x) for n,x in zip(NAMES,frames)})
            self.pending.put_nowait((meta, [x.copy() for x in frames]))
            self.accepted += 1
        except queue.Full:
            self.dropped += 1
        except Exception as error:
            self.error = str(error)

    def _run(self):
        while True:
            item = self.pending.get()
            try:
                if item is None:
                    return
                meta, frames = item
                name = f"{meta['completed_monotonic_ns']}-{uuid.uuid4().hex}.npz"
                tmp = self.directory / (name+'.tmp')
                with tmp.open('wb') as stream:
                    np.savez(stream, metadata=np.array(json.dumps(meta)), **dict(zip(NAMES,frames)))
                os.replace(tmp, self.directory/name)
                self.written += 1
                self.persist()
            except Exception as error:
                self.error = str(error)
            finally:
                self.pending.task_done()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.pending.join()
        self.pending.put(None)
        self.thread.join()
        self.persist()
