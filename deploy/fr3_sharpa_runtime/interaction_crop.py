"""Causal current-frame ROI. Independent locator; no learned FoveaVLA box head.

The r11 action model expects four available RGB views. A tracked crop or current
full-view fallback stays available; local validity is recorded separately.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from .crop_vendor.geometry import native_crop, square_box
from .crop_vendor.track import CausalBoxTrack

LOCATOR_SHA256 = '8b06ab9aa07e513e73d305b7c2f3ca5d060ee0a05ed5ab08a822df144a1d89fb'
PRIMARY_SHA256 = '19069cbd6fb9f1763798585685e3606bbe57e05f2a99ffdecd015e31bc04c305'


def locator_identity(root):
    path = Path(root).expanduser().resolve()
    weight = path / 'model.pt'
    sha = hashlib.sha256(weight.read_bytes()).hexdigest()
    if sha != LOCATOR_SHA256:
        raise ValueError('Locator weights differ from the reviewed hard-case repair locator; select its model.pt directory')
    if not (path / 'receipt.json').is_file():
        raise ValueError('Locator receipt.json required beside model.pt')
    if hashlib.sha256((path/'primary/model.pt').read_bytes()).hexdigest() != PRIMARY_SHA256:
        raise ValueError('Original locator dependency missing or changed in primary/')
    return str(path), sha


class InteractionCropper:
    def __init__(self, locator, state_scale, *, status_dir=None):
        from .crop_selector import CurrentInteractionSelector
        self.locator = locator
        self.state_scale = state_scale
        self.selector_type = CurrentInteractionSelector
        self.status_dir = Path(status_dir) if status_dir else None
        self.counts = Counter()
        self.last = {}
        self.reset()

    def reset(self):
        self.selector = self.selector_type(self.state_scale)
        self.track = None
        self.previous_stamp = None
        self.previous_shape = None
        self.previous_session = None
        self.previous_arrival = None
        self.track_seed_stamp = None
        self.track_roles = None
        self.frame = 0

    @classmethod
    def from_environment(cls, norm_stats):
        from .crop_vendor.repair import RgbLocator
        root, sha = locator_identity(os.environ.get('SHARPA_CROP_LOCATOR', ''))
        import torch
        torch.set_num_threads(4)
        cv2.setNumThreads(1)
        locator = RgbLocator(root, device=os.environ.get('SHARPA_CROP_DEVICE', 'cuda'))
        stats = json.loads(Path(norm_stats).read_text())['norm_stats']['state']
        instance = cls(locator, np.asarray(stats['q99'])-np.asarray(stats['q01']),
                       status_dir=os.environ.get('SHARPA_CROP_STATUS_DIR'))
        instance.locator_sha256 = sha
        return instance

    def apply(self, observation):
        start = time.perf_counter()
        image = np.asarray(observation['observation/image'])
        state = np.asarray(observation['observation/state'])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 32:
            raise ValueError('ROI requires native HWC uint8 RGB camera input')
        if state.shape != (58,) or not np.isfinite(state).all():
            raise ValueError('ROI requires measured canonical58 state')
        context = observation.get('_crop_context', {})
        stamp = context.get('source_monotonic_ns')
        session = context.get('session')
        now = time.monotonic()
        reset = (self.previous_shape != image.shape or self.previous_session != session
                 or self.previous_arrival is None or now-self.previous_arrival > 0.75
                 or stamp is None or self.previous_stamp is None
                 or stamp <= self.previous_stamp or stamp-self.previous_stamp > 750_000_000)
        if reset:
            self.reset()
        self.previous_stamp, self.previous_shape = stamp, image.shape
        self.previous_session, self.previous_arrival = session, now
        self.frame += 1
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        detector_start = time.perf_counter()
        detections = self.locator.predict([image])[0]  # Batch=1, current RGB only.
        detector_ms = (time.perf_counter()-detector_start)*1000
        roi = self.selector.select(detections, state, width, height, rgb=image)
        mode = 'detected_local'
        if roi['valid']:
            tight = np.asarray(roi['interaction_box'])
            # r11 windows retain at least 55% of source image height as context.
            scale = max(1.5, .55*height/max(tight[2:]-tight[:2]))
            box = square_box(tight, width, height, scale=scale)
            self.track = CausalBoxTrack(gray, roi['interaction_box'], self.frame, max_age=8)
            self.track_seed_stamp = stamp
            self.track_roles = tuple(roi["roles"])
        else:
            same_role = self.track_roles == tuple(roi['roles'])
            tracked = self.track.update(gray, self.frame) if self.track and same_role else None
            track_fresh = (stamp is not None and self.track_seed_stamp is not None
                           and 0 <= stamp-self.track_seed_stamp <= 750_000_000)
            if tracked and tracked['valid'] and track_fresh:
                tight = np.asarray(tracked['box'])
                box = square_box(tight, width, height, scale=max(1.5,.55*height/max(tight[2:]-tight[:2])))
                mode = 'tracked_local'
            else:
                box = None
                mode = 'current_full_fallback'
                self.track = None
        pixels = native_crop(image, box) if box is not None else image
        if box is not None:
            crop = cv2.resize(pixels, (224, 224), interpolation=cv2.INTER_AREA)
        else:
            # Preserve aspect ratio, matching the full-view input convention.
            ratio = 224/max(width,height)
            rw,rh = round(width*ratio),round(height*ratio)
            crop = np.zeros((224,224,3),np.uint8)
            x,y = (224-rw)//2,(224-rh)//2
            crop[y:y+rh,x:x+rw] = cv2.resize(image,(rw,rh),interpolation=cv2.INTER_AREA)
        self.counts[mode] += 1
        self.last = dict(mode=mode, local_roi_valid=box is not None, image_available=True,
                         source_camera='cam0', source_size=[width,height], crop_box=box,
                         source_monotonic_ns=stamp, uses_future_frames=False,
                         reason=roi['reason'], roles=roi['roles'], reset=reset, detector_ms=detector_ms,
                         crop_total_ms=(time.perf_counter()-start)*1000,
                         counts=dict(self.counts), updated_wall_time=time.time(),
                         locator_sha256=getattr(self, 'locator_sha256', None),
                         training_roi='offline_bidirectional_r11', independent_semantic_accuracy_measured=False)
        if self.status_dir:
            self.status_dir.mkdir(parents=True, exist_ok=True)
            preview = cv2.resize(image, (448, 252))
            if box is not None:
                x1,y1,x2,y2 = box
                cv2.rectangle(preview,(int(x1*448/width),int(y1*252/height)),
                              (int(x2*448/width),int(y2*252/height)),(0,220,190),2)
            canvas = np.zeros((252,672,3),np.uint8)
            canvas[:,:448] = preview
            canvas[14:238,448:] = crop
            encoded = cv2.imencode('.jpg',cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))[1]
            temporary = self.status_dir/'preview.tmp'
            temporary.write_bytes(encoded.tobytes());temporary.replace(self.status_dir/'preview.jpg')
            temporary = self.status_dir/'status.tmp'
            temporary.write_text(json.dumps(self.last));temporary.replace(self.status_dir/'status.json')
        return {**observation, 'observation/interaction_image':crop,
                'observation/interaction_valid':np.bool_(True)}


def attach_cropper(policy, cropper):
    transform = policy._input_transform
    policy._input_transform = lambda observation: transform(cropper.apply(observation))
    policy.interaction_cropper = cropper
