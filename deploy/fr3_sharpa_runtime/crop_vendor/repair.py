"""Conservative current-frame repair: retain strong original detections by role."""
import json
import time
from pathlib import Path

import hashlib

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
from .locator import RgbLocator as OriginalLocator


def merge_detections(primary, repair):
    result = []
    for role in (1,2,3):
        original = [d for d in primary if d['label']==role]
        additions = [d for d in repair if d['label']==role and d['score']>=.85]
        # The repair model is only used for roles without a strong current
        # original-model detection. No low-score threshold relaxation.
        if max((d['score'] for d in original),default=0)>=.75 or not additions:
            result.extend(original)
        else:
            result.extend(additions)
    return result


class RgbLocator:
    def __init__(self,root,device='cuda'):
        root = Path(root)
        receipt = json.loads((root/'receipt.json').read_text())
        parent = root / 'primary'
        if digest(parent/'model.pt')!=receipt['parent_model_sha256']:
            raise ValueError('Changed original-model repair dependency')
        self.primary = OriginalLocator(parent,device)
        self.repair = OriginalLocator(root,device)
        self.last_timing = {}

    def predict(self,images):
        return self._predict(images)

    def predict_detail(self,images):
        """Reacquire from current RGB at 480px; preserve scores and labels."""
        return self._predict(images,min_size=480)

    def _predict(self,images,*,min_size=None):
        options={} if min_size is None else dict(min_size=min_size)
        start = time.perf_counter()
        primary = self.primary.predict(images,**options)
        primary_end = time.perf_counter()
        requested = [i for i,detections in enumerate(primary)
                     if any(max((d['score'] for d in detections if d['label']==role),default=0)<.75 for role in (1,2))]
        repair_start = time.perf_counter()
        repairs = dict(zip(requested,self.repair.predict([images[i] for i in requested],**options),strict=True)) if requested else {}
        repair_end = time.perf_counter()
        merged = [merge_detections(detections,repairs.get(i,[])) for i,detections in enumerate(primary)]
        self.last_timing = dict(primary_ms=(primary_end-start)*1000,
                                repair_ms=(repair_end-repair_start)*1000 if requested else 0.,
                                repair_images=len(requested), total_ms=(time.perf_counter()-start)*1000)
        return merged
