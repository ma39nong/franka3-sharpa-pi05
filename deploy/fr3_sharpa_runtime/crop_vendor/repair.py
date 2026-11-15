"""Conservative current-frame repair: retain strong original detections by role."""
import json
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

    def predict(self,images):
        primary = self.primary.predict(images)
        requested = [i for i,detections in enumerate(primary)
                     if any(max((d['score'] for d in detections if d['label']==role),default=0)<.75 for role in (1,2))]
        repairs = dict(zip(requested,self.repair.predict([images[i] for i in requested]),strict=True)) if requested else {}
        return [merge_detections(detections,repairs.get(i,[])) for i,detections in enumerate(primary)]
