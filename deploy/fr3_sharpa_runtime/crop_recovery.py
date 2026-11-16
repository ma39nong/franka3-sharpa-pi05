"""Past-only, role-specific recovery using flow and current-pixel re-detection.

Detection confidence thresholds are shared with the selector. A flow estimate
never refreshes a detection deadline. Reacquisition requires new RGB evidence.
"""
import numpy as np

from .crop_vendor.geometry import square_box, native_crop
from .crop_vendor.track import CausalBoxTrack


def confident_hands(detections, width, height):
    best = {}
    for detection in detections:
        role, score = detection['label'], detection['score']
        box = np.asarray(detection['box'], float)
        if (role not in (1, 2) or not np.isfinite(score) or not .75 <= score <= 1
                or box.shape != (4,) or not np.isfinite(box).all()):
            continue
        if (not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)
                or min(box[2:] - box[:2]) < 15
                or np.prod(box[2:] - box[:2]) > width * height * .13):
            continue
        if role not in best or score > best[role]['score']:
            best[role] = dict(label=role, score=float(score), box=box.tolist())
    return best


class CausalHandRecovery:
    flow_seconds = .75
    search_seconds = 1.5

    def __init__(self):
        self.hands = {}

    @staticmethod
    def unambiguous(candidates, role, width, height):
        strong = [d for d in candidates if role in confident_hands([d], width, height)]
        found = confident_hands(strong, width, height)
        if role not in found:
            return None, False
        chosen = np.asarray(found[role]['box'])
        for other in strong:
            b = np.asarray(other['box'])
            intersection = np.prod(np.maximum(0, np.minimum(b[2:], chosen[2:]) - np.maximum(b[:2], chosen[:2])))
            if intersection < .5 * min(np.prod(b[2:]-b[:2]), np.prod(chosen[2:]-chosen[:2])):
                return None, True
        return found[role], False

    def recover(self, image, gray, detections, stamp, frame, locator, required_roles):
        height, width = gray.shape
        current = confident_hands(detections, width, height)
        sources = {role: 'detected' for role in current}
        attempts = []
        # Flow is advanced separately for each hand, including a hand temporarily
        # outside the selected interaction. No union-box identity switch.
        for role, history in list(self.hands.items()):
            if role in current:
                continue
            age = (stamp - history['stamp']) / 1e9 if stamp is not None else float('inf')
            if 0 < age <= self.flow_seconds and history['track'] is not None:
                tracked = history['track'].update(gray, frame)
                if tracked['valid']:
                    found = confident_hands([dict(label=role, score=.75, box=tracked['box'])], width, height)
                    if role in found:
                        current[role] = found[role]
                        sources[role] = 'tracked'
                        history['box'] = tracked['box']
            # Prefer actual current-frame evidence over tracking, but only incur
            # extra detector work for a role needed by the current interaction.
            if role not in required_roles or not 0 < age <= self.search_seconds:
                continue
            region = square_box(history['box'], width, height, scale=2.0 + age)
            pixels = native_crop(image, region)
            candidates = locator.predict([pixels])[0]
            mapped = []
            for candidate in candidates:
                if candidate['label'] != role:
                    continue
                b = np.asarray(candidate['box'], float)
                if b.shape != (4,):
                    continue
                b = b + np.array([region[0], region[1], region[0], region[1]])
                mapped.append({**candidate, 'box': b.tolist()})
            # A crop is a spatial prior, not a new identity oracle. Refuse
            # ambiguous same-role candidates with disjoint high-confidence boxes.
            found, ambiguous = self.unambiguous(mapped, role, width, height)
            if found is not None:
                current[role] = found
                sources[role] = 'reacquired'
            attempts.append(dict(role=role, search_box=list(region), accepted=found is not None,
                                 ambiguous=ambiguous))
        missing = [role for role in required_roles if sources.get(role) not in ('detected', 'reacquired')]
        if missing and hasattr(locator, 'predict_detail'):
            # The global model normally downsizes cam0 to 320px. Re-run at
            # 480px to recover small/occluded hands without a stale spatial prior.
            proposals = locator.predict_detail([image])[0]
            for role in missing:
                found, ambiguous = self.unambiguous(proposals, role, width, height)
                if found is not None:
                    current[role] = found
                    sources[role] = 'reacquired'
                attempts.append(dict(role=role, accepted=found is not None,
                                     ambiguous=ambiguous, method='current_detail_480'))
        for role, detection in current.items():
            if sources[role] in ('detected', 'reacquired'):
                self.hands[role] = dict(box=detection['box'], stamp=stamp,
                                       track=CausalBoxTrack(gray, detection['box'], frame, max_age=30))
        # Expired history cannot seed another recovery on later frames.
        self.hands = {role: history for role, history in self.hands.items()
                      if stamp is not None and history['stamp'] is not None
                      and 0 <= stamp-history['stamp'] <= self.search_seconds*1e9}
        return list(current.values()), sources, attempts


class CausalCropWindow:
    """Expand immediately to contain evidence; ease inward using source time."""
    def __init__(self):
        self.box = None
        self.stamp = None

    def select(self, tight, width, height, stamp, uncertain=False):
        tight = np.asarray(tight, float)
        scale = max(1.5, .55*height/max(tight[2:]-tight[:2]))
        target = square_box(tight, width, height, scale=scale)
        if self.box is not None and stamp is not None and self.stamp is not None:
            dt = min(.75, max(0., (stamp-self.stamp)/1e9))
            old = np.asarray(self.box, float)
            side = max(target[2]-target[0], (old[2]-old[0])*np.exp(-.25*dt))
            center = (old[:2]+old[2:])/2
            goal = (np.asarray(target[:2])+target[2:])/2
            center += (goal-center)*(1-np.exp(-4*dt))
            if uncertain:
                side = max(side, min(min(width,height), (old[2]-old[0])*(1+.15*dt)))
            # Constrain the smoothed center to include this frame's entire tight
            # evidence. Containment takes priority over smoothing at transitions.
            side = min(max(side, max(tight[2:]-tight[:2])), max(width,height))
            center = np.maximum(tight[2:]-side/2, np.minimum(center, tight[:2]+side/2))
            side = int(np.ceil(side))
            side = max(side, int(np.max(np.ceil(tight[2:])-np.floor(tight[:2]))))
            low = np.maximum(0, np.ceil(tight[2:]-side))
            high = np.minimum(np.array([width,height])-side, np.floor(tight[:2]))
            # Match square_box padding for interactions wider than image height.
            if side > height:
                low[1], high[1] = height-side, 0
            origin = np.minimum(high, np.maximum(low, np.floor(center-side/2))).astype(int)
            target = tuple(int(v) for v in np.r_[origin, origin+side])
        self.box, self.stamp = target, stamp
        return target
