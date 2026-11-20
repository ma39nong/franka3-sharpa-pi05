"""Causal counterpart of training's full-block-right-truss-180-v1 ROI.

Anchor/context rules match scripts.tomato_full_block_roi.interaction. Window
geometry uses the same 180..300 native-pixel bounds and 1.12 context at 640x400.
Offline future interpolation is replaced by current detections / bounded past
tracking. Missing right hand falls back to this frame's full RGB, never left-only.
"""
import cv2
import numpy as np

from .crop_selector import gap


def union(boxes):
    b = np.asarray(boxes, float)
    return np.r_[b[:, :2].min(0), b[:, 2:].max(0)]


class InHandInteractionSelector:
    def __init__(self, state_scale=None):
        self.roles = [1]

    def select(self, detections, state, width, height, rgb=None):
        # Run exactly the training extent rule at its native reference size.
        scale = np.array([640 / width, 400 / height] * 2)
        best = {}
        for d in detections:
            b = np.asarray(d['box'], float)
            label, score = d['label'], d['score']
            if (label not in (1, 2, 3) or not np.isfinite(score)
                    or not (.85 if label == 3 else .75) <= score <= 1
                    or b.shape != (4,) or not np.isfinite(b).all()
                    or not (0 <= b[0] < b[2] <= width and 0 <= b[1] < b[3] <= height)):
                continue
            b = b * scale
            if np.prod(b[2:] - b[:2]) > .13 * 640 * 400 or min(b[2:] - b[:2]) < 10:
                continue
            if label not in best or score > best[label][0]:
                best[label] = (score, b)
        if 1 not in best:
            return dict(valid=False, interaction_box=None, roles=[1], reason='right_truss_anchor_missing')
        image = cv2.resize(rgb, (640, 400), interpolation=cv2.INTER_AREA) if (width,height) != (640,400) else rgb
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = (((hsv[:,:,0] < 13) | (hsv[:,:,0] > 169)) & (hsv[:,:,1] > 90) & (hsv[:,:,2] > 60)).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        candidates = [np.array([x,y,x+w,y+h],float) for x,y,w,h,area in stats[1:]
                      if 8 <= area <= 3000 and min(w,h) >= 3]
        boxes = [best[1][1]]
        active = boxes[0].copy()
        for _ in range(4):
            retained, changed = [], False
            for b in candidates:
                merged = union([active, b])
                if gap(active,b) <= (35 if len(boxes) == 1 else 15) and max(merged[2:] - merged[:2]) <= 240:
                    boxes.append(b); active = merged; changed = True
                else:
                    retained.append(b)
            candidates = retained
            if not changed:
                break
        if 3 in best:
            obj = best[3][1]
            if gap(active,obj) < 30 and max(union([active,obj])[2:] - union([active,obj])[:2]) <= 240:
                boxes.append(obj)
        tight = union(boxes) + np.array([-12,-18,12,18])
        tight[:2] = np.maximum(tight[:2], 0)
        tight[2:] = np.minimum(tight[2:], [640,400])
        return dict(valid=True, interaction_box=(tight/scale).tolist(), roles=[1], reason=None)


class InHandCropWindow:
    """Contain current evidence; gently contract using source time, no lookahead."""
    def __init__(self):
        self.box = self.stamp = None

    def select(self, tight, width, height, stamp, uncertain=False):
        tight = np.asarray(tight, float)
        unit = height / 400
        minimum, maximum = 180 * unit, min(300 * unit, width, height)
        extent = max(tight[2:] - tight[:2])
        if minimum > maximum or extent > maximum - 2:
            self.box = self.stamp = None
            return None
        side = min(maximum, np.ceil(max(minimum, 1.12 * extent, extent + 2)))
        goal = (tight[:2] + tight[2:]) / 2
        if self.box is not None and stamp is not None and self.stamp is not None:
            dt = min(.75, max(0., (stamp-self.stamp)/1e9))
            old = np.asarray(self.box, float)
            side = min(maximum, np.ceil(max(side, (old[2]-old[0])*np.exp(-.24*dt))))
            center = (old[:2] + old[2:]) / 2
            center += (goal-center)*(1-np.exp(-4*dt))
        else:
            center = goal
        # Current containment has priority over smoothing; training could pan
        # before a transition using future boxes, online operation cannot.
        low = np.maximum(0, np.ceil(tight[2:]-side))
        high = np.minimum(np.array([width,height])-side, np.floor(tight[:2]))
        origin = np.minimum(high, np.maximum(low, np.floor(center-side/2))).astype(int)
        result = tuple(int(v) for v in np.r_[origin, origin+side])
        self.box, self.stamp = result, stamp
        return result
