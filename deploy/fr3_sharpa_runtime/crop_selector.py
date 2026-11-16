"""Current interaction geometry for the old truss task, independent of phase labels.

Right truss anchors the view. Nearby left hand joins the interaction. Following
left transport requires measured finger curl plus RGB co-motion, never arm speed
alone. These are ROI cues, not a claim of successful grasp or detachment.
"""
import numpy as np
from .crop_vendor.geometry import square_box
from .protocol import LEFT_SHARPA_NAMES

MCP = [7+LEFT_SHARPA_NAMES.index(f'left_{finger}_MCP_FE') for finger in ('index','middle','ring','pinky')]


def gap(a,b):
    return float(np.linalg.norm(np.maximum(0,np.maximum(a[:2],b[:2])-np.minimum(a[2:],b[2:]))))


class CurrentInteractionSelector:
    def __init__(self, state_scale=None):
        self.previous_left = None
        self.previous_red = []
        self.roles = [1]

    def select(self,detections,state,width,height,rgb=None):
        import cv2
        state=np.asarray(state)
        if state.shape!=(58,) or not np.isfinite(state).all():
            raise ValueError('Finite canonical58 measured state required')
        from .crop_recovery import confident_hands
        best={role:(d['score'],np.asarray(d['box'],float))
              for role,d in confident_hands(detections,width,height).items()}
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        mask=(((hsv[:,:,0]<13)|(hsv[:,:,0]>169))&(hsv[:,:,1]>100)&(hsv[:,:,2]>65)).astype(np.uint8)
        _,_,components,_=cv2.connectedComponentsWithStats(mask,8)
        regions=[np.array([x,y,x+w,y+h],float) for x,y,w,h,area in components[1:] if 12<=area<=2500 and min(w,h)>=3]
        left=best.get(2,(None,None))[1]
        near_left=[] if left is None else [b for b in regions if gap(b,left)<.12*max(left[2:]-left[:2])]
        attached=False
        if left is not None and self.previous_left is not None and float(state[MCP].mean())>=.70:
            motion=(left[:2]+left[2:]-self.previous_left[:2]-self.previous_left[2:])/2
            for old in self.previous_red:
                for current in near_left:
                    shift=(current[:2]+current[2:]-old[:2]-old[2:])/2
                    if (np.linalg.norm(motion)>=2 and np.linalg.norm(shift)>=2
                            and np.linalg.norm(shift-motion)<max(5.,.12*max(left[2:]-left[:2]))):
                        attached=True
        close=False
        if 1 in best and 2 in best:
            a,b=best[1][1],best[2][1]
            close=gap(a,b)<(.85 if self.roles==[1,2] else .75)*max(*(a[2:]-a[:2]),*(b[2:]-b[:2]))
        self.roles=[1,2] if close else [2] if attached else [1]
        self.previous_left=None if left is None else left.copy()
        self.previous_red=[b.copy() for b in near_left]
        if any(role not in best for role in self.roles):
            return dict(valid=False,crop_box=None,interaction_box=None,roles=self.roles,reason='required_hand_not_confidently_located')
        hands=[best[role][1] for role in self.roles]
        boxes=hands+[b for b in regions if any(gap(b,hand)<.12*max(hand[2:]-hand[:2]) for hand in hands)]
        array=np.asarray(boxes);tight=np.r_[array[:,:2].min(0),array[:,2:].max(0)]
        scale=max(1.5,.55*height/max(tight[2:]-tight[:2]))
        return dict(valid=True,crop_box=list(square_box(tight,width,height,scale)),interaction_box=tight.tolist(),roles=self.roles,reason=None)
