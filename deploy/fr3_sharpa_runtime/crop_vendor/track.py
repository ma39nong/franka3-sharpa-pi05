"""Seeded, forward-only optical-flow ROI prototype; no detector or automatic retargeting.

valid means geometric tracking checks passed, not verified object identity.
Once lost, a track stays invalid until an explicit new seed is supplied.
"""
import cv2
import numpy as np


class CausalBoxTrack:
    def __init__(self, gray, box, frame_index, *, max_age=300):
        from .geometry import valid_box
        h,w=gray.shape
        self.box=valid_box(box,w,h).copy()
        x1,y1,x2,y2=self.box
        self.corners=np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],dtype=float)
        self.seed_area=(self.box[2]-self.box[0])*(self.box[3]-self.box[1])
        self.previous=gray.copy();self.frame_index=frame_index;self.age=0;self.max_age=max_age
        mask=np.zeros_like(gray);x1,y1,x2,y2=np.round(self.box).astype(int);mask[y1:y2,x1:x2]=255
        self.points=cv2.goodFeaturesToTrack(gray,100,.02,4,mask=mask)
        self.lost_reason=None
        if self.points is None or len(self.points)<6:self.lost_reason='insufficient_seed_features'

    def invalid(self, reason):
        self.lost_reason=reason
        return dict(valid=False,box=None,reason=reason)

    def update(self, gray, frame_index):
        if frame_index!=self.frame_index+1:raise ValueError('Consecutive increasing frames required')
        self.frame_index=frame_index;self.age+=1
        if self.lost_reason:return self.invalid(self.lost_reason)
        if self.age>self.max_age:return self.invalid('track_age_limit')
        if gray.shape!=self.previous.shape:raise ValueError('Image dimensions changed')
        new,ok,_=cv2.calcOpticalFlowPyrLK(self.previous,gray,self.points,None,winSize=(21,21),maxLevel=3)
        if new is None:return self.invalid('forward_flow_failed')
        back,reverse,_=cv2.calcOpticalFlowPyrLK(gray,self.previous,new,None,winSize=(21,21),maxLevel=3)
        if back is None:return self.invalid('backward_flow_failed')
        keep=(ok.ravel()==1)&(reverse.ravel()==1)&(np.linalg.norm(back-self.points,axis=2).ravel()<1.5)
        if keep.sum()<6 or keep.mean()<.5:return self.invalid('feature_loss_or_forward_backward_error')
        before=self.points[keep].reshape(-1,2);after=new[keep].reshape(-1,2)
        matrix,inliers=cv2.estimateAffinePartial2D(before,after,method=cv2.RANSAC,ransacReprojThreshold=2)
        if matrix is None or not np.isfinite(matrix).all() or inliers.sum()<6 or inliers.mean()<.6:
            return self.invalid('inconsistent_motion')
        scale=np.linalg.norm(matrix[:,0])
        if not .85<=scale<=1.15 or np.linalg.norm(np.median(after-before,axis=0))>25:
            return self.invalid('implausible_scale_or_motion')
        # Transform the persistent polygon, not the previous axis-aligned envelope:
        # repeated small rotations must not grow the box on every frame.
        corners=np.c_[self.corners,np.ones(4)]@matrix.T
        candidate=np.r_[corners.min(0),corners.max(0)]
        h,w=gray.shape
        area=np.prod(candidate[2:]-candidate[:2])
        if (candidate[:2]<0).any() or candidate[2]>w or candidate[3]>h or not .5<=area/self.seed_area<=2:
            return self.invalid('outside_image_or_area_drift')
        self.box=candidate;self.corners=corners;self.previous=gray.copy()
        self.points=new[keep][inliers.ravel().astype(bool)].reshape(-1,1,2)
        return dict(valid=True,box=self.box.tolist(),reason=None,
                    retained_features=len(self.points),inlier_fraction=float(inliers.mean()))
