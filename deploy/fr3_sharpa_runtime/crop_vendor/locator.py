"""Experimental current-RGB locator and causal measured-state role selection.

Algorithmic validity is not independent semantic quality approval. No stage
labels, future frames, or action commands are accepted by the selector.
"""
from pathlib import Path
import json
import numpy as np
from .geometry import square_box


def make_locator_model():
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model=fasterrcnn_mobilenet_v3_large_320_fpn(weights=None,weights_backbone=None)
    model.roi_heads.box_predictor=FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features,4)
    return model


class RgbLocator:
    def __init__(self,root,device='cuda'):
        import torch
        root=Path(root);receipt=json.loads((root/'receipt.json').read_text())
        if receipt.get('status')!='experimental_training_finished_no_independent_quality_acceptance':
            raise ValueError('Completed isolated locator receipt required')
        payload=torch.load(root/'model.pt',map_location='cpu',weights_only=True)
        if payload['classes']!={1:'right_hand',2:'left_hand',3:'held_object'}:
            raise ValueError('Unexpected locator class contract')
        self.model=make_locator_model();self.model.load_state_dict(payload['state_dict'])
        self.model=self.model.to(device).eval();self.device=device

    def predict(self,images):
        import torch
        inputs=[torch.from_numpy(np.array(im,copy=True)).permute(2,0,1).float().div(255).to(self.device) for im in images]
        with torch.inference_mode():outputs=self.model(inputs)
        return [[dict(label=int(label),score=float(score),box=box.tolist())
                 for box,label,score in zip(p['boxes'].cpu(),p['labels'].cpu(),p['scores'].cpu(),strict=True) if score>=.5]
                for p in outputs]
