"""Four-view inference adapter, canonical measured58 in and absolute58 out."""
import dataclasses

import numpy as np

from openpi import transforms
from openpi.policies.fr3_sharpa_policy import Fr3SharpaInputs
from openpi.training.config import LeRobotFr3SharpaDataConfig

from .paper_model import PaperSharpaData


@dataclasses.dataclass(frozen=True)
class CropInputs(Fr3SharpaInputs):
    def __call__(self, data):
        result = super().__call__(data)
        image = np.asarray(data['observation/interaction_image'])
        if image.dtype != np.uint8 or image.shape != (224, 224, 3):
            raise ValueError('Current native RGB crop224 required')
        valid = np.asarray(data['observation/interaction_valid'])
        if valid.shape != () or valid.dtype != np.bool_ or not valid:
            raise ValueError('Fourth view must be available (local or explicitly reported current full view)')
        result['image']['interaction_0_rgb'] = image
        result['image_mask']['interaction_0_rgb'] = valid
        return result


@dataclasses.dataclass(frozen=True)
class CropRuntimeData(LeRobotFr3SharpaDataConfig):
    def create(self, assets_dirs, model_config):
        base = super().create(assets_dirs, model_config)
        inputs = list(base.data_transforms.inputs)
        inputs[0] = CropInputs(model_config.model_type)
        return dataclasses.replace(base, data_transforms=transforms.Group(
            inputs=inputs, outputs=base.data_transforms.outputs))


@dataclasses.dataclass(frozen=True)
class PaperCropRuntimeData(PaperSharpaData):
    """Retain Paper normalization in both directions while adding the fourth RGB."""
    def create(self, assets_dirs, model_config):
        base = super().create(assets_dirs, model_config)
        inputs = list(base.data_transforms.inputs)
        inputs[0] = CropInputs(model_config.model_type)
        return dataclasses.replace(base, data_transforms=transforms.Group(
            inputs=inputs, outputs=base.data_transforms.outputs))
