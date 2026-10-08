"""π0.5 transforms for the dual FR3 + dual Sharpa 58D protocol."""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

from .fr3_sharpa_protocol import ACTION_DIM


def _image(image):
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class Fr3SharpaInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        result = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": _image(data["observation/image"]),
                "left_wrist_0_rgb": _image(data["observation/left_wrist_image"]),
                "right_wrist_0_rgb": _image(data["observation/right_wrist_image"]),
            },
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.True_},
        }
        if result["state"].shape[-1] != ACTION_DIM:
            raise ValueError(f"Expected measured 58D state, got {result['state'].shape}")
        if "actions" in data:
            result["actions"] = np.asarray(data["actions"], dtype=np.float32)
            if result["actions"].shape[-1] != ACTION_DIM:
                raise ValueError(f"Expected commanded 58D action, got {result['actions'].shape}")
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class Fr3SharpaOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.shape[-1] < ACTION_DIM:
            raise ValueError(f"Expected at least 58 model action values, got {actions.shape}")
        return {"actions": actions[..., :ACTION_DIM]}
