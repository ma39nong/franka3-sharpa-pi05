"""Shared pixel geometry and explicit v1 task settings for training and serving."""

import numpy as np
from PIL import Image

VERSION = "tomato-foveavla-rgb-v1"
IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "interaction_0_rgb")
EXECUTE_STEPS = 8
ACTION_HORIZON = 50
FPS = 30
LOCATION_START = 256000
# Normalized source xyxy. Fixed camera/workspace priors, not human box annotations.
INITIAL_BOX = (0.25, 0.22, 0.83, 0.86)
LEFT_BASKET = (0.25, 0.58, 0.43, 0.85)
RIGHT_PLATE = (0.60, 0.58, 0.80, 0.83)


def validate_box(box):
    b = np.asarray(box, dtype=np.float64)
    if b.shape != (4,) or not np.isfinite(b).all() or np.any(b < 0) or np.any(b > 1) or np.any(b[2:] <= b[:2]):
        raise ValueError("Expected finite nonempty source-normalized xyxy box")
    return b


def union_boxes(boxes):
    b = np.stack([validate_box(x) for x in boxes])
    return np.r_[b[:, :2].min(axis=0), b[:, 2:].max(axis=0)]


def pad_geometry(width, height):
    ratio = max(width / 224, height / 224)
    rw, rh = int(width / ratio), int(height / ratio)
    return np.array([rw, rh, rw, rh]), np.array([(224 - rw) // 2, (224 - rh) // 2] * 2)


def encode_box(box, width, height):
    size, pad = pad_geometry(width, height)
    xyxy = (validate_box(box) * size + pad) / 224
    return np.rint(xyxy[[1, 0, 3, 2]] * 1023).astype(np.int32) + LOCATION_START


def decode_box(tokens, width, height):
    t = np.asarray(tokens)
    if t.shape != (4,) or t.dtype.kind not in "iu" or np.any(t < LOCATION_START) or np.any(t >= LOCATION_START + 1024):
        raise ValueError("Expected four PaliGemma location tokens")
    xyxy = ((t - LOCATION_START) / 1023 * 224)[[1, 0, 3, 2]]
    size, pad = pad_geometry(width, height)
    return validate_box(np.clip((xyxy - pad) / size, 0, 1))


def crop_rgb(image, box, *, perturb_seed=None):
    """Square 1.5x crop from original pixels, zero padding, PIL bilinear to224."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("Source RGB uint8 HWC required")
    h, w = image.shape[:2]
    b = validate_box(box) * [w, h, w, h]
    m = max(b[2:] - b[:2])
    center = (b[:2] + b[2:]) / 2
    side = 1.5 * m
    if perturb_seed is not None:
        rng = np.random.default_rng(perturb_seed)
        center += rng.uniform(-0.25, 0.25, 2) * m
        side *= np.exp(rng.uniform(0, np.log(4 / 3)))
    # Keep the annotated region after center perturbation.
    side = max(side, 2 * np.max(np.maximum(center - b[:2], b[2:] - center)))
    side = int(np.ceil(side))
    if side <= min(w, h):
        corner = np.clip(np.floor(center - side / 2), 0, np.array([w, h]) - side).astype(int)
    else:
        corner = np.floor(center - side / 2).astype(int)
    x, y = corner
    return np.asarray(
        Image.fromarray(image).crop((x, y, x + side, y + side)).resize((224, 224), Image.Resampling.BILINEAR)
    )
