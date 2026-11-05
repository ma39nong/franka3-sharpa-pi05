"""Native FLASH preprocessing and strict full-parameter inference, isolated env."""

import json
import os
from pathlib import Path
import time

import numpy as np

from deploy.fr3_sharpa_runtime.flash_contract import flash_settings
from deploy.fr3_sharpa_runtime.flash_contract import validate_flash_observation

IMAGE_KEYS = {
    "base_0_rgb": "observation/image",
    "left_wrist_0_rgb": "observation/left_wrist_image",
    "right_wrist_0_rgb": "observation/right_wrist_image",
}


def resize_image(rgb):
    # Intentionally matches pinned NativeDataset.resize_image, including dtype,
    # integer resize rounding, interpolation antialias and -1 padding.
    import torch
    import torch.nn.functional as F  # noqa: N812 -- match pinned training formula

    pixels = rgb.astype(np.float32) / 127.5 - 1
    height, width = pixels.shape[:2]
    ratio = max(width / 224, height / 224)
    resized_h, resized_w = int(height / ratio), int(width / ratio)
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)[None]
    value = (
        F.interpolate(tensor, size=(resized_h, resized_w), mode="bilinear", align_corners=False, antialias=True)[0]
        .permute(1, 2, 0)
        .numpy()
    )
    top, left = (224 - resized_h) // 2, (224 - resized_w) // 2
    return np.pad(value, ((top, 224 - resized_h - top), (left, 224 - resized_w - left), (0, 0)), constant_values=-1)[
        None
    ]


def preprocess(clean, stats, tokenizer):
    # Preserve numpy stats dtype until after tokenization, as NativeDataset does.
    stat = stats["state"]
    state = clean["observation/state"]
    normalized = (state - np.asarray(stat["q01"])) / (np.asarray(stat["q99"]) - stat["q01"] + 1e-6) * 2 - 1
    tokens, mask = tokenizer.tokenize(clean["prompt"], normalized)
    images = {key: resize_image(clean[field]) for key, field in IMAGE_KEYS.items()}
    return {
        "images": images,
        "image_masks": {key: np.ones(1, bool) for key in images},
        "state": np.asarray(normalized[None], np.float32),
        "tokenized_prompt": np.asarray(tokens[None], np.int32),
        "tokenized_prompt_mask": np.asarray(mask[None]),
    }


def load_tokenizer(path):
    import sentencepiece

    from openpi.models.tokenizer import PaligemmaTokenizer

    # Only replace asset acquisition, reusing the exact training tokenize method.
    # No global patch and no network access for a tokenizer at service startup.
    tokenizer = PaligemmaTokenizer.__new__(PaligemmaTokenizer)
    tokenizer._max_len = 256  # noqa: SLF001 -- bind local asset, keep upstream tokenize unchanged
    tokenizer._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=Path(path).read_bytes())  # noqa: SLF001
    return tokenizer


class FlashPolicy:
    def __init__(self, checkpoint, metadata):
        import jax

        from openpi.models import model as model_lib
        from openpi.models.pi0_flash import Pi0FlashConfig
        from openpi.shared import nnx_utils

        root = Path(checkpoint)
        settings = flash_settings(root, norm_stats=metadata["norm_stats"], backend=os.environ.get("SHARPA_FLASH_REPO"))
        options = settings["model_config"].copy()
        for key in ("action_q01", "action_q99"):
            options[key] = tuple(options[key])
        model_config = Pi0FlashConfig(**options)
        # No base-model merge, no random projection fallback, no dtype coercion.
        params = model_lib.restore_params(root / "params")
        self.model = model_config.load(params, remove_extra_params=False)
        self.sample = nnx_utils.module_jit(self.model.sample_actions, static_argnames=("num_steps",))
        self.stats = json.loads(Path(metadata["norm_stats"]).read_text())["norm_stats"]
        self.tokenizer = load_tokenizer(settings["tokenizer"])
        self.rng = jax.random.key(0)
        self.metadata = metadata

    def infer(self, observation):
        import jax

        from openpi.models.model import Observation

        started = time.monotonic_ns()
        clean = validate_flash_observation(observation)
        arrays = preprocess(clean, self.stats, self.tokenizer)
        preprocessed = time.monotonic_ns()
        arrays = jax.tree.map(jax.device_put, arrays)
        history = jax.device_put(clean["observation/history"][None])
        jax.block_until_ready((arrays, history))
        copied = time.monotonic_ns()
        # NFE is a Python static int. No history perturbation or external output
        # Unnormalize/AbsoluteActions is ever applied here.
        output = self.sample(self.rng, Observation(**arrays), history=history, num_steps=1)
        output.block_until_ready()
        sampled = time.monotonic_ns()
        actions = np.asarray(output[0], dtype=np.float32)
        if actions.shape != (50, 58) or not np.isfinite(actions).all():
            raise ValueError("FLASH returned invalid physical 50x58 actions")
        ctx = clean["flash_context"]
        return {
            "actions": actions,
            "flash_receipt": {
                key: ctx[key]
                for key in ("protocol_version", "request_id", "session_id", "observation_anchor_source_ns")
            }
            | {"model_identity": self.metadata["flash_manifest_sha256"]},
            "diagnostic_timing": {
                "cpu_preprocessing_ms": (preprocessed - started) / 1e6,
                "host_device_copy_ms": (copied - preprocessed) / 1e6,
                "fused_vlm_expert_polynomial_ms": (sampled - copied) / 1e6,
                "total_inference_ms": (time.monotonic_ns() - started) / 1e6,
            },
        }
