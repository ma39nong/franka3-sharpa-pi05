"""Stateless current-frame cropping; the client commits predicted boxes after execution."""

from pathlib import Path
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms
from openpi.models import model as model_lib
from openpi.models.pi0_fovea import FoveaConfig
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.policies.fr3_sharpa_policy import Fr3SharpaInputs
from openpi.policies.fr3_sharpa_policy import Fr3SharpaOutputs
from openpi.shared import nnx_utils
from openpi.shared import normalize
from openpi.subtasks import fovea

from .fovea_contract import checkpoint_contract


class FoveaPolicy:
    def __init__(self, model, stats, metadata):
        self.model = model
        self.metadata = metadata
        self.sample = nnx_utils.module_jit(model.sample_joint)
        self.rng = jax.random.key(0)
        self.lock = threading.Lock()
        self.inputs = transforms.compose(
            [
                Fr3SharpaInputs(model_lib.ModelType.PI05),
                transforms.Normalize(stats, use_quantiles=True),
                transforms.ResizeImages(224, 224),
                transforms.TokenizePrompt(PaligemmaTokenizer(256), discrete_state_input=True),
                transforms.PadStatesAndActions(58),
            ]
        )
        self.outputs = transforms.compose(
            [
                transforms.Unnormalize(stats, use_quantiles=True),
                transforms.AbsoluteActions(transforms.make_bool_mask(7, -22, 7, -22)),
                Fr3SharpaOutputs(),
            ]
        )

    def infer(self, observation):
        with self.lock:
            return self._infer(observation)

    def _infer(self, observation):
        from .runtime import validate_observation

        clean = validate_observation(observation)
        context = observation.get("_fovea", {})
        index = context.get("query_index")
        session = context.get("session_id")
        if (
            type(index) is not int
            or index < 0
            or not isinstance(session, str)
            or not 1 <= len(session) <= 128
            or context.get("executed_steps") != (0 if index == 0 else fovea.EXECUTE_STEPS)
        ):
            raise ValueError("Fovea session, query index and committed execution count required")
        box = fovea.validate_box(context.get("current_box"))
        if index == 0 and not np.array_equal(box, np.asarray(fovea.INITIAL_BOX)):
            raise ValueError("First query must use the checkpoint fixed initial box")
        image = clean["observation/image"]
        h, w = image.shape[:2]
        if [w, h] != [640, 400]:
            raise ValueError("Source-resolution 640x400 cam0 required")
        crop = fovea.crop_rgb(image, box)
        inputs = self.inputs(clean)
        inputs["image"]["interaction_0_rgb"] = crop
        inputs["image_mask"]["interaction_0_rgb"] = np.True_
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
        obs = model_lib.Observation.from_dict(inputs)
        self.rng, key = jax.random.split(self.rng)
        start = time.monotonic()
        result = jax.tree.map(lambda x: np.asarray(x[0]), self.sample(key, obs))
        elapsed = (time.monotonic() - start) * 1000
        actions = self.outputs({"actions": result["actions"], "state": np.asarray(inputs["state"][0])})["actions"]
        if actions.shape != (50, 58) or not np.isfinite(actions).all():
            raise ValueError("Invalid Fovea action chunk")
        valid = True
        try:
            next_box = fovea.decode_box(result["foveation_tokens"], w, h)
        except ValueError:
            next_box = box
            valid = False
        response = {
            "actions": actions,
            "policy_timing": {"infer_ms": elapsed},
            "foveation_tokens": result["foveation_tokens"],
            "foveation": {
                "session_id": session,
                "query_index": index,
                "used_box": box.tolist(),
                "next_box": next_box.tolist(),
                "prediction_valid": valid,
                "execute_steps": fovea.EXECUTE_STEPS,
                "source_size": [w, h],
            },
        }
        if context.get("record_roi") is True:
            from .four_view_recording import pixel_sha
            response["recording_roi"] = {"pixels": crop, "metadata": {
                "mode": "fovea-predicted-box", "roi_pixel_sha256": pixel_sha(crop),
                "source_monotonic_ns": context.get("source_monotonic_ns"),
                "used_box": box.tolist(), "next_box": next_box.tolist(),
                "prediction_valid": valid,
            }}
        return response


def load_policy(checkpoint):
    metadata = checkpoint_contract(checkpoint)
    config = FoveaConfig(action_dim=58, action_horizon=50, max_token_len=256, image_keys=fovea.IMAGE_KEYS)
    model = config.load(model_lib.restore_params(Path(metadata["checkpoint"]) / "params", dtype=jnp.bfloat16))
    stats = normalize.load(Path(metadata["norm_stats"]).parent)
    return FoveaPolicy(model, stats, metadata)
