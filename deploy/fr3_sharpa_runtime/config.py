"""30 Hz FR3 + Sharpa deployment contract shared by client and server."""

import os
from pathlib import Path

from .protocol import ACTION_DIM, ACTION_HORIZON, FPS
from .model_profiles import get_model_profile

MODEL_PROFILE = get_model_profile(os.environ.get("SHARPA_MODEL_PROFILE", "full"))
CONFIG_NAME = MODEL_PROFILE.config_name
# LoRA deliberately ignores generic/full checkpoint settings from local.env.
# It has no default weight, so a missing LoRA path never falls back to full.
CHECKPOINT_ENV = "SHARPA_LORA_CHECKPOINT" if MODEL_PROFILE.name == "lora" else "SHARPA_CHECKPOINT"
NORM_STATS_ENV = "SHARPA_LORA_NORM_STATS" if MODEL_PROFILE.name == "lora" else "SHARPA_NORM_STATS"
_checkpoint_value = os.environ.get(CHECKPOINT_ENV, "" if MODEL_PROFILE.name == "lora" else
    "/media/user/8C0F29AFFD6CD874/sharpa_tomato_ckpt_4000")
CHECKPOINT = Path(_checkpoint_value).expanduser().resolve() if _checkpoint_value else None


def _norm_stats_path():
    if os.environ.get(NORM_STATS_ENV):
        return Path(os.environ[NORM_STATS_ENV]).expanduser().resolve()
    if CHECKPOINT is not None:
        candidates = sorted((CHECKPOINT / "assets").rglob("norm_stats.json"))
        if len(candidates) == 1:
            return candidates[0].resolve()
    return None


# Prefer checkpoint-owned statistics. Never reuse another experiment's stats
# just because its data was sampled at the same rate.
NORM_STATS = _norm_stats_path()
POLICY_PORT = int(os.environ.get("SHARPA_LORA_PORT", 18766)) if MODEL_PROFILE.name == "lora" else 18765
if not 1 <= POLICY_PORT <= 65535:
    raise ValueError("Invalid model service port")
POLICY_URL = f"ws://127.0.0.1:{POLICY_PORT}"
ACTION_HZ = FPS
SOURCE = "openpi"
FR3_TOPIC = "/teleop/arm_commands"
SHARPA_LEFT_TOPIC = "/teleop/sharpa/left/openpi_target"
SHARPA_RIGHT_TOPIC = "/teleop/sharpa/right/openpi_target"
STATE_DIM = ACTION_DIM
CHUNK_LENGTH = ACTION_HORIZON
ACTION_ORDER = ["left_arm_7", "left_hand_22", "right_arm_7", "right_hand_22"]
CAMERA_MAPPING = {"head": "cam0", "left_wrist": "cam2", "right_wrist": "cam1"}
EXECUTION_TRACE_DIR = Path(os.environ.get(
    "SHARPA_LORA_EXECUTION_TRACE_DIR" if MODEL_PROFILE.name == "lora" else "SHARPA_EXECUTION_TRACE_DIR",
    str(Path.home() / (".local/state/fr3_sharpa_lora/execution_traces" if MODEL_PROFILE.name == "lora"
                       else ".local/state/fr3_sharpa_runtime/execution_traces"))))


def require_checkpoint() -> Path:
    if CHECKPOINT is None:
        raise ValueError(f"Set {CHECKPOINT_ENV} to the complete trained 30 Hz step directory in both terminals")
    if NORM_STATS is None:
        raise ValueError(f"Set {NORM_STATS_ENV} to matching norm_stats.json, or include exactly one in checkpoint assets/")
    return CHECKPOINT


def validate_training_contract(training):
    if training.data.action_hz != ACTION_HZ:
        raise ValueError(f"Training action_hz={training.data.action_hz} does not match deployment {ACTION_HZ} Hz")
    if (training.model.paligemma_variant != MODEL_PROFILE.paligemma_variant
            or training.model.action_expert_variant != MODEL_PROFILE.action_expert_variant
            or training.data.extra_delta_transform):
        label = "full-finetune" if MODEL_PROFILE.name == "full" else "LoRA"
        raise ValueError(f"Deployment requires {label} Gemma variants and absolute actions (no delta transform)")
    if training.model.action_dim != ACTION_DIM or training.model.action_horizon != ACTION_HORIZON:
        raise ValueError("Training action shape does not match the 50x58 deployment contract")


def validate_policy_metadata(metadata):
    expected = {"config_name": CONFIG_NAME, "checkpoint": str(require_checkpoint()),
                "norm_stats": str(NORM_STATS), "action_dim": ACTION_DIM,
                "action_horizon": ACTION_HORIZON, "action_hz": ACTION_HZ,
                "action_representation": "absolute_joint_position", "warmed_up": True,
                "action_order": ACTION_ORDER, "camera_mapping": CAMERA_MAPPING}
    # Robot execution depends on physical actions and checkpoint identity,
    # not on whether the model was trained with adapters or full parameters.
    manifest = CHECKPOINT / "params/manifest.ocdbt"
    if manifest.is_file():
        import hashlib
        expected["checkpoint_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if NORM_STATS.is_file():
        import hashlib
        expected["normalization_sha256"] = hashlib.sha256(NORM_STATS.read_bytes()).hexdigest()
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Policy server {key}={metadata.get(key)!r} does not match deployment {value!r}")


DEFAULT_PROMPT = (
    "Pick up a tomato truss with the right hand, then pick a cherry tomato "
    "with the left hand and place it in the left basket."
)
