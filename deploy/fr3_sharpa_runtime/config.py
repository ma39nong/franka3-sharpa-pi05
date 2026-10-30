"""30 Hz FR3 + Sharpa deployment contract shared by client and server."""

import os
from pathlib import Path
import threading

from .protocol import ACTION_DIM, ACTION_HORIZON, FPS
from .model_profiles import get_model_profile

MODEL_PROFILE = get_model_profile(os.environ.get("SHARPA_MODEL_PROFILE", "full"))
CONFIG_NAME = MODEL_PROFILE.config_name
# Alternate profiles deliberately ignore absolute checkpoint settings from local.env.
# They have no default weights; missing paths never fall back to the absolute model.
CHECKPOINT_ENV = MODEL_PROFILE.checkpoint_env
NORM_STATS_ENV = MODEL_PROFILE.norm_stats_env
_checkpoint_value = os.environ.get(CHECKPOINT_ENV, MODEL_PROFILE.default_checkpoint)
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
POLICY_PORT = int(os.environ.get(MODEL_PROFILE.port_env, MODEL_PROFILE.default_port))
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
EXECUTION_TRACE_DIR = Path(os.environ.get(MODEL_PROFILE.trace_env, str(Path.home() / MODEL_PROFILE.trace_subdir)))


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
            or training.data.extra_delta_transform != MODEL_PROFILE.extra_delta_transform):
        label = "full-finetune" if MODEL_PROFILE.finetune_mode == "full" else "LoRA"
        action_label = "arm delta, hands absolute" if MODEL_PROFILE.extra_delta_transform else "absolute actions (no delta transform)"
        raise ValueError(f"Deployment requires {label} Gemma variants and {action_label}")
    if training.model.action_dim != ACTION_DIM or training.model.action_horizon != ACTION_HORIZON:
        raise ValueError("Training action shape does not match the 50x58 deployment contract")


_validation_lock = threading.Lock()
_validation_cache = None


def _asset_signature(paths):
    signature = []
    for path in paths:
        try:
            info = path.stat()
        except FileNotFoundError:
            signature.append(None)
        else:
            signature.append((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
    return tuple(signature)


def prepare_policy_validation():
    """Fully verify once, then invalidate on replacement/change of any contract asset.

    Run before acquiring the first observation. Each handshake still checks local
    file identities and compares the verified hashes/algorithm against the server.
    Checkpoint assets must remain immutable for the duration of execution.
    """
    global _validation_cache
    checkpoint = require_checkpoint()
    paths = (checkpoint / "params/manifest.ocdbt", checkpoint / "params/_METADATA", NORM_STATS,
             NORM_STATS.parent / "delta_paper.json", NORM_STATS.parent / "delta_paper.npz",
             NORM_STATS.parent / "norm_metadata.json")
    context = (str(checkpoint.resolve()), str(NORM_STATS.resolve()), MODEL_PROFILE, CONFIG_NAME)
    with _validation_lock:
        signature = _asset_signature(paths)
        if _validation_cache is not None:
            previous_context, previous_signature, verified = _validation_cache
            if context == previous_context and signature == previous_signature:
                return verified
            if context == previous_context and previous_signature[1] is not None and signature[1] is None:
                raise ValueError("Checkpoint metadata removed after validation; execution blocked")
        # Never cache failed/incomplete validation, or assets modified during it.
        if signature[1] is not None:
            from .model_loader import checkpoint_contract
            contract = checkpoint_contract()
            verified = {key: contract[key] for key in (
                "checkpoint_manifest_sha256", "normalization_sha256", "paper_algorithm")}
        else:
            # Keep metadata-free protocol fixtures/legacy checks compatible.
            import hashlib
            verified = {"paper_algorithm": None}
            for path, key in ((paths[0], "checkpoint_manifest_sha256"), (NORM_STATS, "normalization_sha256")):
                if path.is_file():
                    verified[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        if signature != _asset_signature(paths):
            raise ValueError("Checkpoint assets changed during validation; execution blocked")
        _validation_cache = (context, signature, verified)
        return verified


def validate_policy_metadata(metadata):
    expected = {"config_name": CONFIG_NAME, "checkpoint": str(require_checkpoint()),
                "norm_stats": str(NORM_STATS), "action_dim": ACTION_DIM,
                "action_horizon": ACTION_HORIZON, "action_hz": ACTION_HZ,
                "action_representation": "absolute_joint_position", "warmed_up": True,
                "action_order": ACTION_ORDER, "camera_mapping": CAMERA_MAPPING}
    if MODEL_PROFILE.extra_delta_transform:
        expected["training_action_supervision"] = MODEL_PROFILE.training_action_supervision
    # Robot execution depends on physical actions and checkpoint identity,
    # not on whether the model was trained with adapters or full parameters.
    expected.update(prepare_policy_validation())
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Policy server {key}={metadata.get(key)!r} does not match deployment {value!r}")


DEFAULT_PROMPT = (
    "Pick up a tomato truss with the right hand, then pick a cherry tomato "
    "with the left hand and place it in the left basket."
)
