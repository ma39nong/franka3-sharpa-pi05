"""FLASH v1 package and wire contracts. Metadata only: no JAX, ROS or SDK."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .dataset_sample import ORDER

VERSION = 1
RESAMPLER = "causal-zoh-30hz-v1"
PREPROCESSOR = "native-flash-rgb-state-v1"
CONFIG_NAME = "pi05_fr3_sharpa_flash_absolute_k6_s1_deploy"
BACKEND_FILES = (
    "src/openpi/models/pi0_flash.py",
    "src/openpi/models/pi0_config.py",
    "src/openpi/models/pi0.py",
    "src/openpi/models/model.py",
    "src/openpi/models/gemma.py",
    "src/openpi/models/siglip.py",
    "src/openpi/models/tokenizer.py",
    "src/openpi/shared/flash_polynomial.py",
    "src/openpi/shared/nnx_utils.py",
    "src/openpi/training/flash_native.py",
    "src/openpi/policies/fr3_sharpa_protocol.py",
)
CAPABILITIES = {
    "algorithm": "flash",
    "model_profile": "flash-absolute",
    "model_class": "Pi0Flash",
    "action_dim": 58,
    "action_horizon": 50,
    "action_hz": 30,
    "action_representation": "absolute_joint_position",
    "action_unit": "rad",
    "poly_degree": 6,
    "coefficient_tokens": 7,
    "history_steps": 8,
    "stride": 1,
    "inference_num_steps": 1,
    "requires_history": True,
    "rtc_supported": False,
    "supported_modes": ["serial"],
    "action_decoding": "flash_internal_denormalize",
}
CAMERAS = {"head": "cam0", "left_wrist": "cam2", "right_wrist": "cam1"}
GROUPS = ("left_arm", "left_hand", "right_arm", "right_hand")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def child(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Missing/outside FLASH package: {relative}")
    return path


def validate_stats(path):
    stats = json.loads(Path(path).read_text())["norm_stats"]
    for group in ("state", "actions"):
        for key in ("mean", "std", "q01", "q99"):
            values = np.asarray(stats[group][key])
            if values.shape != (58,) or not np.isfinite(values).all():
                raise ValueError(f"FLASH requires finite 58D {group}.{key}")
        if np.any(np.asarray(stats[group]["q99"]) < stats[group]["q01"]):
            raise ValueError("Invalid FLASH quantiles")
    return stats


def declares_flash(checkpoint):
    root = Path(checkpoint)
    if (root / "flash_deployment.json").exists():
        return True
    for path in (root / "assets").rglob("flash_manifest.json"):
        try:
            value = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        contract = value.get("contract", {})
        if (
            value.get("algorithm") == "flash"
            or value.get("model_class") == "Pi0Flash"
            or ("K" in contract and "history" in contract)
        ):
            return True
    return False


# Defaults match the audited 8000-step FullFT checkpoint. They configure the
# inference architecture only; action quantiles always come from its own stats.
MODEL_DEFAULTS = {
    "pi05": True, "dtype": "bfloat16", "action_dim": 58, "action_horizon": 50,
    "max_token_len": 256, "discrete_state_input": True, "freeze_vlm": False,
    "poly_degree": 6, "history_steps": 8, "rtc_max_delay": 0,
    "paligemma_variant": "gemma_2b", "action_expert_variant": "gemma_300m",
    "history_reg_lambda": 0.1, "boundary_constraint": True,
    "history_noise_std": 0.5, "consistency_weight": 1.0,
}


def flash_settings(checkpoint, *, norm_stats=None, backend=None):
    """Read original assets in place; never export, copy or modify a checkpoint."""
    root = Path(checkpoint).expanduser().resolve()
    sealed = root / "flash_deployment.json"
    manifest = json.loads(sealed.read_text()) if sealed.is_file() else None
    candidates = sorted((root / "assets").rglob("norm_stats.json"))
    if norm_stats:
        stats_path = Path(norm_stats).expanduser().resolve()
        if not stats_path.is_relative_to(root / "assets"):
            raise ValueError("FLASH statistics must belong to selected checkpoint assets/")
    elif len(candidates) == 1:
        stats_path = candidates[0].resolve()
    else:
        raise ValueError("FLASH checkpoint assets/ must contain unique norm_stats.json")
    stats = validate_stats(stats_path)
    explicit = os.environ.get("SHARPA_FLASH_TRAINING_CONFIG")
    records = [Path(explicit).expanduser()] if explicit else [
        root / "flash_training_contract.json", root.parent / "flash_training_contract.json"]
    record = next((p.resolve() for p in records if p.is_file()), None)
    if explicit and record is None:
        raise ValueError(f"Missing explicitly selected FLASH training configuration: {explicit}")
    if manifest:
        options = manifest["model_config"].copy()
        tokenizer = root / manifest["tokenizer"]
        source = "existing_deployment_manifest"
    elif record:
        options = parse_model_repr(json.loads(record.read_text())["model"])
        tokenizer = None
        source = "original_training_config"
    else:
        options = {**MODEL_DEFAULTS, **{
            "action_" + key: stats["actions"][key] for key in ("q01", "q99")}}
        tokenizer = None
        source = "manual_flash_k6_s1_nfe1_history8"
    for key, expected in MODEL_DEFAULTS.items():
        if options.get(key) != expected:
            raise ValueError(f"FLASH v1 model_config mismatch: {key} (expected {expected!r})")
    for key in ("q01", "q99"):
        if not np.array_equal(options.get("action_" + key), stats["actions"][key]):
            raise ValueError("FLASH model quantiles must match checkpoint actions statistics")
    selected_tokenizer = os.environ.get("SHARPA_FLASH_TOKENIZER")
    if selected_tokenizer:
        tokenizer = Path(selected_tokenizer).expanduser().resolve()
    if tokenizer is None:
        tokenizers = [root / "tokenizer.model",
                      Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model"]
        if backend:
            tokenizers.append(Path(backend) / ".cache/openpi/big_vision/paligemma_tokenizer.model")
        tokenizer = next((p.resolve() for p in tokenizers if p.is_file()), tokenizers[1])
    # A missing tokenizer can be configured before model startup; inspection of
    # the selected weights itself does not import a tokenizer or model runtime.
    return {"model_config": options, "norm_stats": str(stats_path),
            "tokenizer": str(tokenizer), "configuration_source": source}


def inspect_native_flash(checkpoint, *, norm_stats=None, backend=None):
    """Check a native training step without any generated deployment sidecar."""
    root = Path(checkpoint).expanduser().resolve()
    settings = flash_settings(root, norm_stats=norm_stats, backend=backend)
    completion = json.loads((root / "_CHECKPOINT_METADATA").read_text())
    if not completion.get("commit_timestamp_nsecs"):
        raise ValueError("FLASH checkpoint save has not committed")
    for name in ("params/_METADATA", "params/manifest.ocdbt"):
        if not (root / name).is_file():
            raise ValueError(f"Missing FLASH checkpoint {name}")
    if not any(p.is_file() and p.stat().st_size > 0 and "d" in p.relative_to(root / "params").parts[:-1]
               for p in (root / "params").rglob("*")):
        raise ValueError("FLASH requires full parameter data, not a metadata-only download")
    entries = json.loads((root / "params/_METADATA").read_text())["tree_metadata"]
    projections = {}
    for entry in entries.values():
        names = [part["key"] for part in entry.get("key_metadata", ())]
        if any("lora" in str(name).lower() for name in names):
            raise ValueError("FLASH v1 requires FullFT parameters, excludes LoRA")
        if "action_in_proj" in names and "kernel" in names:
            projections["input"] = entry["value_metadata"]["write_shape"][0]
        if "action_out_proj" in names and "bias" in names:
            projections["output"] = entry["value_metadata"]["write_shape"]
    if projections != {"input": 58, "output": [58]}:
        raise ValueError("FLASH requires native 58D projections; global tree checked during restore")
    data_path = Path(settings["norm_stats"]).parent / "flash_manifest.json"
    if data_path.is_file():
        data = json.loads(data_path.read_text()).get("contract", {})
        for key, expected in {"action_semantics": "absolute", "horizon": 50, "history": 8,
                              "target": 54, "K": 6, "stride": 1, "joint_names": list(ORDER)}.items():
            if data.get(key) != expected:
                raise ValueError(f"FLASH training data configuration mismatch: {key}")
    backend_identity = None
    if backend:
        repo = Path(backend).expanduser().resolve()
        backend_identity = digest_json({name: sha(child(repo, name)) for name in BACKEND_FILES})
    adapter_root = Path(__file__).resolve().parents[2]
    adapter = digest_json({name: sha(adapter_root / name) for name in (
        "deploy/fr3_sharpa_flash/policy.py", "deploy/fr3_sharpa_flash/serve.py",
        "deploy/fr3_sharpa_flash/codec.py", "deploy/fr3_sharpa_runtime/flash_contract.py")})
    params_hash = sha(root / "params/manifest.ocdbt")
    stats_hash = sha(settings["norm_stats"])
    # Historical wire field name retained for compatibility. This is an
    # in-memory identity, not a file that the operator must create.
    identity = digest_json({"params": params_hash, "stats": stats_hash, "model": settings["model_config"]})
    return {**CAPABILITIES, "config_name": CONFIG_NAME, "checkpoint": str(root),
            "norm_stats": settings["norm_stats"], "checkpoint_manifest_sha256": params_hash,
            "normalization_sha256": stats_hash, "flash_manifest_sha256": identity,
            "flash_backend_identity": backend_identity, "flash_adapter_identity": adapter,
            "flash_implementation_version": "fr3-flash-v1", "configuration_source": settings["configuration_source"],
            "action_order": ["left_arm_7", "left_hand_22", "right_arm_7", "right_hand_22"],
            "camera_mapping": CAMERAS, "paper_algorithm": None, "train_rtc": None,
            "training_action_supervision": "absolute", "model_finetune_mode": "full", "warmed_up": False}


def inspect_flash(checkpoint, *, norm_stats=None, backend=None):
    root = Path(checkpoint).expanduser().resolve()
    path = root / "flash_deployment.json"
    if not path.is_file():
        return inspect_native_flash(root, norm_stats=norm_stats, backend=backend)
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != VERSION:
        raise ValueError("Unsupported FLASH manifest schema")
    for key, value in CAPABILITIES.items():
        if manifest.get(key) != value:
            raise ValueError(f"FLASH manifest {key} must be {value!r}")
    if manifest.get("joint_names") != list(ORDER) or manifest.get("camera_mapping") != CAMERAS:
        raise ValueError("FLASH joint/camera order mismatch")
    if manifest.get("preprocessing_version") != PREPROCESSOR or manifest.get("history_resampler_version") != RESAMPLER:
        raise ValueError("Unsupported FLASH preprocessing/history rule")
    completion = json.loads(child(root, "_CHECKPOINT_METADATA").read_text())
    if (
        not completion.get("commit_timestamp_nsecs")
        or type(manifest.get("checkpoint_step")) is not int
        or manifest["checkpoint_step"] < 0
    ):
        raise ValueError("FLASH package is not a completed matching Orbax step")
    files = manifest["files"]
    if "_CHECKPOINT_METADATA" not in files:
        raise ValueError("FLASH checkpoint completion marker must be sealed")
    actual_params = {str(p.relative_to(root)) for p in (root / "params").rglob("*") if p.is_file()}
    sealed_params = {name for name in files if name.startswith("params/")}
    if (
        not actual_params
        or actual_params != sealed_params
        or not {"params/_METADATA", "params/manifest.ocdbt"} <= actual_params
    ):
        raise ValueError("FLASH parameter inventory incomplete/changed")
    if not any("/d/" in name and files[name]["size"] > 0 for name in actual_params):
        raise ValueError("FLASH params must include Orbax/OCDBT data files, not metadata alone")
    entries = json.loads((root / "params/_METADATA").read_text())["tree_metadata"]
    projections = {}
    for entry in entries.values():
        names = [part["key"] for part in entry.get("key_metadata", ())]
        if any("lora" in str(name).lower() for name in names):
            raise ValueError("FLASH FullFT excludes LoRA parameter branches")
        if "action_in_proj" in names and "kernel" in names:
            projections["input"] = entry["value_metadata"]["write_shape"][0]
        if "action_out_proj" in names and "bias" in names:
            projections["output"] = entry["value_metadata"]["write_shape"]
    if projections != {"input": 58, "output": [58]}:
        raise ValueError("FLASH requires native 58D projections; strict global tree checked during restore")
    for name, identity in files.items():
        if name == manifest.get("execution_validation"):
            continue  # A historical validation report is not a runtime dependency.
        item = child(root, name)
        if item.stat().st_size != identity["size"] or sha(item) != identity["sha256"]:
            raise ValueError(f"FLASH asset/parameter hash mismatch: {name}")
    stats_path = child(root, manifest["norm_stats"])
    if norm_stats is not None and Path(norm_stats).resolve() != stats_path:
        raise ValueError("FLASH statistics must belong to the sealed checkpoint")
    required_assets = [
        stats_path.parent / name
        for name in ("norm_stats.json", "flash_manifest.json", "flash_seal.json", "flash_validation.json")
    ]
    for item in [*required_assets, child(root, manifest["tokenizer"]), child(root, manifest["training_contract"])]:
        if str(item.relative_to(root)) not in files:
            raise ValueError(f"Unsealed FLASH dependency: {item.name}")
    stats = validate_stats(stats_path)
    training_assets = json.loads(required_assets[1].read_text())
    seal = json.loads(required_assets[2].read_text())
    if (
        digest_json({k: v for k, v in training_assets.items() if k != "sha256"}) != training_assets["sha256"]
        or seal["manifest_sha256"] != training_assets["sha256"]
        or seal["norm_sha256"] != sha(stats_path)
        or seal["validation_sha256"] != digest_json(json.loads(required_assets[3].read_text()))
    ):
        raise ValueError("FLASH training asset seal mismatch")
    data = training_assets["contract"]
    for key, value in {
        "action_semantics": "absolute",
        "horizon": 50,
        "history": 8,
        "target": 54,
        "K": 6,
        "stride": 1,
        "joint_names": list(ORDER),
    }.items():
        if data.get(key) != value:
            raise ValueError(f"FLASH training data contract mismatch: {key}")
    model = manifest["model_config"]
    for key, value in {
        "pi05": True,
        "action_dim": 58,
        "action_horizon": 50,
        "max_token_len": 256,
        "discrete_state_input": True,
        "freeze_vlm": False,
        "poly_degree": 6,
        "history_steps": 8,
        "rtc_max_delay": 0,
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
    }.items():
        if model.get(key) != value:
            raise ValueError(f"FLASH model_config mismatch: {key}")
    for key in ("history_reg_lambda", "boundary_constraint", "history_noise_std", "consistency_weight", "dtype"):
        if key not in model:
            raise ValueError(f"Missing model static configuration: {key}")
    for key in ("q01", "q99"):
        if not np.array_equal(model.get("action_" + key), stats["actions"][key]):
            raise ValueError("FLASH model quantiles must match checkpoint actions statistics")
    training = json.loads(child(root, manifest["training_contract"]).read_text())
    if (
        training.get("dataset", {}).get("normalization_sha256") != sha(stats_path)
        or training.get("dataset", {}).get("manifest_sha256") != training_assets["sha256"]
    ):
        raise ValueError("FLASH training run/data/statistics provenance mismatch")
    if parse_model_repr(training["model"]) != model:
        raise ValueError("FLASH model configuration differs from training contract")
    source = manifest["backend"]
    if (
        not isinstance(source.get("commit"), str)
        or len(source["commit"]) != 40
        or set(source["files"]) != set(BACKEND_FILES)
    ):
        raise ValueError("Missing complete FLASH backend identity")
    if backend is not None:
        repo = Path(backend).expanduser().resolve()
        for name, expected in source["files"].items():
            if sha(child(repo, name)) != expected:
                raise ValueError(f"FLASH backend code mismatch: {name}")
    # Old sealed packages remain readable. Execution reports are optional
    # historical artifacts and no longer decide whether this model can run.
    if manifest.get("history_max_sample_age_ns") != 75_000_000 or manifest.get("history_max_gap_ns") != 100_000_000:
        raise ValueError("Unsupported FLASH history bounds; use a new resampler version")
    adapter_root = Path(__file__).resolve().parents[2]
    adapter_files = (
        "deploy/fr3_sharpa_flash/policy.py",
        "deploy/fr3_sharpa_flash/serve.py",
        "deploy/fr3_sharpa_flash/codec.py",
        "deploy/fr3_sharpa_runtime/flash_contract.py",
    )
    adapter_identity = digest_json({name: sha(adapter_root / name) for name in adapter_files})
    return {
        **CAPABILITIES,
        "flash_adapter_identity": adapter_identity,
        "config_name": CONFIG_NAME,
        "checkpoint": str(root),
        "norm_stats": str(stats_path),
        "checkpoint_manifest_sha256": sha(root / "params/manifest.ocdbt"),
        "normalization_sha256": sha(stats_path),
        "flash_manifest_sha256": sha(path),
        "flash_backend_identity": digest_json(source),
        "flash_implementation_version": "fr3-flash-v1",
        "action_order": ["left_arm_7", "left_hand_22", "right_arm_7", "right_hand_22"],
        "camera_mapping": CAMERAS,
        "paper_algorithm": None,
        "train_rtc": None,
        "training_action_supervision": "absolute",
        "model_finetune_mode": "full",
        "warmed_up": False,
    }


def parse_model_repr(value):
    """Literal-only dataclass repr parser: no eval, imports or function calls."""
    import ast

    expr = ast.parse(value, mode="eval").body
    if (
        not isinstance(expr, ast.Call)
        or not isinstance(expr.func, ast.Name)
        or expr.func.id != "Pi0FlashConfig"
        or expr.args
    ):
        raise ValueError("Expected literal Pi0FlashConfig training contract")
    result = {}
    for entry in expr.keywords:
        if entry.arg is None or entry.arg in result:
            raise ValueError("Invalid FLASH model configuration")
        result[entry.arg] = json.loads(json.dumps(ast.literal_eval(entry.value)))
    return result


def validate_flash_observation(observation):
    from .runtime import validate_observation

    allowed = {
        "observation/state",
        "observation/history",
        "observation/image",
        "observation/left_wrist_image",
        "observation/right_wrist_image",
        "prompt",
        "flash_context",
    }
    unknown = [key for key in observation if key not in allowed and not key.startswith("_")]
    if unknown:
        raise ValueError(f"FLASH rejects extra inputs (RTC/noise/future labels): {unknown}")
    clean = validate_observation(observation)
    history = np.asarray(observation["observation/history"], dtype=np.float32)
    if (
        history.shape != (8, 58)
        or not np.isfinite(history).all()
        or not np.array_equal(history[-1], clean["observation/state"])
    ):
        raise ValueError("FLASH requires finite 8x58 history with last row == measured state")
    ctx = observation["flash_context"]
    if ctx.get("protocol_version") != VERSION or ctx.get("history_resampler_version") != RESAMPLER:
        raise ValueError("FLASH history protocol/resampler mismatch")
    for key in ("session_id", "request_id"):
        if not isinstance(ctx.get(key), str) or not ctx[key]:
            raise ValueError(f"Missing FLASH {key}")
    anchor = ctx["observation_anchor_source_ns"]
    if (
        type(anchor) is not int
        or anchor <= 0
        or type(ctx.get("captured_monotonic_ns")) is not int
        or ctx["captured_monotonic_ns"] <= 0
    ):
        raise ValueError("Invalid FLASH capture/anchor")
    grid = np.asarray(ctx["history_grid_source_ns"])
    source = np.asarray(ctx["history_source_ns"])
    expected_grid = [anchor - round((7 - i) * 1e9 / 30) for i in range(8)]
    if grid.dtype.kind not in "iu" or grid.shape != (8,) or grid.tolist() != expected_grid:
        raise ValueError("FLASH grid must be eight quantized 30Hz causal ticks")
    if (
        source.dtype.kind not in "iu"
        or source.shape != (8, 4)
        or (source <= 0).any()
        or (np.diff(source, axis=0) < 0).any()
        or (source > grid[:, None]).any()
        or (grid[:, None] - source > 75_000_000).any()
        or (np.ptp(source, axis=1) > 100_000_000).any()
    ):
        raise ValueError("FLASH history source times invalid/too old")
    if observation.get("_captured_monotonic_ns", ctx["captured_monotonic_ns"]) != ctx["captured_monotonic_ns"]:
        raise ValueError("FLASH capture identity mismatch")
    clean.update({"observation/history": history, "flash_context": dict(ctx)})
    return clean


def execution_budget(checkpoint=None):
    """Optional operator age bound; paired-validation reports are not required.

    0/unset disables this additional FLASH experiment. Existing sensor freshness,
    joint limits, command deadlines and shutdown checks remain independent.
    """
    value = float(os.environ.get("SHARPA_FLASH_MAX_PLAN_AGE_MS", "0"))
    if not np.isfinite(value) or value < 0:
        raise ValueError("SHARPA_FLASH_MAX_PLAN_AGE_MS must be finite and nonnegative (0 disables)")
    return value or None


def check_plan_identity(plan):
    if plan.get("protocol_version") != VERSION or not plan.get("request_id") or not plan.get("session_id"):
        raise ValueError("Missing FLASH execution identity")


def check_plan_age(plan, *, bridge_steps, speed_scale, now_ns):
    check_plan_identity(plan)
    budget = plan.get("max_plan_age_ms")
    captured = plan.get("captured_monotonic_ns")
    anchor_age = plan.get("anchor_age_at_capture_ns")
    if (
        not isinstance(budget, float | int)
        or not np.isfinite(budget)
        or budget <= 0
        or type(captured) is not int
        or type(anchor_age) is not int
        or anchor_age < 0
    ):
        raise ValueError("FLASH execution requires finite positive plan age budget")
    age = (now_ns - captured + anchor_age) / 1e6
    predicted = age + bridge_steps / 30 * 1000  # Bridge always emits at 30Hz; scale stretches policy nodes only.
    if age < 0 or predicted > budget:
        raise ValueError(
            f"FLASH plan expired before any frame: predicted_first_target_age_ms={predicted:.3f}, budget={budget}"
        )
    return {"predicted_first_target_age_ms": predicted, "max_plan_age_ms": budget}
