"""Train RTC configuration from optional archives or explicit operator settings."""

import hashlib
import json
import os
from pathlib import Path


def resolve_training_record(checkpoint, record_path=None):
    """Explicit/local evidence wins; shipped records require exact checkpoint fingerprints."""
    if os.environ.get("SHARPA_DEPLOY_MODEL_TYPE", "auto") != "auto":
        path = record_path or os.environ.get("SHARPA_DEPLOY_TRAINING_RECORD")
        return Path(path).expanduser().resolve() if path else None
    path = record_path or os.environ.get("SHARPA_TRAIN_RTC_RECORD")
    if path:
        return Path(path).expanduser().resolve()
    root = Path(checkpoint)
    candidate = root / "train_rtc_record.json"
    if candidate.is_file():
        return candidate.resolve()
    # New training archives keep the effective dataclass config alongside the
    # step directory, or inside the run's reproducibility directory.
    for candidate in (root / "effective_config.json", root / "reproducibility/effective_config.json",
                      root.parent / "reproducibility/effective_config.json"):
        if candidate.is_file():
            return candidate.resolve()
    manifest = root / "params/manifest.ocdbt"
    metadata = root / "params/_METADATA"
    if not manifest.is_file() or not metadata.is_file():
        return None
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for candidate in sorted((Path(__file__).parent / "training_records").glob("*.json")):
        record = json.loads(candidate.read_text())
        if record.get("checkpoint_manifest_sha256") == digest:
            if record.get("checkpoint_metadata_sha256") != hashlib.sha256(metadata.read_bytes()).hexdigest():
                raise ValueError("Known Train RTC checkpoint metadata hash mismatch")
            return candidate.resolve()
    return None


def load_training_record(path, stats_path, *, paper_algorithm=None):
    """Normalize legacy run JSON and new effective_config.json archives."""
    record = json.loads(Path(path).read_text())
    if "config" in record:
        return record
    fields = record.get("fields")
    if not isinstance(fields, dict) or "model" not in fields or "data" not in fields:
        raise ValueError("训练JSON必须是运行记录或effective_config.json")
    model = fields["model"]["fields"]
    # An effective config lacks legacy inputs hashes. Bind it to the verified
    # Paper bundle, or compare the adjacent archived statistics for plain RTC.
    if paper_algorithm:
        if model.get("paper_asset_sha256") != paper_algorithm["bundle_sha256"]:
            raise ValueError("训练配置的Paper资产摘要与checkpoint不一致")
    else:
        adjacent = Path(path).parent / "norm_stats.json"
        if not adjacent.is_file():
            raise ValueError("effective_config旁缺少norm_stats.json；请提供运行JSON或使用手动训练确认")
        expected = json.loads(adjacent.read_text())
        actual = json.loads(Path(stats_path).read_text())
        expected = expected.get("norm_stats", expected)
        # Training reproducibility serializes NormStats dataclasses with their
        # type/fields envelope; checkpoint assets serialize their numeric fields.
        # Unwrap only this exact type and retain exact numeric equality.
        expected = {
            key: value["fields"] if isinstance(value, dict)
            and value.get("type") == "openpi.shared.normalize.NormStats"
            and set(value) == {"type", "fields"} else value
            for key, value in expected.items()
        }
        if expected != actual.get("norm_stats", actual):
            raise ValueError("effective_config归档统计与checkpoint不一致")
    return {"config": record, "inputs": {
        "norm_stats_sha256": hashlib.sha256(Path(stats_path).read_bytes()).hexdigest(),
        "paper_bundle_sha256": paper_algorithm["bundle_sha256"] if paper_algorithm else None}}


def inspect_train_rtc(stats_path, checkpoint, record_path=None, *, paper_algorithm=None,
                      manual_delay=None, manual_semantics=None, model_type=None):
    selected = model_type or os.environ.get("SHARPA_DEPLOY_MODEL_TYPE", "auto")
    manual = selected != "auto"
    train_types = ("train-rtc-absolute", "train-rtc-delta", "train-paper-rtc")
    # Explicit selection does not discover or copy adjacent training records.
    # An explicitly supplied record remains an optional cross-check.
    explicit = record_path or os.environ.get("SHARPA_DEPLOY_TRAINING_RECORD")
    path = (Path(explicit).expanduser().resolve() if explicit else None) if manual else resolve_training_record(checkpoint, record_path)
    if manual and selected not in train_types:
        if path:
            fields = load_training_record(path, stats_path, paper_algorithm=paper_algorithm)["config"]["fields"]
            if fields["model"]["fields"].get("rtc_max_delay", 0):
                raise ValueError("所选模型类型与提供的Train RTC训练记录冲突")
        return None
    if manual and selected in train_types:
        if (selected == "train-paper-rtc") != bool(paper_algorithm):
            raise ValueError("手动模型类型与Paper算法资产不一致")
        delay = manual_delay if manual_delay is not None else int(os.environ.get("SHARPA_TRAIN_RTC_MAX_DELAY", "0"))
        if not delay:
            raise ValueError("手动选择Train RTC后，请填写训练RTC最大延迟（1..49）")
    if path is None:
        delay = manual_delay if manual_delay is not None else int(os.environ.get("SHARPA_TRAIN_RTC_MAX_DELAY", "0"))
        if not delay:
            return None
        if type(delay) is not int or not 0 < delay < 50:
            raise ValueError("手动训练RTC最大延迟必须为1..49整数")
        semantics = ("absolute" if selected == "train-rtc-absolute" else "delta") if manual else (
            "delta" if paper_algorithm else (manual_semantics or os.environ.get("SHARPA_TRAIN_RTC_SEMANTICS")))
        if semantics not in ("absolute", "delta"):
            raise ValueError("手动训练RTC需明确选择Absolute或Delta动作类型")
        binding = {"max_delay": delay, "semantics": semantics, "paper": paper_algorithm,
                   "manifest": hashlib.sha256((Path(checkpoint) / "params/manifest.ocdbt").read_bytes()).hexdigest(),
                   "stats": hashlib.sha256(Path(stats_path).read_bytes()).hexdigest()}
        return {"max_delay": delay, "record": None, "manual_confirmation": True,
                "record_sha256": hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest(),
                "semantics": semantics}
    path = Path(path).expanduser().resolve()
    record = load_training_record(path, stats_path, paper_algorithm=paper_algorithm)
    fields = record["config"]["fields"]
    model, data = fields["model"]["fields"], fields["data"]["fields"]
    delay = model.get("rtc_max_delay", 0)
    if manual_delay and manual_delay != delay:
        raise ValueError("手动RTC延迟与训练记录冲突")
    if delay == 0:
        return None
    if type(delay) is not int or not 0 < delay < 50:
        raise ValueError("Train RTC rtc_max_delay must be an integer in 1..49")
    if (
        model.get("pi05"),
        model["action_dim"],
        model["action_horizon"],
        model["max_token_len"],
        model["paligemma_variant"],
        model["action_expert_variant"],
        data["action_hz"],
    ) != (True, 58, 50, 256, "gemma_2b", "gemma_300m", 30):
        raise ValueError("Train RTC requires full pi05 58D/50/256/30Hz training evidence")
    digest = hashlib.sha256(Path(stats_path).read_bytes()).hexdigest()
    if record["inputs"].get("norm_stats_sha256") != digest:
        raise ValueError("Train RTC training record normalization hash mismatch")
    for name, relative in (("checkpoint_manifest_sha256", "params/manifest.ocdbt"),
                           ("checkpoint_metadata_sha256", "params/_METADATA")):
        if record.get(name) and record[name] != hashlib.sha256((Path(checkpoint) / relative).read_bytes()).hexdigest():
            raise ValueError("Train RTC checkpoint fingerprint mismatch")
    paper_keys = ("paper_mode", "paper_beta", "paper_std_floor", "paper_asset_sha256")
    has_paper = any(key in model for key in paper_keys)
    if has_paper != bool(paper_algorithm):
        raise ValueError("Train RTC training record Paper algorithm mismatch")
    if paper_algorithm:
        if not data["extra_delta_transform"]:
            raise ValueError("Train Paper RTC requires arm delta supervision")
        for key, asset_key in zip(paper_keys, ("mode", "beta", "std_floor", "bundle_sha256"), strict=True):
            if model.get(key) != paper_algorithm[asset_key]:
                raise ValueError(f"Train Paper RTC training record {key} mismatch")
            if key in data and data[key] != model[key]:
                raise ValueError(f"Train Paper RTC model/data {key} mismatch")
        if record["inputs"].get("paper_bundle_sha256") != paper_algorithm["bundle_sha256"]:
            raise ValueError("Train Paper RTC training bundle hash mismatch")
    return {
        "max_delay": delay,
        "record": str(path),
        "record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "semantics": "delta" if data["extra_delta_transform"] else "absolute",
    }


def require_train_capabilities(metadata):
    if (
        metadata.get("rtc_algorithm") != "trained_prefix_conditioning"
        or metadata.get("rtc_warmed_up") is not True
        or metadata.get("rtc_supported") is not True
        or metadata.get("rtc_protocol_version") != 1
        or metadata.get("rtc_action_dim") != 58
        or not metadata.get("train_rtc")
        or metadata.get("rtc_warmed_prefix_lengths") != list(range(metadata["train_rtc"]["max_delay"] + 1))
    ):
        raise ValueError("Train RTC requires --enable-train-rtc with all supported delays warmed")


def require_train_paper_capabilities(metadata):
    require_train_capabilities(metadata)
    from .rtc_stream import PAPER_RTC_ADAPTER
    if (not metadata.get("paper_algorithm") or metadata.get("rtc_input_adapter") != PAPER_RTC_ADAPTER
            or metadata.get("rtc_short_prefix_warmed_up") is not True):
        raise ValueError("Train Paper RTC requires --enable-train-paper-rtc with warmed short references")
