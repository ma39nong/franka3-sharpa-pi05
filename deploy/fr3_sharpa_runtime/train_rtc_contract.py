"""Train RTC needs training evidence: tensor shapes cannot identify this algorithm."""

import hashlib
import json
import os
from pathlib import Path


def resolve_training_record(checkpoint, record_path=None):
    """Explicit/local evidence wins; shipped records require exact checkpoint fingerprints."""
    path = record_path or os.environ.get("SHARPA_TRAIN_RTC_RECORD")
    if path:
        return Path(path).expanduser().resolve()
    root = Path(checkpoint)
    candidate = root / "train_rtc_record.json"
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


def inspect_train_rtc(stats_path, checkpoint, record_path=None, *, paper_algorithm=None):
    path = resolve_training_record(checkpoint, record_path)
    if path is None:
        return None
    path = Path(path).expanduser().resolve()
    record = json.loads(path.read_text())
    fields = record["config"]["fields"]
    model, data = fields["model"]["fields"], fields["data"]["fields"]
    delay = model.get("rtc_max_delay", 0)
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
