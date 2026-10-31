"""Train RTC needs training evidence: tensor shapes cannot identify this algorithm."""

import hashlib
import json
import os
from pathlib import Path


def inspect_train_rtc(stats_path, checkpoint, record_path=None):
    path = record_path or os.environ.get("SHARPA_TRAIN_RTC_RECORD")
    if not path:
        candidate = Path(checkpoint) / "train_rtc_record.json"
        if not candidate.is_file():
            return None
        path = candidate
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
