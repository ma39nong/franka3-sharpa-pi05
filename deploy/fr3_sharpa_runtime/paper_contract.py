"""Inspect versioned paper assets without importing JAX, ROS or loading weights."""
import hashlib
import json
from pathlib import Path

import numpy as np

from openpi.shared import delta_paper as paper


def inspect_paper(stats_path, entries):
    directory = Path(stats_path).parent
    names = (paper.MANIFEST, paper.BUNDLE, "norm_metadata.json")
    has_state = any("paper_noise_cholesky" in key for key in entries)
    present = [(directory / name).is_file() for name in names[:2]]
    if not any(present) and not has_state:
        return None
    if not all((directory / name).is_file() for name in names):
        raise ValueError("Paper checkpoint缺少delta_paper.json/npz或norm_metadata.json；不能套用普通delta")
    metadata = json.loads((directory / paper.MANIFEST).read_text())
    mode, beta, floor = metadata.get("mode"), metadata.get("beta"), metadata.get("std_floor")
    if mode not in paper.MODES or not isinstance(beta, (int, float)) or not 0 <= beta < 1:
        raise ValueError("不支持的Paper归一化/相关噪声算法")
    if not isinstance(floor, (int, float)) or not np.isfinite(floor) or floor <= 0:
        raise ValueError("Invalid paper std_floor")
    metadata, _ = paper.load_bundle(directory, mode=mode, beta=beta, horizon=50, std_floor=floor)
    contract = metadata["source_contract"]
    if (contract.get("action_dim"), contract.get("action_horizon"), contract.get("action_hz")) != (58, 50, 30):
        raise ValueError("Paper assets必须是30Hz/50x58")
    declared = contract.get("vector_transforms", {}).get("delta_paper", {})
    for key, value in (("version", metadata["version"]), ("mode", mode), ("beta", beta), ("std_floor", floor)):
        if declared.get(key) != value:
            raise ValueError("Paper算法与归一化训练契约冲突")
    vectors = contract.get("vector_transforms", {}).get("vectors", [])
    expected = [i < 7 or 29 <= i < 36 for i in range(58)]
    masks = [v.get("fields", {}).get("mask") for v in vectors if v.get("type") == "openpi.transforms.DeltaActions"]
    if masks != [expected]:
        raise ValueError("Paper assets缺少双臂delta/双手absolute训练证据")
    shapes = [v["value_metadata"]["write_shape"] for k, v in entries.items() if "paper_noise_cholesky" in k]
    if shapes != ([[2900, 2900]] if beta else []):
        raise ValueError("Paper噪声参数与beta或50x58维度不一致")
    return {"version": metadata["version"], "mode": mode, "beta": beta, "std_floor": floor,
            "bundle_sha256": metadata["bundle_sha256"],
            "manifest_sha256": paper.sha256(directory / paper.MANIFEST),
            "audit_sha256": paper.sha256(directory / "norm_metadata.json")}


def validate_stats(stats, *, paper_algorithm=None):
    for group in ("state", "actions"):
        expected = (50, 58) if group == "actions" and paper_algorithm else (58,)
        for field in ("mean", "std", "q01", "q99"):
            value = np.asarray(stats[group][field], dtype=float)
            if value.shape != expected or not np.isfinite(value).all():
                raise ValueError(f"{group}.{field}需要{expected}有限统计；仅支持普通58D或有完整资产的Paper50x58")
        if np.any(np.asarray(stats[group]["std"]) < 0) or np.any(np.asarray(stats[group]["q01"]) > stats[group]["q99"]):
            raise ValueError(f"Invalid {group} normalization range")


def algorithm_identity(algorithm):
    return hashlib.sha256(json.dumps(algorithm, sort_keys=True).encode()).hexdigest() if algorithm else ""
