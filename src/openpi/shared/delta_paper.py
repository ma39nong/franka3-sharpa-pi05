"""Versioned numerical helpers for opt-in Sharpa Delta paper experiments."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

VERSION = "sharpa-delta-paper-v1"
ARMS = np.r_[0:7, 29:36]
MODES = ("global_quantile", "global_zscore_arms", "timestamp_zscore_arms")
BUNDLE = "delta_paper.npz"
MANIFEST = "delta_paper.json"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_bundle(directory, *, mode, beta, horizon, std_floor, expected_sha256=None):
    directory = Path(directory)
    metadata = json.loads((directory / MANIFEST).read_text())
    expected = {
        "version": VERSION,
        "mode": mode,
        "beta": beta,
        "shape": [horizon, 58],
        "std_floor": std_floor,
        "flatten_order": "time-major",
        "windows": "complete_only",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Paper asset {key} differs from experiment; use matching new statistics")
    digest = sha256(directory / BUNDLE)
    if metadata.get("bundle_sha256") != digest or (expected_sha256 and expected_sha256 != digest):
        raise ValueError("Paper asset digest mismatch")
    if metadata.get("norm_stats_sha256") != sha256(directory / "norm_stats.json"):
        raise ValueError("Paper normalization digest mismatch")
    audit = json.loads((directory / "norm_metadata.json").read_text())
    if audit.get("norm_stats_sha256") != metadata["norm_stats_sha256"] or audit.get("contract") != metadata.get(
        "source_contract"
    ):
        raise ValueError("Paper bundle and normalization provenance differ")
    with np.load(directory / BUNDLE, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    for key in ("arm_mean", "arm_std"):
        if arrays[key].shape != (horizon, 14) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid paper {key}")
    if np.any(arrays["arm_std"] < 0):
        raise ValueError("Negative paper arm standard deviation")
    chol = arrays["noise_cholesky"]
    expected_shape = (horizon * 58, horizon * 58) if beta else (0, 0)
    if chol.shape != expected_shape or not np.isfinite(chol).all():
        raise ValueError("Invalid paper noise shape/values")
    if beta and (np.any(np.diag(chol) <= 0) or not np.allclose(chol, np.tril(chol), atol=1e-7)):
        raise ValueError("Paper noise factor must be lower triangular with positive diagonal")
    if metadata.get("training_windows", 0) < 2:
        raise ValueError("Paper statistics require at least two complete windows")
    return metadata, arrays


@dataclass(frozen=True)
class ArmNormalization:
    """Bridge existing global quantile normalization and opt-in arm z-scores.

    Hands are never changed. Input/output bridge around the unchanged shared
    Normalize/Unnormalize, so inverse order remains normalization then Absolute.
    """

    mode: str
    arm_mean: np.ndarray
    arm_std: np.ndarray
    q01: np.ndarray
    q99: np.ndarray
    std_floor: float
    bundle_sha256: str
    inverse: bool = False

    def __call__(self, data):
        if "actions" not in data:
            return data
        if self.mode not in MODES:
            raise ValueError("Unknown paper normalization mode")
        x = np.asarray(data["actions"])
        if x.shape[-2:] != self.q01.shape or not np.isfinite(x).all():
            raise ValueError("Paper normalization requires finite full-horizon actions")
        if self.mode == "global_quantile":
            return data
        result = x.copy()
        low, scale = self.q01[:, ARMS], self.q99[:, ARMS] - self.q01[:, ARMS] + 1e-6
        std = np.maximum(self.arm_std, self.std_floor)
        if self.inverse:
            raw = x[..., ARMS] * std + self.arm_mean
            result[..., ARMS] = (raw - low) / scale * 2 - 1
        else:
            raw = (x[..., ARMS] + 1) / 2 * scale + low
            result[..., ARMS] = (raw - self.arm_mean) / std
        return {**data, "actions": result}


class Moments:
    """Stable batch-Welford moments; first axis is samples, remaining axes survive."""

    def __init__(self):
        self.count = 0
        self.mean = self.m2 = None

    def update(self, value):
        value = np.asarray(value, np.float64)
        n = len(value)
        if n == 0 or not np.isfinite(value).all():
            raise ValueError("Moments require nonempty finite samples")
        mean = value.mean(axis=0)
        m2 = np.square(value - mean).sum(axis=0)
        if self.count:
            delta = mean - self.mean
            self.m2 += m2 + delta**2 * self.count * n / (self.count + n)
            self.mean += delta * n / (self.count + n)
        else:
            self.mean, self.m2 = mean, m2
        self.count += n

    def std(self):
        return np.sqrt(np.maximum(0, self.m2 / self.count))


def correlation_factor(samples, beta):
    """Unit-diagonal full correlation, constant dimensions independent, then shrink."""
    if not 0 <= beta <= 1:
        raise ValueError("Correlation beta must be in [0,1]")
    x = np.asarray(samples, np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Correlation needs at least two finite flattened windows")
    x = x - x.mean(axis=0)
    std = x.std(axis=0)
    active = std > 1e-8
    x[:, active] /= std[active]
    x[:, ~active] = 0
    correlation = (x.T @ x) / len(x)
    correlation = (correlation + correlation.T) * 0.5
    np.fill_diagonal(correlation, 1)
    regularized = beta * correlation + (1 - beta) * np.eye(x.shape[1])
    try:
        chol = np.linalg.cholesky(regularized)
    except np.linalg.LinAlgError as error:
        raise ValueError("Correlation shrinkage is not positive definite; reduce beta") from error
    return chol.astype(np.float32)
