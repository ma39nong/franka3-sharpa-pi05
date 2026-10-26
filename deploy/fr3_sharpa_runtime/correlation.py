"""Versioned covariance assets for normalized training-space actions, not new noise."""
# ruff: noqa: PLC0415
# Lazy imports keep pure helpers hardware/framework-free and test patches local.

import hashlib
import json
from pathlib import Path

import numpy as np

from .rolling import model_identity

ASSET_VERSION = "sharpa-normalized-action-covariance-v1"


def correction_matrix(covariance, observed_count):
    sigma = np.asarray(covariance, dtype=np.float64)
    if (
        sigma.ndim != 2
        or sigma.shape[0] != sigma.shape[1]
        or not np.isfinite(sigma).all()
        or not np.allclose(sigma, sigma.T, atol=1e-6, rtol=1e-6)
        or np.any(np.diag(sigma) < 0)
        or not 0 < observed_count < len(sigma)
    ):
        raise ValueError("Invalid finite symmetric covariance")
    try:
        np.linalg.cholesky(sigma + np.eye(len(sigma)) * 1e-6)
        regularized = 0.5 * sigma + 0.5 * np.eye(len(sigma))
        # Solve OO * X = OU; never form the inverse.
        return np.linalg.solve(
            regularized[:observed_count, :observed_count], regularized[:observed_count, observed_count:]
        ).T.astype(np.float32)
    except np.linalg.LinAlgError as error:
        raise ValueError("Covariance is not positive semidefinite") from error


def load_correction(path, metadata):
    path = Path(path)
    with np.load(path, allow_pickle=False) as asset:
        info = json.loads(str(asset["metadata"].item()))
        if (
            info.get("version") != ASSET_VERSION
            or info.get("model_identity") != model_identity(metadata)
            or info.get("shape") != [50, 58]
            or info.get("flatten_order") != "time-major"
            or info.get("space") != "normalized_training_actions"
            or info.get("action_order") != metadata["action_order"]
            or info.get("training_action_supervision") != metadata["training_action_supervision"]
            or info.get("action_hz") != 30
            or info.get("sample_count", 0) < 2
        ):
            raise ValueError("Covariance asset identity/space does not match selected model")
        covariance = asset["covariance"]
        if covariance.shape != (2900, 2900):
            raise ValueError("Covariance must cover full 50x58 horizon")
        matrix = correction_matrix(covariance, 4 * 58)
    return matrix, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "beta": 0.5,
        "noise": "independent",
        "sample_count": info["sample_count"],
    }


def main():
    """Explicit NPZ input: absolute_actions (N,50,58), states (N,58). No devices."""
    import argparse

    from . import config
    from .model_loader import checkpoint_contract
    from .model_service import encode_reference

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    metadata = checkpoint_contract()
    from openpi import transforms
    from openpi.shared import normalize

    stats = normalize.load(config.NORM_STATS.parent)
    normalize_actions = transforms.Normalize(stats, use_quantiles=True)
    with np.load(args.input, allow_pickle=False) as data:
        actions, states = data["absolute_actions"], data["states"]
        if actions.ndim != 3 or actions.shape[1:] != (50, 58) or states.shape != (len(actions), 58) or len(actions) < 2:
            raise ValueError("Expected absolute_actions (N,50,58), states (N,58), N>=2")
        encoded = np.stack(
            [
                encode_reference(a, s, delta=config.MODEL_PROFILE.extra_delta_transform, normalizer=normalize_actions)
                for a, s in zip(actions, states, strict=True)
            ]
        )
    info = {
        "version": ASSET_VERSION,
        "model_identity": model_identity(metadata),
        "shape": [50, 58],
        "flatten_order": "time-major",
        "space": "normalized_training_actions",
        "action_order": metadata["action_order"],
        "action_hz": 30,
        "training_action_supervision": metadata["training_action_supervision"],
        "sample_count": len(encoded),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
    }
    covariance = np.cov(encoded.reshape(len(encoded), -1).astype(np.float64), rowvar=False)
    correction_matrix(covariance, 232)  # Validate before publishing a new asset.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, covariance=covariance, metadata=json.dumps(info))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
