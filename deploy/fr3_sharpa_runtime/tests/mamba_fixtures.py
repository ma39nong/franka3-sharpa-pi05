"""Explicitly synthetic fixtures. No fixture is evidence of trained task performance."""

from contextlib import contextmanager
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
from unittest.mock import patch

from flax import nnx
from flax import traverse_util
import jax
import numpy as np

from deploy.fr3_sharpa_runtime.mamba_contract import SCHEMA
from deploy.fr3_sharpa_runtime.mamba_contract import UPSTREAM
from openpi.models import gemma
from openpi.models import siglip
from openpi.models.mamba3.config import Mamba3Config
from openpi.models.pi0_mamba_config import MambaPi0Config


def tiny_config(**kwargs):
    return MambaPi0Config(
        diagnostic_only=True,
        paligemma_variant="dummy",
        dtype="float32",
        max_token_len=8,
        mamba=Mamba3Config(hidden_size=16, state_size=8, head_dim=8, dtype="float32"),
        mamba_depth=2,
        mamba_mlp_dim=32,
        film_mlp_dim=32,
        context_heads=2,
        context_vision_tokens=5,
        context_language_tokens=3,
        context_global_tokens=2,
        **kwargs,
    )


@contextmanager
def tiny_network():
    original_gemma, original_siglip = gemma.get_config, siglip.Module

    def small_gemma(variant):
        if variant == "dummy":
            return replace(original_gemma(variant), width=16, depth=2, mlp_dim=32, num_heads=2, head_dim=8)
        return original_gemma(variant)

    def small_siglip(*args, **kwargs):
        return original_siglip(
            *args,
            **{
                **kwargs,
                "width": 16,
                "depth": 2,
                "mlp_dim": 32,
                "num_heads": 2,
                "patch_size": (112, 112),
                "head_zeroinit": False,
            },
        )

    with (
        patch.object(gemma, "get_config", small_gemma),
        patch.object(gemma, "PALIGEMMA_VOCAB_SIZE", 128),
        patch.object(siglip, "Module", small_siglip),
    ):
        yield


@lru_cache
def production_entries():
    model = nnx.eval_shape(MambaPi0Config().create, jax.random.key(0))
    flat = traverse_util.flatten_dict(nnx.state(model).to_pure_dict())
    return {
        str(path): {
            "key_metadata": [{"key": key} for key in ("params", *path, "value")],
            "value_metadata": {
                "value_type": "None" if value is None else "jax.Array",
                "skip_deserialize": value is None,
                **({"write_shape": list(value.shape)} if value is not None else {}),
            },
        }
        for path, value in flat.items()
    }


def checkpoint(root, *, paper=False, delay=0):
    if paper:
        from deploy.fr3_sharpa_runtime.tests.test_paper_deployment import checkpoint as paper_fixture

        root, assets, _ = paper_fixture.__wrapped__(root)
    else:
        root = root / "20"
        assets = root / "assets/absolute"
        assets.mkdir(parents=True)
        (root / "params").mkdir()
        (root / "params/manifest.ocdbt").write_bytes(b"synthetic contract fixture, no weights")
        stats = {
            k: {
                "mean": np.linspace(-0.1, 0.1, 58).tolist(),
                "std": np.linspace(0.2, 2.0, 58).tolist(),
                "q01": [-2.0] * 58,
                "q99": [2.0] * 58,
            }
            for k in ("state", "actions")
        }
        (assets / "norm_stats.json").write_text(json.dumps({"norm_stats": stats}))
    model = json.loads(SCHEMA.read_text())["paper" if paper else "plain"]
    model["rtc_max_delay"] = delay
    if paper:
        algorithm = json.loads((assets / "delta_paper.json").read_text())
        model.update(
            paper_mode=algorithm["mode"],
            paper_beta=algorithm["beta"],
            paper_asset_sha256=algorithm["bundle_sha256"],
            paper_asset_dir="/archived/training/machine",
        )
    record = {
        "schema_version": 1,
        "upstream": UPSTREAM,
        "model": model,
        "norm_stats_sha256": hashlib.sha256((assets / "norm_stats.json").read_bytes()).hexdigest(),
    }
    if paper:
        record["paper_files_sha256"] = {
            n: hashlib.sha256((assets / n).read_bytes()).hexdigest()
            for n in ("delta_paper.json", "delta_paper.npz", "norm_metadata.json")
        }
    (assets / "mamba_architecture.json").write_text(json.dumps(record))
    (root / "params/_METADATA").write_text(json.dumps({"tree_metadata": production_entries()}))
    return root, assets, record


def observation():
    obs = {"observation/state": np.linspace(-0.4, 0.4, 58, dtype=np.float32), "prompt": "pick tomato"}
    for i, key in enumerate(("observation/image", "observation/left_wrist_image", "observation/right_wrist_image")):
        obs[key] = np.full((224, 224, 3), i * 70, np.uint8)
    return obs


def select(monkeypatch, root, assets, name):
    from deploy.fr3_sharpa_runtime import config
    from deploy.fr3_sharpa_runtime.model_profiles import get_model_profile

    profile = get_model_profile(name)
    for key, value in {
        "CHECKPOINT": root,
        "NORM_STATS": assets / "norm_stats.json",
        "MODEL_PROFILE": profile,
        "CONFIG_NAME": profile.config_name,
        "_validation_cache": None,
    }.items():
        monkeypatch.setattr(config, key, value)


def record_evidence(name, values):
    import os
    from pathlib import Path

    directory = os.environ.get("MAMBA_SMOKE_EVIDENCE")
    if directory:
        path = Path(directory) / "numerical_evidence.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        records = json.loads(path.read_text()) if path.exists() else {}
        records[name] = values
        path.write_text(json.dumps(records, indent=2) + "\n")
