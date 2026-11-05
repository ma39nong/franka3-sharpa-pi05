"""Disk-only Mamba identity gate. No JAX, model initialization, ROS or SDK imports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ARCHITECTURE = "pi05_mamba3_jax"
SOURCE_COMMIT = "0117bb5bdc26874082cb462764131cfad037df7a"
UPSTREAM = {
    "mamba3_jax": "ffa55848fc513ca91aa330bda4c33c86dab74827",
    "MambaAct": "97414cfffc82225c6be34abf486f06c7eb11d22d",
}
IDENTITY_KEYS = (
    "model_architecture",
    "mamba_manifest_sha256",
    "mamba_stage",
    "mamba_config_hash",
    "mamba_normalization",
    "mamba_provenance",
    "checkpoint_metadata_sha256",
)
SCHEMA = Path(__file__).resolve().parents[2] / "third_party/mamba3/config_schema.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_hash(model):
    return hashlib.sha256(
        json.dumps(model, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def architecture_files(checkpoint):
    return sorted((Path(checkpoint) / "assets").rglob("mamba_architecture.json"))


def _model_fields(model, *, paper):
    template = json.loads(SCHEMA.read_text())["paper" if paper else "plain"]
    if not isinstance(model, dict) or set(model) != set(template):
        raise ValueError("Mamba model schema fields differ from the supported training config")
    fixed = (
        "architecture",
        "stage",
        "action_representation",
        "normalization",
        "pi05",
        "discrete_state_input",
        "diagnostic_only",
        "action_dim",
        "action_horizon",
        "max_token_len",
        "paligemma_variant",
        "action_expert_variant",
    )
    for key in fixed:
        if model[key] != template[key] or type(model[key]) is not type(template[key]):
            raise ValueError(f"Unsupported Mamba {key}: {model[key]!r}")
    if model["dtype"] not in ("bfloat16", "float32") or model["pytorch_compile_mode"] not in (
        None,
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    ):
        raise ValueError("Unsupported Mamba dtype/compile mode")
    delay = model["rtc_max_delay"]
    if type(delay) is not int or not 0 <= delay < 50:
        raise ValueError("Mamba rtc_max_delay must be an integer in 0..49")
    for key in (
        "mamba_depth",
        "mamba_mlp_dim",
        "film_mlp_dim",
        "context_heads",
        "context_vision_tokens",
        "context_language_tokens",
        "context_global_tokens",
    ):
        if type(model[key]) is not int or model[key] <= 0:
            raise ValueError(f"Invalid Mamba {key}")
    mixer = model["mamba"]
    if not isinstance(mixer, dict) or set(mixer) != set(template["mamba"]):
        raise ValueError("Unsupported nested Mamba config schema")
    for key in ("hidden_size", "state_size", "head_dim", "expand", "ngroups", "chunk_size"):
        if type(mixer[key]) is not int or mixer[key] <= 0:
            raise ValueError(f"Invalid Mamba mixer {key}")
    for key in ("rope_fraction", "A_floor", "bcnorm_eps", "time_step_min", "time_step_max", "time_step_floor"):
        if type(mixer[key]) not in (int, float) or not np.isfinite(mixer[key]) or mixer[key] <= 0:
            raise ValueError(f"Invalid Mamba mixer {key}")
    width, head, groups = mixer["hidden_size"] * mixer["expand"], mixer["head_dim"], mixer["ngroups"]
    if (
        width % head
        or (width // head) % groups
        or mixer["hidden_size"] % model["context_heads"]
        or mixer["state_size"] % 2
        or mixer["rope_fraction"] not in (0.5, 1.0)
        or int(mixer["state_size"] * mixer["rope_fraction"]) // 2 <= 0
        or not 0 < mixer["time_step_floor"] <= mixer["time_step_min"] <= mixer["time_step_max"]
        or mixer["dtype"] != model["dtype"]
    ):
        raise ValueError("Invalid Mamba dimensions, precision or SSM numerics")
    if paper:
        from openpi.shared import delta_paper

        if (
            model["paper_mode"] not in delta_paper.MODES
            or type(model["paper_beta"]) not in (int, float)
            or not 0 <= model["paper_beta"] < 1
            or type(model["paper_std_floor"]) not in (int, float)
            or not np.isfinite(model["paper_std_floor"])
            or model["paper_std_floor"] <= 0
            or not isinstance(model["paper_asset_dir"], str)
            or not isinstance(model["paper_asset_sha256"], str)
            or len(model["paper_asset_sha256"]) != 64
        ):
            raise ValueError("Invalid Mamba Paper config")


def reject_mamba_without_manifest(entries):
    for entry in entries.values():
        names = [str(part["key"]) for part in entry.get("key_metadata", ())]
        if "context_compressor" in names or "action_expert" in names:
            raise ValueError("Mamba parameter tree requires its architecture manifest and explicit Mamba profile")


def validate_layout(entries, model):
    paths = set()
    for key, entry in entries.items():
        names = tuple(str(part["key"]) for part in entry["key_metadata"])
        if names[:1] == ("params",):
            names = names[1:]
        if names[-1:] == ("value",):
            names = names[:-1]
        if not names or "lora" in str(key).lower() or any("lora" in n.lower() for n in names):
            raise ValueError("Mamba FullFT rejects LoRA/empty parameter paths")
        if names in paths:
            raise ValueError("Duplicate Mamba parameter path")
        paths.add(names)
    roots = {p[0] for p in paths}
    expected = {"PaliGemma", "action_expert", "context_compressor"}
    if model.get("paper_beta", 0):
        expected.add("paper_noise_cholesky")
    if roots != expected:
        raise ValueError(f"Mamba parameter roots differ: {roots}")
    if {p[1] for p in paths if p[0] == "PaliGemma"} != {"llm", "img"}:
        raise ValueError("Mamba requires complete PaliGemma VLM")
    if any(part.endswith("_1") for p in paths if p[:2] == ("PaliGemma", "llm") for part in p[2:]):
        raise ValueError("Mamba checkpoint contains a Transformer Action Expert")
    required = {
        (root, name, leaf)
        for root, names in (
            ("action_expert", ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out")),
            ("context_compressor", ("context_proj", "query", "key", "value", "out", "mlp_in", "mlp_out")),
        )
        for name in names
        for leaf in ("kernel", "bias")
    }
    required.add(("action_expert", "final_norm", "weight"))
    for index in range(model["mamba_depth"]):
        prefix = ("action_expert", "blocks", f"block_{index}")
        for name in ("cond_proj", "film_in", "film_out", "ffn_in", "ffn_out"):
            required.update({(*prefix, name, "kernel"), (*prefix, name, "bias")})
        required.update({(*prefix, name, "weight") for name in ("mamba_norm", "film_norm", "ffn_norm")})
        required.update(
            {(*prefix, "mamba", name, leaf) for name in ("in_proj", "out_proj") for leaf in ("kernel", "bias")}
        )
        required.update({(*prefix, "mamba", name) for name in ("B_bias", "C_bias", "dt_bias", "D")})
        required.update({(*prefix, "mamba", name, "weight") for name in ("B_norm", "C_norm")})
    required.update(
        {
            ("context_compressor", n)
            for n in ("vision_type_embed", "language_type_embed", "global_type_embed", "global_queries")
        }
    )
    required.add(("context_compressor", "norm", "weight"))
    for entry in entries.values():
        names = tuple(str(part["key"]) for part in entry["key_metadata"])
        if (
            "mamba" in names
            and names[-2:] == ("bias", "value")
            and any(n in names for n in ("in_proj", "out_proj"))
            and entry["value_metadata"].get("value_type") != "None"
        ):
            raise ValueError("Mamba mixer projections must have no bias")
    non_vlm = {p for p in paths if p[0] not in ("PaliGemma", "paper_noise_cholesky")}
    if non_vlm != required:
        raise ValueError(
            f"Mamba expert/compressor tree differs: missing={sorted(required - non_vlm)}, extra={sorted(non_vlm - required)}"
        )
    # write_shape is a shard extent, never evidence of global tensor shape.
    # Complete VLM and global shape/dtype equality are checked again by model.load.


def inspect_mamba(checkpoint, stats_path, *, profile=None):
    checkpoint, stats_path = Path(checkpoint).resolve(), Path(stats_path).resolve()
    files = architecture_files(checkpoint)
    if len(files) != 1 or files[0].parent.resolve() != stats_path.parent:
        raise ValueError("Mamba requires exactly one checkpoint-owned mamba_architecture.json next to norm_stats.json")
    norm_files = list((checkpoint / "assets").rglob("norm_stats.json"))
    if len(norm_files) != 1 or norm_files[0].resolve() != stats_path:
        raise ValueError("Mamba statistics must be uniquely owned by the checkpoint")
    path = files[0]
    record = json.loads(path.read_text())
    if (
        type(record.get("schema_version")) is not int
        or record["schema_version"] != 1
        or record.get("upstream") != UPSTREAM
    ):
        raise ValueError("Unsupported Mamba schema/upstream revisions")
    model = record.get("model")
    paper = isinstance(model, dict) and model.get("stage") == "paper_delta_fm"
    expected_record = {"schema_version", "model", "upstream", "norm_stats_sha256"}
    if paper:
        expected_record.add("paper_files_sha256")
    if set(record) != expected_record:
        raise ValueError("Mamba architecture record schema differs")
    _model_fields(model, paper=paper)
    selected = "mamba-delta" if paper else "mamba"
    if profile is not None and profile.name != selected:
        raise ValueError(f"Mamba checkpoint requires --model {selected}")
    if sha256(stats_path) != record["norm_stats_sha256"]:
        raise ValueError("Mamba normalization SHA256 mismatch")
    if (checkpoint / "model.safetensors").exists():
        raise ValueError("Mamba deployment requires an Orbax JAX checkpoint, excludes PyTorch weights")
    manifest, tree = checkpoint / "params/manifest.ocdbt", checkpoint / "params/_METADATA"
    if not manifest.is_file() or not tree.is_file() or not manifest.stat().st_size:
        raise ValueError("Missing complete Orbax params metadata/manifest")
    entries = json.loads(tree.read_text())["tree_metadata"]
    validate_layout(entries, model)
    from .paper_contract import inspect_paper
    from .paper_contract import validate_stats

    if paper:
        names = {"delta_paper.npz", "delta_paper.json", "norm_metadata.json"}
        hashes = record["paper_files_sha256"]
        if set(hashes) != names or any(sha256(stats_path.parent / name) != hashes[name] for name in names):
            raise ValueError("Mamba Paper asset SHA256 mismatch")
        # Legacy Paper's disk inspector expects an unsharded noise shape. Its
        # algorithm/bundle checks remain useful; global shapes are validated on restore.
        paper_entries = {
            k: {**v, "value_metadata": {**v.get("value_metadata", {}), "write_shape": [2900, 2900]}}
            if "paper_noise_cholesky" in k
            else v
            for k, v in entries.items()
        }
        algorithm = inspect_paper(stats_path, paper_entries)
        for field, key in (
            ("paper_mode", "mode"),
            ("paper_beta", "beta"),
            ("paper_std_floor", "std_floor"),
            ("paper_asset_sha256", "bundle_sha256"),
        ):
            if model[field] != algorithm[key]:
                raise ValueError(f"Mamba {field} differs from authenticated Paper assets")
    else:
        if any(p.name.startswith("delta_paper") or "paper" in p.name.lower() for p in stats_path.parent.iterdir()):
            raise ValueError("Plain Mamba rejects Paper assets")
        algorithm = None
    stats = json.loads(stats_path.read_text())["norm_stats"]
    if set(stats) != {"state", "actions"}:
        raise ValueError("Mamba requires state/actions statistics only")
    validate_stats(stats, paper_algorithm=algorithm)
    digest = sha256(path)
    trained = (
        {
            "max_delay": model["rtc_max_delay"],
            "record": str(path),
            "record_sha256": digest,
            "semantics": "delta" if paper else "absolute",
            "evidence": "mamba_architecture_schema_v1",
        }
        if model["rtc_max_delay"]
        else None
    )
    return {
        "model_architecture": ARCHITECTURE,
        "mamba_manifest_sha256": digest,
        "mamba_stage": model["stage"],
        "mamba_config_hash": config_hash(model),
        "mamba_normalization": model["normalization"],
        "mamba_provenance": {"supported_source_commit": SOURCE_COMMIT, "upstream": record["upstream"]},
        "checkpoint_manifest_sha256": sha256(manifest),
        "checkpoint_metadata_sha256": sha256(tree),
        "normalization_sha256": sha256(stats_path),
        "paper_algorithm": algorithm,
        "train_rtc": trained,
        "model_profile": selected,
        "params_leaves": len(entries),
        "mamba_model": model,
    }


def reconstruct_model(metadata, stats_path):
    """Imports JAX only after the disk gate succeeds; never imports training UI/loaders."""
    from openpi.models.mamba3.config import Mamba3Config
    from openpi.models.pi0_mamba_config import MambaPi0Config
    from openpi.models.pi0_mamba_paper import PaperMambaPi0Config

    values = dict(metadata["mamba_model"])
    values["mamba"] = Mamba3Config(**values["mamba"])
    cls = MambaPi0Config
    if metadata["paper_algorithm"]:
        cls = PaperMambaPi0Config
        values["paper_asset_dir"] = str(Path(stats_path).parent)
    return cls(**values)
