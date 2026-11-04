"""Seal an already completed training step; no model creation or dataset scan."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from deploy.fr3_sharpa_runtime.dataset_sample import ORDER
from deploy.fr3_sharpa_runtime.flash_contract import BACKEND_FILES
from deploy.fr3_sharpa_runtime.flash_contract import CAMERAS
from deploy.fr3_sharpa_runtime.flash_contract import CAPABILITIES
from deploy.fr3_sharpa_runtime.flash_contract import PREPROCESSOR
from deploy.fr3_sharpa_runtime.flash_contract import RESAMPLER
from deploy.fr3_sharpa_runtime.flash_contract import inspect_flash
from deploy.fr3_sharpa_runtime.flash_contract import parse_model_repr
from deploy.fr3_sharpa_runtime.flash_contract import sha


def export(checkpoint, repo, tokenizer, training_contract, validation=None, step=None):
    root, repo = Path(checkpoint).resolve(), Path(repo).resolve()
    if (root / "flash_deployment.json").exists():
        raise FileExistsError("FLASH package already sealed; export to another complete package to revise it")
    if not (root / "_CHECKPOINT_METADATA").is_file():
        raise ValueError("Missing completed Orbax step _CHECKPOINT_METADATA; do not export an in-progress save")
    completion = json.loads((root / "_CHECKPOINT_METADATA").read_text())
    if step is None:
        step = int(root.name) if root.name.isdigit() else None
    if type(step) is not int or step < 0 or not completion.get("commit_timestamp_nsecs"):
        raise ValueError("FLASH export requires a numbered, committed Orbax step")
    if any("tmp" in p.name.lower() for p in (root / "params").rglob("*")):
        raise ValueError("Temporary parameter files present")
    paths = list((root / "assets").rglob("norm_stats.json"))
    if len(paths) != 1:
        raise ValueError("FLASH requires unique checkpoint-owned statistics")
    stats = paths[0]
    training = json.loads(Path(training_contract).read_text())
    model = parse_model_repr(training["model"])
    # Include actual provenance and tokenizer bytes, rather than server-only paths.
    dependencies = root / "flash_assets"
    dependencies.mkdir(exist_ok=True)
    for source, name in ((training_contract, "flash_training_contract.json"), (tokenizer, "tokenizer.model")):
        destination = dependencies / name
        if destination.exists():
            if sha(destination) != sha(source):
                raise ValueError(f"Refusing to replace existing {name}")
        else:
            shutil.copyfile(source, destination)
    validation_name = None
    if validation:
        validation_name = "flash_assets/execution_validation.json"
        shutil.copyfile(validation, root / validation_name)
    files = [p for p in (root / "params").rglob("*") if p.is_file()]
    files += [
        stats.parent / name
        for name in ("norm_stats.json", "flash_manifest.json", "flash_seal.json", "flash_validation.json")
    ]
    files += [*list(dependencies.iterdir()), root / "_CHECKPOINT_METADATA"]
    # Check that these are exactly the source files used by train_flash.py's seal.
    source_files = (
        "scripts/train_flash.py",
        "src/openpi/training/flash_native.py",
        "src/openpi/models/pi0_flash.py",
        "src/openpi/shared/flash_polynomial.py",
        "src/openpi/training/optimizer.py",
        "src/openpi/models/pi0.py",
        "src/openpi/models/gemma.py",
        "src/openpi/models/siglip.py",
        "scripts/train.py",
    )
    from deploy.fr3_sharpa_runtime.flash_contract import digest_json

    if digest_json({name: (repo / name).read_text() for name in source_files}) != training["training_code_sha256"]:
        raise ValueError("Training source fingerprint differs from supplied FLASH repository")
    manifest = {
        **CAPABILITIES,
        "schema_version": 1,
        "checkpoint_step": step,
        "model_config": model,
        "joint_names": list(ORDER),
        "camera_mapping": CAMERAS,
        "norm_stats": str(stats.relative_to(root)),
        "tokenizer": "flash_assets/tokenizer.model",
        "training_contract": "flash_assets/flash_training_contract.json",
        "preprocessing_version": PREPROCESSOR,
        "history_resampler_version": RESAMPLER,
        "history_max_sample_age_ns": 75_000_000,
        "history_max_gap_ns": 100_000_000,
        "execution_validation": validation_name,
        "backend": {
            "commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
            "files": {name: sha(repo / name) for name in BACKEND_FILES},
        },
        "files": {str(p.relative_to(root)): {"sha256": sha(p), "size": p.stat().st_size} for p in files},
    }
    path = root / "flash_deployment.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    try:
        return inspect_flash(root, backend=repo)
    except BaseException:
        path.unlink()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-repo", type=Path, required=True)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--execution-validation", type=Path)
    parser.add_argument("--step", type=int, help="Actual saved training step, required when directory was renamed")
    args = parser.parse_args()
    print(
        json.dumps(
            export(
                args.checkpoint,
                args.training_repo,
                args.tokenizer,
                args.training_contract,
                args.execution_validation,
                args.step,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
