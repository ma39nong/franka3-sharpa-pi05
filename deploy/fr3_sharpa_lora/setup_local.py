"""Save LoRA-only model defaults, without changing hardware/full-model files."""
import argparse
import json

from deploy.fr3_sharpa_runtime.model_profiles import get_model_profile, validate_parameter_layout
from pathlib import Path
import shlex


def write_defaults(checkpoint, *, output, norm_stats=None, port=18766, force=False):
    checkpoint = Path(checkpoint).expanduser().resolve()
    output = Path(output).expanduser().absolute()
    if not 1 <= port <= 65535:
        raise ValueError("Invalid LoRA service port")
    for file in ("params/manifest.ocdbt", "params/_METADATA"):
        if not (checkpoint / file).is_file():
            raise FileNotFoundError(checkpoint / file)
    entries = json.loads((checkpoint / "params/_METADATA").read_text())["tree_metadata"]
    validate_parameter_layout(entries, get_model_profile("lora"))
    if norm_stats is not None:
        norm_stats = Path(norm_stats).expanduser().resolve()
        if not norm_stats.is_file():
            raise FileNotFoundError(norm_stats)
    elif len(list((checkpoint / "assets").rglob("norm_stats.json"))) != 1:
        raise ValueError("Include exactly one checkpoint-owned norm_stats.json or pass --norm-stats")
    values = {"SHARPA_LORA_CHECKPOINT": str(checkpoint), "SHARPA_LORA_PORT": str(port)}
    if norm_stats is not None:
        values["SHARPA_LORA_NORM_STATS"] = str(norm_stats)
    text = "# LoRA-only defaults; existing caller environment takes precedence.\n"
    for name, value in values.items():
        text += f'if [[ -z "${{{name}:-}}" ]]; then export {name}={shlex.quote(value)}; fi\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if force else "x", encoding="utf-8") as stream:
        stream.write(text)
    return output


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument("--port", type=int, default=18766)
    parser.add_argument("--output", type=Path, default=root / ".deployment/lora.env")
    parser.add_argument("--force", action="store_true", help="Replace an existing LoRA defaults file")
    args = parser.parse_args()
    output = write_defaults(args.checkpoint, output=args.output, norm_stats=args.norm_stats,
                            port=args.port, force=args.force)
    print(f"Prepared {output}; full local.env, Home and gateway configurations unchanged")
    print("Next: bash deploy/fr3_sharpa_lora/serve.sh --check-only")


if __name__ == "__main__":
    main()
