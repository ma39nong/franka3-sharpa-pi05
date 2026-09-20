"""Serve the 30000v2 full-finetune checkpoint with the 54D hardware contract."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from deploy.fr3_wuji_models.model_30000 import serve as base
from deploy.fr3_wuji_models.model_30000v2.contract import DEFAULT_CHECKPOINT
from deploy.fr3_wuji_models.model_30000v2.contract import checkpoint_contract


def with_defaults(argv):
    args = list(argv)
    if "--checkpoint" not in args and not any(arg.startswith("--checkpoint=") for arg in args):
        args.extend(("--checkpoint", str(DEFAULT_CHECKPOINT)))
    if "--port" not in args and not any(arg.startswith("--port=") for arg in args):
        args.extend(("--port", "8003"))
    if "--full-finetune" not in args:
        args.append("--full-finetune")
    return args


def main(argv=None):
    # create_policy resolves this global when composing server metadata. The
    # replacement is limited to this dedicated server process.
    original_contract = base.checkpoint_contract
    base.checkpoint_contract = checkpoint_contract
    try:
        sys.argv = [sys.argv[0], *with_defaults(sys.argv[1:] if argv is None else argv)]
        base.main()
    finally:
        base.checkpoint_contract = original_contract


if __name__ == "__main__":
    main()
