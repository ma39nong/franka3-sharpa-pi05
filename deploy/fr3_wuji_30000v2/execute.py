"""Run the existing slow executor against the isolated 30000v2 service."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deploy.fr3_wuji_30000v2.contract import DEFAULT_CHECKPOINT
from deploy.fr3_wuji_30000v2.contract import DEFAULT_URI
from deploy.fr3_wuji_30000v2.contract import checkpoint_contract


def with_defaults(argv):
    args = list(argv)
    if "--checkpoint" not in args and not any(arg.startswith("--checkpoint=") for arg in args):
        args.extend(("--checkpoint", str(DEFAULT_CHECKPOINT)))
    if "--uri" not in args and not any(arg.startswith("--uri=") for arg in args):
        args.extend(("--uri", DEFAULT_URI))
    return args


def main(argv=None):
    from deploy.fr3_wuji_30000.planning_profile import install
    from experiments.weight_motion_eval.oneshot import deploy

    install()
    original_contract = deploy.checkpoint_contract
    deploy.checkpoint_contract = checkpoint_contract
    try:
        sys.argv = [sys.argv[0], *with_defaults(sys.argv[1:] if argv is None else argv)]
        deploy.main()
    finally:
        deploy.checkpoint_contract = original_contract


if __name__ == "__main__":
    main()
