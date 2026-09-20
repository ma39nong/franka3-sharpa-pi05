"""Run the 25000 transformation with 25000-single's distinct identity."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from deploy.fr3_wuji_models.model_25000 import serve as base
from deploy.fr3_wuji_models.model_25000_single.contract import DEFAULT_CHECKPOINT
from deploy.fr3_wuji_models.model_25000_single.contract import checkpoint_contract


def main(argv=None):
    return base.main(
        argv,
        contract=checkpoint_contract,
        default_checkpoint=DEFAULT_CHECKPOINT,
        default_port=8005,
    )


if __name__ == "__main__":
    main()
