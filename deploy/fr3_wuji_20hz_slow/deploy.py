"""Slow deployment for the 50x54 tomato policy recorded at 20 Hz."""

from pathlib import Path

from deploy.fr3_wuji_runtime import hardware
from deploy.fr3_wuji_slow import deploy as slow_deploy

from .contract import checkpoint_contract

DEFAULT_CHECKPOINT = hardware.ROOT / "checkpoints/tomato_lora_0918_20hz/19999"
CONFIG = Path(__file__).with_name("config.yaml")


def parse_args(argv=None):
    return slow_deploy.parse_args(
        argv,
        default_checkpoint=DEFAULT_CHECKPOINT,
        output_prefix="slow20-",
    )


def main(argv=None):
    return slow_deploy.main(
        argv,
        config_path=CONFIG,
        contract=checkpoint_contract,
        default_checkpoint=DEFAULT_CHECKPOINT,
        output_prefix="slow20-",
        profile_name="20hz-slow",
    )


if __name__ == "__main__":
    main()
