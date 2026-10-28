"""Use the existing hardware-container ownership and confirmed-shutdown lifecycle."""

import sys

from .hardware_session import HardwareSession
from .hardware_session import run_session
from .shutdown import DisableSettings


def main():
    grace = DisableSettings.from_env().session_grace_s
    hardware = HardwareSession()
    raise SystemExit(
        run_session(
            [sys.executable, "-m", "deploy.fr3_sharpa_runtime.async_soft_run", *sys.argv[1:]],
            hardware,
            shutdown_grace_s=grace,
        )
    )


if __name__ == "__main__":
    main()
