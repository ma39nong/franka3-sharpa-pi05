"""Release this deployment's exact containers after its execute process exits."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from .shutdown import stderr


class HardwareSession:
    def __init__(self):
        root = Path(__file__).resolve().parents[2]
        gello = Path(os.environ["SHARPA_GELLO_REPO"])
        self.overlay = str(root / "deploy/fr3_sharpa_runtime/compose.hardware.yaml")
        self.gateway = str(Path(os.environ["SHARPA_GATEWAY_CONFIG"]).resolve())
        self.compose = [
            "docker",
            "compose",
            "--profile",
            "openpi-hardware",
            "--project-directory",
            str(gello / "docker"),
            "--env-file",
            str(gello / "docker/.env"),
            "-f",
            str(gello / "docker/compose.yaml"),
            "-f",
            self.overlay,
        ]
        self.owned = self.inventory()

    def inventory(self):
        result = subprocess.run(
            [*self.compose, "ps", "-a", "-q", "teleop-control", "openpi-cameras", "franka-control"],
            check=True,
            capture_output=True,
            text=True,
        )
        ids = result.stdout.split()
        if not ids:
            return {}
        inspected = subprocess.run(["docker", "inspect", *ids], check=True, capture_output=True, text=True)
        return {item["Config"]["Labels"]["com.docker.compose.service"]: item for item in json.loads(inspected.stdout)}

    def is_openpi(self, item, service):
        files = item["Config"]["Labels"].get("com.docker.compose.project.config_files", "").split(",")
        if self.overlay not in files:
            return False
        command = item["Config"].get("Cmd") or []
        if service == "teleop-control":
            return "config:=/etc/openpi/teleop_control.yaml" in command and any(
                mount.get("Source") == self.gateway and mount.get("Destination") == "/etc/openpi/teleop_control.yaml"
                for mount in item.get("Mounts", [])
            )
        return service == "openpi-cameras" and "triple_camera.launch.py" in command

    def cleanup(self):
        current = self.inventory()
        released_gateway = False
        errors = []
        for service in ("openpi-cameras", "teleop-control"):
            original, latest = self.owned.get(service), current.get(service)
            if not original or not latest or original["Id"] != latest["Id"] or not self.is_openpi(latest, service):
                continue
            try:
                # Address immutable IDs, never a service name that another launcher may have replaced.
                subprocess.run(["docker", "stop", "--time", "30", latest["Id"]], check=True, capture_output=True)
                subprocess.run(["docker", "rm", latest["Id"]], check=True, capture_output=True)
                released_gateway |= service == "teleop-control"
            except subprocess.SubprocessError as error:
                errors.append(f"{service}: {error}")
        # Reinspect after gateway removal: a teleop launcher may already have taken over.
        after = self.inventory() if released_gateway else {}
        replacement = after.get("teleop-control")
        if replacement and replacement["Id"] != self.owned["teleop-control"]["Id"]:
            released_gateway = False
        original, latest = self.owned.get("franka-control"), after.get("franka-control")
        if released_gateway and original and latest and original["Id"] == latest["Id"]:
            try:
                subprocess.run(["docker", "stop", "--time", "30", latest["Id"]], check=True, capture_output=True)
            except subprocess.SubprocessError as error:
                errors.append(f"franka-control: {error}")
        if errors:
            raise RuntimeError("OpenPI cleanup incomplete: " + "; ".join(errors))


def run_session(command, hardware, *, shutdown_grace_s=15):
    """Forward termination to the runtime tree, then release its hardware containers."""
    child = None
    interrupted = None

    def forward(number, _frame):
        nonlocal interrupted
        if interrupted is not None:
            return
        interrupted = number
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, number)
            except ProcessLookupError:
                pass

    previous = {number: signal.signal(number, forward) for number in (signal.SIGINT, signal.SIGTERM)}
    status = 1
    try:
        child = subprocess.Popen(command, start_new_session=True, stdin=subprocess.DEVNULL)
        if interrupted is not None:
            forward(interrupted, None)
        while True:
            try:
                status = child.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                if interrupted is not None:
                    try:
                        status = child.wait(timeout=shutdown_grace_s)
                    except subprocess.TimeoutExpired:
                        stderr("[sharpa-shutdown] ERROR 执行端退出超时，将终止进程组；未确认设备失能。")
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            status = child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            status = child.wait()
                    break
    finally:
        try:
            hardware.cleanup()
        except Exception as error:
            stderr(f"ERROR: {error}; run hardware.sh stop to retry cleanup")
            status = 1
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
    return 128 + interrupted if interrupted is not None and status != 1 else (128 - status if status < 0 else status)


def main():
    hardware = HardwareSession()  # Snapshot IDs before creating any runtime/SDK worker.
    if sys.argv[1:] == ["stop"]:
        hardware.cleanup()
        return
    import argparse
    from .shutdown import DisableSettings
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sharpa-backend", default="sdk_direct")
    args, _ = parser.parse_known_args()
    grace = DisableSettings.from_env().session_grace_s if args.sharpa_backend == "sdk_direct" else 15
    raise SystemExit(run_session([sys.executable, "-m", "deploy.fr3_sharpa_runtime.run", *sys.argv[1:]],
                                 hardware, shutdown_grace_s=grace))


if __name__ == "__main__":
    main()
