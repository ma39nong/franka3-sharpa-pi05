"""Shared deployment startup, ownership and runtime manifest helpers.

These functions are used by all speed entry points. Motion planning and
speed-dependent device checks remain in their respective profiles.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

import yaml


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path("/home/user/lpy/gello-retarget")
SDK = Path("/home/user/miniconda3/envs/gello-upper-body-teleop/lib/python3.10/site-packages")
OVERLAY = ROOT / ".deployment/oneshot-overlay"

# Keep the three latency-sensitive Python/ROS boundaries off one another and
# off the host-side control/observation processes. CPUs 8-15 remain reserved
# for the Franka controller container.
HELPER_CPUSETS = {
    "gateway": "2-3",
    "splitter": "4-5",
    "devices": "6-7,16-19",
}


def verify_overlay():
    manifest = json.loads((OVERLAY / "source-manifest.json").read_text())
    if manifest.get("build_completed") is not True:
        raise ValueError("Build the isolated controller overlay first")
    for root, key in (
        (Path(manifest["reference"]), "source_hashes"),
        (OVERLAY / "src/franka_fr3_arm_controllers", "overlay_source_hashes"),
    ):
        for name, expected in manifest[key].items():
            if digest(root / name) != expected:
                raise ValueError("Controller source changed since build: " + str(root / name))
    if digest(manifest["built_library"]) != manifest["built_library_sha256"]:
        raise ValueError("Installed controller differs from recorded build")
    return manifest


def write_runtime(args, output, contract):
    from experiments.weight_motion_eval.reference import reference_limits

    limits, names = reference_limits()
    manifest = verify_overlay()
    model, _ = contract(args.checkpoint)
    workcell = yaml.safe_load((REFERENCE / "config/workcell/current.yaml").read_text())
    workcell["LEFT"]["robot_ip"], workcell["RIGHT"]["robot_ip"] = args.left_arm_ip, args.right_arm_ip
    for side in ("LEFT", "RIGHT"):
        if workcell[side]["use_fake_hardware"] != "false":
            raise ValueError("Expected the reference real workcell profile")
    (output / "workcell.yaml").write_text(yaml.safe_dump(workcell))
    runtime = {
        "schema_version": 1,
        "reference": str(REFERENCE),
        "hands": {"left": args.wuji_left_address, "right": args.wuji_right_address},
        "arms": {"left": args.left_arm_ip, "right": args.right_arm_ip},
        "limits": {k: limits[k].tolist() for k in ("lower", "upper")},
        "reference_hashes": limits["sources"],
        "joint_names": names,
        "controller_library": manifest["built_library"],
        "controller_sha256": manifest["built_library_sha256"],
        "checkpoint": model,
        "finish_policy": args.finish_policy,
        "hand_control": args.hand_control,
        "continuous": args.continuous,
    }
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    return runtime, limits, names


def docker_command(name, output, ipc, command, *, controller=False, sdk=False, cpuset=None):
    # Match reference compose: DDS only on loopback; keep SDK/ROS helpers off FCI CPUs.
    args = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "host",
        "--ipc",
        "host",
        "--cpuset-cpus",
        cpuset or ("8-9,12-13" if controller else "0-7,16-23"),
        "-v",
        f"{REFERENCE}:/workspace/franka_upper_body_teleop:ro",
        "-v",
        f"{REFERENCE}:{REFERENCE}:ro",
        "-v",
        f"{ROOT}:{ROOT}:ro",
        "-v",
        f"{output}:{output}",
        "-v",
        f"{ipc}:{ipc}",
        "-w",
        str(ROOT),
    ]
    for value in (
        "ROS_LOCALHOST_ONLY=1",
        "ROS_DOMAIN_ID=0",
        "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp",
        "CYCLONEDDS_URI=file:///workspace/franka_upper_body_teleop/config/cyclonedds.xml",
        "PYTHONDONTWRITEBYTECODE=1",
        "OPENBLAS_NUM_THREADS=1",
        "OMP_NUM_THREADS=1",
        f"ROS_LOG_DIR={output}/ros",
        f"PYTHONPATH={ROOT}" + (":/sdk" if sdk else ""),
    ):
        args += ["-e", value]
    if controller:
        args += [
            "--privileged",
            "--ulimit",
            "rtprio=99",
            "--ulimit",
            "rttime=-1",
            "--ulimit",
            "memlock=-1",
            "-v",
            "/dev:/dev",
            "-v",
            f"{OVERLAY}:/overlay:ro",
        ]
    else:
        args += ["--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp"]
    if sdk:
        for package in ("wuji_sdk", "wuji_sdk.libs"):
            if not (SDK / package).is_dir():
                raise ValueError("Missing installed SDK dependency: " + str(SDK / package))
            args += ["-v", f"{SDK / package}:/sdk/{package}:ro"]
    return [*args, "franka-upper-body-teleop:latest", *command]


def launch_commands(args, output, ipc):
    prefix = "pi05-single-" + uuid.uuid4().hex[:8]
    commands = {}
    if args.execute:
        commands["arms"] = docker_command(
            prefix + "-arms",
            output,
            ipc,
            [
                "bash",
                "-c",
                'source "$1"; exec ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py robot_config_file:="$2"',
                "pi05",
                "/overlay/install/setup.bash",
                str(output / "workcell.yaml"),
            ],
            controller=True,
        )
        for role in ("gateway", "splitter"):
            commands[role] = docker_command(
                prefix + "-" + role,
                output,
                ipc,
                [
                    "python3",
                    "-m",
                    "deploy.fr3_wuji_runtime.ros_boundary",
                    role,
                    "--diagnostics",
                    str(output / (role + "-diagnostics.json")),
                ],
                cpuset=HELPER_CPUSETS[role],
            )
    bridge = [
        "python3",
        "-m",
        "deploy.fr3_wuji_runtime.bridge",
        "--runtime",
        str(output / "runtime.json"),
        "--socket",
        str(ipc / "devices.sock"),
    ]
    if args.execute:
        bridge += ["--execute"]
        bridge += (
            ["--supervised-trial"] if args.supervised_trial else ["--qualification", str(args.qualification.resolve())]
        )
    commands["devices"] = docker_command(
        prefix + "-devices", output, ipc, bridge, sdk=True, cpuset=HELPER_CPUSETS["devices"]
    )
    if args.execute and not args.supervised_trial:
        index = commands["devices"].index("franka-upper-body-teleop:latest")
        directory = args.qualification.resolve().parent
        commands["devices"][index:index] = ["-v", f"{directory}:{directory}:ro"]
    return commands


def preflight_owners(*, execute):
    processes = subprocess.check_output(["ps", "-eo", "args="], text=True)
    if any(
        marker in processes
        for marker in (
            "apps.operator_gui",
            "apps.wuji_ui",
            "adapters.wuji.hand_only",
            "hand_read.py",
            "observation_sources.py hands",
        )
    ):
        raise RuntimeError("An existing hand SDK owner must be closed before this deployment acquires devices")
    containers = subprocess.check_output(
        ["docker", "ps", "--no-trunc", "--format", "{{.Names}} {{.Command}}"], text=True
    )
    if any(marker in containers for marker in ("oneshot.bridge", "fr3_wuji_runtime.bridge", "hand-control", "adapters.wuji")):
        raise RuntimeError("An existing hand device container must exit before acquiring its SDK devices")
    if execute:
        if any(
            marker in containers
            for marker in (
                "franka-control",
                "robot_control.launch",
                "franka_fr3_arm_controllers.launch",
                "moveit-real",
                "oneshot.bridge",
                "fr3_wuji_runtime.bridge",
            )
        ):
            raise RuntimeError("Existing device/control owner detected; refusing a second FCI or SDK owner")
        if "read_arms" in processes:
            raise RuntimeError("An existing direct FCI reader must stop before controller acquisition")


class Children:
    def __init__(self, output):
        self.output, self.children, self.streams = output, [], []

    def launch(self, label, command):
        stream = (self.output / (label + ".log")).open("x")
        self.streams.append(stream)
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        self.children.append((command[command.index("--name") + 1], child))

    def check(self):
        if any(child.poll() is not None for _, child in self.children):
            raise RuntimeError("A deployment process exited; inspect per-process logs")

    def close(self):
        for name, child in reversed(self.children):
            subprocess.run(
                ["docker", "stop", "--signal", "SIGINT", "--timeout", "30", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=5)
        for stream in self.streams:
            stream.close()


