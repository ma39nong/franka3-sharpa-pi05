"""Standalone tomato Home for both FR3 arms and both Wuji hands.

The default mode is a file/configuration check. Real motion requires the
explicit execution path used by ``home.sh``. No model, camera, GELLO, MANUS,
teleop backend, or operator UI is started.
"""

import argparse
from contextlib import ExitStack
import copy
import fcntl
import hashlib
import ipaddress
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
import uuid

import numpy as np
import yaml

from deploy.fr3_wuji_home.planner import ARM_HOME_TRACKING_TOLERANCE_RAD
from deploy.fr3_wuji_home.planner import build_home_plan
from deploy.fr3_wuji_home.poses import DEFAULT_POSES
from deploy.fr3_wuji_home.poses import load_home
from deploy.fr3_wuji_runtime import hardware
from deploy.fr3_wuji_slow.core import Admission
from deploy.fr3_wuji_slow.core import OneShot
from deploy.fr3_wuji_slow.ipc import RemoteDevices
from deploy.fr3_wuji_slow.runner import run_live

ROOT = hardware.ROOT
ARM_RUNTIME_SPEED_RAD_S = 0.7


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Validate and render launch files; never connect devices")
    mode.add_argument("--execute", action="store_true", help="Move real hardware")
    parser.add_argument("--home", default=None, help="Named Home from poses.yaml; default is tomato")
    parser.add_argument("--poses", type=Path, default=DEFAULT_POSES)
    parser.add_argument("--left-arm-ip", default="172.16.0.2")
    parser.add_argument("--right-arm-ip", default="172.16.1.2")
    parser.add_argument("--wuji-left-address", default="192.168.1.110:7447")
    parser.add_argument("--wuji-right-address", default="192.168.2.111:7447")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument(
        "--supervised-trial",
        action="store_true",
        help="Operator-attended Home without claiming historical commissioning evidence",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "logs/weight_motion_eval" / ("home-" + uuid.uuid4().hex[:10]),
    )
    args = parser.parse_args(argv)
    if args.supervised_trial and (not args.execute or args.qualification is not None):
        parser.error("--supervised-trial requires --execute and cannot be combined with --qualification")
    if args.execute and not args.supervised_trial and args.qualification is None:
        parser.error("--execute requires --qualification or --supervised-trial")
    for value in (args.left_arm_ip, args.right_arm_ip):
        ipaddress.IPv4Address(value)
    for value in (args.wuji_left_address, args.wuji_right_address):
        address, port = value.rsplit(":", 1)
        ipaddress.IPv4Address(address)
        if not 1 <= int(port) <= 65535:
            parser.error("Invalid Wuji hand port")
    if args.left_arm_ip == args.right_arm_ip or args.wuji_left_address == args.wuji_right_address:
        parser.error("Left/right devices must have distinct addresses")
    # Attributes used by the shared isolated hardware launcher.
    args.finish_policy = "disable"
    args.hand_control = "slider"
    args.continuous = False
    return args


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_runtime(args, output, home):
    from experiments.weight_motion_eval.reference import reference_limits

    limits, names = reference_limits()
    manifest = hardware.verify_overlay()
    workcell = yaml.safe_load((hardware.REFERENCE / "config/workcell/current.yaml").read_text())
    workcell["LEFT"]["robot_ip"], workcell["RIGHT"]["robot_ip"] = args.left_arm_ip, args.right_arm_ip
    for side in ("LEFT", "RIGHT"):
        if workcell[side]["use_fake_hardware"] != "false":
            raise ValueError("Expected the reference real workcell profile")
    (output / "workcell.yaml").write_text(yaml.safe_dump(workcell))
    runtime = {
        "schema_version": 1,
        "mode": "standalone_home",
        "reference": str(hardware.REFERENCE),
        "hands": {"left": args.wuji_left_address, "right": args.wuji_right_address},
        "arms": {"left": args.left_arm_ip, "right": args.right_arm_ip},
        "limits": {key: limits[key].tolist() for key in ("lower", "upper")},
        "reference_hashes": limits["sources"],
        "joint_names": names,
        "controller_library": manifest["built_library"],
        "controller_sha256": manifest["built_library_sha256"],
        "finish_policy": "disable",
        "hand_control": "slider",
        "continuous": False,
        "arm_speed_rad_s": ARM_RUNTIME_SPEED_RAD_S,
        "arm_tracking_rad": ARM_HOME_TRACKING_TOLERANCE_RAD,
        "home": {
            "name": home.name,
            "description": home.description,
            "pose_file": str(home.source),
            "pose_file_sha256": _sha256(home.source),
            "provenance": home.provenance,
            "target": home.target.tolist(),
        },
    }
    (output / "runtime.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2) + "\n")
    return runtime, limits, names


def reject_other_owners():
    hardware.preflight_owners(execute=True)
    processes = subprocess.check_output(["ps", "-eo", "args="], text=True)
    if any(marker in processes for marker in ("teleop_runtime.cli", "apps.operator_gui.operator_gui")):
        raise RuntimeError("Teleop backend/UI is running; stop it before standalone Home")


def _save_plan(plan, output):
    np.savez_compressed(output / "home-plan.npz", **plan.arrays())
    report = dict(plan.report)
    report["artifact_sha256"] = _sha256(output / "home-plan.npz")
    (output / "home-plan.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def _save_motion(history, output):
    if not history:
        return
    np.savez_compressed(
        output / "home-motion.npz",
        time=np.asarray([item[0] for item in history]),
        phase=np.asarray([item[1] for item in history]),
        command=np.asarray([item[2] for item in history]),
        feedback=np.asarray([item[3] for item in history]),
    )


def main(argv=None):
    args = parse_args(argv)
    from experiments.weight_motion_eval.cli import safe_output

    home = load_home(args.home, args.poses)
    output = safe_output(args.output)
    output.mkdir(parents=True)
    runtime, limits, names = write_runtime(args, output, home)
    if not args.execute:
        # With no live start, a zero-distance plan still applies the complete
        # joint-limit, schema and derivative contract to the stored target.
        checked_target = build_home_plan(home.target, home.target, limits, names)
        preview = copy.copy(args)
        preview.execute, preview.supervised_trial, preview.qualification = True, True, None
        commands = hardware.launch_commands(preview, output, Path("/tmp/pi05-home-RUNTIME"))
        (output / "commands-preview.json").write_text(json.dumps(commands, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "mode": "check",
                    "hardware_output": False,
                    "home": home.name,
                    "pose_file": str(home.source),
                    "pose_file_sha256": runtime["home"]["pose_file_sha256"],
                    "stored_target_checks_passed": checked_target.report["planning_checks_passed"],
                    "arms": runtime["arms"],
                    "hands": runtime["hands"],
                    "sequence": ["arms", "hands", "disable_and_exit"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    reject_other_owners()
    interrupted = threading.Event()
    previous_handlers = {}

    def stop_requested(*_):
        interrupted.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, stop_requested)
    history = []
    report = {
        "schema_version": 1,
        "mode": "standalone_home",
        "hardware_output": True,
        "home": home.name,
        "completed": False,
        "output": str(output),
    }
    try:
        with ExitStack() as stack:
            lock = stack.enter_context((ROOT / ".deployment/oneshot-owner.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ipc = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="pi05-home-")))
            children = hardware.Children(output)
            stack.callback(children.close)
            commands = hardware.launch_commands(args, output, ipc)
            (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
            print("【Home】启动最小双臂/双手控制栈 (不启动遥操、相机或模型)", flush=True)
            for label, command in commands.items():
                children.launch(label, command)
            deadline = time.monotonic() + 45
            while not (ipc / "devices.sock").exists():
                children.check()
                if interrupted.is_set():
                    raise KeyboardInterrupt
                if time.monotonic() > deadline:
                    raise RuntimeError("Home device bridge startup timed out")
                time.sleep(0.1)
            devices = RemoteDevices(
                ipc / "devices.sock",
                execute=True,
                finish_policy="disable",
                arm_speed_rad_s=ARM_RUNTIME_SPEED_RAD_S,
            )
            stack.callback(devices.close)
            inventory = devices.call("inventory", timeout=3)
            (output / "devices-inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
            feedback = None
            while time.monotonic() < deadline:
                children.check()
                if interrupted.is_set():
                    raise KeyboardInterrupt
                try:
                    feedback = devices.feedback(time.monotonic())
                    break
                except (RuntimeError, ValueError):
                    time.sleep(0.05)
            if feedback is None:
                raise RuntimeError("Missing fresh feedback from both arms and both hands")

            plan = build_home_plan(feedback.positions, home.target, limits, names)
            plan.report["home"] = runtime["home"]
            _save_plan(plan, output)
            prepared = time.monotonic()
            admission = Admission(
                prepared,
                prepared,
                prepared,
                "home-" + uuid.uuid4().hex,
                f"home:{home.name}",
                feedback.epoch,
            )
            player = OneShot(plan, admission, prepared)
            print(
                f"【Home】规划通过: 先归双臂 {plan.approach.duration:.1f}s, 稳定后归双手 {plan.playback.duration:.1f}s",
                flush=True,
            )

            def record(frame, measured):
                history.append((frame.created, frame.phase, frame.positions.copy(), measured.positions.copy()))

            run_live(player, devices, stop_requested=interrupted.is_set, emit=record)
            _save_motion(history, output)
            report.update(
                completed=player.state == "complete",
                player_state=player.state,
                transitions=player.transitions,
                frames=len(history),
                plan_sha256=_sha256(output / "home-plan.npz"),
                motion_sha256=_sha256(output / "home-motion.npz"),
                finish_policy="disable",
            )
            print("【Home】tomato 双臂和双手均已到位; 设备已释放", flush=True)
    except BaseException as error:
        report["error"] = str(error) or type(error).__name__
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if history and not (output / "home-motion.npz").exists():
            _save_motion(history, output)
        (output / "home-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
