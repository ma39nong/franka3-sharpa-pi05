"""Independent medium-speed deployment. Default --check never starts devices."""

import argparse
from contextlib import ExitStack
import copy
import fcntl
import ipaddress
import json
import math
from pathlib import Path
import tempfile
import time
import uuid

import yaml

from deploy.fr3_wuji_runtime import hardware
from deploy.fr3_wuji_models.registry import MODELS, profile

from .profile import ARM_SPEED_RAD_S
from .profile import checkpoint_contract
from .profile import planning_config
from .profile import session_limits
from .runtime.ipc import RemoteDevices
from .runtime.limits import ARM_TRACKING_TOLERANCE_RAD
from .runtime.limits import HAND_CONTACT_SECONDS
from .runtime.limits import HAND_CURRENT_A
from .runtime.limits import HAND_KP
from .runtime.limits import HAND_SPEED_RAD_S
from .runtime.limits import SLIDER_CURRENT_A
from .runtime.limits import SLIDER_SPEED_RAD_S
from .runtime.qualification import Qualification
from .runtime.qualification import SupervisedTrial

ROOT, REFERENCE = hardware.ROOT, hardware.REFERENCE
verify_overlay, preflight_owners, Children = hardware.verify_overlay, hardware.preflight_owners, hardware.Children


def launch_commands(args, output, ipc):
    commands = hardware.launch_commands(args, output, ipc)
    if "gateway" in commands:
        commands["gateway"] += ["--arm-speed-rad-s", str(ARM_SPEED_RAD_S)]
    # Route every speed-sensitive boundary to the medium-owned runtime.
    modules = {
        "deploy.fr3_wuji_runtime.ros_boundary": "deploy.fr3_wuji_medium.runtime.ros_boundary",
        "deploy.fr3_wuji_runtime.bridge": "deploy.fr3_wuji_medium.runtime.bridge",
    }
    return {role: [modules.get(arg, arg) for arg in command] for role, command in commands.items()}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--read-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--left-arm-ip", default="172.16.0.2")
    parser.add_argument("--right-arm-ip", default="172.16.1.2")
    parser.add_argument("--wuji-sides", choices=("both",), default="both")
    parser.add_argument("--wuji-left-address", default="192.168.1.110:7447")
    parser.add_argument("--wuji-right-address", default="192.168.2.111:7447")
    parser.add_argument("--model", choices=tuple(MODELS), default="30000")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--uri")
    parser.add_argument("--start-cameras", action="store_true")
    parser.add_argument("--finish-policy", choices=("hold", "disable"), default="hold")
    parser.add_argument("--hand-control", choices=("slider", "strict"), default="slider")
    parser.add_argument(
        "--continuous", action="store_true", help="Execute fresh prediction prefixes in one held session"
    )
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--replan-steps", type=int, default=20)
    parser.add_argument("--minimum-time-scale", type=float, default=0.625)
    parser.add_argument(
        "--slow-after-seconds",
        type=float,
        default=25.0,
        help="Switch to slow motion at the first new prediction segment after this many motion seconds; 0 disables",
    )
    parser.add_argument("--qualification", type=Path)
    parser.add_argument(
        "--supervised-trial",
        action="store_true",
        help="One operator-supervised 50-step run without historical commissioning evidence",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "logs/weight_motion_eval" / ("medium-" + uuid.uuid4().hex[:10])
    )
    args = parser.parse_args(argv)
    args.checkpoint = args.checkpoint or ROOT / profile(args.model).checkpoint
    args.uri = args.uri or profile(args.model).uri
    if args.rounds < 1 or not 2 <= args.replan_steps <= 50:
        parser.error("rounds must be positive; replan-steps must be between 2 and 50")
    if not 0.5 <= args.minimum_time_scale <= 100:
        parser.error("minimum-time-scale must be between 0.5 and 100")
    if not math.isfinite(args.slow_after_seconds) or args.slow_after_seconds < 0:
        parser.error("slow-after-seconds must be finite and nonnegative")
    if args.continuous and args.execute and not args.supervised_trial:
        parser.error("Continuous execution currently requires --supervised-trial")
    if args.supervised_trial and (not args.execute or args.qualification is not None):
        parser.error("--supervised-trial requires --execute and cannot be combined with --qualification")
    for value in (args.left_arm_ip, args.right_arm_ip):
        ipaddress.IPv4Address(value)
    for value in (args.wuji_left_address, args.wuji_right_address):
        address, port = value.rsplit(":", 1)
        ipaddress.IPv4Address(address)
        if not 1 <= int(port) <= 65535:
            parser.error("Invalid hand port")
    if args.left_arm_ip == args.right_arm_ip or args.wuji_left_address == args.wuji_right_address:
        parser.error("Left/right devices must have distinct addresses")
    if args.execute and not args.supervised_trial:
        parser.error("Medium speed requires --supervised-trial; slow commissioning is not medium qualification")
    return args


def write_runtime(args, output):
    from experiments.weight_motion_eval.reference import reference_limits

    limits, names = reference_limits()
    limits = session_limits(limits)
    manifest = verify_overlay()
    model, _ = checkpoint_contract(args.model, args.checkpoint)
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
        "hand_kp": {"left": 5.0 if args.model == "25000-single" else HAND_KP, "right": HAND_KP},
        "continuous": args.continuous,
        "arm_speed_rad_s": ARM_SPEED_RAD_S,
        "arm_tracking_rad": ARM_TRACKING_TOLERANCE_RAD,
        "deployment_mode": "medium",
        "planning": planning_config(args.model, minimum_time_scale=args.minimum_time_scale),
        "slow_after_seconds": args.slow_after_seconds if args.continuous else None,
    }
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    return runtime, limits, names


def main(argv=None):
    args = parse_args(argv)
    from experiments.weight_motion_eval.cli import safe_output
    from experiments.weight_motion_eval.reference import deployment_module

    output = safe_output(args.output)
    output.mkdir(parents=True)
    runtime, limits, names = write_runtime(args, output)
    if not args.read_only and not args.execute:
        preview = copy.copy(args)
        preview.execute = True
        preview.supervised_trial, preview.qualification = True, None
        commands = launch_commands(preview, output, Path("/tmp/pi05-medium-RUNTIME"))
        (output / "commands-preview.json").write_text(json.dumps(commands, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "mode": "check",
                    "hardware_output": False,
                    "runtime": str(output / "runtime.json"),
                    "arms": runtime["arms"],
                    "hands": runtime["hands"],
                    "controller_verified": True,
                    "execution_requires": "--execute plus --supervised-trial; no devices started",
                },
                indent=2,
            )
        )
        return
    qualification = None
    if args.execute:
        qualification = (
            SupervisedTrial(hand_control=args.hand_control)
            if args.supervised_trial
            else Qualification(
                args.qualification, controller_sha256=runtime["controller_sha256"], hand_control=args.hand_control
            )
        )
        (output / "execution-admission.json").write_text(
            json.dumps(
                {
                    "execution_mode": "supervised_trial" if args.supervised_trial else "commissioned",
                    "historical_stop_evidence_waived": args.supervised_trial,
                    "hand_control": args.hand_control,
                    "hand_kp": runtime["hand_kp"],
                    "hand_velocity_rad_s": SLIDER_SPEED_RAD_S if args.hand_control == "slider" else HAND_SPEED_RAD_S,
                    "hand_measured_speed_stop": args.hand_control == "strict",
                    "hand_current_limit_a": SLIDER_CURRENT_A if args.hand_control == "slider" else HAND_CURRENT_A,
                    "hand_contact_duration_s": HAND_CONTACT_SECONDS if args.hand_control == "slider" else None,
                    "single_prediction_steps": 50,
                    "continuous": args.continuous,
                    "rounds": args.rounds if args.continuous else 1,
                    "execution_steps": args.replan_steps if args.continuous else 50,
                    "physical_stop_qualified": not args.supervised_trial,
                },
                indent=2,
            )
            + "\n"
        )
    preflight_owners(execute=args.execute)
    with ExitStack() as stack:
        lock = stack.enter_context((ROOT / ".deployment/oneshot-owner.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ws, model = None, None
        if args.execute:
            from openpi_client import msgpack_numpy
            from websockets.sync.client import connect

            ws = stack.enter_context(
                connect(args.uri, compression=None, max_size=None, open_timeout=5, close_timeout=2, proxy=None)
            )
            model = msgpack_numpy.unpackb(ws.recv(timeout=3))
            if (
                any(model.get(key) != value for key, value in runtime["checkpoint"].items())
                or model.get("warmed_up") is not True
            ):
                raise ValueError("Wrong checkpoint/normalization or model server not warmed up")
        ipc = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="pi05-one-")))
        children = Children(output)
        stack.callback(children.close)
        commands = launch_commands(args, output, ipc)
        (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        print("【阶段】启动机械臂、手部设备及网关通信", flush=True)
        for label, command in commands.items():
            children.launch(label, command)
        deadline = time.monotonic() + 45
        while not (ipc / "devices.sock").exists():
            children.check()
            if time.monotonic() > deadline:
                raise RuntimeError("Device bridge startup timed out")
            time.sleep(0.1)
        devices = RemoteDevices(
            ipc / "devices.sock",
            execute=args.execute,
            finish_policy=args.finish_policy,
            arm_speed_rad_s=ARM_SPEED_RAD_S,
        )
        stack.callback(devices.close)
        try:
            inventory = devices.call("inventory", timeout=3)
            (output / "devices-inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
            feedback = None
            while time.monotonic() < deadline:
                children.check()
                try:
                    feedback = devices.feedback(time.monotonic())
                    break
                except RuntimeError:
                    time.sleep(0.05)
            if feedback is None:
                readiness = devices.call("readiness")
                (output / "readiness.json").write_text(json.dumps(readiness, indent=2) + "\n")
                raise RuntimeError(
                    "Missing live arm/hand feedback; read-only mode requires existing arm state publishers"
                )
            if args.read_only:
                (output / "feedback.json").write_text(
                    json.dumps(
                        {
                            "hardware_output": False,
                            "positions": feedback.positions.tolist(),
                            "velocities": feedback.velocities.tolist(),
                        },
                        indent=2,
                    )
                )
                print("Four-device measured feedback received; no motor enable or command sent.")
                return
            from .live import LiveConsumer

            recorder_module = deployment_module("executor.async_recorder")
            recorder = recorder_module.AsyncRecorder(output)
            consumer = None
            try:
                config = planning_config(args.model, minimum_time_scale=args.minimum_time_scale)
                config["hand_control"] = args.hand_control
                config["minimum_time_scale"] = args.minimum_time_scale
                config["hand_raw_initial_delta_rad"] = qualification.data["hand_raw_initial_delta_rad"]
                if args.continuous:
                    from .continuous import ContinuousConsumer

                    consumer = ContinuousConsumer(
                        devices,
                        ws,
                        model,
                        config,
                        limits,
                        names,
                        output,
                        recorder,
                        rounds=args.rounds,
                        replan_steps=args.replan_steps,
                        slow_after_seconds=args.slow_after_seconds or None,
                    )
                    stack.callback(consumer.close)
                else:
                    consumer = LiveConsumer(devices, ws, model, config, limits, names, output, recorder)
                observe = deployment_module("observe")
                options = [
                    "--arm-source",
                    "ros",
                    "--hand-source",
                    "ros",
                    "--duration",
                    str(args.rounds * 90 + 60) if args.continuous else "3",
                    "--poll-interval",
                    "0.005",
                    "--output",
                    str(output / "observation"),
                ]
                if args.start_cameras:
                    options += ["--start-cameras", "--camera-profile", "rgb"]
                camera_wait = getattr(observe, "CAMERA_WARMUP_SECONDS", 18)
                print(
                    "【阶段】启动实时观测" + (f"，等待相机稳定 {camera_wait} 秒" if args.start_cameras else ""),
                    flush=True,
                )
                observe.main(options, consumer=consumer, single_shot=True)
                if consumer.completed != 1:
                    raise RuntimeError("No fresh inference was admitted")
                print(
                    (
                        f"{args.rounds} prediction rounds completed. "
                        if args.continuous
                        else "One 50-step trajectory completed. "
                    )
                    + (
                        "Hands holding; Ctrl-C stops and releases after feedback confirmation."
                        if args.finish_policy == "hold"
                        else "Hands released after settling."
                    ),
                    flush=True,
                )
                while args.finish_policy == "hold":
                    children.check()
                    if args.continuous:
                        consumer.poll()
                    if devices.hold_error:
                        raise RuntimeError("Hold monitoring failed: " + devices.hold_error)
                    time.sleep(0.1)
            finally:
                # Stop motion/hold before potentially blocking on recorder drain.
                try:
                    if args.continuous and consumer is not None:
                        consumer.close()
                    devices.stop()
                finally:
                    recorder.close()
        except KeyboardInterrupt:
            print("【阶段】收到停止请求，等待设备停止及清理", flush=True)
        finally:
            if args.execute:
                try:
                    result = devices.stop()
                    (output / "stop-request.json").write_text(json.dumps(result, indent=2) + "\n")
                except Exception as error:
                    (output / "stop-request.json").write_text(
                        json.dumps({"physical_stop_confirmed": False, "error": str(error)}) + "\n"
                    )


if __name__ == "__main__":
    main()
