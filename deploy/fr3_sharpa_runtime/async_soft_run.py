"""Fourth mode CLI: asynchronous Soft only; original run.py is unchanged."""

import argparse
import atexit
import json
import math
import os
from pathlib import Path
import shlex
import time

import numpy as np

from . import config
from .async_soft_client import AsyncSoftClient
from .async_soft_client import default_port


def execute(args):
    from . import run as existing
    from .async_soft_runtime import run_async_soft
    from .motion_worker import MotionWorker

    client = AsyncSoftClient(args.policy_url)
    client.inspect_metadata()  # Reject another mode before opening SDK devices.
    setup = args.ros_overlay / "setup.bash"
    if not setup.is_file():
        raise FileNotFoundError(setup)
    home_arms = None
    if args.initialization == "home":
        from .home_pose import load_home

        home_arms, _ = load_home(args.home_poses, args.home_task, hand_path=args.hand_home)
    root = Path(__file__).resolve().parents[2]
    core = existing.GELLO_REPO / "ros_ws/src/teleop_core"
    inner = (
        f"source {shlex.quote(str(setup))}; "
        f"export SHARPA_SHUTDOWN_REPORT_DIR={shlex.quote(str(args.execution_trace_dir.expanduser().resolve()))}; "
        f"export PYTHONPATH={shlex.quote(str(root))}:{shlex.quote(str(existing.GELLO_REPO))}:"
        f'{shlex.quote(str(core))}:"${{PYTHONPATH:-}}"; '
        "exec python -m deploy.fr3_sharpa_runtime.async_soft_motion_ros --enable-hardware"
    )
    worker = MotionWorker(
        ["pixi", "run", "-e", "ros2", "bash", "-lc", inner], cwd=existing.ROS_PROJECT, protocol_fd=True
    )
    atexit.register(worker.close)
    try:
        ready = worker.request({"operation": "ready"})
        if ready.get("operation") != "ready":
            raise RuntimeError("Async Soft SDK worker did not become ready")
        observation = existing.live_observation(args.ros_overlay, args.prompt)
        if home_arms is not None:
            from .home_pose import home_target

            report = worker.request(
                {
                    "operation": "prepare",
                    "target": home_target(home_arms, observation["observation/state"]).tolist(),
                    "speed_rad_s": args.prepare_speed_rad_s,
                    "tolerance_rad": 0.02,
                }
            )
            observation = existing.observation_after_completion(args, report)
        for attempt in range(2):
            first = client.infer_chunk(observation)
            if 0 <= time.monotonic_ns() - observation["_captured_monotonic_ns"] <= 200_000_000:
                break
            if attempt:
                raise ValueError("Async Soft initial observation expired during inference")
            observation = existing.live_observation(args.ros_overlay, args.prompt)
        return run_async_soft(
            client,
            lambda **options: existing.live_observation(args.ros_overlay, args.prompt, **options),
            worker,
            first,
            observation,
            args,
        )
    finally:
        worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--enable-hardware", action="store_true")
    parser.add_argument("--sharpa-backend", choices=("sdk_direct",), default="sdk_direct")
    parser.add_argument("--initialization", choices=("current", "home"), default="current")
    speeds = parser.add_mutually_exclusive_group()
    speeds.add_argument("--speed", choices=("slow", "medium", "fast"), default=None)
    speeds.add_argument("--execute-speed-scale", type=float)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument(
        "--execute-steps",
        type=int,
        choices=(26,),
        default=26,
        help="Fixed consumed model prefix; compressed to 20 targets",
    )
    parser.add_argument("--prediction-clip-rad", type=float, default=0.05)
    parser.add_argument("--prepare-speed-rad-s", type=float, default=0.03)
    parser.add_argument("--policy-url", default=f"ws://127.0.0.1:{default_port()}")
    parser.add_argument("--ros-overlay", type=Path)
    parser.add_argument("--execution-trace-dir", type=Path, default=config.EXECUTION_TRACE_DIR)
    parser.add_argument("--prompt", default=config.DEFAULT_PROMPT)
    parser.add_argument(
        "--home-poses",
        type=Path,
        default=Path(
            os.environ.get(
                "SHARPA_HOME_POSES", str(Path(__file__).resolve().parents[2] / ".deployment/operator_poses.yaml")
            )
        ),
    )
    parser.add_argument("--home-task", default="powder_weighing")
    parser.add_argument(
        "--hand-home", type=Path, default=Path(__file__).resolve().parents[2] / ".deployment/hand_home.json"
    )
    args = parser.parse_args()
    from .profiles import resolve_speed

    profile = resolve_speed(args.speed, args.execute_speed_scale)
    args.execute_speed_scale = profile.execute_speed_scale
    if config.MODEL_PROFILE.name not in ("full", "full-delta"):
        parser.error("Async Soft supports full/full-delta only")
    if (
        args.execute_steps != 26
        or args.num_chunks < 0
        or not np.isfinite(args.execute_speed_scale)
        or not 0 < args.execute_speed_scale <= 1
        or not np.isfinite(args.prediction_clip_rad)
        or args.prediction_clip_rad < 0
        or not np.isfinite(args.prepare_speed_rad_s)
        or args.prepare_speed_rad_s <= 0
    ):
        parser.error("Invalid async Soft count, delay, speed or prediction clipping")
    # Validate loopback even in --check, without opening a socket.
    AsyncSoftClient(args.policy_url)
    if args.check:
        if args.enable_hardware:
            parser.error("--check cannot enable hardware")
        print(
            json.dumps(
                {
                    "mode": "check",
                    "broker_mode": "async-soft",
                    "hardware_output": False,
                    "model_profile": config.MODEL_PROFILE.name,
                    "policy_url": args.policy_url,
                    "action_shape": [50, 58],
                    "execute_steps": args.execute_steps,
                    "num_chunks": args.num_chunks,
                    "speed": profile.name,
                    "node_hz": 30.0 / math.ceil(1 / args.execute_speed_scale),
                    "soft_reference_count": 4,
                    "effective_horizon": 30,
                    "execution_target_count": 20,
                    "discarded_model_steps": 20,
                    "alignment": "trajectory_phase",
                    "replan_trigger": "segment_start",
                    "rtc_guidance": False,
                    "compression": "cubic",
                    "correlation": False,
                    "checkpoint": str(config.CHECKPOINT) if config.CHECKPOINT else None,
                },
                indent=2,
            )
        )
        return
    if not args.enable_hardware or args.ros_overlay is None:
        parser.error("--execute needs --enable-hardware and --ros-overlay")
    print(json.dumps(execute(args), indent=2))


if __name__ == "__main__":
    main()
