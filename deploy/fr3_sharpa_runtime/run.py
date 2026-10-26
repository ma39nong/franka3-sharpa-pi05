"""Prepare and chunk-loop CLI. Offline modes never publish commands."""

from __future__ import annotations

import argparse
import atexit
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from . import config
from .runtime import SharpaRuntime
from .preflight import offline_sanity_check

DEFAULT_PROMPT = config.DEFAULT_PROMPT
GELLO_REPO = Path(os.environ.get("SHARPA_GELLO_REPO", "/home/user/gello_upper_body_teleop"))
ROS_PROJECT = GELLO_REPO / "portable_deps/litchi_hardware"
_observation_worker = None
_observation_overlay = None


def load_sample(path: Path, prompt: str, frame: int = 0) -> dict:
    """Offline LeRobot sample for dry-run and ros-test only."""
    from .dataset_sample import load_sample as read_sample
    return read_sample(path, prompt, frame)


def _start_observation_worker(overlay: Path):
    """Create the runtime-owned ROS worker once, before preflight acquisition."""
    global _observation_worker, _observation_overlay
    import shlex
    from .observation_worker import ObservationWorker

    setup = overlay / "setup.bash"
    if not setup.is_file():
        raise FileNotFoundError(setup)
    repo = str(Path(__file__).resolve().parents[2])
    inner = (
        f"source {shlex.quote(str(setup))}; "
        f"export PYTHONPATH={shlex.quote(repo)}:\"${{PYTHONPATH:-}}\"; "
        "exec python -m deploy.fr3_sharpa_runtime.ros_observation --persistent"
    )
    if _observation_worker is None:
        _observation_worker = ObservationWorker(
            ["pixi", "run", "--environment", "ros2", "bash", "-lc", inner], cwd=ROS_PROJECT)
        _observation_overlay = overlay.resolve()
        atexit.register(_observation_worker.close)
    elif overlay.resolve() != _observation_overlay:
        raise ValueError("Cannot change ROS overlay within one observation runtime")
    return _observation_worker


def live_observation(overlay: Path, prompt: str, **snapshot_options) -> dict:
    """Reuse the runtime's read-only ROS node across snapshot requests."""
    line = _start_observation_worker(overlay).snapshot(**snapshot_options)
    boundary_return_ns = time.monotonic_ns()
    start = time.monotonic_ns()
    snapshot = json.loads(line)
    if "error" in snapshot:
        raise RuntimeError(f"ROS observation failed: {snapshot['error']}")
    json_decode_ms = (time.monotonic_ns()-start)/1e6
    decode_start = time.monotonic_ns()

    def decode(key):
        payload = snapshot["images"][key]
        shape = tuple(payload["shape"])
        image = np.frombuffer(base64.b64decode(payload["rgb_b64"]), dtype=np.uint8)
        if len(shape) != 3 or shape[-1] != 3 or image.size != int(np.prod(shape)):
            raise ValueError(f"Invalid live {key} image")
        return image.reshape(shape).copy()

    observation = {
        "observation/state": np.asarray(snapshot["state"], dtype=np.float32),
        "observation/image": decode("cam0"),
        "observation/left_wrist_image": decode("cam2"),
        "observation/right_wrist_image": decode("cam1"),
        "prompt": prompt,
        "_captured_monotonic_ns": snapshot["captured_monotonic_ns"],
        "_topic_stamp_ns": snapshot["topic_stamp_ns"],
        "_alignment": snapshot.get("alignment", {}),
    }
    observation["_timing"] = {**snapshot.get("timing", {}), **snapshot.get("serialization_timing", {}),
                              "ros_boundary_return_monotonic_ns": boundary_return_ns,
                              "json_decode_ms": json_decode_ms,
                              "observation_base64_decode_ms": (time.monotonic_ns()-decode_start)/1e6}
    return observation


def prepare_before_observation(args):
    """Prepare must succeed before a newly sampled policy observation is used."""
    import shlex
    from .prepare import PrepareConfig
    PrepareConfig.load(args.prepare_config)  # Fail before ROS for missing target angles.
    setup = args.ros_overlay / "setup.bash"
    if not setup.is_file():
        raise FileNotFoundError(setup)
    repo = str(Path(__file__).resolve().parents[2])
    core = str(GELLO_REPO / "ros_ws/src/teleop_core")
    command = ["python", "-m", "deploy.fr3_sharpa_runtime.prepare_ros", "--config", str(args.prepare_config.resolve())]
    inner = (f"source {shlex.quote(str(setup))}; export PYTHONPATH={shlex.quote(repo)}:{shlex.quote(str(GELLO_REPO))}:{shlex.quote(core)}:\"${{PYTHONPATH:-}}\"; exec {shlex.join(command)}")
    result = subprocess.run(["pixi", "run", "-e", "ros2", "bash", "-lc", inner],
                            cwd=ROS_PROJECT, capture_output=True, text=True, timeout=90)
    if result.returncode:
        raise RuntimeError(f"PREPARE FAIL; execute blocked: {result.stderr[-1800:]}")
    report = json.loads(result.stdout.splitlines()[-1])
    if report.get("status") != "PASS":
        raise RuntimeError("PREPARE FAIL; execute blocked")
    return report, observation_after_completion(args, report)


def observation_after_completion(args, report):
    # Require all seven real sensor stamps to follow settling, not a cached
    # pre-prepare snapshot. Existing freshness/sync checks still apply.
    deadline = time.monotonic()+5
    while time.monotonic()<deadline:
        observation = live_observation(args.ros_overlay, args.prompt)
        stamps = observation["_topic_stamp_ns"]
        if (len(stamps) == 7 and min(stamps.values()) >= report["completed_ros_ns"]
                and observation["_captured_monotonic_ns"] >= report["completed_monotonic_ns"]):
            return observation
    raise TimeoutError("Prepare settled, but no new post-prepare observation arrived; execute blocked")


def execute_policy_prepare(args):
    """Live-state execution, with optional Home/policy prepare before prediction prefixes."""
    import shlex
    from .motion_worker import MotionWorker
    from .policy_prepare import prepare_and_run_chunks
    from .execution_trace import ExecutionTraceWriter
    setup = args.ros_overlay / "setup.bash"
    if not setup.is_file():
        raise FileNotFoundError(setup)
    repo = str(Path(__file__).resolve().parents[2])
    core = str(GELLO_REPO / "ros_ws/src/teleop_core")
    worker_module = ("deploy.fr3_sharpa_runtime.sdk_direct_motion_ros"
                     if args.sharpa_backend == "sdk_direct"
                     else "deploy.fr3_sharpa_runtime.guarded_motion_ros")
    home_arms = home_metadata = None
    initialization = getattr(args, "initialization", "current") if args.sharpa_backend == "sdk_direct" else "policy"
    if initialization == "home":
        from .home_pose import load_home
        home_arms, home_metadata = load_home(args.home_poses, args.home_task, hand_path=args.hand_home)
    if getattr(args, "broker_mode", "serial") == "rtc":
        SharpaRuntime(args.policy_url).inspect_metadata(require_rtc=True)
    inner = (f"source {shlex.quote(str(setup))}; "
             f"export SHARPA_SHUTDOWN_REPORT_DIR={shlex.quote(str(args.execution_trace_dir.expanduser().resolve()))}; "
             f"export PYTHONPATH={shlex.quote(repo)}:{shlex.quote(str(GELLO_REPO))}:{shlex.quote(str(core))}:\"${{PYTHONPATH:-}}\"; "
             f"exec python -m {worker_module} --enable-hardware")
    worker = None
    trace_writer = (ExecutionTraceWriter(args.execution_trace_dir)
                    if args.sharpa_backend == "sdk_direct" and getattr(args, "broker_mode", "serial") == "serial" else None)
    try:
        if args.sharpa_backend == "sdk_direct":
            # Direct SDK owns the Wave devices and publishes their feedback before
            # the first observation is captured.  This avoids starting a second
            # SDK connection after a legacy sharpa_driver snapshot.
            worker = MotionWorker(["pixi", "run", "-e", "ros2", "bash", "-lc", inner],
                                  cwd=ROS_PROJECT, protocol_fd=True)
            atexit.register(worker.close)
            ready = worker.request({"operation": "ready"})
            if ready.get("operation") != "ready":
                raise RuntimeError(f"sdk_direct worker did not become ready: {ready}")
        _start_observation_worker(args.ros_overlay)
        runtime = SharpaRuntime(args.policy_url)
        def infer(observation):
            # Policy cold start can make the first read-only inference stale.  Drop
            # that result, reacquire all sensors, and retry once.  The 200 ms gate
            # itself remains unchanged for every command-capable result.
            current = observation
            for attempt in range(2):
                result = runtime.infer_chunk(current)
                age = time.monotonic_ns() - current['_captured_monotonic_ns']
                if 0 <= age <= 200_000_000:
                    # Keep the trace paired with the observation actually used
                    # after a warmup retry, rather than the expired original.
                    if current is not observation:
                        observation.clear()
                        observation.update(current)
                    return result
                if attempt == 0:
                    print("POLICY WARMUP: discarded stale inference; reacquiring observation", file=sys.stderr)
                    current = live_observation(args.ros_overlay, args.prompt)
                    if "_inference_context" in observation:
                        current["_inference_context"] = observation["_inference_context"]
            raise RuntimeError("Observation expired during inference; no new target/chunk will be sent")
        initial = live_observation(args.ros_overlay, args.prompt)
        # Only legacy policy prepare consumes an inference before the chunk loop.
        # Current mode uses this fresh observation for its first executable prediction.
        first = infer(initial) if initialization == "policy" else None
        initial_used = False
        def cached_first_infer(observation):
            nonlocal initial_used
            if not initial_used and first is not None:
                initial_used = True
                return first
            return infer(observation)
        if args.sharpa_backend == "litchibot" and args.handoff_socket.exists():
            from .handoff import HandoffClient
            worker=HandoffClient(args.handoff_socket,enable_hardware=True)
        elif worker is None:
            worker = MotionWorker(["pixi", "run", "-e", "ros2", "bash", "-lc", inner],
                                  cwd=ROS_PROJECT, protocol_fd=args.sharpa_backend == "sdk_direct")
        initial_target = None
        if home_arms is not None:
            from .home_pose import home_target
            initial_target = home_target(home_arms, initial["observation/state"])
        if getattr(args, "broker_mode", "serial") == "rtc":
            if initial_target is not None:
                prepared = worker.request({"operation": "prepare", "target": initial_target.tolist(),
                    "speed_rad_s": args.prepare_speed_rad_s, "tolerance_rad": 0.02})
                if prepared.get("status") != "PASS":
                    raise RuntimeError("Home failed before RTC")
                initial = observation_after_completion(args, prepared)
            from .rtc_runtime import run_rtc
            return run_rtc(runtime, lambda **options: live_observation(args.ros_overlay, args.prompt, **options),
                           worker, infer(initial), initial, args)
        result = prepare_and_run_chunks(cached_first_infer,
            lambda completed: observation_after_completion(args, completed), worker, initial,
            num_chunks=args.num_chunks, speed_rad_s=args.prepare_speed_rad_s,
            execute_speed_scale=args.execute_speed_scale,
            trace_writer=trace_writer,
            execute_steps=getattr(args, "execute_steps", 50),
            initial_target=initial_target, home_metadata=home_metadata,
            prepare_initial=initialization != "current",
            prediction_clip_rad=getattr(args, "prediction_clip_rad", 0.05),
            rolling_options=getattr(args, "rolling_options", None))
        if trace_writer is not None:
            result["execution_traces"] = trace_writer.paths
            result["run_report"] = str(trace_writer.run_path)
        return result
    except BaseException as error:
        if trace_writer is not None and not trace_writer.run_path.exists():
            trace_writer.finish_run({"status": "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                                     "stage": "startup", "error": str(error) or type(error).__name__,
                                     "num_chunks": 0})
        raise
    finally:
        if worker is not None:
            worker.close()


def verify_handoff(args):
    from .handoff import HandoffClient
    client=HandoffClient(args.handoff_socket)
    try:
        before=client.request({'operation':'inspect'})
        observation=live_observation(args.ros_overlay,args.prompt)
        metadata,actions,_=SharpaRuntime(args.policy_url).infer_chunk(observation)
        age=time.monotonic_ns()-observation['_captured_monotonic_ns']
        warmup_discarded=age>200_000_000
        if warmup_discarded:
            # Read-only cold-start warmup: discard its output, reacquire all
            # live sensors and keep the original 200ms freshness requirement.
            observation=live_observation(args.ros_overlay,args.prompt)
            metadata,actions,_=SharpaRuntime(args.policy_url).infer_chunk(observation)
            age=time.monotonic_ns()-observation['_captured_monotonic_ns']
        if not 0<=age<=200_000_000:
            raise RuntimeError('Observation expired during handoff policy verification')
        after=client.request({'operation':'inspect'})
        if (before['session_id']!=after['session_id'] or after['sequence_next']<=before['sequence_next']
                or after['phase']!='READY' or after['motion_started'] or after['arm_publishers']!=0
                or after['supervisor'].get('hardware_send')!=0
                or any(after['supervisor']['hands'][s]['permit'] is not False for s in ('left','right'))):
            raise RuntimeError('Handoff verification requires continuously sampled READY, same session, disabled hands and zero hardware sends')
        if before['hand_publishers']!=after['hand_publishers'] or any(
                len(v)!=1 or v[0]['node']!='fr3_sharpa_runtime' for v in after['hand_publishers'].values()):
            raise RuntimeError('Handoff publisher ownership/GUID changed or duplicate source found')
        return {'status':'PASS','mode':'handoff-verify','policy_action_shape':list(actions.shape),
                'finite':bool(np.isfinite(actions).all()),'checkpoint':metadata['checkpoint'],
                'norm_stats':metadata['norm_stats'],'transport':after,'executed':False,
                'snapshot_to_action_ms':age/1e6,
                'cold_start_output_discarded':warmup_discarded,
                'sequence_before':before['sequence_next'],'sequence_after':after['sequence_next']}
    finally:client.close()


def ros_boundary(actions: np.ndarray | None, session_id: str, mode: str, overlay: Path | None,
                 enabled: bool, sharpa_backend: str = "litchibot"):
    import shlex

    transport = ["python", "-m", "deploy.fr3_sharpa_runtime.ros_transport", "--mode", mode]
    if enabled:
        transport.append("--enable-hardware")
    transport.extend(["--sharpa-backend", sharpa_backend])
    env = os.environ.copy()
    repo = str(Path(__file__).resolve().parents[2])
    setup_prefix = ""
    if overlay is not None:
        setup = overlay / "setup.bash"
        if not setup.is_file():
            raise FileNotFoundError(setup)
        setup_prefix = f"source {shlex.quote(str(setup))}; "
    # Pixi resets PYTHONPATH during activation, so source the ROS overlay inside it.
    gello = str(GELLO_REPO)
    core = f"{gello}/ros_ws/src/teleop_core"
    inner = f"{setup_prefix}export PYTHONPATH={shlex.quote(repo)}:{shlex.quote(gello)}:{shlex.quote(core)}:\"${{PYTHONPATH:-}}\"; exec {shlex.join(transport)}"
    command = ["pixi", "run", "--environment", "ros2", "bash", "-lc", inner]
    # Episode/inference state is deliberately absent: transport subscribes to
    # fresh ROS feedback after inference and checks again immediately at dispatch.
    payload = json.dumps({"actions": None if actions is None else actions.tolist(), "session_id": session_id})
    result = subprocess.run(command, cwd=ROS_PROJECT, env=env, input=payload, text=True, capture_output=True, timeout=30)
    if result.returncode and mode != "preflight-only":
        raise RuntimeError(f"ROS boundary failed ({result.returncode}): {result.stderr[-1500:]}")
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if not result.stdout.strip():
        raise RuntimeError(f"ROS boundary produced no report ({result.returncode}): {result.stderr[-1500:]}")
    report = json.loads(result.stdout.splitlines()[-1])
    if mode == "preflight-only":
        report["ros_probe_exit_code"] = result.returncode
    return report


def run_preflight_only(args) -> dict:
    """Real graph → real observations → policy → live acquisition check; no send."""
    report = ros_boundary(None, "preflight-readiness", "preflight-only", args.ros_overlay, False,
                          args.sharpa_backend)
    report["checks"]["policy"] = {"status": "NOT_CHECKED", "reason": "waiting for live graph and observations"}
    if not report["readiness_pass"]:
        return report
    try:
        observation = live_observation(args.ros_overlay, args.prompt)
    except Exception as error:
        report["checks"]["observation"] = {"status": "FAIL", "reason": str(error)}
        report["status"] = "FAIL"
        report["failures"]["observation"] = report["checks"]["observation"]
        return report
    try:
        runtime = SharpaRuntime(args.policy_url)
        metadata, actions, chunk = runtime.infer_chunk(observation)
    except Exception as error:
        report["checks"]["policy"] = {"status": "FAIL", "reason": str(error)}
        report["failures"]["policy"] = report["checks"]["policy"]
        report["status"] = "FAIL"
        return report
    observation_age_ns = time.monotonic_ns() - observation["_captured_monotonic_ns"]
    # Exactly the existing execute observation-age check, without hardware output.
    fresh = 0 <= observation_age_ns <= 200_000_000
    report = ros_boundary(actions, chunk.session_id, "preflight-only", args.ros_overlay, False,
                          args.sharpa_backend)
    report["timing"] = {"observation": observation.get("_timing", {}), "policy_client": runtime.last_timing,
                        "snapshot_to_action_ms": observation_age_ns/1e6}
    report["checks"]["policy"] = {"status": "PASS", "checkpoint": metadata["checkpoint"],
                                    "norm_stats": metadata["norm_stats"], "action_shape": list(actions.shape)}
    report["checks"]["observation"] = {
        "status": "PASS" if fresh else "FAIL", "age_after_inference_ms": observation_age_ns / 1e6,
        "reason": "fresh realtime observation" if fresh else "Observation expired during policy inference",
    }
    if not fresh:
        report["status"] = "FAIL"
        report["failures"]["observation"] = report["checks"]["observation"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(epilog="Shell launcher options: --model {full,lora,full-delta}, --list-models")
    parser.add_argument("--mode", choices=("dry-run", "ros-test", "preflight-only", "prepare", "execute", "handoff-verify", "check"), default="dry-run")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="Print model/execution settings without ROS, SDK or model loading")
    modes.add_argument("--execute", action="store_true", help="Alias for --mode execute; requires --enable-hardware")
    modes.add_argument("--preflight-only", action="store_true", help="Inspect live graph, feedback and policy without command publishers")
    modes.add_argument("--prepare", action="store_true", help="Verify configured task pose and settling; motion backend pending")
    modes.add_argument('--handoff-verify',action='store_true',help='Verify persistent acquisition IPC + policy without enabling motion')
    from .handoff import DEFAULT_SOCKET
    parser.add_argument('--handoff-socket',type=Path,default=DEFAULT_SOCKET)
    parser.add_argument("--prepare-config", type=Path, help="Legacy litchibot only: configured 58D task pose")
    parser.add_argument("--prepare-policy-first", action="store_true",
                        help="Execute-only: slowly approach initial policy first target through existing guards, then re-infer")
    parser.add_argument("--prepare-speed-rad-s", type=float, default=0.03,
                        help="Joint-space approach speed (rad/s); native guarded limits remain authoritative")
    speeds = parser.add_mutually_exclusive_group()
    speeds.add_argument("--speed", choices=("slow", "medium", "fast"),
                        help="Sharpa time-stretch preset; independent of the checkpoint training method")
    speeds.add_argument("--execute-speed-scale", type=float,
                       help="Custom time scale in (0,1]; alternative to --speed; default is fast (1.0)")
    parser.add_argument("--execute-steps", type=int,
                        help="Execute first N of 50 predicted actions, then reobserve/reinfer (default 15)")
    parser.add_argument("--prediction-clip-rad", type=float, default=0.05,
                        help="Maximum per-value correction to hand limits for predicted actions only; 0 is strict")
    parser.add_argument("--broker-mode", choices=("serial", "rtc"), default="serial",
                        help="serial keeps existing behavior; rtc overlaps inference and accepted motion")
    parser.add_argument("--rtc-initial-delay-steps", type=int, default=5)
    parser.add_argument("--rtc-delay-buffer-size", type=int, default=10)
    parser.add_argument("--rtc-prefix-attention-schedule", choices=("ones", "zeros", "linear", "exp"), default="exp")
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=5.0)
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--initialization", choices=("current", "home", "policy"),
                                help="sdk_direct: current live state (default), fixed Home, or legacy policy first target")
    initialization.add_argument("--home", dest="initialization", action="store_const", const="home",
                                help="sdk_direct: go to fixed arm/hand Home, then reobserve and execute")
    parser.add_argument("--home-poses", type=Path, default=Path(os.environ.get(
        "SHARPA_HOME_POSES", str(Path(__file__).resolve().parents[2] / ".deployment/operator_poses.yaml"))))
    parser.add_argument("--home-task", default="powder_weighing")
    parser.add_argument("--hand-home", type=Path, default=Path(__file__).resolve().parents[2] / ".deployment/hand_home.json",
                        help="Named 22D targets for both hands; required in fixed Home mode")
    parser.add_argument("--num-chunks", type=int, default=1,
                        help="Number of replanning rounds; 0 repeats until Ctrl-C")
    parser.add_argument("--policy-url", default=config.POLICY_URL)
    parser.add_argument("--episode-parquet", type=Path)
    parser.add_argument("--sample-frame", type=int, default=0, help="Offline episode row to read")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--ros-overlay", type=Path, help="ROS overlay containing teleop_interfaces for Pixi ROS Python")
    parser.add_argument("--enable-hardware", action="store_true")
    parser.add_argument("--sharpa-backend", choices=("litchibot", "sdk_direct"), default="sdk_direct",
                        help="Sharpa command backend; sdk_direct bypasses the legacy litchibot validation path")
    parser.add_argument("--execution-trace-dir", type=Path, default=config.EXECUTION_TRACE_DIR,
                        help="Directory for sdk_direct per-chunk JSON execution traces")
    parser.add_argument("--inference-preset", choices=("paper-rolling",))
    parser.add_argument("--inpainting", choices=("off", "soft", "correlation"), default="off")
    parser.add_argument("--compression", choices=("off", "cubic"), default="off")
    parser.add_argument("--reference-max-age-s", type=float, default=1.0)
    parser.add_argument("--hand-variation-rad", type=float, default=0.05)
    args = parser.parse_args()
    args.rolling_options = None
    if args.inference_preset:
        if args.broker_mode != "serial":
            parser.error("paper-rolling requires --broker-mode serial; RTC is a separate experiment")
        from .rolling import RollingOptions
        if config.MODEL_PROFILE.name not in ("full", "full-delta") or args.sharpa_backend != "sdk_direct":
            parser.error("paper-rolling requires full/full-delta and sdk_direct")
        if args.initialization == "policy" or args.prepare_policy_first:
            parser.error("paper-rolling supports current or Home initialization")
        if args.execute_steps not in (None, 26):
            parser.error("paper-rolling requires --execute-steps 26")
        args.execute_steps = 26
        try:
            args.rolling_options = RollingOptions(args.inpainting, args.compression,
                args.reference_max_age_s, args.hand_variation_rad)
        except ValueError as error:
            parser.error(str(error))
    elif args.inpainting != "off" or args.compression != "off":
        parser.error("--inpainting/--compression require --inference-preset paper-rolling")
    if not np.isfinite(args.prediction_clip_rad) or args.prediction_clip_rad < 0:
        parser.error("--prediction-clip-rad must be finite and nonnegative")
    if args.prepare_policy_first and args.sharpa_backend == "sdk_direct":
        if args.initialization not in (None, "policy"):
            parser.error("--prepare-policy-first conflicts with --home or --initialization current/home")
        args.initialization = "policy"
    args.initialization = args.initialization or "current"
    if args.sharpa_backend == "litchibot" and args.initialization == "home":
        parser.error("--home/--initialization home requires sdk_direct")
    if args.execute_steps is None:
        args.execute_steps = 15 if args.sharpa_backend == "sdk_direct" else 50
    if not 1 <= args.execute_steps <= config.ACTION_HORIZON:
        parser.error("--execute-steps must be in [1, 50]")
    if args.sharpa_backend == "litchibot" and args.execute_steps != 50:
        parser.error("legacy litchibot requires --execute-steps 50; prefix execution uses sdk_direct")
    if args.broker_mode == "rtc":
        from .rtc_stream import RTCSettings
        try:
            RTCSettings(args.rtc_initial_delay_steps, args.rtc_delay_buffer_size,
                        args.rtc_prefix_attention_schedule, args.rtc_max_guidance_weight)
        except ValueError as error:
            parser.error(str(error))
        if args.sharpa_backend != "sdk_direct" or config.MODEL_PROFILE.finetune_mode != "full":
            parser.error("RTC currently supports only sdk_direct full/full-delta JAX deployments")
        if args.initialization == "policy":
            parser.error("RTC supports current or Home initialization, not legacy policy prepare")
        if args.num_chunks == 1 or args.num_chunks < 0 or not 1 <= args.execute_steps <= 24:
            parser.error("RTC needs --num-chunks 0 or >=2 and --execute-steps in [1, 24] (replan interval)")
    from .profiles import resolve_speed
    profile = resolve_speed(args.speed, args.execute_speed_scale)
    args.speed = profile.name
    args.execute_speed_scale = profile.execute_speed_scale
    if args.check:
        if args.mode != "dry-run":
            parser.error("Use --check or --mode, not both")
        args.mode = "check"
    if args.mode == "check":
        from .policy_prepare import HOME_TOLERANCE_RAD
        if (args.enable_hardware or args.execute or args.preflight_only or args.prepare
                or args.handoff_verify or args.episode_parquet):
            parser.error("check does not accept hardware, live modes or recorded inputs")
        if (not np.isfinite(args.execute_speed_scale) or not 0 < args.execute_speed_scale <= 1
                or not np.isfinite(args.prepare_speed_rad_s) or args.prepare_speed_rad_s <= 0
                or args.num_chunks < 0):
            parser.error("Invalid speed, prepare speed or chunk count")
        print(json.dumps({"mode": "check", "hardware_output": False, "broker_mode": args.broker_mode,
                          "speed": args.speed, "execute_speed_scale": args.execute_speed_scale,
                          "prediction_clip_rad": args.prediction_clip_rad,
                          "prepare_speed_rad_s": args.prepare_speed_rad_s,
                          "home_tolerance_rad": HOME_TOLERANCE_RAD,
                          "num_chunks": args.num_chunks, "action_hz": config.ACTION_HZ,
                          "execute_steps": args.execute_steps, "initialization": args.initialization,
                          "home_poses": str(args.home_poses), "home_task": args.home_task,
                          "hand_initialization": ("fixed_named_hand_home" if args.initialization == "home"
                                                  else "policy_first_target" if args.initialization == "policy"
                                                  else "current_measured_positions"),
                          "hand_home": str(args.hand_home),
                          "action_shape": [config.ACTION_HORIZON, config.ACTION_DIM],
                          "checkpoint": str(config.CHECKPOINT) if config.CHECKPOINT is not None else None,
                          "model_profile": config.MODEL_PROFILE.name, "policy_url": args.policy_url,
                          "training_action_supervision": config.MODEL_PROFILE.training_action_supervision,
                          "action_representation": "absolute_joint_position",
                          "sharpa_backend": args.sharpa_backend,
                          "inference_experiment": args.rolling_options.as_dict() if args.rolling_options else None}, indent=2))
        return
    if args.execute or args.preflight_only or args.prepare or args.handoff_verify:
        if args.mode != "dry-run":
            parser.error("Use a mode flag or --mode, not both")
        args.mode = "execute" if args.execute else "prepare" if args.prepare else "handoff-verify" if args.handoff_verify else "preflight-only"
    if args.broker_mode == "rtc" and args.mode != "execute":
        parser.error("--broker-mode rtc is for execute or offline --check only")
    if args.rolling_options and args.mode != "execute":
        parser.error("paper-rolling requires --check or --execute; use paper_replay for synthetic evaluation")
    if args.mode == "execute":
        if not args.enable_hardware or args.episode_parquet:
            parser.error("execute requires --enable-hardware and live ROS observations; recorded samples are forbidden")
        if args.sharpa_backend == "sdk_direct":
            if args.prepare_config is not None:
                parser.error("sdk_direct uses --initialization current/home/policy; --prepare-config belongs to the legacy backend")
            args.prepare_policy_first = True
        elif args.prepare_config is None and not args.prepare_policy_first:
            parser.error("legacy execute requires --prepare-config or --prepare-policy-first")
    elif args.mode == "prepare":
        if args.enable_hardware or args.episode_parquet or args.prepare_config is None:
            parser.error("prepare requires --prepare-config and forbids recorded input/hardware enable; guarded motion backend pending")
    elif args.mode in ("preflight-only","handoff-verify"):
        if args.enable_hardware or args.episode_parquet:
            parser.error("preflight-only accepts live ROS observations only and forbids --enable-hardware")
    elif args.enable_hardware:
        parser.error("offline modes do not accept hardware enable")
    elif not args.episode_parquet:
        parser.error("offline modes require --episode-parquet")
    if args.prepare_policy_first and (args.mode != 'execute' or args.prepare_config is not None):
        parser.error("--prepare-policy-first requires execute and cannot be mixed with --prepare-config")
    if (args.num_chunks < 0 or not np.isfinite(args.prepare_speed_rad_s)
            or args.prepare_speed_rad_s <= 0
            or not np.isfinite(args.execute_speed_scale)
            or not 0 < args.execute_speed_scale <= 1):
        parser.error("Positive finite prepare speed, execute speed scale in (0,1], and nonnegative num-chunks required (0 = continuous)")
    if args.num_chunks != 1 and not args.prepare_policy_first:
        parser.error("Multiple chunks require the persistent --prepare-policy-first path")
    if args.mode != "dry-run" and args.ros_overlay is None:
        parser.error("ROS modes require --ros-overlay with teleop_interfaces built for the Pixi ROS Python")
    if args.mode=='handoff-verify':
        try:
            print(json.dumps(verify_handoff(args),indent=2))
        except Exception as error:
            print(json.dumps({'status':'FAIL','mode':'handoff-verify','executed':False,'reason':str(error)},indent=2))
            raise SystemExit(1)
        return
    if args.mode == "prepare":
        try:
            report, observation = prepare_before_observation(args)
            report["post_prepare_observation"] = "PASS"
            print(json.dumps(report, indent=2))
        except Exception as error:
            print(json.dumps({"status": "FAIL", "mode": "prepare", "published": False, "reason": str(error)}))
            raise SystemExit(1)
        return
    if args.mode == "preflight-only":
        try:
            _start_observation_worker(args.ros_overlay)
            report = run_preflight_only(args)
        except Exception as error:
            report = {"mode": "preflight-only", "status": "FAIL", "published": False,
                      "command_publishers_created": 0, "reason": str(error)}
        print(json.dumps(report, indent=2))
        if report["status"] != "PASS":
            raise SystemExit(1)
        return
    preparation = None
    if args.mode == 'execute' and args.prepare_policy_first:
        try:
            print(json.dumps(execute_policy_prepare(args), indent=2))
        except KeyboardInterrupt:
            print("Execution interrupted; worker released. See the latest *-run.json report.", file=sys.stderr)
            raise SystemExit(130)
        return
    if args.mode == "execute":
        preparation, observation = prepare_before_observation(args)
    else:
        observation = load_sample(args.episode_parquet, args.prompt, args.sample_frame)
    metadata, actions, chunk = SharpaRuntime(args.policy_url).infer_chunk(observation)
    if args.mode == "execute" and not 0 <= time.monotonic_ns() - observation["_captured_monotonic_ns"] <= 200_000_000:
        raise RuntimeError("Observation expired during policy inference; no command will be published")
    offline_result = {"status": "NOT RUN", "blocking": False}
    if args.mode != "execute":
        offline_result = offline_sanity_check(actions, observation["observation/state"])
        print(f"OFFLINE SANITY: {offline_result['status']} (diagnostic only)", file=sys.stderr)
        print("EXECUTE PREFLIGHT: WAITING FOR LIVE ROS STATE", file=sys.stderr)
    result = {
        "mode": args.mode,
        "prepare": preparation,
        "policy_url": args.policy_url,
        "checkpoint": metadata["checkpoint"],
        "norm_stats": metadata["norm_stats"],
        "action_shape": list(actions.shape),
        "finite": bool(np.isfinite(actions).all()),
        "frames": len(chunk.frames),
        "fr3_dim": len(chunk.frames[0].fr3.positions),
        "sharpa_dims": [len(chunk.frames[0].left_sharpa.position), len(chunk.frames[0].right_sharpa.position)],
        "session_id": chunk.session_id,
        "sequence": [chunk.frames[0].sequence, chunk.frames[-1].sequence],
        "published": False,
        "offline_sanity": offline_result,
        "execute_preflight": {"status": "WAITING FOR LIVE ROS STATE"},
        "sharpa_backend": args.sharpa_backend,
    }
    if args.mode != "dry-run":
        result["ros"] = ros_boundary(actions, chunk.session_id, args.mode, args.ros_overlay,
                                      args.enable_hardware, args.sharpa_backend)
        result["published"] = result["ros"]["published"]
        result["execute_preflight"] = result["ros"]["execute_preflight"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
