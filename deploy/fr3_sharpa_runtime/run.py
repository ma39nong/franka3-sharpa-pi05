"""One-chunk CLI. Default and ros-test modes cannot publish ROS commands."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from openpi.policies.fr3_sharpa_protocol import POLICY_CAMERA_KEYS

from . import config
from .preflight import offline_sanity_check
from .runtime import SharpaRuntime

DEFAULT_PROMPT = (
    "Pick up a tomato with the right hand, move it to the center, "
    "let the left hand receive it, and place it in the bowl."
)
GELLO_REPO = Path(os.environ.get("SHARPA_GELLO_REPO", "/home/user/gello_upper_body_teleop"))
ROS_PROJECT = GELLO_REPO / "portable_deps/litchi_hardware"


def load_sample(path: Path, prompt: str) -> dict:
    """Offline LeRobot sample for dry-run and ros-test only."""
    from PIL import Image
    import pyarrow.parquet as pq

    table = pq.read_table(path).slice(0, 1)

    def image(key):
        return np.asarray(Image.open(io.BytesIO(table[key][0].as_py()["bytes"])).convert("RGB"))

    return {
        "observation/state": np.asarray(table["observation.state"][0].as_py(), dtype=np.float32),
        **{key: image(f"observation.images.{camera}") for key, camera in POLICY_CAMERA_KEYS.items()},
        "prompt": prompt,
    }


def live_observation(overlay: Path, prompt: str) -> dict:
    """Read a single fresh ROS snapshot in Pixi, then cross the environment boundary."""
    import shlex

    setup = overlay / "setup.bash"
    if not setup.is_file():
        raise FileNotFoundError(setup)
    repo = str(Path(__file__).resolve().parents[2])
    inner = (
        f"source {shlex.quote(str(setup))}; "
        f"export PYTHONPATH={shlex.quote(repo)}:\"${{PYTHONPATH:-}}\"; "
        "exec python -m deploy.fr3_sharpa_runtime.ros_observation"
    )
    result = subprocess.run(
        ["pixi", "run", "--environment", "ros2", "bash", "-lc", inner],
        cwd=ROS_PROJECT, capture_output=True, text=True, timeout=15,
    )
    if result.returncode:
        raise RuntimeError(f"ROS observation failed ({result.returncode}): {result.stderr[-1500:]}")
    snapshot = json.loads(result.stdout.splitlines()[-1])

    def decode(key):
        payload = snapshot["images"][key]
        shape = tuple(payload["shape"])
        image = np.frombuffer(base64.b64decode(payload["rgb_b64"]), dtype=np.uint8)
        if len(shape) != 3 or shape[-1] != 3 or image.size != int(np.prod(shape)):
            raise ValueError(f"Invalid live {key} image")
        return image.reshape(shape).copy()

    return {
        "observation/state": np.asarray(snapshot["state"], dtype=np.float32),
        **{key: decode(camera) for key, camera in POLICY_CAMERA_KEYS.items()},
        "prompt": prompt,
        "_captured_monotonic_ns": snapshot["captured_monotonic_ns"],
    }


def ros_boundary(actions: np.ndarray | None, session_id: str, mode: str, overlay: Path | None,
                 enabled: bool):
    import shlex

    transport = ["python", "-m", "deploy.fr3_sharpa_runtime.ros_transport", "--mode", mode]
    if enabled:
        transport.append("--enable-hardware")
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
    report = ros_boundary(None, "preflight-readiness", "preflight-only", args.ros_overlay, False)
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
        metadata, actions, chunk = SharpaRuntime(args.policy_url).infer_chunk(observation)
    except Exception as error:
        report["checks"]["policy"] = {"status": "FAIL", "reason": str(error)}
        report["failures"]["policy"] = report["checks"]["policy"]
        report["status"] = "FAIL"
        return report
    observation_age_ns = time.monotonic_ns() - observation["_captured_monotonic_ns"]
    # Exactly the existing execute observation-age check, without hardware output.
    fresh = 0 <= observation_age_ns <= 200_000_000
    report = ros_boundary(actions, chunk.session_id, "preflight-only", args.ros_overlay, False)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry-run", "ros-test", "preflight-only", "execute"), default="dry-run")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true", help="Alias for --mode execute; requires --enable-hardware")
    modes.add_argument("--preflight-only", action="store_true", help="Inspect live graph, feedback and policy without command publishers")
    parser.add_argument("--policy-url", default=config.POLICY_URL)
    parser.add_argument("--episode-parquet", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--ros-overlay", type=Path, help="ROS overlay containing teleop_interfaces for Pixi ROS Python")
    parser.add_argument("--enable-hardware", action="store_true")
    args = parser.parse_args()
    if args.execute or args.preflight_only:
        if args.mode != "dry-run":
            parser.error("Use a mode flag or --mode, not both")
        args.mode = "execute" if args.execute else "preflight-only"
    if args.mode == "execute":
        if not args.enable_hardware or args.episode_parquet:
            parser.error("execute requires --enable-hardware and live ROS observations; recorded samples are forbidden")
    elif args.mode == "preflight-only":
        if args.enable_hardware or args.episode_parquet:
            parser.error("preflight-only accepts live ROS observations only and forbids --enable-hardware")
    elif args.enable_hardware:
        parser.error("offline modes do not accept hardware enable")
    elif not args.episode_parquet:
        parser.error("offline modes require --episode-parquet")
    if args.mode != "dry-run" and args.ros_overlay is None:
        parser.error("ROS modes require --ros-overlay with teleop_interfaces built for the Pixi ROS Python")
    if args.mode == "preflight-only":
        try:
            report = run_preflight_only(args)
        except Exception as error:
            report = {"mode": "preflight-only", "status": "FAIL", "published": False,
                      "command_publishers_created": 0, "reason": str(error)}
        print(json.dumps(report, indent=2))
        if report["status"] != "PASS":
            raise SystemExit(1)
        return
    observation = (
        live_observation(args.ros_overlay, args.prompt)
        if args.mode == "execute" else load_sample(args.episode_parquet, args.prompt)
    )
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
    }
    if args.mode != "dry-run":
        result["ros"] = ros_boundary(actions, chunk.session_id, args.mode, args.ros_overlay, args.enable_hardware)
        result["published"] = result["ros"]["published"]
        result["execute_preflight"] = result["ros"]["execute_preflight"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
