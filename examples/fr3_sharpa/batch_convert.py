"""Batch-check ROS bags, optionally convert and export one LeRobot dataset."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .convert_rosbag import convert
except ImportError:  # direct execution from the documented ROS environment
    from convert_rosbag import convert


def _load_clips(path: Path | None) -> dict[str, list[tuple[float, float]]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("clips JSON must be an object keyed by episode name")
    result: dict[str, list[tuple[float, float]]] = {}
    for episode, intervals in data.items():
        if not isinstance(intervals, list):
            raise ValueError(f"clips for {episode} must be a list")
        result[str(episode)] = [(float(item[0]), float(item[1])) for item in intervals]
    return result


def _valid_intermediate(destination: Path, bag: Path, hz: float, causal: bool) -> bool:
    report_path = destination / "report.json"
    if not report_path.is_file() or not (destination / "episode.npz").is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return (report.get("status") == "converted_intermediate" and report.get("dropped_frames", 1) == 0
                and report.get("bag") == str(bag) and float(report.get("fps")) == float(hz)
                and bool(report.get("causal")) == bool(causal))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _valid_export(path: Path) -> bool:
    try:
        return json.loads((path / "export_report.json").read_text(encoding="utf-8")).get("status") == "exported"
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _git_commit(project: Path) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(project), "rev-parse", "HEAD"],
                                text=True, capture_output=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except OSError:
        return None


def main(input_root: Path, output_root: Path, hz: float = 30.0, tolerance_ms: float = 80.0,
         include_unfinalized: bool = False, causal: bool = False, export_lerobot: bool = False,
         lerobot_output: Path | None = None, repo_id: str = "fr3_sharpa/batch",
         task: str = "right hand picks tomato, left hand places it in the bowl",
         overwrite: bool = False, skip_processed: bool = True, clips_json: Path | None = None,
         uv_project: Path | None = None) -> None:
    episodes = sorted(
        (p for p in input_root.glob("episode*") if (p / "metadata.yaml").is_file() and list(p.glob("*.db3"))),
        key=lambda p: (int("".join(c for c in p.name if c.isdigit()) or "-1"), p.name),
    )
    if not episodes:
        raise FileNotFoundError(f"No ROS bags found under {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    clips = _load_clips(clips_json)
    project_root = Path(__file__).resolve().parents[2]
    manifest: dict[str, Any] = {
        "version": "fr3_sharpa_v2", "protocol_version": "fr3_sharpa_58d_v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit(project_root),
        "camera_config": {"camera_count": 3, "keys": ["cam0", "cam1", "cam2"], "encoding": "RGB"},
        "state_dim": 58, "action_dim": 58, "hz": hz, "tolerance_ms": tolerance_ms, "causal": causal,
        "task": task, "input_root": str(input_root), "episodes": [], "rejected": [],
        "lerobot_export": {"enabled": export_lerobot, "repo_id": repo_id,
                            "output": str(lerobot_output) if lerobot_output else None, "status": "pending"},
        "clips_json": str(clips_json) if clips_json else None,
    }
    accepted_sources: list[Path] = []
    accepted_intervals: dict[str, list[tuple[float, float]]] = {}
    for bag in episodes:
        state_path = bag / "collection_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
        if not include_unfinalized and state.get("finalized") is not True:
            manifest["rejected"].append({"episode": bag.name, "bag": str(bag), "reason": "not_finalized",
                                         "conversion": {"status": "SKIP", "exit_code": None}})
            continue
        destination = output_root / bag.name
        try:
            reused = skip_processed and _valid_intermediate(destination, bag, hz, causal)
            report = json.loads((destination / "report.json").read_text(encoding="utf-8")) if reused else convert(
                bag, destination, hz=hz, tolerance_ms=tolerance_ms, causal=causal)
            quality = {"dropped_frames": report.get("dropped_frames"),
                       "converted_frames": report.get("converted_frames"),
                       "observation_age_ms": report.get("observation_age_ms"),
                       "stream_alignment_ms": report.get("stream_alignment_ms"),
                       "reused": reused}
            if report.get("status") != "converted_intermediate" or report.get("dropped_frames", 0):
                manifest["rejected"].append({"episode": bag.name, "bag": str(bag), "reason": "quality_gate",
                                             "conversion": {"status": "FAIL", "exit_code": 1},
                                             "quality": quality, "report": str(destination / "report.json")})
                continue
            intervals = clips.get(bag.name, [])
            success = state.get("success") if isinstance(state.get("success"), bool) else None
            manifest["episodes"].append({"episode_id": bag.name, "episode": bag.name,
                                          "source_bag": str(bag), "bag": str(bag), "source": str(destination),
                                          "frames": report["converted_frames"], "task": task,
                                          "crop_intervals_s": [list(x) for x in intervals], "quality": quality,
                                          "status": "accepted", "conversion_status": "PASS",
                                          "conversion": {"status": "PASS", "exit_code": 0},
                                          "lerobot_export_status": "pending", "success": success, "error": ""})
            accepted_sources.append(destination)
            if intervals:
                accepted_intervals[bag.name] = intervals
        except Exception as exc:  # keep the batch moving; record the exact failure
            manifest["rejected"].append({"episode": bag.name, "bag": str(bag), "reason": str(exc),
                                         "conversion": {"status": "FAIL", "exit_code": 1, "error": str(exc)},
                                         "report": str(destination / "report.json")})
    if export_lerobot:
        if lerobot_output is None:
            raise ValueError("--lerobot-output is required with --export-lerobot")
        if skip_processed and _valid_export(lerobot_output) and not overwrite:
            export_result = json.loads((lerobot_output / "export_report.json").read_text(encoding="utf-8"))
            manifest["lerobot_export"].update({"status": "skipped_existing", "validation": export_result.get("validation")})
            for item in manifest["episodes"]:
                item["lerobot_status"] = "skipped_existing"
                item["lerobot_export_status"] = "skipped_existing"
        elif accepted_sources:
            project = uv_project or Path(__file__).resolve().parents[2]
            job_path = output_root / "lerobot_export_job.json"
            job_path.write_text(json.dumps({"sources": [str(path) for path in accepted_sources],
                                            "intervals": {name: [list(pair) for pair in pairs]
                                                          for name, pairs in accepted_intervals.items()}},
                                           indent=2, ensure_ascii=False), encoding="utf-8")
            command = ["uv", "run", "--project", str(project), "python",
                       str(Path(__file__).with_name("export_lerobot_batch.py")),
                       "--sources-json", str(job_path), "--output", str(lerobot_output),
                       "--repo-id", repo_id, "--task", task]
            if overwrite:
                command.append("--overwrite")
            export_env = os.environ.copy()
            # Pixi exports ROS Python paths into its environment.  They must
            # not leak into the Python 3.11 uv process (especially NumPy's
            # compiled extensions).
            export_env["PYTHONPATH"] = str(project / "src")
            for variable in ("PYTHONHOME", "AMENT_PREFIX_PATH", "CMAKE_PREFIX_PATH",
                             "COLCON_PREFIX_PATH", "LD_LIBRARY_PATH"):
                export_env.pop(variable, None)
            export_env["PATH"] = ":".join(part for part in export_env.get("PATH", "").split(":")
                                           if ".pixi/envs/ros2" not in part)
            completed = subprocess.run(command, cwd=project, env=export_env, text=True,
                                       capture_output=True, check=False)
            export_record = {"command": command, "exit_code": completed.returncode,
                             "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]}
            if completed.returncode == 0 and _valid_export(lerobot_output):
                export_result = json.loads((lerobot_output / "export_report.json").read_text(encoding="utf-8"))
                manifest["lerobot_export"].update({"status": "PASS", "validation": export_result.get("validation"),
                                                    "episodes": export_result.get("episodes"), "process": export_record})
                for item in manifest["episodes"]:
                    item["lerobot_status"] = "exported"
                    item["lerobot_export_status"] = "PASS"
            else:
                manifest["lerobot_export"].update({"status": "FAIL", "process": export_record,
                                                    "error": "uv exporter failed or export_report.json was not PASS"})
                for item in manifest["episodes"]:
                    item["lerobot_status"] = "export_failed"
                    item["lerobot_export_status"] = "FAIL"
        else:
            manifest["lerobot_export"]["status"] = "no_accepted_sources"
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"accepted": len(manifest["episodes"]), "rejected": len(manifest["rejected"]),
                      "manifest": str(manifest_path), "lerobot_export": manifest["lerobot_export"]["status"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--tolerance-ms", type=float, default=80.0)
    parser.add_argument("--include-unfinalized", action="store_true")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--export-lerobot", action="store_true")
    parser.add_argument("--lerobot-output", type=Path)
    parser.add_argument("--repo-id", default="fr3_sharpa/batch")
    parser.add_argument("--task", default="right hand picks tomato, left hand places it in the bowl")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-skip-processed", dest="skip_processed", action="store_false")
    parser.add_argument("--clips-json", type=Path,
                        help="JSON object such as {\"episode284\": [[12.0, 45.0], [50.0, 80.0]]}")
    parser.add_argument("--uv-project", type=Path,
                        help="OpenPI project root for the uv LeRobot exporter (defaults to this repository)")
    args = parser.parse_args()
    main(**vars(args))
