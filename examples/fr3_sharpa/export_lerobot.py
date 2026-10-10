"""Export validated FR3/Sharpa intermediates into LeRobot v2.1 datasets."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable
from fractions import Fraction

import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from openpi.policies.fr3_sharpa_protocol import ACTION_DIM, ORDER

CAMERAS = ("cam0", "cam1", "cam2")
CAMERA_SIZES = {"cam0": (400, 640, 3), "cam1": (480, 640, 3), "cam2": (480, 640, 3)}


def _read_source(source: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    report_path = source / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "converted_intermediate":
        raise ValueError(f"{source}: source report is not a validated Sharpa conversion")
    if report.get("dropped_frames", 0):
        raise ValueError(f"{source}: contains {report['dropped_frames']} dropped frames")
    data = np.load(source / "episode.npz")
    state, action, times = data["state"], data["action"], data["time_ns"]
    if state.shape != action.shape or state.ndim != 2 or state.shape[1:] != (ACTION_DIM,) or len(times) != len(state):
        raise ValueError(f"{source}: invalid intermediate shapes state={state.shape}, action={action.shape}, time={times.shape}")
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"{source}: timestamps must be finite and strictly increasing")
    fps = float(report.get("fps", 0))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"{source}: report.fps must be positive")
    observed_dt = np.diff(times).astype(np.float64) / 1e9
    expected_dt = 1.0 / fps
    max_dt_error = float(np.max(np.abs(observed_dt - expected_dt)))
    if max_dt_error > 2e-8:
        raise ValueError(f"{source}: report fps={fps} disagrees with timestamp grid (max error {max_dt_error:.3g}s)")
    for camera in CAMERAS:
        image_dir = source / "images" / camera
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)
        missing = [i for i in range(len(state)) if not (image_dir / f"{i:06d}.jpg").is_file()]
        if missing:
            raise ValueError(f"{source}: {camera} missing {len(missing)} image frames")
    report.setdefault("export_validation", {})["source_timestamp_max_error_s"] = max_dt_error
    report["export_validation"]["source_duration_s"] = float((times[-1] - times[0]) / 1e9)
    return report, state.astype(np.float32), action.astype(np.float32), times.astype(np.int64)


def _features() -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "observation.state": {"dtype": "float32", "shape": (ACTION_DIM,), "names": list(ORDER)},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": list(ORDER)},
    }
    for camera, size in CAMERA_SIZES.items():
        features[f"observation.images.{camera}"] = {
            "dtype": "video", "shape": size, "names": ["height", "width", "channel"]
        }
    return features


def _interval_indices(times: np.ndarray, interval: tuple[float, float] | None) -> np.ndarray:
    if interval is None:
        return np.arange(len(times), dtype=np.int64)
    start_s, end_s = interval
    if not (math.isfinite(start_s) and math.isfinite(end_s) and 0 <= start_s < end_s):
        raise ValueError(f"invalid crop interval {interval}; expected 0 <= start < end seconds")
    relative = (times - times[0]) / 1e9
    indices = np.flatnonzero((relative >= start_s) & (relative < end_s))
    if len(indices) < 2:
        raise ValueError(f"crop interval {interval} contains fewer than two frames")
    return indices.astype(np.int64)


def _add_segment(dataset: LeRobotDataset, source: Path, state: np.ndarray, action: np.ndarray,
                 indices: np.ndarray, task: str) -> dict[str, Any]:
    first, last = int(indices[0]), int(indices[-1])
    segment_state = state[indices]
    segment_action = action[indices]
    state_delta = np.diff(segment_state, axis=0)
    action_delta = np.diff(segment_action, axis=0)
    from PIL import Image
    for original_index in indices:
        original_index = int(original_index)
        frame = {"observation.state": state[original_index], "action": action[original_index], "task": task}
        for camera in CAMERAS:
            path = source / "images" / camera / f"{original_index:06d}.jpg"
            frame[f"observation.images.{camera}"] = np.asarray(Image.open(path).convert("RGB"))
        dataset.add_frame(frame)
    dataset.save_episode()
    return {
        "source": str(source), "source_start_index": first, "source_end_index": last,
        "frames": len(indices), "task": task,
        "state_max_abs_step": float(np.max(np.abs(state_delta))) if len(state_delta) else 0.0,
        "action_max_abs_step": float(np.max(np.abs(action_delta))) if len(action_delta) else 0.0,
    }


def export_sources(sources: Iterable[Path], repo_id: str, output: Path, *, task: str,
                   overwrite: bool = False, intervals: dict[str, list[tuple[float, float]]] | None = None) -> dict[str, Any]:
    sources = [Path(s) for s in sources]
    if not sources:
        raise ValueError("no sources to export")
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite")
    loaded = [(source, *_read_source(source)) for source in sources]
    fps_values = {float(report["fps"]) for _, report, *_ in loaded}
    if len(fps_values) != 1:
        raise ValueError(f"all sources in one LeRobot dataset must use one FPS, got {sorted(fps_values)}")
    fps = fps_values.pop()
    dataset_fps = int(fps) if fps.is_integer() else Fraction(str(fps))
    dataset = LeRobotDataset.create(repo_id=repo_id, root=output, fps=dataset_fps, robot_type="dual_fr3_sharpa",
                                    features=_features(), use_videos=True, image_writer_processes=4,
                                    image_writer_threads=4)
    episodes = []
    source_reports = []
    expected_states = []
    for source, report, state, action, times in loaded:
        source_reports.append({"source": str(source), "bag": report.get("bag"), "fps": report["fps"],
                               "causal": report.get("causal"), "task": task})
        source_intervals = (intervals or {}).get(source.name, [])
        if not source_intervals:
            source_intervals = [None]
        for interval in source_intervals:
            indices = _interval_indices(times, interval)
            episode = _add_segment(dataset, source, state, action, indices, task)
            expected_states.append(state[indices])
            episode["crop_interval_s"] = list(interval) if interval is not None else None
            boundary_checks = []
            if interval is not None:
                first, last = int(indices[0]), int(indices[-1])
                if first > 0:
                    boundary_checks.append({"side": "start", "source_gap_before_s": float((times[first] - times[first - 1]) / 1e9),
                                            "state_max_abs_jump_rad": float(np.max(np.abs(state[first] - state[first - 1]))),
                                            "action_max_abs_jump_rad": float(np.max(np.abs(action[first] - action[first - 1])))})
                if last + 1 < len(times):
                    boundary_checks.append({"side": "end", "source_gap_after_s": float((times[last + 1] - times[last]) / 1e9),
                                            "state_max_abs_jump_rad": float(np.max(np.abs(state[last + 1] - state[last]))),
                                            "action_max_abs_jump_rad": float(np.max(np.abs(action[last + 1] - action[last])))})
            episode["crop_boundary_checks"] = boundary_checks
            episodes.append(episode)
    validation = verify_export(output, expected_fps=fps, expected_frames=np.concatenate(expected_states, axis=0))
    result = {"status": "exported" if validation["status"] == "PASS" else "FAIL", "repo_id": repo_id,
              "root": str(output), "fps": fps, "task": task, "episodes": episodes,
              "sources": source_reports, "validation": validation}
    (output / "export_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if result["status"] != "exported":
        raise ValueError(f"LeRobot export validation failed; see {output / 'export_report.json'}")
    return result


def verify_export(root: Path, *, expected_fps: float, expected_frames: np.ndarray | None = None,
                  expected_sources: list[tuple] | None = None) -> dict[str, Any]:
    """Cross-check info, parquet timestamps, videos, and representative samples."""
    failures: list[str] = []
    checks: dict[str, Any] = {}
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    info_fps = float(info["fps"])
    if not math.isclose(info_fps, expected_fps, rel_tol=0, abs_tol=1e-9):
        failures.append(f"info fps {info_fps} != expected {expected_fps}")
    checks["info_fps"] = info_fps
    import pyarrow.parquet as pq
    parquet_paths = sorted((root / "data").glob("**/*.parquet"))
    table = pq.read_table(parquet_paths)
    timestamps = np.asarray(table["timestamp"].to_numpy(zero_copy_only=False), dtype=np.float64)
    frame_count = len(timestamps)
    episode_indices = np.asarray(table["episode_index"].to_numpy(zero_copy_only=False), dtype=np.int64)
    timestamp_errors = []
    for episode_index in sorted(set(episode_indices.tolist())):
        rows = np.flatnonzero(episode_indices == episode_index)
        timestamp_errors.append(float(np.max(np.abs(timestamps[rows] - np.arange(len(rows)) / expected_fps))))
    timestamp_error = max(timestamp_errors, default=math.inf)
    if timestamp_error > 2e-6:
        failures.append(f"parquet timestamp max error {timestamp_error:.3g}s")
    episode_frame_counts = {int(e): int(np.sum(episode_indices == e)) for e in sorted(set(episode_indices.tolist()))}
    episode_durations = {str(e): float((count - 1) / expected_fps) for e, count in episode_frame_counts.items()}
    checks["parquet"] = {"frames": frame_count, "episodes": len(episode_frame_counts),
                          "first_timestamp_s": float(timestamps[0]), "last_timestamp_s": float(timestamps[-1]),
                          "max_grid_error_s": timestamp_error,
                          "episode_durations_s": episode_durations,
                          "duration_s": float(timestamps[-1] - timestamps[0]) if frame_count else None}
    import av
    videos = {}
    for video_path in sorted((root / "videos").glob("**/*.mp4")):
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        rate = float(stream.average_rate) if stream.average_rate else 0.0
        count = int(stream.frames)
        videos[str(video_path.relative_to(root))] = {"fps": rate, "frames": count,
                                                     "duration_s": float((count - 1) / rate) if count and rate else None}
        if not math.isclose(rate, expected_fps, rel_tol=0, abs_tol=1e-6):
            failures.append(f"video {video_path.name} fps {rate} != expected {expected_fps}")
        match = re.search(r"episode_(\d+)\.mp4$", video_path.name)
        expected_count = episode_frame_counts.get(int(match.group(1))) if match else frame_count
        if count != expected_count:
            failures.append(f"video {video_path.name} frames {count} != parquet episode {expected_count}")
        container.close()
    checks["videos"] = videos
    if expected_frames is None and expected_sources:
        expected_frames = np.concatenate([source[2] for source in expected_sources], axis=0)
    if expected_frames is not None:
        # pyav is available in the offline ROS/LeRobot environment; avoid
        # making validation depend on torchcodec's system FFmpeg libraries.
        ds = LeRobotDataset(repo_id=info.get("repo_id", ""), root=root, video_backend="pyav")
        sample_indices = [0, frame_count // 2, frame_count - 1]
        sample_checks = []
        expected_state = expected_frames
        if len(expected_state) == frame_count:
            for index in sorted(set(sample_indices)):
                item = ds[index]
                state = np.asarray(item["observation.state"])
                state_error = float(np.max(np.abs(state - expected_state[index])))
                images = {camera: tuple(np.asarray(item[f"observation.images.{camera}"]).shape) for camera in CAMERAS}
                sample_checks.append({"index": index, "timestamp_s": float(item["timestamp"]),
                                      "state_max_abs_error": state_error, "image_shapes": images})
                if state_error > 1e-5:
                    failures.append(f"dataset state mismatch at frame {index}: {state_error}")
        checks["sample_frames"] = sample_checks
    checks["episode_duration_s"] = checks["parquet"]["duration_s"]
    return {"status": "PASS" if not failures else "FAIL", "failures": failures, "checks": checks}


def main(source: Path, repo_id: str = "fr3_sharpa/episode0", output: Path | None = None,
         overwrite: bool = False, task: str = "right hand picks tomato, left hand places it in the bowl"):
    destination = output or Path.home() / ".cache/huggingface/lerobot" / repo_id
    result = export_sources([source], repo_id, destination, task=task, overwrite=overwrite)
    print(json.dumps({"status": result["status"], "repo_id": repo_id, "root": str(destination),
                      "fps": result["fps"], "episodes": len(result["episodes"]),
                      "frames": sum(e["frames"] for e in result["episodes"]),
                      "validation": result["validation"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    tyro.cli(main)
