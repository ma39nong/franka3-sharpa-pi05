"""Compare two causal intermediate reports without reopening the ROS bag."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tyro


def _summary(source: Path) -> dict:
    report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    topics = report["observation_age_ms"]
    return {
        "source": str(source),
        "fps": report["fps"],
        "frames": report["converted_frames"],
        "dropped_frames": report["dropped_frames"],
        "causal": report["causal"],
        "observation_age_ms": {topic: {key: value for key, value in age.items()}
                                for topic, age in topics.items()},
        "repetition": report.get("stream_repetition", {}),
        "action_motion": report.get("action_motion", {}),
    }


def _top_event_retention(low: Path, high: Path, percentile: float = 95.0) -> dict:
    low_report = json.loads((low / "report.json").read_text(encoding="utf-8"))
    high_report = json.loads((high / "report.json").read_text(encoding="utf-8"))
    low_data, high_data = np.load(low / "episode.npz"), np.load(high / "episode.npz")
    low_delta = np.linalg.norm(np.diff(low_data["action"], axis=0), axis=1)
    high_delta = np.linalg.norm(np.diff(high_data["action"], axis=0), axis=1)
    low_events = np.flatnonzero(low_delta >= np.percentile(low_delta, percentile))
    high_events = np.flatnonzero(high_delta >= np.percentile(high_delta, percentile))
    low_times = (low_events + 1) / float(low_report["fps"])
    high_times = (high_events + 1) / float(high_report["fps"])
    retained = sum(np.min(np.abs(high_times - time)) <= 1.0 / float(high_report["fps"]) + 1e-9 for time in low_times)
    return {"percentile": percentile, "low_events": int(len(low_events)), "high_events": int(len(high_events)),
            "retained_low_events": int(retained),
            "retention_rate": float(retained / len(low_events)) if len(low_events) else None,
            "low_peak_delta_l2_rad": float(np.max(low_delta)) if len(low_delta) else 0.0,
            "high_peak_delta_l2_rad": float(np.max(high_delta)) if len(high_delta) else 0.0}


def main(low: Path, high: Path, output: Path | None = None) -> None:
    result = {"low": _summary(low), "high": _summary(high),
              "top_action_event_retention": [_top_event_retention(low, high, p) for p in (90.0, 95.0, 99.0)]}
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if output is None:
        print(text)
    else:
        output.write_text(text, encoding="utf-8")
        print(json.dumps({"status": "written", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    tyro.cli(main)
