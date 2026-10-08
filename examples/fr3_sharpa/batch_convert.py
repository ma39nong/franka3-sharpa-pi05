"""Batch-check and convert finalized Sharpa ROS bags without modifying them."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from .convert_rosbag import convert
except ImportError:  # direct execution from the documented ROS environment
    from convert_rosbag import convert


def main(input_root: Path, output_root: Path, hz: float = 30.0, tolerance_ms: float = 80.0,
         include_unfinalized: bool = False, causal: bool = False) -> None:
    episodes = sorted(
        (p for p in input_root.glob("episode*") if (p / "metadata.yaml").is_file() and list(p.glob("*.db3"))),
        key=lambda p: (int("".join(c for c in p.name if c.isdigit()) or "-1"), p.name),
    )
    if not episodes:
        raise FileNotFoundError(f"No ROS bags found under {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"version": "fr3_sharpa_v1", "hz": hz, "tolerance_ms": tolerance_ms,
                "episodes": [], "rejected": []}
    for bag in episodes:
        state_path = bag / "collection_state.json"
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        if not include_unfinalized and state.get("finalized") is not True:
            manifest["rejected"].append({"episode": bag.name, "reason": "not_finalized"})
            continue
        destination = output_root / bag.name
        try:
            report = convert(bag, destination, hz=hz, tolerance_ms=tolerance_ms, causal=causal)
        except Exception as exc:  # keep the batch moving; report the exact failure
            manifest["rejected"].append({"episode": bag.name, "reason": str(exc)})
            continue
        if report.get("status") != "converted_intermediate" or report.get("dropped_frames", 0):
            manifest["rejected"].append({"episode": bag.name, "reason": "quality_gate", "report": str(destination / "report.json")})
        else:
            manifest["episodes"].append({"episode": bag.name, "frames": report["converted_frames"],
                                          "report": str(destination / "report.json")})
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"accepted": len(manifest["episodes"]), "rejected": len(manifest["rejected"]),
                      "manifest": str(output_root / "manifest.json")}, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--tolerance-ms", type=float, default=80.0)
    parser.add_argument("--include-unfinalized", action="store_true")
    parser.add_argument("--causal", action="store_true")
    args = parser.parse_args()
    main(**vars(args))
