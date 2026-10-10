"""Small uv-side entry point for batch LeRobot export.

The ROS bag reader lives in the Pixi environment. This wrapper is intentionally
separate so batch_convert.py can invoke it in the repository's uv environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from export_lerobot import export_sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    job = json.loads(args.sources_json.read_text(encoding="utf-8"))
    sources = [Path(item) for item in job["sources"]]
    intervals = {
        str(name): [tuple(float(value) for value in pair) for pair in values]
        for name, values in job.get("intervals", {}).items()
    }
    result = export_sources(sources, args.repo_id, args.output, task=args.task,
                            overwrite=args.overwrite, intervals=intervals)
    print(json.dumps({"status": result["status"], "repo_id": args.repo_id,
                      "output": str(args.output), "episodes": len(result["episodes"]),
                      "frames": sum(item["frames"] for item in result["episodes"]),
                      "validation": result["validation"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
