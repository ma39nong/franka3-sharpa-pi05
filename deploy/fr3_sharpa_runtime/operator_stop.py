"""Identify explicit UI stops and seal interrupted reports after verified cleanup."""

import json
import os
from pathlib import Path


def operator_stop_requested():
    configured = os.environ.get("SHARPA_OPERATOR_STOP_FILE")
    return bool(configured and Path(configured).is_file())


def finish_operator_reports(directory):
    """Called by the lifecycle owner only after all cleanup checks succeed."""
    root = Path(directory)
    paths = [*root.glob("*-run.json"), *root.glob("rtc-*/run.json")]
    for path in paths:
        report = json.loads(path.read_text())
        if report.get("status") != "INTERRUPTED":
            continue
        report.update(status="FINISHED", reason="operator stop",
                      interruption=report.pop("error", "KeyboardInterrupt"), cleanup_verified=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)
