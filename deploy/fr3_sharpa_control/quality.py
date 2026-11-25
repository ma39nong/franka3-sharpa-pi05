"""Persistent operator ratings, independent of execution and bag integrity.

Run directories stay fixed while video export works in the background. A grade
never renames/deletes a bag or changes its recorded failure/cleanup evidence.
"""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import re
import time
import uuid

from .recording import write_json

GRADES = {"A": "优等", "B": "一般", "C": "报错", "D": "放弃"}
FILE = "data_quality.json"
RUN_PATH = re.compile(r"\d{4}-\d{2}-\d{2}/\d{2}\.\d{2}(?:-\d+)?")


@contextmanager
def locked(directory):
    with (Path(directory) / ".data_quality.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def read(directory):
    return json.loads((Path(directory) / FILE).read_text())


def begin(directory):
    directory = Path(directory)
    with locked(directory):
        if (directory / FILE).exists():
            raise ValueError("不能复用已有录制的评级记录")
        write_json(directory / FILE, {
            "schema": "sharpa.recording-quality.v1", "run_id": uuid.uuid4().hex,
            "grade": None, "label": None, "note": "", "revision": 0, "history": [],
            "review_ready": False, "execution": {"outcome": "RUNNING"},
        })


def finish(directory, *, outcome, reason, returncode, recorder_closed):
    """Only the lifecycle owner seals evidence after recorder/robot cleanup."""
    directory = Path(directory)
    with locked(directory):
        value = read(directory)
        value.update(review_ready=bool(recorder_closed), execution={
            "outcome": outcome, "reason": reason, "returncode": returncode,
            "recorder_closed": bool(recorder_closed), "ended_unix_ns": time.time_ns(),
        })
        write_json(directory / FILE, value)


def resolve_run(output, relative):
    if not isinstance(relative, str) or RUN_PATH.fullmatch(relative) is None:
        raise ValueError("录制编号无效，请刷新录制列表")
    root = Path(output).expanduser().resolve()
    directory = (root / relative).resolve()
    if not directory.is_relative_to(root) or not directory.is_dir():
        raise ValueError("录制目录不在所选归档父目录内")
    return directory


def summary(directory):
    directory = Path(directory)
    value = read(directory)
    recording = {}
    evidence_error = None
    try:
        if (directory / "recording.json").exists():
            recording = json.loads((directory / "recording.json").read_text())
    except (OSError, ValueError) as error:
        evidence_error = str(error)
    bag = directory / "rosbag"
    started = recording.get("started_unix_ns") is not None
    available = bag.is_dir() or started
    bag_status = recording.get("status", "UNKNOWN")
    warnings = []
    if value["execution"].get("outcome") != "FINISHED":
        warnings.append("本次执行未正常完成，保留原始中断原因")
    if bag_status != "COMPLETE":
        warnings.append("录制未确认完整：" + str(bag_status))
    if not (bag / "metadata.yaml").is_file():
        warnings.append("ROSbag 缺少 metadata.yaml，需单独检查可读性")
    if evidence_error:
        warnings.append("录制报告无法读取：" + evidence_error)
    if available and recording.get("ended_unix_ns") is None:
        warnings.append("录制器未留下完整收尾报告，人工评级不会补全该证据")
    return {**value, "directory": str(directory), "recording_status": bag_status,
            "recording_error": recording.get("error"), "counts": recording.get("counts", {}),
            "has_recording": available,
            "can_rate": bool(value["review_ready"] and available),
            "warnings": warnings,
            "training_validation": "UNVERIFIED"}


def list_recordings(output, *, limit=100):
    root = Path(output).expanduser().resolve()
    if not root.is_dir():
        return []
    # Bounded to the date/minute layout; never recursively scan bag payloads.
    paths = sorted(root.glob("????-??-??/*/" + FILE), reverse=True)
    result = []
    for path in paths:
        relative = path.parent.relative_to(root).as_posix()
        if RUN_PATH.fullmatch(relative) is None or not path.parent.resolve().is_relative_to(root):
            continue
        try:
            result.append({"relative": relative, **summary(path.parent)})
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if len(result) >= limit:
            break
    return result


def rate(output, relative, *, run_id, grade, note="", expected_revision):
    if grade not in GRADES:
        raise ValueError("请选择 A 优等、B 一般、C 报错或 D 放弃")
    if not isinstance(note, str) or len(note) > 2000:
        raise ValueError("评级备注最多2000字")
    if type(expected_revision) is not int:
        raise ValueError("缺少评级版本，请刷新列表")
    directory = resolve_run(output, relative)
    with locked(directory):
        value = read(directory)
        if value["run_id"] != run_id or value["revision"] != expected_revision:
            raise ValueError("录制或评级版本已变化，请刷新后再保存")
        if not summary(directory)["can_rate"]:
            raise ValueError("录制尚未收尾或没有录制数据，暂不能评级")
        event = {"grade": grade, "label": GRADES[grade], "note": note.strip(),
                 "revision": value["revision"] + 1, "rated_unix_ns": time.time_ns()}
        value.update(event)
        value["history"].append(event)
        write_json(directory / FILE, value)
    return summary(directory)
