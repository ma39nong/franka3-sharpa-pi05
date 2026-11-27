"""Persistent ratings and deferred DAgger archive placement.

Deployment reports stay in their date directory. The rosbag link follows the
episode between pending and grade folders after background readers close.
"""
from contextlib import contextmanager
import fcntl
import json
import os
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


def payload(directory, value=None):
    value = read(directory) if value is None else value
    return Path(directory) / ("mcap" if value.get("recording_format") == "mcap" else "rosbag")


def begin(directory, *, archive_parent=None, recording_format="rosbag"):
    if recording_format not in ("rosbag", "mcap"):
        raise ValueError("Unsupported recording format")
    directory = Path(directory)
    with locked(directory):
        if (directory / FILE).exists():
            raise ValueError("不能复用已有录制的评级记录")
        storage = None
        if archive_parent is not None:
            root = Path(archive_parent).expanduser().resolve()
            if root == directory.absolute() or root.is_relative_to(directory.absolute()):
                raise ValueError("DAgger 保存父目录不能位于本轮归档目录内部")
            for folder in (*GRADES, "pending"):
                (root / folder).mkdir(parents=True, exist_ok=True)
            with locked(root):
                counter = root / ".episode_sequence.json"
                last = json.loads(counter.read_text())["last"] if counter.exists() else -1
                for folder in (*GRADES, "pending"):
                    for path in (root / folder).iterdir():
                        match = re.fullmatch(r"episode(\d+)", path.name)
                        if match:
                            last = max(last, int(match[1]))
                episode = f"episode{last + 1}"
                write_json(counter, {"last": last + 1})
            target = root / "pending" / episode
            (directory / ("mcap" if recording_format == "mcap" else "rosbag")).symlink_to(target, target_is_directory=True)
            storage = {"root": str(root), "episode": episode, "layout": "bag-episode-v1"}
        write_json(directory / FILE, {
            "archive": storage, "recording_format": recording_format,
            "schema": "sharpa.recording-quality.v1", "run_id": uuid.uuid4().hex,
            "grade": None, "label": None, "note": "", "revision": 0, "history": [],
            "review_ready": False, "execution": {"outcome": "RUNNING"},
        })


def save_quality(directory, value):
    write_json(directory / FILE, value)
    bag = payload(directory)
    if bag.is_dir():
        write_json(bag / FILE, value)
        if value.get("recording_format") == "mcap" and (bag / "manifest.json").is_file():
            from .mcap_storage import seal
            seal(bag, json.loads((bag / "manifest.json").read_text())["status"])


def finish(directory, *, outcome, reason, returncode, recorder_closed):
    """Only the lifecycle owner seals evidence after recorder/robot cleanup."""
    directory = Path(directory)
    with locked(directory):
        value = read(directory)
        value.update(review_ready=bool(recorder_closed), execution={
            "outcome": outcome, "reason": reason, "returncode": returncode,
            "recorder_closed": bool(recorder_closed), "ended_unix_ns": time.time_ns(),
        })
        bag = payload(directory, value)
        if value.get("recording_format") == "mcap" and (bag / "episode.json").is_file():
            episode = json.loads((bag / "episode.json").read_text())
            episode["end_reason"] = reason
            if outcome == "FAILED":
                episode["task_result"] = "failure"
            write_json(bag / "episode.json", episode)
        save_quality(directory, value)
    archive_when_ready(directory)


def archive_when_ready(directory):
    """Keep a requested grade durable until all background writers have ended."""
    directory = Path(directory).absolute()
    if not (directory / FILE).is_file():
        return
    with locked(directory):
        value = read(directory)
        storage = value.get("archive")
        if not storage or not value.get("grade") or not value["review_ready"]:
            return
        for name in ("video_export.json", "dagger/labels_export.json"):
            report = directory / name
            if report.exists():
                try:
                    status = json.loads(report.read_text()).get("status")
                except (OSError, ValueError):
                    return
                if status in ("QUEUED", "EXPORTING", "RUNNING"):
                    return
        # Older whole-run archives keep their existing location.
        if storage.get("layout") != "bag-episode-v1":
            return
        link = payload(directory, value)
        current = link.resolve()
        destination = Path(storage["root"]) / value["grade"] / storage["episode"]
        if current == destination or not current.is_dir():
            return
        try:
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"目标 episode 已存在，保留暂存数据：{destination}")
            current.rename(destination)
            replacement = directory / (".payload-" + uuid.uuid4().hex)
            try:
                replacement.symlink_to(destination, target_is_directory=True)
                os.replace(replacement, link)
            except OSError:
                destination.rename(current)
                replacement.unlink(missing_ok=True)
                raise
            value.pop("archive_error", None)
            save_quality(directory, value)
        except OSError as error:
            value["archive_error"] = str(error)
            save_quality(directory, value)


def resolve_run(output, relative):
    if not isinstance(relative, str) or RUN_PATH.fullmatch(relative) is None:
        raise ValueError("录制编号无效，请刷新录制列表")
    root = Path(output).expanduser().resolve()
    directory = (root / relative).resolve()
    alias = root / relative
    if not directory.is_dir():
        raise ValueError("录制目录不在所选归档父目录内")
    if not directory.is_relative_to(root):
        storage = read(alias).get("archive") or {}
        if storage.get("alias") != str(alias) or not directory.is_relative_to(Path(storage.get("root", "/nonexistent"))):
            raise ValueError("录制目录不在所选归档父目录内")
    return alias


def summary(directory):
    directory = Path(directory)
    value = read(directory)
    if (value["execution"].get("outcome") == "STOPPED"
            and value["execution"].get("reason") == "operator stop"
            and value["execution"].get("recorder_closed")
            and value["execution"].get("returncode") in (0, 130)):
        value["execution"]["outcome"] = "FINISHED"
    recording = {}
    evidence_error = None
    try:
        if (directory / "recording.json").exists():
            recording = json.loads((directory / "recording.json").read_text())
    except (OSError, ValueError) as error:
        evidence_error = str(error)
    bag = payload(directory)
    started = recording.get("started_unix_ns") is not None
    available = bag.is_dir() or started
    bag_status = recording.get("status", "UNKNOWN")
    warnings = []
    if value["execution"].get("outcome") != "FINISHED":
        warnings.append("本次执行未正常完成，保留原始中断原因")
    if bag_status != "COMPLETE":
        warnings.append("录制未确认完整：" + str(bag_status))
    if not (bag / "metadata.yaml").is_file():
        warnings.append(value.get("recording_format", "rosbag") + " 缺少 metadata.yaml，需单独检查可读性")
    if evidence_error:
        warnings.append("录制报告无法读取：" + evidence_error)
    if available and recording.get("ended_unix_ns") is None:
        warnings.append("录制器未留下完整收尾报告，人工评级不会补全该证据")
    storage = value.get("archive")
    archive_directory = (bag.resolve() if storage.get("layout") == "bag-episode-v1"
                         else directory.resolve()) if storage else None
    archive_pending = bool(storage and value["grade"] and archive_directory.parent.name != value["grade"])
    return {**value, "directory": str(directory),
            "archive_directory": str(archive_directory) if storage else None,
            "archive_pending": archive_pending, "recording_status": bag_status,
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
        if RUN_PATH.fullmatch(relative) is None:
            continue
        try:
            resolve_run(root, relative)
            archive_when_ready(path.parent)
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
        save_quality(directory, value)
    archive_when_ready(directory)
    return summary(directory)


def update_note(output, relative, *, run_id, note, expected_revision):
    directory = resolve_run(output, relative)
    value = read(directory)
    if not value.get("grade"):
        raise ValueError("请先在质量弹窗中选择评级")
    return rate(output, relative, run_id=run_id, grade=value["grade"],
                note=note, expected_revision=expected_revision)
