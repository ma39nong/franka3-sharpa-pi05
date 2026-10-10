"""Read-only ROS 2 bag -> synchronized FR3/Sharpa intermediate dataset.

The script intentionally has no ROS node, publisher, or hardware API.  It uses
rosbag2_py only as a sequential reader and writes an auditable report before
writing the synchronized rows.  A later LeRobot exporter can consume episode.npz
and the JPEG frames without reopening the original bag.
"""

from __future__ import annotations

import dataclasses
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from openpi.policies.fr3_sharpa_protocol import ACTION_DIM, SHARPA_JOINTS, join_action, join_state, reorder_named

ARM_TOPICS = {"left": "/left/franka/joint_states", "right": "/right/franka/joint_states"}
HAND_STATE_TOPICS = {"left": "/sharpa/left/joint_states", "right": "/sharpa/right/joint_states"}
HAND_ACTION_TOPICS = {"left": "/sharpa/left/command", "right": "/sharpa/right/command"}
CAMERA_TOPICS = {"cam0": "/cam0/color/image_raw", "cam1": "/cam1/color/image_raw", "cam2": "/cam2/color/image_raw"}
REQUIRED = (*ARM_TOPICS.values(), *HAND_STATE_TOPICS.values(), *HAND_ACTION_TOPICS.values(), *CAMERA_TOPICS.values())


@dataclasses.dataclass
class Sample:
    time_ns: int
    value: Any


def _stamp(message, fallback_ns: int) -> int:
    header = getattr(message, "header", None)
    if header is None:
        return fallback_ns
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def _image_bytes(message) -> bytes:
    if message.encoding.lower() not in {"rgb8", "bgr8"} or message.step < message.width * 3:
        raise ValueError(f"unsupported camera encoding/stride: {message.encoding}/{message.step}")
    rows = np.frombuffer(bytes(message.data), dtype=np.uint8).reshape(message.height, message.step)
    array = rows[:, : message.width * 3].reshape(message.height, message.width, 3)
    if message.encoding.lower() == "bgr8":
        array = array[..., ::-1]
    out = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(out, format="JPEG", quality=85, optimize=True)
    return out.getvalue()


class _CdrReader:
    """Small CDR reader for the recorded ArmCommand message.

    The custom teleop_interfaces package is built for Python 3.10 on the
    capture workstation, while this offline ROS environment is Python 3.12.
    Parsing this fixed message locally avoids importing or rebuilding a ROS
    package and does not change the bag.
    """

    def __init__(self, payload: bytes):
        self.data = memoryview(payload)[4:]  # CDR encapsulation header
        self.pos = 0

    def align(self, size: int):
        self.pos = (self.pos + size - 1) // size * size

    def raw(self, size: int) -> bytes:
        value = self.data[self.pos : self.pos + size].tobytes()
        self.pos += size
        return value

    def u32(self) -> int:
        self.align(4); return int.from_bytes(self.raw(4), "little")

    def i32(self) -> int:
        self.align(4); return int.from_bytes(self.raw(4), "little", signed=True)

    def u64(self) -> int:
        self.align(8); return int.from_bytes(self.raw(8), "little")

    def string(self) -> str:
        size = self.u32()
        value = self.raw(size)
        if not value or value[-1] != 0:
            raise ValueError("invalid CDR string")
        return value[:-1].decode("utf-8")

    def strings(self) -> list[str]:
        return [self.string() for _ in range(self.u32())]

    def doubles(self) -> list[float]:
        count = self.u32(); self.align(8)
        return [np.frombuffer(self.raw(8), dtype="<f8")[0].item() for _ in range(count)]


def _deserialize_arm_command(payload: bytes) -> dict:
    reader = _CdrReader(payload)
    reader.i32(); reader.u32(); reader.string()  # std_msgs/Header
    # The recorded bag predates the current ArmCommand.msg and contains the
    # deployed legacy layout: Header, joint_names, positions.  Detect it from
    # the bounded joint-name sequence.  The current extended layout remains
    # documented above and can be added without changing the protocol output.
    checkpoint = reader.pos
    count = reader.u32()
    if 1 <= count <= 32:
        try:
            joint_names = [reader.string() for _ in range(count)]
            positions = reader.doubles()
            if len(joint_names) == len(positions) and all("joint" in name for name in joint_names):
                return {"source": "legacy", "session_id": "", "sequence": 0,
                        "active_sides": [], "joint_names": joint_names, "positions": positions}
        except (IndexError, UnicodeDecodeError, ValueError):
            pass
    reader.pos = checkpoint
    source = reader.string(); session_id = reader.string(); sequence = reader.u64()
    active_sides = reader.strings(); joint_names = reader.strings(); positions = reader.doubles()
    if len(joint_names) != len(positions):
        raise ValueError("ArmCommand joint_names/positions mismatch")
    return {"source": source, "session_id": session_id, "sequence": sequence,
            "active_sides": active_sides, "joint_names": joint_names, "positions": positions}


def _nearest(samples: list[Sample], target: int, tolerance_ns: int) -> tuple[Sample | None, int | None]:
    if not samples:
        return None, None
    idx = int(np.searchsorted([s.time_ns for s in samples], target))
    choices = samples[max(0, idx - 1) : min(len(samples), idx + 1)]
    item = min(choices, key=lambda s: abs(s.time_ns - target))
    error = item.time_ns - target
    return (item, error) if abs(error) <= tolerance_ns else (None, error)


def _causal(samples: list[Sample], target: int, tolerance_ns: int) -> tuple[Sample | None, int | None]:
    """Choose only the newest sample at or before target; never use future data."""
    if not samples:
        return None, None
    times = [s.time_ns for s in samples]
    idx = int(np.searchsorted(times, target, side="right")) - 1
    if idx < 0:
        return None, samples[0].time_ns - target
    item = samples[idx]
    age = item.time_ns - target
    return (item, age) if -age <= tolerance_ns else (None, age)


def _joint(message, *, expected_kind: str) -> np.ndarray:
    names = list(message.name)
    values = list(message.position)
    if expected_kind == "arm":
        expected = tuple(names)  # arm names are checked by explicit side mapping below
    else:
        expected = tuple(names)
    return np.asarray(reorder_named(values, names, expected), dtype=np.float32)


def read_bag(bag: Path):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"), rosbag2_py.ConverterOptions("", ""))
    topic_meta = {item.name: item for item in reader.get_all_topics_and_types()}
    wanted_types = {name: topic_meta[name].type for name in REQUIRED if name in topic_meta}
    message_types = {topic: get_message(kind) for topic, kind in wanted_types.items() if kind in ("sensor_msgs/msg/JointState", "sensor_msgs/msg/Image")}
    samples: dict[str, list[Sample]] = {topic: [] for topic in (*REQUIRED, "/teleop/validated_arm_commands")}
    raw_counts: dict[str, int] = {topic: 0 for topic in topic_meta}
    raw_first: dict[str, int] = {}
    raw_last: dict[str, int] = {}
    unsupported = []
    while reader.has_next():
        topic, data, bag_time = reader.read_next()
        raw_counts[topic] = raw_counts.get(topic, 0) + 1
        raw_first.setdefault(topic, bag_time)
        raw_last[topic] = bag_time
        if topic == "/teleop/validated_arm_commands":
            try:
                message = _deserialize_arm_command(data)
            except (IndexError, UnicodeDecodeError, ValueError) as exc:
                unsupported.append({"topic": topic, "error": str(exc)})
                continue
            samples[topic].append(Sample(bag_time, message))
            continue
        if topic not in message_types:
            if topic in REQUIRED:
                unsupported.append({"topic": topic, "type": wanted_types.get(topic)})
            continue
        message = deserialize_message(data, message_types[topic])
        time_ns = _stamp(message, bag_time)
        if topic in (*ARM_TOPICS.values(), *HAND_STATE_TOPICS.values(), *HAND_ACTION_TOPICS.values()):
            if not message.position or not np.isfinite(np.asarray(message.position, dtype=np.float64)).all():
                continue
        samples[topic].append(Sample(time_ns, message))
    for values in samples.values():
        values.sort(key=lambda x: x.time_ns)
    return samples, raw_counts, raw_first, raw_last, unsupported, topic_meta


def convert(bag: Path, output: Path, *, hz: float = 30.0, tolerance_ms: float = 80.0, causal: bool = False) -> dict:
    samples, counts, first, last, unsupported, topic_meta = read_bag(bag)
    required_with_arm_action = (*REQUIRED, "/teleop/validated_arm_commands")
    missing_topics = [topic for topic in required_with_arm_action if not samples[topic]]
    if missing_topics:
        raise ValueError(f"required topic has no valid messages: {missing_topics}")
    start = max(samples[topic][0].time_ns for topic in required_with_arm_action)
    end = min(samples[topic][-1].time_ns for topic in required_with_arm_action)
    step = int(round(1_000_000_000 / hz))
    tolerance = int(tolerance_ms * 1_000_000)
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    rows_state, rows_action, rows_time = [], [], []
    arm_action_sets = {}
    image_errors = {name: [] for name in CAMERA_TOPICS}
    stream_errors = {topic: [] for topic in required_with_arm_action}
    stream_selected_times = {topic: [] for topic in required_with_arm_action}
    dropped = []
    cursor = start
    while cursor <= end:
        chosen = {}
        failures = []
        for topic in required_with_arm_action:
            item, error = (_causal if causal else _nearest)(samples[topic], cursor, tolerance)
            if item is None:
                failures.append({"topic": topic, "nearest_error_ms": None if error is None else error / 1e6})
            else:
                chosen[topic] = item
                stream_errors[topic].append(item.time_ns - cursor)
                stream_selected_times[topic].append(item.time_ns)
        if failures:
            dropped.append({"time_ns": cursor, "missing": failures})
            cursor += step
            continue
        try:
            arm_l = reorder_named(chosen[ARM_TOPICS["left"]].value.position, chosen[ARM_TOPICS["left"]].value.name,
                                  tuple(f"left_fr3_joint{i}" for i in range(1, 8)))
            arm_r = reorder_named(chosen[ARM_TOPICS["right"]].value.position, chosen[ARM_TOPICS["right"]].value.name,
                                  tuple(f"right_fr3_joint{i}" for i in range(1, 8)))
            hand_l = reorder_named(chosen[HAND_STATE_TOPICS["left"]].value.position,
                                   chosen[HAND_STATE_TOPICS["left"]].value.name,
                                   tuple(f"left_{name}" for name in SHARPA_JOINTS))
            hand_r = reorder_named(chosen[HAND_STATE_TOPICS["right"]].value.position,
                                   chosen[HAND_STATE_TOPICS["right"]].value.name,
                                   tuple(f"right_{name}" for name in SHARPA_JOINTS))
            cmd_l = reorder_named(chosen[HAND_ACTION_TOPICS["left"]].value.position,
                                  chosen[HAND_ACTION_TOPICS["left"]].value.name, SHARPA_JOINTS)
            cmd_r = reorder_named(chosen[HAND_ACTION_TOPICS["right"]].value.position,
                                  chosen[HAND_ACTION_TOPICS["right"]].value.name, SHARPA_JOINTS)
            arm_action = chosen["/teleop/validated_arm_commands"].value
            arm_action_sets[tuple(arm_action["joint_names"])] = arm_action_sets.get(tuple(arm_action["joint_names"]), 0) + 1
            action_l = reorder_named(arm_action["positions"], arm_action["joint_names"],
                                     tuple(f"left_fr3v2_joint{i}" for i in range(1, 8)))
            action_r = reorder_named(arm_action["positions"], arm_action["joint_names"],
                                     tuple(f"right_fr3v2_joint{i}" for i in range(1, 8)))
            rows_state.append(join_state(arm_l, hand_l, arm_r, hand_r))
            rows_action.append(join_action(action_l, cmd_l, action_r, cmd_r))
            rows_time.append(cursor)
            for name, cam_topic in CAMERA_TOPICS.items():
                image_path = output / "images" / name
                image_path.mkdir(exist_ok=True)
                (image_path / f"{len(rows_time)-1:06d}.jpg").write_bytes(_image_bytes(chosen[cam_topic].value))
                image_errors[name].append(chosen[cam_topic].time_ns - cursor)
            cursor += step
        except ValueError as exc:
            dropped.append({"time_ns": cursor, "error": str(exc)})
            cursor += step
            continue
    report = {
        "bag": str(bag), "output": str(output), "fps": hz, "tolerance_ms": tolerance_ms, "causal": causal,
        "required_topics": required_with_arm_action, "unsupported_required_topics": unsupported,
        "topic_counts": counts, "topic_time_range_ns": {t: [first.get(t), last.get(t)] for t in counts},
        "topic_rates_hz": {t: (float(counts[t] - 1) / ((last[t] - first[t]) / 1e9)
                               if t in first and last[t] > first[t] else None) for t in counts},
        "source_time_range_ns": [start, end], "candidate_frames": int(max(0, (end - start) // step + 1)),
        "converted_frames": len(rows_time), "dropped_frames": len(dropped), "drop_examples": dropped[:20],
        "state_shape": [len(rows_state), ACTION_DIM], "action_shape": [len(rows_action), ACTION_DIM],
        "action_source": "/teleop/validated_arm_commands + /sharpa/{left,right}/command",
        "state_source": "measured FR3 and Sharpa joint_states",
        "image_alignment_ms": {name: {"min": float(np.min(errors) / 1e6) if errors else None,
                                       "max": float(np.max(errors) / 1e6) if errors else None,
                                       "mean_abs": float(np.mean(np.abs(errors)) / 1e6) if errors else None}
                               for name, errors in image_errors.items()},
        "stream_alignment_ms": {topic: {"min": float(np.min(errors) / 1e6) if errors else None,
                                         "max": float(np.max(errors) / 1e6) if errors else None,
                                         "mean_abs": float(np.mean(np.abs(errors)) / 1e6) if errors else None}
                                for topic, errors in stream_errors.items()},
        "stream_repetition": {topic: {
            "selected_frames": len(times),
            "unique_source_samples": len(set(times)),
            "repeated_frames": len(times) - len(set(times)),
            "repeated_rate": float((len(times) - len(set(times))) / len(times)) if times else None,
        } for topic, times in stream_selected_times.items()},
        "causal_future_sample_count": {topic: int(sum(error > 0 for error in errors))
                                       for topic, errors in stream_errors.items()},
        "camera_formats": {name: {"encoding": getattr(samples[topic][0].value, "encoding", None),
                                   "height": getattr(samples[topic][0].value, "height", None),
                                   "width": getattr(samples[topic][0].value, "width", None),
                                   "step": getattr(samples[topic][0].value, "step", None)}
                           for name, topic in CAMERA_TOPICS.items()},
        "joint_protocol": [dataclasses.asdict(spec) for spec in __import__("openpi.policies.fr3_sharpa_protocol", fromlist=["JOINT_SPECS"]).JOINT_SPECS],
        "arm_action_joint_set_counts": {"|".join(names): count for names, count in arm_action_sets.items()},
        "joint_value_ranges": {
            "state": {name: {"min": float(np.min(np.asarray(rows_state)[:, i])) if rows_state else None,
                             "max": float(np.max(np.asarray(rows_state)[:, i])) if rows_state else None}
                       for i, name in enumerate(__import__("openpi.policies.fr3_sharpa_protocol", fromlist=["ORDER"]).ORDER)},
            "action": {name: {"min": float(np.min(np.asarray(rows_action)[:, i])) if rows_action else None,
                              "max": float(np.max(np.asarray(rows_action)[:, i])) if rows_action else None}
                        for i, name in enumerate(__import__("openpi.policies.fr3_sharpa_protocol", fromlist=["ORDER"]).ORDER)},
        },
        "observation_age_ms": {topic: {"p50": float(np.percentile(-np.asarray(errors) / 1e6, 50)) if errors else None,
                                        "p95": float(np.percentile(-np.asarray(errors) / 1e6, 95)) if errors else None,
                                        "max": float(np.max(-np.asarray(errors) / 1e6)) if errors else None,
                                        "timeout_samples": int(sum(error is None or error > tolerance for error in errors))}
                           for topic, errors in stream_errors.items()},
        "action_motion": {
            "delta_l2_threshold_rad": 0.01,
            "active_frames": int(np.sum(np.linalg.norm(np.diff(np.asarray(rows_action), axis=0), axis=1) > 0.01)) if len(rows_action) > 1 else 0,
            "total_delta_frames": max(0, len(rows_action) - 1),
            "active_fraction": float(np.mean(np.linalg.norm(np.diff(np.asarray(rows_action), axis=0), axis=1) > 0.01)) if len(rows_action) > 1 else 0.0,
            "max_delta_l2_rad": float(np.max(np.linalg.norm(np.diff(np.asarray(rows_action), axis=0), axis=1))) if len(rows_action) > 1 else 0.0,
        },
        "unit": "rad", "nan_inf_rejected": True, "status": "converted_intermediate",
    }
    np.savez_compressed(output / "episode.npz", time_ns=np.asarray(rows_time, dtype=np.int64),
                        state=np.asarray(rows_state, dtype=np.float32), action=np.asarray(rows_action, dtype=np.float32))
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--tolerance-ms", type=float, default=80.0)
    parser.add_argument("--causal", action="store_true", help="use only samples at or before each target time")
    args = parser.parse_args()
    print(json.dumps(convert(args.bag, args.output, hz=args.hz, tolerance_ms=args.tolerance_ms, causal=args.causal), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
