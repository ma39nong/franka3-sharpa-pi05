"""Read-only ROS history sidecar: source-time causal 10Hz/16-frame windows."""

import argparse
from bisect import bisect_right
from collections import deque
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
from .ros_observation import ObservationProvider, TOPICS, stamp_ns, ordered_state, rgb_image, timing_issues
from .protocol import SHARPA_JOINT_NAMES

JOINT_NAMES = tuple(
    n + ".position"
    for side in ("left", "right")
    for n in ([f"{side}_fr3_joint{i}" for i in range(1, 8)] + [f"{side}_{n}" for n in SHARPA_JOINT_NAMES])
)


def select_history(histories, anchor_ns, *, tolerance_ns=40_000_000):
    """Original samples only; no future frames, interpolation or timestamp repair."""
    entries = {}
    timestamps = {}
    for key in TOPICS:
        entries[key] = {}
        for entry in histories[key]:
            stamp = stamp_ns(entry[0])
            if 0 < stamp <= anchor_ns:
                entries[key].setdefault(stamp, entry)
        timestamps[key] = sorted(entries[key])
    rows = []
    for offset in range(15, -1, -1):
        target = anchor_ns - offset * 100_000_000
        row = {}
        for key in TOPICS:
            index = bisect_right(timestamps[key], target) - 1
            if index < 0:
                raise ValueError("历史预热中：" + key)
            stamp = timestamps[key][index]
            if target - stamp > tolerance_ns:
                raise ValueError("历史缺帧：" + key)
            row[key] = entries[key][stamp]
        stamps = [stamp_ns(v[0]) for v in row.values()]
        if max(stamps) - min(stamps) > tolerance_ns:
            raise ValueError("历史多模态未对齐")
        rows.append(row)
    return rows


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    os.replace(temp, path)


class HistoryBuffer:
    """Keep receiving original messages while another thread prepares windows."""

    def __init__(self, capacity):
        self.lock = threading.Lock()
        self.latest = {}
        self.histories = {key: deque(maxlen=capacity) for key in TOPICS}

    def remember(self, key, message):
        now = time.monotonic_ns()
        entry = (message, now)
        with self.lock:
            self.latest[key] = entry
            history = self.histories[key]
            history.append(entry)
            while history and now - history[0][1] > 3_000_000_000:
                history.popleft()

    def snapshot(self):
        with self.lock:
            return dict(self.latest), {key: tuple(rows) for key, rows in self.histories.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    roi = parser.add_mutually_exclusive_group()
    roi.add_argument("--shared-roi-directory", type=Path)
    roi.add_argument("--native-agent-roi", action="store_true")
    args = parser.parse_args()
    import cv2
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor

    provider = ObservationProvider(node_name="fr3_sharpa_agent_observation")
    buffer = HistoryBuffer(1024 if args.native_agent_roi else 256)
    provider.remember = buffer.remember
    executor = SingleThreadedExecutor(context=provider.node.context)
    executor.add_node(provider.node)
    spin_finished = threading.Event()
    spin_errors = []

    def receive():
        try:
            executor.spin()
        except ExternalShutdownException:
            pass
        except Exception as error:
            spin_errors.append(error)
        finally:
            spin_finished.set()

    receiver = threading.Thread(target=receive, name="agent-ros-receiver")
    directory = args.directory
    directory.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    next_emit = time.monotonic()
    receiver.start()
    try:
        while not spin_finished.wait(max(0.0, next_emit - time.monotonic())):
            cycle_start = time.monotonic()
            next_emit = max(next_emit + 0.5, cycle_start)
            try:
                latest, histories = buffer.snapshot()
                if len(latest) != len(TOPICS):
                    raise ValueError("等待三路相机和四组关节反馈")
                ros_now = provider.node.get_clock().now().nanoseconds
                captured_unix = time.time()
                issues = timing_issues(latest, ros_now, time.monotonic_ns())
                if issues:
                    raise ValueError("; ".join(issues))
                anchor = min(stamp_ns(v[0]) for v in latest.values())
                rows = select_history(
                    histories, anchor,
                    tolerance_ns=75_000_000 if args.native_agent_roi else 40_000_000,
                )
                rgb = []
                native_cam0 = []
                for camera in ("cam0", "cam1", "cam2"):
                    frames = []
                    for row in rows:
                        message = row[camera][0]
                        # Match training's RGB full-frame INTER_AREA resize.
                        payload = rgb_image(message)
                        import base64

                        pixels = np.frombuffer(base64.b64decode(payload["rgb_b64"]), np.uint8).reshape(payload["shape"])
                        if camera == "cam0" and (args.native_agent_roi or args.shared_roi_directory):
                            native_cam0.append(pixels)
                        frames.append(cv2.resize(pixels, (256, 256), interpolation=cv2.INTER_AREA))
                    rgb.append(np.stack(frames))
                joints = [
                    sum(
                        (
                            ordered_state(row[f"{side}_{kind}"][0], side, kind == "hand")
                            for side, kind in (("left", "arm"), ("left", "hand"), ("right", "arm"), ("right", "hand"))
                        ),
                        [],
                    )
                    for row in rows
                ]
                oid = str(anchor)
                path = directory / (oid + ".npz")
                temp = directory / (oid + ".tmp.npz")
                extras = {}; shared_meta = None
                if args.native_agent_roi:
                    extras = dict(
                        native_cam0=np.stack(native_cam0),
                        cam0_stamp_ns=np.array([stamp_ns(row["cam0"][0]) for row in rows], dtype=np.int64),
                    )
                if args.shared_roi_directory:
                    from .shared_roi import read_history
                    stamps=[{k:stamp_ns(v[0]) for k,v in row.items()} for row in rows]
                    local,roi_meta,shared_meta=read_history(args.shared_roi_directory,stamps,native_rgb=native_cam0,now_unix=captured_unix,anchor_ns=anchor)
                    rgb.append(np.stack([cv2.resize(im,(256,256),interpolation=cv2.INTER_AREA) for im in local]))
                    extras=dict(roi_metadata=roi_meta,roi_contract=np.array(shared_meta['roi_contract']))
                np.savez(temp,rgb=np.stack(rgb),joints=np.asarray(joints,np.float32),joint_names=np.array(JOINT_NAMES),**extras)
                os.replace(temp, path)
                # Preserve ROS epoch and source age. Never retimestamp an old frame as fresh.
                metadata = dict(
                    valid=True,
                    observation_id=oid,
                    observation_npz=str(path),
                    timestamp=anchor / 1e9,
                    ros_now=ros_now / 1e9,
                    captured_unix=captured_unix,
                    source_age_s=(ros_now - anchor) / 1e9,
                    shared_roi=shared_meta,
                    history_source_ns={k: [stamp_ns(r[k][0]) for r in rows] for k in TOPICS},
                    preparation_ms=(time.monotonic() - cycle_start) * 1000,
                )
                atomic_json(directory / "latest.json", metadata)
                for old in directory.glob("*.npz"):
                    if time.time() - old.stat().st_mtime > 10:
                        old.unlink(missing_ok=True)
            except (ValueError, TimeoutError) as error:
                atomic_json(directory / "latest.json", dict(valid=False, error=str(error), captured_unix=time.time()))
        if spin_errors:
            raise spin_errors[0]
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        receiver.join()
        provider.close()


if __name__ == "__main__":
    main()
