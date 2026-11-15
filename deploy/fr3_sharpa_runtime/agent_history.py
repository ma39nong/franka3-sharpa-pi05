"""Read-only ROS history sidecar: source-time causal 10Hz/16-frame windows."""

import argparse
from collections import deque
import json
import os
from pathlib import Path
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
    rows = []
    for offset in range(15, -1, -1):
        target = anchor_ns - offset * 100_000_000
        row = {}
        for key in TOPICS:
            candidates = [v for v in histories[key] if 0 < stamp_ns(v[0]) <= target]
            if not candidates:
                raise ValueError("历史预热中：" + key)
            entry = max(candidates, key=lambda v: stamp_ns(v[0]))
            if target - stamp_ns(entry[0]) > tolerance_ns:
                raise ValueError("历史缺帧：" + key)
            row[key] = entry
        stamps = [stamp_ns(v[0]) for v in row.values()]
        if max(stamps) - min(stamps) > tolerance_ns:
            raise ValueError("历史多模态未对齐")
        rows.append(row)
    return rows


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    os.replace(temp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    import cv2
    from rclpy.executors import ExternalShutdownException

    provider = ObservationProvider(node_name="fr3_sharpa_agent_observation")
    provider.histories = {k: deque(maxlen=256) for k in TOPICS}

    # Extend ONLY this independent subscriber's buffer; original runtime untouched.
    def remember(key, message):
        now = time.monotonic_ns()
        entry = (message, now)
        provider.latest[key] = entry
        history = provider.histories[key]
        history.append(entry)
        while history and now - history[0][1] > 3_000_000_000:
            history.popleft()

    # Subscription lambdas look up provider.remember at callback time.
    provider.remember = remember
    directory = args.directory
    directory.mkdir(parents=True, exist_ok=True)
    last_emit = 0.0
    try:
        while True:
            provider.rclpy.spin_once(provider.node, timeout_sec=0.005)
            now = time.monotonic()
            if now - last_emit < 0.5:
                continue
            last_emit = now
            try:
                if len(provider.latest) != len(TOPICS):
                    raise ValueError("等待三路相机和四组关节反馈")
                ros_now = provider.node.get_clock().now().nanoseconds
                captured_unix = time.time()
                issues = timing_issues(provider.latest, ros_now, time.monotonic_ns())
                if issues:
                    raise ValueError("; ".join(issues))
                anchor = min(stamp_ns(v[0]) for v in provider.latest.values())
                rows = select_history(provider.histories, anchor)
                rgb = []
                for camera in ("cam0", "cam1", "cam2"):
                    frames = []
                    for row in rows:
                        message = row[camera][0]
                        # Match training's RGB full-frame INTER_AREA resize.
                        payload = rgb_image(message)
                        import base64

                        pixels = np.frombuffer(base64.b64decode(payload["rgb_b64"]), np.uint8).reshape(payload["shape"])
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
                np.savez(
                    temp, rgb=np.stack(rgb), joints=np.asarray(joints, np.float32), joint_names=np.array(JOINT_NAMES)
                )
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
                    history_source_ns={k: [stamp_ns(r[k][0]) for r in rows] for k in TOPICS},
                )
                atomic_json(directory / "latest.json", metadata)
                for old in directory.glob("*.npz"):
                    if time.time() - old.stat().st_mtime > 10:
                        old.unlink(missing_ok=True)
            except (ValueError, TimeoutError) as error:
                atomic_json(directory / "latest.json", dict(valid=False, error=str(error), captured_unix=time.time()))
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        provider.close()


if __name__ == "__main__":
    main()
