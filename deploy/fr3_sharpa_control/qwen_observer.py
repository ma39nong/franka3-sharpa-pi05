"""Observe the existing ROS history and task socket; never request robot motion."""

import argparse
from collections import OrderedDict
import fcntl
import hashlib
import itertools
import json
from pathlib import Path
import threading
import time
import uuid

from deploy.fr3_sharpa_runtime.task_control import task_rpc

from .qwen_client import MODELS
from .qwen_client import build_payload
from .qwen_client import recognize
from .qwen_memory import HarvestPlan
from .qwen_memory import TaskMemory
from .qwen_replay import atomic_json


def read_frames(directory, *, now):
    import cv2
    import numpy as np

    metadata = json.loads((Path(directory) / "latest.json").read_text())
    if (
        metadata.get("valid") is not True
        or not 0 <= now - metadata["captured_unix"] <= 1
        or abs(metadata["ros_now"] - metadata["captured_unix"]) > 0.25
        or not 0 <= metadata["source_age_s"] <= 0.5
    ):
        raise ValueError("实时观测无效、过期或时钟域不一致")
    frames = []
    with np.load(metadata["observation_npz"], allow_pickle=False) as data:
        rgb = data["rgb"]
        if rgb.shape not in ((3, 16, 256, 256, 3), (4, 16, 256, 256, 3)) or rgb.dtype != np.uint8:
            raise ValueError("实时RGB历史结构无效")
        for camera, index in (("cam0", 0), ("cam2", 2)):
            stamps = metadata["history_source_ns"][camera]
            if (
                len(stamps) != 16
                or any(type(t) is not int or t <= 0 for t in stamps)
                or any(not 0 < b - a <= 200_000_000 for a, b in itertools.pairwise(stamps))
                or not 0 <= metadata["timestamp"] - stamps[-1] / 1e9 <= 0.15
                or not 0 <= now - stamps[-1] / 1e9 <= 1.5
            ):
                raise ValueError("相机源时间不连续")
            if camera == "cam0" and "native_cam0" in data:
                native = data["native_cam0"]
                if native.ndim != 4 or native.shape[0] != 16 or native.shape[-1] != 3 or native.dtype != np.uint8:
                    raise ValueError("原分辨率主相机历史无效")
            for i in (0, 5, 10, 15):
                image = data["native_cam0"][i] if camera == "cam0" and "native_cam0" in data else rgb[index, i]
                height, width = image.shape[:2]
                if width > 640:
                    image = cv2.resize(image, (640, round(height * 640 / width)), interpolation=cv2.INTER_AREA)
                ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                if not ok:
                    raise ValueError("实时图像编码失败")
                frames.append(
                    ({"id": f"{camera}:{stamps[i]}", "timestamp": stamps[i] / 1e9, "camera": camera}, jpeg.tobytes())
                )
    return frames


class FrameBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.frames = OrderedDict()
        self.error = None

    def collect(self, directory, stop):
        while not stop.is_set():
            try:
                rows = read_frames(directory, now=time.time())
                with self.lock:
                    for frame, image in rows:
                        self.frames[frame["id"]] = (frame, image)
                    self.frames = OrderedDict(
                        sorted(self.frames.items(), key=lambda pair: pair[1][0]["timestamp"])[-256:]
                    )
                    self.error = None
            except (OSError, ValueError, KeyError) as error:
                with self.lock:
                    self.error = str(error)
            stop.wait(0.2)

    def window(self, memory):
        with self.lock:
            if self.error:
                raise ValueError(self.error)
            rows = list(self.frames.values())
        if not rows:
            return [], {}
        end = rows[-1][0]["timestamp"]
        if memory.last_end is not None and end <= memory.last_end + 0.5:
            return [], {}
        start = max(end - 6, memory.stage_started_at or 0)
        rows = [(frame, image) for frame, image in rows if start <= frame["timestamp"] <= end]
        # Keep the same original frame identity across overlapping snapshots.
        selected, last = [], {}
        for frame, image in rows:
            if frame["timestamp"] - last.get(frame["camera"], -float("inf")) >= 0.4:
                selected.append((frame, image))
                last[frame["camera"]] = frame["timestamp"]
        return [f for f, _ in selected], {f["id"]: image for f, image in selected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-directory", required=True, type=Path)
    parser.add_argument("--task-socket", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--quotas", nargs="+", type=int, default=[2, 2])
    parser.add_argument("--model", choices=MODELS, default=MODELS[0])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-requests", type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lease = (args.output / "observer.lock").open("a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = HarvestPlan(tuple(args.quotas))
    path = args.output / "memory.json"
    memory = TaskMemory.restore(json.loads(path.read_text())) if path.exists() else TaskMemory(args.run_id, plan)
    if memory.run_id != args.run_id or memory.plan != plan:
        raise ValueError("记忆属于其他会话或任务计划")
    buffer, stop = FrameBuffer(), threading.Event()
    worker = threading.Thread(target=buffer.collect, args=(args.history_directory, stop), daemon=True)
    worker.start()
    count = 0
    try:
        while not args.max_requests or count < args.max_requests:
            try:
                task = task_rpc(args.task_socket, args.run_id, renew_contact=False)
                if (
                    task["task_profile"] != "tomato-in-hand119"
                    or task["execution_state"] != "RUNNING"
                    or task["active_stage"] != task["requested_stage"]
                ):
                    raise ValueError("等待稳定运行的手中调整任务")
                active = task["active_stage"]
                if active == memory.stage and memory.stage_started_at is None:
                    memory.stage_started_at = task["active_applied_unix_s"]
                    atomic_json(path, memory.snapshot())
                if active != memory.stage:
                    memory.acknowledge(
                        active,
                        run_id=args.run_id,
                        expected_revision=memory.revision,
                        applied_at=task["active_applied_unix_s"],
                    )
                    atomic_json(path, memory.snapshot())
                frames, images = buffer.window(memory)
                if len(frames) < 4 or frames[-1]["timestamp"] - frames[0]["timestamp"] < 1:
                    stop.wait(0.2)
                    continue
                request = {
                    "id": uuid.uuid4().hex,
                    "run_id": memory.run_id,
                    "revision": memory.revision,
                    "active_stage": memory.stage,
                    "start": frames[0]["timestamp"],
                    "end": frames[-1]["timestamp"],
                    "frames": frames,
                }
                directory = args.output / request["id"]
                payload = build_payload(memory.snapshot(), request, images, model=args.model)
                atomic_json(
                    directory / "request.json",
                    {
                        "request": request,
                        "memory": memory.snapshot(),
                        "execution_version": task["stage_version"],
                        "model": args.model,
                        "enable_thinking": False,
                    },
                )
                for frame in frames:
                    blob = images[frame["id"]]
                    (directory / (hashlib.sha256(frame["id"].encode()).hexdigest() + ".jpg")).write_bytes(blob)
                count += 1
                result = recognize(payload)
                atomic_json(directory / "model-result.json", result)
                current = task_rpc(args.task_socket, args.run_id, renew_contact=False)
                if any(
                    current.get(k) != task.get(k)
                    for k in ("stage_version", "active_stage", "execution_state", "requested_stage")
                ):
                    raise ValueError("模型推理期间执行状态变化,旧回答未采用")
                evaluated = memory.evaluate(request, result["answer"], now=time.time())
                atomic_json(directory / "evaluation.json", evaluated)
                atomic_json(path, memory.snapshot())
                atomic_json(
                    args.output / "status.json",
                    {
                        "evaluation": evaluated,
                        "model": args.model,
                        "elapsed_s": result["elapsed_s"],
                        "control_eligible": False,
                    },
                )
            except (OSError, ValueError, RuntimeError, KeyError) as error:
                atomic_json(args.output / "status.json", {"error": str(error), "control_eligible": False})
                stop.wait(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        worker.join(timeout=2)


if __name__ == "__main__":
    main()
