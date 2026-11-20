"""Causal single-window Qwen replay; emits advice, never hardware commands."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from .qwen_client import MODELS
from .qwen_client import build_payload
from .qwen_client import recognize
from .qwen_memory import HarvestPlan
from .qwen_memory import TaskMemory


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def video_window(dataset, episode, start, end, *, period=0.5, cameras=("cam0", "cam2")):
    import cv2

    if not 0 <= start < end or end - start > 12 or period < 0.2:
        raise ValueError("回放窗口须在0到12秒内,采样间隔至少0.2秒")
    dataset = Path(dataset)
    info = json.loads((dataset / "meta/info.json").read_text())
    fps = info["fps"]
    frames, images, sources = [], {}, {}
    for name in cameras:
        camera = "observation.images." + name
        path = dataset / info["video_path"].format(
            episode_chunk=episode // info.get("chunks_size", 1000),
            video_key=camera,
            episode_index=episode,
        )
        sources[camera] = hashlib.sha256(path.read_bytes()).hexdigest()
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened() or round(end * fps) >= int(capture.get(cv2.CAP_PROP_FRAME_COUNT)):
                raise ValueError("视频不可读或窗口超出视频末尾")
            index = 0
            while start + index * period <= end + 1e-6:
                frame_number = round((start + index * period) * fps)
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ok, pixels = capture.read()
                if not ok:
                    raise ValueError("视频缺帧")
                height, width = pixels.shape[:2]
                if width > 640:
                    pixels = cv2.resize(pixels, (640, round(height * 640 / width)), interpolation=cv2.INTER_AREA)
                ok, jpeg = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    raise ValueError("JPEG编码失败")
                fid = f"{camera}:{frame_number}"
                frames.append({"id": fid, "timestamp": frame_number / fps, "camera": camera})
                images[fid] = jpeg.tobytes()
                index += 1
        finally:
            capture.release()
    frames.sort(key=lambda f: (f["timestamp"], f["camera"]))
    return frames, images, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--episode", required=True, type=int)
    parser.add_argument("--start", required=True, type=float)
    parser.add_argument("--end", required=True, type=float)
    parser.add_argument("--quotas", nargs="+", type=int, default=[2, 2])
    parser.add_argument("--model", choices=MODELS, default=MODELS[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--cameras", nargs="+", choices=("cam0", "cam1", "cam2"), default=["cam0", "cam2"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = HarvestPlan(tuple(args.quotas))
    frames, images, sources = video_window(args.dataset, args.episode, args.start, args.end, cameras=args.cameras)
    source_id = hashlib.sha256(json.dumps([args.episode, sources], sort_keys=True).encode()).hexdigest()
    state_path = args.output / "memory.json"
    if state_path.exists():
        memory = TaskMemory.restore(json.loads(state_path.read_text()))
        if memory.run_id != source_id or memory.plan != plan:
            raise ValueError("已有记忆属于其他视频或计划,请使用独立输出目录")
    else:
        memory = TaskMemory(source_id, plan)
        memory.acknowledge("harvest_1", run_id=source_id, expected_revision=0)
    request = {
        "id": uuid.uuid4().hex,
        "run_id": memory.run_id,
        "revision": memory.revision,
        "active_stage": memory.stage,
        "start": frames[0]["timestamp"],
        "end": frames[-1]["timestamp"],
        "frames": frames,
    }
    payload = build_payload(memory.snapshot(), request, images, model=args.model)
    directory = args.output / request["id"]
    atomic_json(
        directory / "request.json",
        {
            "request": request,
            "memory": memory.snapshot(),
            "source_sha256": sources,
            "model": args.model,
            "enable_thinking": False,
            "replay": True,
        },
    )
    if args.dry_run:
        print(json.dumps({"dry_run": True, "frames": len(frames), "directory": str(directory)}))
        return
    start = time.monotonic()
    try:
        result = recognize(payload, timeout=args.timeout)
        atomic_json(directory / "model-result.json", result)
        evaluated = memory.evaluate(request, result["answer"], now=request["end"] + result["elapsed_s"])
        atomic_json(directory / "evaluation.json", evaluated)
        atomic_json(state_path, memory.snapshot())
        print(
            json.dumps(
                {
                    "model": args.model,
                    "elapsed_s": result["elapsed_s"],
                    "answer": result["answer"],
                    "evaluation": evaluated,
                    "directory": str(directory),
                },
                ensure_ascii=False,
            )
        )
    except Exception as error:
        atomic_json(directory / "failure.json", {"error": str(error), "elapsed_s": time.monotonic() - start})
        raise


if __name__ == "__main__":
    main()
