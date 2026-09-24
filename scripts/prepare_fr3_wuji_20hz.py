"""Prepare long prompts and 20 Hz Pi0.5 normalization for the 0918 datasets."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openpi import transforms
from openpi.policies.fr3_wuji_policy import _reorder_actions
from openpi.policies.fr3_wuji_policy import _reorder_state
from openpi.shared import normalize

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/home/descfly/lpy/Convert_data/outputs/fr3_wuji/0918")
DATASETS = (DATA_ROOT / "tomato_A_cleaned", DATA_ROOT / "tomato_B_cleaned")
OUTPUT = ROOT / "assets/pi05_fr3_wuji_20hz/fr3_wuji/0918_20hz"
PROMPT = (
    "Pick up a tomato truss with the right hand, then pick a cherry tomato with the left hand "
    "and place it in the left basket."
)
ACTION_HZ = 20
ACTION_HORIZON = 50


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(path)


def rewrite_prompts(dataset: Path) -> None:
    tasks_path = dataset / "meta/tasks.jsonl"
    tasks_backup = tasks_path.with_suffix(tasks_path.suffix + ".before_20hz_prompt")
    if not tasks_backup.exists():
        tasks_backup.write_bytes(tasks_path.read_bytes())
    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines()]
    for task in tasks:
        task["task"] = PROMPT
    _atomic_jsonl(tasks_path, tasks)

    episodes_path = dataset / "meta/episodes.jsonl"
    episodes_backup = episodes_path.with_suffix(episodes_path.suffix + ".before_20hz_prompt")
    if not episodes_backup.exists():
        episodes_backup.write_bytes(episodes_path.read_bytes())
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    for episode in episodes:
        episode["tasks"] = [PROMPT]
    _atomic_jsonl(episodes_path, episodes)


def input_fingerprint(datasets: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for dataset in datasets:
        for relative in ("meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl"):
            digest.update((dataset / relative).read_bytes())
    digest.update(f"{ACTION_HZ}:{ACTION_HORIZON}".encode())
    return digest.hexdigest()


def compute_stats(datasets: tuple[Path, ...], output: Path, batch_size: int) -> None:
    infos = [json.loads((dataset / "meta/info.json").read_text()) for dataset in datasets]
    source_fps = infos[0]["fps"]
    if source_fps < ACTION_HZ or any(info["fps"] != source_fps for info in infos):
        raise ValueError("The two datasets must share a source FPS of at least 20 Hz")
    offsets = np.rint(np.arange(ACTION_HORIZON) * source_fps / ACTION_HZ).astype(np.int64)
    delta = transforms.DeltaActions(transforms.make_bool_mask(7, -20, 7, -20))
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    total_episodes = sum(info["total_episodes"] for info in infos)
    completed = 0

    for dataset, info in zip(datasets, infos, strict=True):
        episodes = [json.loads(line) for line in (dataset / "meta/episodes.jsonl").read_text().splitlines()]
        if len(episodes) != info["total_episodes"]:
            raise ValueError(f"Episode metadata mismatch: {dataset}")
        for episode in episodes:
            path = dataset / info["data_path"].format(
                episode_chunk=episode["episode_index"] // info["chunks_size"],
                episode_index=episode["episode_index"],
            )
            table = pq.read_table(path, columns=["observation.state", "action"])
            state = _reorder_state(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32))
            action = _reorder_actions(np.asarray(table["action"].to_pylist(), dtype=np.float32))
            for start in range(0, len(table), batch_size):
                stop = min(len(table), start + batch_size)
                indices = np.minimum(np.arange(start, stop)[:, None] + offsets, len(table) - 1)
                batch = delta({"state": state[start:stop], "actions": action[indices]})
                for key, accumulator in stats.items():
                    accumulator.update(batch[key].astype(np.float64))
            completed += 1
            print(f"20 Hz stats: episode {completed}/{total_episodes}", flush=True)

    result = {key: accumulator.get_statistics() for key, accumulator in stats.items()}
    output.mkdir(parents=True, exist_ok=True)
    normalize.save(output, result)
    provenance = {
        "datasets": [str(path) for path in datasets],
        "source_hz": source_fps,
        "action_hz": ACTION_HZ,
        "action_horizon": ACTION_HORIZON,
        "trajectory_seconds": ACTION_HORIZON / ACTION_HZ,
        "nearest_source_frame_offsets": offsets.tolist(),
        "prompt": PROMPT,
        "input_fingerprint": input_fingerprint(datasets),
        "method": "all rows; nearest 30 Hz source frames on a 20 Hz action grid; episode-clipped; arm delta",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    datasets = tuple(path.resolve() for path in DATASETS)
    with ExitStack() as locks:
        for dataset in sorted(datasets):
            lock = locks.enter_context((dataset / "conversion/lock").open("r"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for dataset in datasets:
            rewrite_prompts(dataset)

        provenance_path = OUTPUT / "provenance.json"
        fingerprint = input_fingerprint(datasets)
        if not args.force and provenance_path.is_file() and (OUTPUT / "norm_stats.json").is_file():
            provenance = json.loads(provenance_path.read_text())
            if provenance.get("input_fingerprint") == fingerprint:
                print(f"20 Hz assets already current: {OUTPUT}")
                return
        compute_stats(datasets, OUTPUT, args.batch_size)
        print(f"Prepared 20 Hz assets: {OUTPUT}")


if __name__ == "__main__":
    main()
