"""Export the validated Sharpa intermediate into a LeRobot v2 dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from openpi.policies.fr3_sharpa_protocol import ACTION_DIM, ORDER


def main(source: Path, repo_id: str = "fr3_sharpa/episode0", output: Path | None = None,
         overwrite: bool = False, task: str = "perform the task with both Sharpa hands"):
    report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "converted_intermediate":
        raise ValueError("source report is not a validated Sharpa conversion")
    data = np.load(source / "episode.npz")
    state, action, times = data["state"], data["action"], data["time_ns"]
    if state.shape != action.shape or state.shape[1:] != (ACTION_DIM,) or len(times) != len(state):
        raise ValueError(f"invalid intermediate shapes: state={state.shape}, action={action.shape}, time={times.shape}")
    root = output or Path.home() / ".cache/huggingface/lerobot" / repo_id
    if root.exists() and not overwrite:
        raise FileExistsError(f"{root} exists; pass --overwrite")
    features = {
        "observation.state": {"dtype": "float32", "shape": (ACTION_DIM,), "names": list(ORDER)},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": list(ORDER)},
    }
    for camera, size in (("cam0", (400, 640, 3)), ("cam1", (480, 640, 3)), ("cam2", (480, 640, 3))):
        features[f"observation.images.{camera}"] = {"dtype": "video", "shape": size,
                                                     "names": ["height", "width", "channel"]}
    dataset = LeRobotDataset.create(repo_id=repo_id, root=root, fps=30, robot_type="dual_fr3_sharpa",
                                    features=features, use_videos=True, image_writer_processes=4,
                                    image_writer_threads=4)
    for i in range(len(state)):
        frame = {"observation.state": state[i], "action": action[i], "task": task}
        for camera in ("cam0", "cam1", "cam2"):
            from PIL import Image
            frame[f"observation.images.{camera}"] = np.asarray(Image.open(source / "images" / camera / f"{i:06d}.jpg").convert("RGB"))
        dataset.add_frame(frame)
    dataset.save_episode()
    print(json.dumps({"repo_id": repo_id, "root": str(root), "episodes": dataset.meta.total_episodes,
                      "frames": dataset.meta.total_frames, "state_shape": list(state.shape),
                      "action_shape": list(action.shape)}, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
