"""Opt-in FLASH offline samples; never synthesize unavailable past frames."""

import time
import uuid

import numpy as np

from .dataset_sample import load_sample
from .flash_contract import RESAMPLER


def load_flash_sample(path, prompt, frame):
    import pyarrow.parquet as pq

    if frame < 7:
        raise ValueError("FLASH offline sample needs seven real preceding frames in the same episode")
    rows = pq.read_table(path).slice(frame - 7, 8).to_pylist()
    if len(rows) != 8 or len({row["episode_index"] for row in rows}) != 1:
        raise ValueError("FLASH history cannot cross episode boundaries")
    indices = [row["frame_index"] for row in rows]
    if indices != list(range(indices[0], indices[0] + 8)):
        raise ValueError("FLASH history has missing/out-of-order frames")
    relative = np.asarray([row["timestamp"] for row in rows])
    if np.max(np.abs(relative - relative[-1] + np.arange(7, -1, -1) / 30)) > 0.001:
        raise ValueError("FLASH history requires real contiguous 30Hz source frames")
    # load_sample retains all existing named-state/116D conversion rules. This
    # offline diagnostic favors correctness; it does not affect live latency.
    observations = [load_sample(path, prompt, i) for i in range(frame - 7, frame + 1)]
    current = observations[-1]
    history = np.stack([item["observation/state"] for item in observations])
    anchor = round((float(relative[-1]) + 1) * 1e9)
    grid = [anchor - round((7 - i) * 1e9 / 30) for i in range(8)]
    current.update(
        {
            "observation/history": history,
            "flash_context": {
                "protocol_version": 1,
                "request_id": uuid.uuid4().hex,
                "session_id": "offline-episode-" + str(rows[0]["episode_index"]),
                "observation_anchor_source_ns": anchor,
                "history_grid_source_ns": grid,
                "history_source_ns": [[stamp] * 4 for stamp in grid],
                "captured_monotonic_ns": time.monotonic_ns(),
                "history_resampler_version": RESAMPLER,
                "source": "offline_lerobot_30hz_not_live_execution",
            },
        }
    )
    return current
