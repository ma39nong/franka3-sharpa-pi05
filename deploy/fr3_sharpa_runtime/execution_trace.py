"""Persistent, read-only execution traces for sdk_direct chunks."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import numpy as np

from .protocol import (
    LEFT_FR3_SLICE,
    LEFT_SHARPA_SLICE,
    RIGHT_FR3_SLICE,
    RIGHT_SHARPA_SLICE,
)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def observation_metadata(observation: dict) -> dict:
    """Keep metadata and timing, without duplicating camera pixel payloads."""
    result = {}
    for key, value in observation.items():
        if key.startswith("observation/") and isinstance(value, np.ndarray):
            result[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        elif key in {"prompt", "_captured_monotonic_ns", "_topic_stamp_ns", "_timing", "_alignment"}:
            result[key] = _json_value(value)
    return result


def adapted_actions(chunk) -> dict:
    rows = {
        "left_fr3": np.stack([frame.fr3.positions[:7] for frame in chunk.frames]),
        "left_sharpa": np.stack([frame.left_sharpa.position for frame in chunk.frames]),
        "right_fr3": np.stack([frame.fr3.positions[7:] for frame in chunk.frames]),
        "right_sharpa": np.stack([frame.right_sharpa.position for frame in chunk.frames]),
    }
    return {key: value.astype(np.float32).tolist() for key, value in rows.items()}


class ExecutionTraceWriter:
    """Write one self-contained JSON record per completed action chunk."""

    def __init__(self, directory: Path | None = None):
        configured = directory or Path(os.environ.get(
            "SHARPA_EXECUTION_TRACE_DIR",
            "/home/user/.local/state/fr3_sharpa_runtime/execution_traces",
        ))
        self.directory = Path(configured)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.trace_id = f"{time.time_ns()}-{uuid.uuid4().hex[:8]}"
        self.paths = []
        self.run_path = self.directory / f"{self.trace_id}-run.json"
        self.events_path = self.directory / f"{self.trace_id}-events.jsonl"

    def event(self, stage, **fields):
        payload = {"stage": stage, "monotonic_ns": time.monotonic_ns(), **fields}
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_json_value(payload), allow_nan=False) + "\n")

    def finish_run(self, report):
        self._write(self.run_path, {"trace_id": self.trace_id, **report})

    @staticmethod
    def _write(destination, payload):
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(destination)

    def fail_chunk(self, path, *, error, stage):
        destination = Path(path)
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if payload["status"] == "IN_FLIGHT":
            payload["status"] = "FAILED"
        payload.update({"error": error, "failure_stage": stage})
        self._write(destination, payload)

    def _base_payload(self, *, chunk_index: int, metadata: dict, observation_before: dict,
                      actions: np.ndarray, adapted_chunk) -> dict:
        action_array = np.asarray(actions, dtype=np.float32)
        if action_array.shape != (50, 58):
            raise ValueError(f"trace action shape must be (50, 58), got {action_array.shape}")
        return {
            "trace_id": self.trace_id,
            "chunk_index": int(chunk_index),
            "created_unix_ns": time.time_ns(),
            "status": "IN_FLIGHT",
            "policy": _json_value(metadata),
            "observation_before": observation_metadata(observation_before),
            "feedback_before": np.asarray(observation_before["observation/state"], dtype=np.float32).tolist(),
            "policy_output_unnormalized": action_array.tolist(),
            # The policy websocket returns post-output-transform values.  At
            # this boundary the output is already unnormalized; retain an
            # explicit second field so replay tools can distinguish the stage.
            "unnormalized_action": action_array.tolist(),
            "adapter_after": adapted_actions(adapted_chunk),
            "observation_after": None,
            "feedback_after": None,
            "chunk_duration_s": None,
            "end_to_end_duration_s": None,
            "action_shape": [50, 58],
            "action_unit": "rad",
            "action_representation": "absolute_joint_position",
            "normalization_stage": "policy websocket output is post-Unnormalize",
        }

    def begin_chunk(self, *, chunk_index: int, metadata: dict, observation_before: dict,
                    actions: np.ndarray, adapted_chunk, execute_steps=50) -> str:
        """Persist inputs before hardware output starts."""
        payload = self._base_payload(
            chunk_index=chunk_index, metadata=metadata,
            observation_before=observation_before, actions=actions,
            adapted_chunk=adapted_chunk)
        payload["planned_execute_steps"] = execute_steps
        payload["planned_action_indices"] = [0, execute_steps - 1]
        destination = self.directory / f"{self.trace_id}-chunk{chunk_index:03d}.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(destination)
        if str(destination) not in self.paths:
            self.paths.append(str(destination))
        return str(destination)

    def record_experiment(self, path, *, context, diagnostics, execution_targets, compression):
        destination = Path(path)
        payload = json.loads(destination.read_text(encoding="utf-8"))
        payload["inference_experiment"] = {"context": context, "diagnostics": diagnostics,
            "compression": compression, "execution_targets": np.asarray(execution_targets).tolist(),
            "model_horizon": 50, "effective_horizon": 30, "consumed_model_steps": 26,
            "overlap_indices": [26, 30], "alignment": "trajectory_phase_not_wall_clock"}
        self._write(destination, payload)

    def finish_chunk(self, path: str, *, observation_after: dict | None,
                     command_duration_s: float, end_to_end_duration_s: float, execution_report=None) -> None:
        destination = Path(path)
        payload = json.loads(destination.read_text(encoding="utf-8"))
        payload.update({
            "status": "COMPLETE",
            "observation_after": observation_metadata(observation_after) if observation_after is not None else None,
            "feedback_after": (np.asarray(observation_after["observation/state"], dtype=np.float32).tolist()
                               if observation_after is not None else None),
            "execution_report": execution_report,
            "chunk_duration_s": float(command_duration_s),
            "end_to_end_duration_s": float(end_to_end_duration_s),
        })
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(destination)

    def write_chunk(self, *, chunk_index: int, metadata: dict, observation_before: dict,
                    actions: np.ndarray, adapted_chunk, observation_after: dict,
                    command_duration_s: float, end_to_end_duration_s: float) -> str:
        path = self.begin_chunk(chunk_index=chunk_index, metadata=metadata,
                                observation_before=observation_before,
                                actions=actions, adapted_chunk=adapted_chunk)
        self.finish_chunk(path, observation_after=observation_after,
                          command_duration_s=command_duration_s,
                          end_to_end_duration_s=end_to_end_duration_s)
        return path
