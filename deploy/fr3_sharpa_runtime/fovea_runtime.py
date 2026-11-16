"""Serial fifteen-step client: commit a future box only after successful execution."""

import time
import uuid

import numpy as np

from openpi.subtasks import fovea

from .action_adapter import SharpaActionAdapter
from .protocol import as_finite_action_chunk
from .runtime import validate_observation


class FoveaRuntime:
    def __init__(self, policy_url, *, execute_steps=fovea.EXECUTE_STEPS):
        if execute_steps != fovea.EXECUTE_STEPS:
            raise ValueError("Fovea requires exactly15 executed control steps")
        self.policy_url = policy_url
        self.session = uuid.uuid4().hex
        self.query_index = 0
        self.box = np.array(fovea.INITIAL_BOX)
        self.pending = None
        self.adapter = SharpaActionAdapter()
        self.last_timing = {}

    def infer_chunk(self, observation, *, response_timeout=120):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect

        clean = validate_observation(observation)
        context = {
            "session_id": self.session,
            "query_index": self.query_index,
            "executed_steps": 0 if self.query_index == 0 else fovea.EXECUTE_STEPS,
            "current_box": self.box.tolist(),
        }
        started = time.monotonic()
        with connect(self.policy_url, compression=None, max_size=None, open_timeout=10, close_timeout=1) as ws:
            metadata = msgpack_numpy.unpackb(ws.recv(timeout=10))
            contract = metadata.get("foveation", {})
            if (
                contract.get("version") != fovea.VERSION
                or contract.get("execute_steps") != fovea.EXECUTE_STEPS
                or contract.get("initial_box") != list(fovea.INITIAL_BOX)
                or not metadata.get("warmed_up")
            ):
                raise ValueError("Warmed matching Fovea service required")
            ws.send(msgpack_numpy.packb({**clean, "_fovea": context}))
            response = ws.recv(timeout=response_timeout)
        if isinstance(response, str):
            raise RuntimeError(response)
        result = msgpack_numpy.unpackb(response)
        return self.accept_result(metadata, result, started=started)

    def accept_result(self, metadata, result, *, started=None):
        receipt = result["foveation"]
        if (
            receipt["session_id"] != self.session
            or receipt["query_index"] != self.query_index
            or receipt["execute_steps"] != fovea.EXECUTE_STEPS
            or receipt["source_size"] != [640, 400]
            or not np.array_equal(receipt["used_box"], self.box)
        ):
            raise ValueError("Mismatched Fovea inference receipt")
        next_box = fovea.validate_box(receipt["next_box"])
        actions = as_finite_action_chunk(result["actions"])
        self.pending = next_box.copy()
        self.last_timing = {"foveation": receipt, "infer_ms": (time.monotonic() - started) * 1000 if started else None}
        chunk = self.adapter.adapt_chunk(
            actions,
            start_timestamp_ns=time.time_ns(),
            session_id=f"fovea-{self.session}-{self.query_index}",
            source="openpi",
        )
        return {**metadata, "foveation_query": receipt}, actions, chunk

    def acknowledge_execution(self, completed_steps):
        if type(completed_steps) is not int or completed_steps != fovea.EXECUTE_STEPS or self.pending is None:
            raise ValueError("Commit requires a successful fifteen-step execution and pending prediction")
        self.box = self.pending
        self.pending = None
        self.query_index += 1
