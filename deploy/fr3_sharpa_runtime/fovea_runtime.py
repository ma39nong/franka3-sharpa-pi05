"""Serial fifteen-step client: commit a future box only after successful execution."""

import os
import time
import uuid

import numpy as np

from openpi.subtasks import fovea

from .action_adapter import SharpaActionAdapter
from .protocol import as_finite_action_chunk
from .runtime import validate_observation


class FoveaRuntime:
    def __init__(self, policy_url, *, execute_steps=fovea.EXECUTE_STEPS, expected_checkpoint=None):
        if execute_steps != fovea.EXECUTE_STEPS:
            raise ValueError("Fovea requires exactly15 executed control steps")
        self.policy_url = policy_url
        self.session = uuid.uuid4().hex
        self.query_index = 0
        self.box = np.array(fovea.INITIAL_BOX)
        self.pending = None
        self.last_receipt = None
        self.adapter = SharpaActionAdapter()
        self.last_timing = {}
        self.four_view_recording = None
        if os.environ.get("SHARPA_FOUR_VIEW_RECORD_DIR"):
            from .four_view_recording import recording_for
            self.four_view_recording = recording_for(os.environ["SHARPA_FOUR_VIEW_RECORD_DIR"])
        self.expected_identity = None
        if expected_checkpoint is not None:
            from .fovea_contract import checkpoint_contract
            self.expected_identity = checkpoint_contract(expected_checkpoint)["fovea_identity"]

    def validate_metadata(self, metadata):
        contract = metadata.get("foveation", {})
        if (contract.get("version") != fovea.VERSION
                or contract.get("execute_steps") != fovea.EXECUTE_STEPS
                or contract.get("initial_box") != list(fovea.INITIAL_BOX)
                or not metadata.get("warmed_up")):
            raise ValueError("Warmed matching Fovea service required")
        if self.expected_identity is not None and metadata.get("fovea_identity") != self.expected_identity:
            raise ValueError("Fovea服务权重身份与当前选择不一致")
        return metadata

    def inspect_metadata(self):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect
        with connect(self.policy_url, compression=None, max_size=None, open_timeout=3, close_timeout=0.1) as ws:
            return self.validate_metadata(msgpack_numpy.unpackb(ws.recv(timeout=3)))

    def fovea_context(self):
        return {
            "session_id": self.session,
            "query_index": self.query_index,
            "executed_steps": 0 if self.query_index == 0 else fovea.EXECUTE_STEPS,
            "current_box": self.box.tolist(),
        }

    def infer_chunk(self, observation, *, response_timeout=120, rtc_context=None):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect

        clean = validate_observation(observation)
        context = self.fovea_context()
        if self.four_view_recording is not None:
            context["record_roi"] = True
            context.setdefault("source_monotonic_ns", observation.get("_captured_monotonic_ns"))
        started = time.monotonic()
        with connect(self.policy_url, compression=None, max_size=None, open_timeout=10, close_timeout=1) as ws:
            metadata = msgpack_numpy.unpackb(ws.recv(timeout=10))
            self.validate_metadata(metadata)
            payload = {**clean, "_fovea": context}
            if rtc_context is not None:
                from openpi_client import rtc
                payload = rtc.make_request(payload, rtc_context)
            ws.send(msgpack_numpy.packb(payload))
            response = ws.recv(timeout=response_timeout)
        if isinstance(response, str):
            raise RuntimeError(response)
        result = msgpack_numpy.unpackb(response)
        if rtc_context is not None:
            from openpi_client import rtc
            rtc.validate_receipt(result.get("rtc"), rtc_context)
        elif "rtc" in result:
            raise ValueError("Unexpected Fovea RTC receipt")
        self.last_receipt = result.get("rtc")
        accepted = self.accept_result(metadata, result, started=started)
        if self.four_view_recording is not None:
            if "recording_roi" not in result:
                self.four_view_recording.error = "Fovea service returned no actual crop pixels"
            else:
                self.four_view_recording.submit(observation, result["recording_roi"], query_id=uuid.uuid4().hex,
                                                task=observation.get("_task_record"))
        return accepted

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

    def reset_stage(self):
        """New prompt starts a new crop session, only at the owner motion boundary."""
        self.session = uuid.uuid4().hex
        self.query_index = 0
        self.box = np.array(fovea.INITIAL_BOX)
        self.pending = None
