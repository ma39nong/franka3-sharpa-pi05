"""Policy client and one-chunk manager. No ROS or SDK imports."""

from __future__ import annotations

import time
import uuid

import numpy as np

from . import config
from .action_adapter import SharpaActionAdapter
from .protocol import as_finite_action_chunk


def validate_observation(observation: dict) -> dict:
    required = (
        "observation/state",
        "observation/image",
        "observation/left_wrist_image",
        "observation/right_wrist_image",
        "prompt",
    )
    missing = [key for key in required if key not in observation]
    if missing:
        raise ValueError(f"Missing policy observations: {missing}")
    state = np.asarray(observation["observation/state"], dtype=np.float32)
    if state.shape != (config.STATE_DIM,) or not np.isfinite(state).all():
        raise ValueError("Expected finite 58D measured state")
    clean = {"observation/state": state}
    for key in required[1:4]:
        image = np.asarray(observation[key])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"{key} must be an HxWx3 uint8 RGB image")
        clean[key] = image
    prompt = observation["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Task prompt is required")
    clean["prompt"] = prompt
    return clean


class SharpaRuntime:
    """Request one 50-step prediction and prepare one independent session."""

    def __init__(self, policy_url: str = config.POLICY_URL):
        if not policy_url.startswith("ws://127.0.0.1:") and not policy_url.startswith("ws://localhost:"):
            raise ValueError("First runtime only accepts a loopback policy server")
        self.policy_url = policy_url
        self.adapter = SharpaActionAdapter()

    def inspect_metadata(self, *, require_rtc=False):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect
        with connect(self.policy_url, compression=None, max_size=None, open_timeout=10, close_timeout=1) as ws:
            metadata = msgpack_numpy.unpackb(ws.recv(timeout=10))
        config.validate_policy_metadata(metadata)
        if require_rtc:
            from .rtc_stream import require_capabilities
            require_capabilities(metadata)
        return metadata

    def infer_chunk(self, observation: dict, *, rtc_context=None, response_timeout=120):
        config.require_checkpoint()
        entered_ns = time.monotonic_ns()
        # Import here so the pure adapter remains importable in Pixi ROS2.
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect

        timing = {"infer_chunk_entered_monotonic_ns": entered_ns,
                  "client_imports_ready_monotonic_ns": time.monotonic_ns()}
        start = time.monotonic_ns()
        clean = validate_observation(observation)
        timing["client_preprocessing_ms"] = (time.monotonic_ns()-start)/1e6
        timing["websocket_connect_start_monotonic_ns"] = time.monotonic_ns()
        with connect(self.policy_url, compression=None, max_size=None, open_timeout=10, close_timeout=5) as ws:
            metadata = msgpack_numpy.unpackb(ws.recv(timeout=10))
            timing["websocket_metadata_ready_monotonic_ns"] = time.monotonic_ns()
            config.validate_policy_metadata(metadata)
            start = time.monotonic_ns()
            if rtc_context is not None:
                from openpi_client import rtc
                from .rtc_stream import require_capabilities
                require_capabilities(metadata)
                clean = rtc.make_request(clean, rtc_context)
            payload = msgpack_numpy.Packer().pack(clean)
            timing["observation_msgpack_ms"] = (time.monotonic_ns()-start)/1e6
            timing["policy_request_send_monotonic_ns"] = time.monotonic_ns()
            ws.send(payload)
            timing["policy_request_sent_monotonic_ns"] = time.monotonic_ns()
            response = ws.recv(timeout=response_timeout)
            timing["response_receive_monotonic_ns"] = time.monotonic_ns()
        if isinstance(response, str):
            raise RuntimeError(f"Policy server error: {response}")
        start = time.monotonic_ns()
        result = msgpack_numpy.unpackb(response)
        timing["response_decode_ms"] = (time.monotonic_ns()-start)/1e6
        timing["server"] = result.get("diagnostic_timing", {})
        timing["existing_server_timing"] = result.get("server_timing", {})
        timing["existing_policy_timing"] = result.get("policy_timing", {})
        if rtc_context is not None:
            from openpi_client import rtc
            rtc.validate_receipt(result.get("rtc"), rtc_context)
        elif "rtc" in result:
            raise ValueError("Unexpected RTC receipt for ordinary inference")
        self.last_receipt = result.get("rtc")
        actions = as_finite_action_chunk(result["actions"])
        # Fresh session for this single chunk; this version never splices chunks.
        session_id = f"openpi-{uuid.uuid4().hex}"
        start = time.monotonic_ns()
        chunk = self.adapter.adapt_chunk(
            actions, start_timestamp_ns=time.time_ns(), session_id=session_id, source=config.SOURCE
        )
        timing["action_adapter_ms"] = (time.monotonic_ns()-start)/1e6
        timing["action_adapter_end_monotonic_ns"] = time.monotonic_ns()
        self.last_timing = timing
        return metadata, actions, chunk
