"""Dedicated async-Soft websocket client; never sends an RTC guidance envelope."""

import time

from . import config
from .async_soft_stream import CONTEXT_KEY
from .async_soft_stream import RECEIPT_KEY
from .async_soft_stream import handover_window
from .async_soft_stream import receipt
from .async_soft_stream import require_capabilities
from .async_soft_stream import validate_context
from .protocol import as_finite_action_chunk
from .rolling import model_identity
from .runtime import validate_observation


def default_port():
    port = config.POLICY_PORT + 100
    if port > 65535:
        raise ValueError("Set an explicit valid async Soft port")
    return port


class AsyncSoftClient:
    def __init__(self, url=None):
        self.url = url or f"ws://127.0.0.1:{default_port()}"
        if not self.url.startswith(("ws://127.0.0.1:", "ws://localhost:")):
            raise ValueError("Async Soft only supports local monotonic clocks/loopback")
        self.identity = None
        self.last_receipt = None

    def _metadata(self, metadata):
        config.validate_policy_metadata(metadata)
        require_capabilities(metadata)
        identity = (model_identity(metadata), metadata["server_instance_id"])
        if self.identity is not None and self.identity != identity:
            raise ValueError("Async Soft server/checkpoint/stats changed; restart execution")
        self.identity = identity

    def inspect_metadata(self):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect

        with connect(self.url, compression=None, max_size=None, open_timeout=10, close_timeout=1) as ws:
            metadata = msgpack_numpy.unpackb(ws.recv(timeout=10))
        self._metadata(metadata)
        return metadata

    def infer_chunk(self, observation, *, context=None, response_timeout=120):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect

        clean = validate_observation(observation)
        deadline = time.monotonic() + response_timeout
        if context is not None:
            context = validate_context(context)
            captured = observation.get("_captured_monotonic_ns")
            if type(captured) is not int or abs(captured / 1e9 - context["observation_monotonic_s"]) > 1e-6:
                raise ValueError("Async Soft context does not match the observation timestamp")
            deadline = min(deadline, handover_window(context)[1])
            clean[CONTEXT_KEY] = context
        if "_inference_context" in observation or "rtc" in observation:
            raise ValueError("Async Soft cannot mix inference protocols")

        def remaining():
            value = deadline - time.monotonic()
            if value <= 0:
                raise TimeoutError("Async Soft response window expired")
            return value

        self.last_receipt = None
        with connect(
            self.url, compression=None, max_size=None, open_timeout=min(10, remaining()), close_timeout=0.1
        ) as ws:
            metadata = msgpack_numpy.unpackb(ws.recv(timeout=min(10, remaining())))
            self._metadata(metadata)
            ws.send(msgpack_numpy.Packer().pack(clean))
            response = ws.recv(timeout=remaining())
        remaining()
        if isinstance(response, str):
            raise RuntimeError(f"Async Soft server rejected request: {response}")
        output = msgpack_numpy.unpackb(response)
        acknowledgement = output.get(RECEIPT_KEY)
        if context is not None:
            expected = {
                "context": receipt(context),
                "server_instance_id": metadata["server_instance_id"],
                "model_identity": model_identity(metadata),
            }
            if acknowledgement != expected:
                raise ValueError("Missing or mismatched async Soft receipt")
            self.last_receipt = acknowledgement["context"]
        elif acknowledgement is not None:
            raise ValueError("Unexpected async Soft receipt")
        return metadata, as_finite_action_chunk(output["actions"]), output.get("async_soft_diagnostics", {})
