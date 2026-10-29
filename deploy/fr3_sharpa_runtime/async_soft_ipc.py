"""Bounded pipe reads: partial JSON messages must not stall the control clock."""

import json
import os
import select


class RequestInbox:
    def __init__(self, stream):
        self.stream = stream
        self.buffer = b""
        try:
            self.fd = stream.fileno()
        except (AttributeError, OSError):
            self.fd = None  # In-memory test transports have no OS descriptor.

    def poll(self, timeout):
        if b"\n" not in self.buffer:
            ready, _, _ = select.select([self.stream], [], [], timeout)
            if not ready:
                return False, None
            if self.fd is None:
                line = self.stream.readline()
                return True, json.loads(line) if line else None
            data = os.read(self.fd, 65536)
            if not data:
                if self.buffer:
                    raise ValueError("Async Soft IPC ended with an incomplete message")
                return True, None
            self.buffer += data
            if len(self.buffer) > 2**20:
                raise ValueError("Async Soft IPC message exceeds 1 MiB")
        if b"\n" not in self.buffer:
            return False, None
        line, self.buffer = self.buffer.split(b"\n", 1)
        return True, json.loads(line)
