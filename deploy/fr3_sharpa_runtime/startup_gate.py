"""One-shot camera admission pipe from the supervising hardware session."""

# ruff: noqa: RUF001 -- Chinese operator diagnostics use Chinese punctuation.

import os
import select
import time

from .camera_clock import CAMERA_PROCESS_TIMEOUT_SEC

CAMERA_GATE_ENV = "SHARPA_CAMERA_GATE_FD"


def wait_for_cameras():
    """Wait after Home, before requesting any task observation or inference.

    EOF is failure, never permission. The parent keeps monitoring cameras after
    PASS and interrupts the runtime on loss, including during Home motion.
    """
    value = os.environ.pop(CAMERA_GATE_ENV, None)
    if value is None:
        return  # Legacy/direct callers retain their existing camera admission.
    fd = int(value)
    if fd < 3:
        raise ValueError("Invalid camera admission descriptor")
    try:
        deadline = time.monotonic() + CAMERA_PROCESS_TIMEOUT_SEC
        reply = b""
        while time.monotonic() < deadline:
            if not select.select([fd], [], [], min(0.1, max(0, deadline - time.monotonic())))[0]:
                continue
            data = os.read(fd, 64)
            if not data:
                raise RuntimeError("相机启动检查取消或失败，任务执行被阻止")
            reply += data
            if reply == b"PASS\n":
                return
            if len(reply) >= 5:
                raise RuntimeError("相机启动检查应答无效，任务执行被阻止")
        raise TimeoutError("等待相机启动检查超时，任务执行被阻止")
    finally:
        os.close(fd)
