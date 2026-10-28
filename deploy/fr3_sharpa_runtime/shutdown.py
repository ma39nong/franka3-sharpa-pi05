"""Bounded disable/readback on existing hand handles; never connect or enable."""

from contextlib import contextmanager
from contextlib import suppress
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time


@dataclass(frozen=True)
class DisableSettings:
    timeout_s: float = 120.0
    max_attempts: int = 120
    retry_interval_s: float = 1.0

    def __post_init__(self):
        if (
            not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0
            or not math.isfinite(self.retry_interval_s)
            or self.retry_interval_s <= 0
            or type(self.max_attempts) is not int
            or self.max_attempts < 1
        ):
            raise ValueError(
                "Disable timeout/interval must be positive and finite; attempts must be a positive integer"
            )

    @classmethod
    def from_env(cls):
        return cls(
            timeout_s=float(os.environ.get("SHARPA_DISABLE_TIMEOUT_S", "120")),
            max_attempts=int(os.environ.get("SHARPA_DISABLE_MAX_ATTEMPTS", "120")),
            retry_interval_s=float(os.environ.get("SHARPA_DISABLE_RETRY_INTERVAL_S", "1")),
        )

    @property
    def worker_grace_s(self):
        return self.timeout_s + 15.0

    @property
    def session_grace_s(self):
        return self.timeout_s + 45.0


def stderr(message):
    try:
        print(message, file=sys.stderr, flush=True)
    except (OSError, ValueError):
        # A Ctrl-C'd tee or closed terminal must never abort device cleanup.
        try:
            with open(os.devnull, "w") as sink:
                os.dup2(sink.fileno(), sys.stderr.fileno())
        except (OSError, ValueError):
            pass


def safe_emit(emit, message):
    # Diagnostics are deliberately outside the disable/readback contract.
    with suppress(Exception):
        emit(message)


def save_shutdown_report(report):
    """Persist independently of console/IPC; never interrupt hand cleanup."""
    directory = os.environ.get("SHARPA_SHUTDOWN_REPORT_DIR")
    if not directory:
        return None
    path = Path(directory) / f"shutdown-{os.getpid()}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w") as stream:
            json.dump({**report, "pid": os.getpid(), "recorded_unix_ns": time.time_ns()}, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        return path
    except (OSError, ValueError, TypeError) as error:
        stderr(f"[sharpa-shutdown] ERROR 无法保存独立失能报告: {error}")
        return None


@contextmanager
def protect_disable_confirmation(emit=stderr):
    """Repeated Ctrl-C must not interrupt the finite disable window."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def waiting(number, _frame):
        safe_emit(emit, f"[sharpa-shutdown] 收到信号 {number}; 正在确认失能, 请等待本次有时限的退出检查。")

    previous = {number: signal.signal(number, waiting) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def read_enable_state(hand):
    # The installed HAL has no public enable getter. Its existing read helper
    # unwraps (Error, bool) and checks the vendor error code without enabling.
    getter = getattr(hand, "read_enable_state", None)
    enabled = getter() if getter is not None else hand._read_setting("get_enable_state")  # noqa: SLF001
    if type(enabled) is not bool:
        raise TypeError(f"Enable readback must be bool, got {enabled!r}")
    return enabled


def disable_and_confirm(hands, settings=None, *, emit=stderr):
    output = emit

    def emit(message):
        safe_emit(output, message)

    settings = settings or DisableSettings.from_env()
    started = time.monotonic()
    deadline = started + settings.timeout_s
    cancel = threading.Event()
    lock = threading.Lock()
    states = {
        side: {
            "status": "PENDING" if hand is not None else "NOT_CONNECTED",
            "confirmed_disabled": False,
            "attempts": 0,
            "last_enabled": None,
            "last_error": None,
            "last_request_error": None,
            "disconnect_error": None,
            "disconnect_complete": False,
        }
        for side, hand in hands.items()
    }

    def update(side, **values):
        with lock:
            states[side].update(values)

    def disable(side, hand):
        try:
            for attempt in range(1, settings.max_attempts + 1):
                if cancel.is_set() or time.monotonic() >= deadline:
                    break
                update(side, attempts=attempt)
                try:
                    hand.stop()  # Existing HAL: set_enable_state(False), no motion target.
                    update(side, last_request_error=None)
                except Exception as error:
                    update(side, last_error=f"disable request: {error}", last_request_error=str(error))
                if cancel.is_set() or time.monotonic() >= deadline:
                    break
                try:
                    enabled = read_enable_state(hand)
                    update(side, last_enabled=enabled)
                    if not enabled:
                        update(side, status="CONFIRMED", confirmed_disabled=True, last_error=None)
                        emit(f"[sharpa-shutdown] {side}: 已读回使能=False, 确认失能(第 {attempt} 次)。")
                        return
                    update(side, last_error="readback still enabled=True")
                except Exception as error:
                    update(side, last_error=f"enable readback: {error}")
                if attempt < settings.max_attempts:
                    cancel.wait(max(0.0, min(settings.retry_interval_s, deadline - time.monotonic())))
            update(side, status="UNCONFIRMED")
        finally:
            try:
                hand.disconnect()
            except Exception as error:
                update(side, disconnect_error=str(error))
            finally:
                update(side, disconnect_complete=True)

    emit(
        f"[sharpa-shutdown] 开始双手失能并读回确认; 总时限 {settings.timeout_s:g}s, "
        f"每手最多 {settings.max_attempts} 次, 间隔 {settings.retry_interval_s:g}s。"
    )
    threads = [
        threading.Thread(target=disable, args=(side, hand), daemon=True)
        for side, hand in hands.items()
        if hand is not None
    ]
    for thread in threads:
        thread.start()
    next_progress = started + 5.0
    while any(thread.is_alive() for thread in threads):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            cancel.set()
            break
        for thread in threads:
            thread.join(timeout=min(0.05, max(0, deadline - time.monotonic())))
        if time.monotonic() >= next_progress:
            with lock:
                progress = {side: value["attempts"] for side, value in states.items()}
            emit(f"[sharpa-shutdown] 仍在等待失能确认, 尝试次数={progress}")
            next_progress = time.monotonic() + 5.0
    with lock:
        result = {side: dict(state) for side, state in states.items()}
    for state in result.values():
        if state["status"] == "PENDING":
            state["status"] = "UNCONFIRMED"
            state["last_error"] = state["last_error"] or "SDK request/readback did not finish before timeout"
        if state["status"] == "CONFIRMED" and not state["disconnect_complete"]:
            state["disconnect_error"] = "SDK disconnect did not finish before timeout"
    connected = [state for state in result.values() if state["status"] != "NOT_CONNECTED"]
    confirmed = bool(connected) and all(state["confirmed_disabled"] for state in result.values())
    errors = [
        f"{side}: {state}"
        for side, state in result.items()
        if state["status"] == "UNCONFIRMED" or state["disconnect_error"]
    ]
    status = "PASS" if confirmed and not errors else "FAIL" if connected else "NOT_CONNECTED"
    for side, state in result.items():
        if state["status"] == "UNCONFIRMED":
            emit(
                f"[sharpa-shutdown] ERROR {side}: 达到次数或时间上限, 未确认失能!"
                f"attempts={state['attempts']}, last_enabled={state['last_enabled']}, error={state['last_error']}"
            )
        if state["status"] == "NOT_CONNECTED":
            emit(f"[sharpa-shutdown] {side}: 没有持有 SDK 连接, 无法读回确认该手状态; 不会重新连接或使能。")
        if state["disconnect_error"]:
            emit(f"[sharpa-shutdown] ERROR {side}: SDK disconnect failed: {state['disconnect_error']}")
    emit(
        f"[sharpa-shutdown] 退出检查结束: {status}。"
        + (
            "双手均已读回确认失能。"
            if status == "PASS"
            else "双手失能已读回确认, 但 SDK 断连未完整完成。"
            if confirmed
            else "无法完整确认双手失能; 程序将继续退出, 请勿据此认定设备已释放。"
        )
    )
    return {
        "status": status,
        "hands": result,
        "elapsed_s": time.monotonic() - started,
        "timeout_s": settings.timeout_s,
        "max_attempts": settings.max_attempts,
    }
