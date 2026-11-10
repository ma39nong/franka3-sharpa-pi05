"""Execution-owned state machine and private, versioned Unix socket control.

IPC threads register intent only. The execution owner serializes all hardware
operations and installation under the same lock as intent registration.
"""

from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import socketserver
import struct
import threading
import time
import uuid

from .task_stages import PHASES, catalog, stage

TRANSITIONS = {
    "IDLE": {"READY", "STOPPING", "FAULT"},
    "READY": {"RESUMING", "SWITCH_REQUESTED", "STOPPING", "FAULT"},
    "RUNNING": {"SWITCH_REQUESTED", "PAUSING", "STOPPING", "FAULT"},
    "SWITCH_REQUESTED": {"SWITCHING", "PAUSING", "STOPPING", "FAULT"},
    "SWITCHING": {"RUNNING", "PAUSING", "STOPPING", "FAULT"},
    "PAUSING": {"PAUSED", "STOPPING", "FAULT"},
    "PAUSED": {"RESUMING", "SWITCH_REQUESTED", "STOPPING", "FAULT"},
    "RESUMING": {"RUNNING", "SWITCH_REQUESTED", "PAUSING", "STOPPING", "FAULT"},
    "STOPPING": {"STOPPED", "COMPLETED", "FAULT"},
    "STOPPED": set(),
    "COMPLETED": set(),
    "FAULT": {"STOPPING"},
}


class TaskControl:
    def __init__(self, run_id=None, initial_stage="pickup_1", *, version=0, emit=None):
        stage(initial_stage)
        self.lock = threading.RLock()
        self.run_id = run_id or uuid.uuid4().hex
        self.emit = emit or (lambda **event: None)
        self.last_contact = time.monotonic()
        self.seen = OrderedDict()
        self.data = dict(
            run_id=self.run_id,
            requested_stage=initial_stage,
            active_stage=None,
            stage_version=version,
            execution_state="IDLE",
            active_prompt=None,
            pending_request_id=None,
            active_chunk_id=None,
            last_transition_reason="created",
            last_transition_time=time.time(),
            last_error=None,
            safe_to_resume=False,
            inference_status="idle",
            chunk_progress=None,
            rtc_mode="soft_prefix_guidance",
            inference_request_id=None,
            request_status="none",
            completed_stages=[],
            cleanup_confirmed=False,
        )

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps({**self.data, "stages": catalog()}))

    def event(self, event, **fields):
        try:
            self.emit(
                event=event,
                unix_ns=time.time_ns(),
                monotonic_ns=time.monotonic_ns(),
                run_id=self.run_id,
                stage_version=self.data["stage_version"],
                **fields,
            )
        except Exception as error:
            # A partially recorded intent must never leave RUNNING with a new
            # prompt but an old timeline. Fail closed without recursive logging.
            with self.lock:
                self.data.update(
                    execution_state="FAULT",
                    safe_to_resume=False,
                    stage_version=self.data["stage_version"] + 1,
                    last_error=f"任务日志写入失败: {error}",
                    request_status="failed",
                )
            raise RuntimeError(f"Task audit unavailable; execution blocked: {error}") from error

    def transition(self, target, reason):
        with self.lock:
            previous = self.data["execution_state"]
            if target not in TRANSITIONS[previous]:
                raise ValueError(f"非法执行状态转换: {previous} → {target}")
            self.event(
                "execution_transition",
                previous_state=previous,
                execution_state=target,
                reason=reason,
                request_id=self.data["pending_request_id"],
                active_stage=self.data["active_stage"],
                requested_stage=self.data["requested_stage"],
            )
            self.data.update(execution_state=target, last_transition_reason=reason, last_transition_time=time.time())

    def ready(self):
        self.transition("READY", "startup checks passed; awaiting explicit start")

    def command(self, command, request_id, *, expected_version, target=None, confirmed=False):
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
            raise ValueError("有效 request_id 必填")
        fingerprint = hashlib.sha256(json.dumps([command, expected_version, target, confirmed]).encode()).hexdigest()
        with self.lock:
            self.last_contact = time.monotonic()
            if request_id in self.seen:
                if self.seen[request_id] != fingerprint:
                    raise ValueError("request_id 被不同命令重复使用")
                return self.snapshot()
            d = self.data
            state = d["execution_state"]
            if command != "stop" and (type(expected_version) is not int or expected_version != d["stage_version"]):
                raise ValueError("阶段版本已改变，请刷新状态后操作")
            if state in ("STOPPING", "STOPPED", "COMPLETED"):
                raise ValueError("任务正在停止或已经结束")
            if command in ("next", "stage", "recover"):
                if state not in ("READY", "RUNNING", "SWITCH_REQUESTED", "SWITCHING", "PAUSED", "RESUMING"):
                    raise ValueError("当前状态不允许切换阶段")
                index = PHASES.index(d["requested_stage"])
                if command == "next":
                    if index == len(PHASES) - 1:
                        raise ValueError("已经是最后阶段，需人工完成任务")
                    target = PHASES[index + 1]
                stage(target)
                if command == "recover":
                    if confirmed is not True:
                        raise ValueError("人工恢复需要明确确认停止和重新观测")
                    if state not in ("READY", "PAUSED"):
                        raise ValueError("跳转或回退必须先独立暂停；在线切换只允许向前推进")
                elif index == len(PHASES) - 1 or target != PHASES[index + 1]:
                    raise ValueError("普通切换仅允许下一阶段；跳转或回退请使用人工恢复")
                previous, old_version = d["requested_stage"], d["stage_version"]
                self.event("request_superseded", request_id=d["pending_request_id"], reason="newer operator intent")
                d.update(
                    requested_stage=target,
                    stage_version=old_version + 1,
                    pending_request_id=request_id,
                    request_status="received",
                    safe_to_resume=d["safe_to_resume"] if state == "PAUSED" else False,
                )
                self.event(
                    "stage_requested",
                    request_id=request_id,
                    operator_action=command,
                    previous_stage=previous,
                    target_stage=target,
                    old_stage_version=old_version,
                )
                if state in ("READY", "PAUSED"):
                    # No motion exists to hand over; changing the selected stage
                    # must not open execution authority without start/resume.
                    d["request_status"] = "selected"
                elif state != "SWITCHING" and state != "SWITCH_REQUESTED":
                    self.transition("SWITCH_REQUESTED", "operator stage request")
            elif command == "start":
                if state != "READY":
                    raise ValueError("任务未就绪，不能开始")
                d.update(pending_request_id=request_id, request_status="received")
                self.transition("RESUMING", "operator starts current stage")
            elif command == "pause":
                if state not in ("RUNNING", "RESUMING", "SWITCHING", "SWITCH_REQUESTED"):
                    raise ValueError("当前状态不能暂停")
                d.update(stage_version=d["stage_version"] + 1, pending_request_id=request_id, safe_to_resume=False)
                self.transition("PAUSING", "operator pause; invalidate outstanding predictions")
            elif command == "resume":
                if state != "PAUSED" or not d["safe_to_resume"] or confirmed is not True:
                    raise ValueError("恢复被阻止：需要安全停止确认、重新授权及硬件检查")
                d.update(stage_version=d["stage_version"] + 1, pending_request_id=request_id, safe_to_resume=False)
                self.transition("RESUMING", "explicit operator reauthorization; fresh observation required")
            elif command in ("stop", "complete"):
                if command == "complete" and (d["active_stage"] != "return_stem" or state not in ("RUNNING", "PAUSED")):
                    raise ValueError("仅最后阶段实际生效后可人工确认完成")
                d.update(
                    stage_version=d["stage_version"] + 1,
                    pending_request_id=request_id,
                    finish_requested=command == "complete",
                    safe_to_resume=False,
                )
                self.transition("STOPPING", "operator " + command)
            else:
                raise ValueError("未知任务命令")
            self.seen[request_id] = fingerprint
            if len(self.seen) > 4096:
                self.seen.popitem(last=False)
            self.event(
                "operator_command",
                operator_action=command,
                request_id=request_id,
                target_stage=target,
                execution_state=d["execution_state"],
            )
            return self.snapshot()

    def ticket(self):
        with self.lock:
            d = self.data
            request_id = uuid.uuid4().hex
            d["inference_request_id"] = request_id
            return dict(
                run_id=self.run_id,
                request_id=request_id,
                stage_id=d["requested_stage"],
                stage_version=d["stage_version"],
                prompt=stage(d["requested_stage"]).prompt,
            )

    def valid(self, ticket):
        with self.lock:
            d = self.data
            return (
                ticket["run_id"] == self.run_id
                and ticket["stage_version"] == d["stage_version"]
                and ticket["request_id"] == d["inference_request_id"]
                and ticket["stage_id"] == d["requested_stage"]
                and d["execution_state"] in ("RUNNING", "RESUMING", "SWITCHING")
            )

    @contextmanager
    def installation(self, ticket):
        # Linearizes acceptance with IPC intent registration, not with GPU work.
        with self.lock:
            yield self.valid(ticket)

    def activate(self, ticket, chunk_id):
        with self.lock:
            if not self.valid(ticket):
                raise ValueError("旧阶段结果不得接入")
            previous = self.data["active_stage"]
            if previous and previous != ticket["stage_id"] and previous not in self.data["completed_stages"]:
                self.data["completed_stages"].append(previous)
            self.data.update(
                active_stage=ticket["stage_id"],
                active_prompt=ticket["prompt"],
                active_chunk_id=chunk_id,
                request_status="applied",
            )
            if self.data["execution_state"] in ("RESUMING", "SWITCHING"):
                self.transition("RUNNING", "execution worker accepted fresh first chunk")
            self.event(
                "chunk_active",
                inference_request_id=ticket["request_id"],
                active_stage=ticket["stage_id"],
                chunk_id=chunk_id,
            )

    def fault(self, error):
        with self.lock:
            previous = self.data["last_error"]
            detail = str(error) if not previous or str(error) in previous else previous + "；" + str(error)
            self.data.update(last_error=detail, safe_to_resume=False, request_status="failed")
            if self.data["execution_state"] != "FAULT":
                self.transition("FAULT", str(error))


class TaskServer:
    def __init__(self, path, control):
        self.path, self.control = Path(path), control
        if self.path.parent.stat().st_mode & 0o077:
            raise PermissionError("任务 socket 目录必须为0700")

        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                self.request.settimeout(1)
                try:
                    _, uid, _ = struct.unpack("3i", self.request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                    if uid != os.getuid():
                        raise PermissionError("任务 socket 仅允许同一用户")
                    raw = self.rfile.readline(16385)
                    if len(raw) > 16384:
                        raise ValueError("任务请求过大")
                    req = json.loads(raw)
                    if req.pop("run_id", None) != control.run_id:
                        raise PermissionError("任务运行身份不匹配")
                    command = req.pop("command")
                    with control.lock:
                        control.last_contact = time.monotonic()
                    result = control.snapshot() if command == "status" else control.command(command, **req)
                    reply = dict(ok=True, task=result)
                except Exception as error:
                    reply = dict(ok=False, error=str(error))
                try:
                    self.wfile.write(json.dumps(reply, allow_nan=False).encode() + b"\n")
                except OSError:
                    pass

        class Server(socketserver.ThreadingUnixStreamServer):
            daemon_threads = True
            block_on_close = False

        self.server = Server(str(self.path), Handler)
        os.chmod(self.path, 0o600)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True, name="task-control-ipc")
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.path.unlink(missing_ok=True)


class TaskLease:
    """Interrupt the owner, including blocking startup/Home, without sending SDK commands."""

    def __init__(self, control, interrupt_owner, *, clock=time.monotonic):
        self.control, self.interrupt_owner, self.clock = control, interrupt_owner, clock
        self.closed = threading.Event()
        self.thread = None

    def poll(self):
        with self.control.lock:
            state = self.control.data["execution_state"]
            expired = self.clock() - self.control.last_contact > 5
            abort = state in ("STOPPING", "FAULT") or (expired and state not in ("STOPPED", "COMPLETED"))
        if abort:
            self.interrupt_owner()
        return abort

    def start(self):
        def monitor():
            while not self.closed.wait(0.05):
                if self.poll():
                    return  # Exactly one interrupt; never interrupt disable confirmation repeatedly.

        self.thread = threading.Thread(target=monitor, daemon=True, name="task-owner-lease")
        self.thread.start()

    def close(self):
        self.closed.set()
        if self.thread:
            self.thread.join(timeout=1)


def task_rpc(path, run_id, command="status", *, timeout=0.75, **fields):
    payload = json.dumps(dict(run_id=run_id, command=command, **fields)).encode() + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(str(path))
        sock.sendall(payload)
        with sock.makefile("rb") as stream:
            raw = stream.readline(65537)
            if len(raw) > 65536:
                raise ValueError("任务响应过大")
            result = json.loads(raw)
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "任务控制失败"))
    return result["task"]
