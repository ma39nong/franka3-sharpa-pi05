"""Deployment-owned B0 process and read-only ROS sidecar; no hardware publishers."""

from contextlib import suppress
import json
import os
from pathlib import Path
import queue
import shlex
import signal
import subprocess
import tempfile
import threading
import time
import uuid

from deploy.fr3_sharpa_runtime.task_control import task_rpc
from deploy.fr3_sharpa_runtime.task_stages import PHASES

LABELS = ("pickup", "harvest", "return", "adjust")
CLASSES = (0, 1, 2, 3, 0, 1, 2)
from .agent_settings import AgentSettings
from dataclasses import asdict


class DecisionGate:
    """Evidence is bound to actual execution run, stage version and mode epoch."""

    def __init__(self, horizon=1.5):
        self.horizon = horizon
        self.reset()

    def reset(self):
        self.key = None
        self.count = 0
        self.last = None

    def update(self, packet, task, *, now, model_version):
        import math

        try:
            if task.get("task_profile", "tomato-b") != "tomato-b":
                self.reset()
                return None
            t = packet["input_timestamp"]
            if (
                packet["protocol"] != "tomato-stage-prediction-v1"
                or packet["labels"] != list(LABELS)
                or packet["valid"] is not True
                or packet.get("error") is not None
                or packet["model_version"] != model_version
                or packet["prediction_mode"] != "joint"
                or not math.isfinite(t)
                or not 0 <= now - t <= 2
                or not t <= packet["result_timestamp"] <= now
            ):
                raise ValueError("Agent 身份或时间戳无效")
            for h in ("current", "future"):
                p = packet[h + "_probabilities"]
                z = packet[h + "_logits"]
                if (
                    len(p) != 4
                    or len(z) != 4
                    or any(not math.isfinite(x) for x in p + z)
                    or any(not 0 <= x <= 1 for x in p)
                    or abs(sum(p) - 1) > 1e-5
                ):
                    raise ValueError("四类概率格式无效")
                exp = [math.exp(x - max(z)) for x in z]
                total = sum(exp)
                if any(abs(a - b / total) > 1e-5 for a, b in zip(p, exp)):
                    raise ValueError("概率与 logits 不一致")
            if abs(packet["future_horizon_s"] - self.horizon) > 1e-6:
                raise ValueError(f"需要 Future{self.horizon}s 权重")
            if packet.get("control_eligible", True) is not True:
                self.reset()
                return None
            key = (task["run_id"], task["stage_version"], task["active_stage"])
            if self.key != key:
                self.reset()
                self.key = key
            if self.last is not None and t <= self.last:
                raise ValueError("重复或乱序观测")
            if self.last is not None and not 0.4 <= t - self.last <= 0.8:
                self.count = 0
            self.last = t
            if (
                task["execution_state"] != "RUNNING"
                or task.get("communication_error")
                or task["active_stage"] != task["requested_stage"]
                or task["active_stage"] not in PHASES
            ):
                self.count = 0
                return None
            phase = PHASES.index(task["active_stage"])
            if phase == 6:
                self.count = 0
                return None
            current = packet["current_probabilities"]
            target = CLASSES[phase + 1]
            qualifies = max(range(4), key=lambda i: current[i]) == target and current[target] >= 0.8
            self.count = self.count + 1 if qualifies else 0
            return PHASES[phase + 1] if self.count >= 2 else None
        except Exception:
            self.reset()
            raise


class AgentSupervisor:
    def __init__(self, controller):
        self.controller = controller
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.thread = None
        self.process = self.collector = None
        self.mode = "manual"
        self.epoch = 0
        self.gate = DecisionGate()
        self.state = dict(status="stopped", mode="manual", message="Agent 未启动", confirmations=0)
        self.directory = None
        self.settings_path = self.controller.root / ".deployment/agent_settings.json"
        self.settings = asdict(AgentSettings.defaults())
        with suppress(OSError, ValueError, TypeError):
            self.settings = asdict(
                AgentSettings.parse(json.loads(self.settings_path.read_text()), self.controller.root)
            )

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps({**self.state, "mode": self.mode}))

    def event(self, event, **fields):
        if self.directory:
            with (self.directory / "events.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(event=event, unix=time.time(), mode=self.mode, **fields),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )

    def set_mode(self, mode):
        if mode not in ("manual", "auto"):
            raise ValueError("请选择手动或 Agent 自动")
        with self.lock:
            if mode == "auto":
                if self.state.get("control_eligible",True) is not True:
                    raise ValueError("研究Memory模型当前仅观察，尚未完成自动控制验收")
                if self.settings["profile"] != "workstation":
                    raise ValueError("开发机离线自检不允许自动控制；请在工作机选择实时模式")
                if self.state["status"] != "ready" or time.time() - self.state.get("received_unix", 0) > 2:
                    raise ValueError("Agent 和实时观测尚未就绪")
                task = self._task()
                if task and task.get("task_profile", "tomato-b") != "tomato-b":
                    raise ValueError("119条五阶段尚未适配旧Agent；请保持手动切换")
                if (
                    not task
                    or task["execution_state"] not in ("READY", "RUNNING")
                    or task["requested_stage"] != task.get("active_stage")
                    and task["execution_state"] != "READY"
                ):
                    raise ValueError("需要就绪或稳定运行中的番茄任务")
            self.mode = mode
            self.epoch += 1
            self.gate.reset()
            self.state.update(confirmations=0)
            if hasattr(self.gate, "describe"):
                self.state.update(self.gate.describe())
            if self.state["status"] == "ready":
                self.state["message"] = "手动控制" if mode == "manual" else "自动切换已启用；等待新鲜连续证据"
            self.event("mode_changed", epoch=self.epoch)
            return self.snapshot()

    def _task(self):
        c = self.controller
        if not c.task_endpoint or not c.state.get("running") or c.operation != "execute":
            return None
        # Agent must never keep the browser/operator lease alive.
        return task_rpc(c.task_endpoint, c.task_run_id, renew_contact=False)

    def start(self, settings):
        options = AgentSettings.parse(settings, self.controller.root)
        report = options.check()
        if not report["ready"]:
            raise ValueError(
                "环境检查未通过：" + "；".join(c["name"] + "：" + c["detail"] for c in report["checks"] if not c["ok"])
            )
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError("Agent 已在运行或正在停止")
            if options.provider == "tairan":
                from .tairan_gate import TairanGate

                self.gate = TairanGate(options.repository, options.handshake_config)
            else:
                self.gate = DecisionGate()
            self.settings = asdict(options)
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.settings_path.with_suffix(".tmp")
            temp.write_text(json.dumps(self.settings, ensure_ascii=False, indent=2))
            os.replace(temp, self.settings_path)
            root, checkpoint, pretrained, python = map(
                Path, (options.repository, options.checkpoint, options.pretrained, options.python)
            )
            self.closed.clear()
            self.mode = "manual"
            self.epoch += 1
            self.gate.reset()
            self.directory = Path(tempfile.mkdtemp(prefix="sharpa-agent-", dir=self.controller.root / ".deployment"))
            self.state = dict(
                status="loading",
                mode="manual",
                message=f"加载 {options.provider} Agent",
                provider=options.provider,
                future_horizon_s=1.0 if options.provider == "tairan" else 1.5,
                profile=options.profile,
                confirmations=0,
                directory=str(self.directory),
            )
            self.thread = threading.Thread(
                target=self._run, args=(root, python, checkpoint, pretrained, options), daemon=True
            )
            self.thread.start()
        return self.snapshot()

    def check_environment(self, values):
        return AgentSettings.parse(values, self.controller.root).check()

    def _development_check(self, replies, options, expected):
        """Actual model smoke test using a recorded NPZ; never calls task_rpc."""
        samples = []
        for index in range(3):
            if self.closed.is_set():
                raise InterruptedError("Agent stopped")
            oid = "development-" + str(index)
            request = dict(
                protocol="tomato-stage-prediction-v1",
                observation_npz=options.sample_npz,
                observation_id=oid,
                session_id=self.directory.name,
                timestamp=time.time(),
            )
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            packet = self._read(replies, 15)
            if packet.get("observation_id") != oid or packet.get("session_id") != self.directory.name:
                raise ValueError("开发机自检响应身份不符")
            placeholder = dict(
                run_id="development",
                stage_version=0,
                active_stage="pickup_1",
                requested_stage="pickup_1",
                execution_state="READY",
            )
            self.gate.update(packet, placeholder, now=time.time(), model_version=expected)
            samples.append(packet["latency_ms"])
            self.event(
                "development_prediction", packet=packet, source="recorded_npz_test_schedule", hardware_connected=False
            )
        with self.lock:
            self.gate.reset()
            self.state.update(
                status="development_ready",
                message=f"{options.provider} 开发机模型自检通过；离线样本3次，无ROS和硬件连接",
                model_version=expected,
                profile="development",
                hardware_connected=False,
                current=dict(zip(LABELS, packet["current_probabilities"])),
                future=dict(zip(LABELS, packet["future_probabilities"])),
                latency_ms=packet["latency_ms"],
                development_latencies_ms=samples,
                confirmations=0,
            )
        while not self.closed.wait(0.1):
            if self.process.poll() is not None:
                raise RuntimeError("开发机模型进程退出")

    def stop(self):
        with self.lock:
            was_auto = self.mode == "auto"
            self.mode = "manual"
            self.epoch += 1
            self.gate.reset()
            self.closed.set()
            self.state.update(status="stopping", message="正在停止 Agent", confirmations=0)
        if was_auto:
            self.controller.stop()
        return self.snapshot()

    def _fail(self, error):
        with self.lock:
            with suppress(Exception):
                self._policy_state()
            was_auto = self.mode == "auto"
            self.mode = "manual"
            self.epoch += 1
            self.gate.reset()
            self.state.update(status="failed", message=str(error), confirmations=0)
            with suppress(Exception):
                self.event("fault", error=str(error))
        if was_auto:
            self.controller.stop()

    def _policy_state(self):
        if hasattr(self.gate, "take_events"):
            for event in self.gate.take_events():
                self.event("tairan_policy", detail=event)
            self.state.update(self.gate.describe())

    def _submit_target(self, target, task):
        if self.mode != "auto" or not target:
            return
        if task.get("task_profile", "tomato-b") != "tomato-b":
            raise ValueError("119条任务禁止旧Agent提交阶段切换")
        result = task_rpc(
            self.controller.task_endpoint,
            self.controller.task_run_id,
            "stage",
            request_id="agent-" + uuid.uuid4().hex,
            expected_version=task["stage_version"],
            target=target,
            source="agent",
        )
        if hasattr(self.gate, "submitted"):
            self.gate.submitted(result)
        else:
            self.gate.reset()
        self.state.update(
            message="已请求 " + target + "，等待执行端实际接入",
            requested_stage=result["requested_stage"],
            confirmations=0,
        )
        self.event("stage_requested", target=target, stage_version=result["stage_version"])

    def _poll_strategy(self):
        with self.lock:
            if self.mode != "auto" or not hasattr(self.gate, "poll"):
                return
            task = self._task()
            if not task:
                raise RuntimeError("tairan 任务执行端失联")
            if task["execution_state"] in ("STOPPING", "STOPPED", "COMPLETED", "FAULT", "PAUSING", "PAUSED"):
                self.set_mode("manual")
                return
            target = self.gate.poll(task, now=time.time())
            self._submit_target(target, task)
            self._policy_state()

    def _read(self, replies, timeout):
        deadline = time.monotonic() + timeout
        while not self.closed.is_set():
            try:
                line = replies.get(timeout=min(0.1, max(0.001, deadline - time.monotonic())))
            except queue.Empty:
                self._poll_strategy()
                if time.monotonic() >= deadline:
                    raise TimeoutError("Agent 模型响应超时")
                continue
            if line is None:
                raise RuntimeError("Agent 模型进程退出")
            return json.loads(line)
        raise InterruptedError("Agent stopped")

    def _run(self, root, python, checkpoint, pretrained, options=None):
        logs = []
        try:
            import hashlib

            expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            log = (self.directory / "model.log").open("w")
            logs.append(log)
            # Avoid another repository's openpi package polluting Agent imports.
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            command = [
                str(python),
                "-m",
                "tools.serve_tomato_agent",
                "--checkpoint",
                str(checkpoint),
                "--pretrained-dir",
                str(pretrained),
            ]
            if options and options.provider == "tairan":
                env["PYTHONPATH"] = str(self.controller.root)
                command = [
                    str(python),
                    "-m",
                    "deploy.fr3_sharpa_control.tairan_service",
                    "--repository",
                    str(root),
                    "--checkpoint",
                    str(checkpoint),
                    "--pretrained-dir",
                    str(pretrained),
                ]
            self.process = subprocess.Popen(
                command,
                cwd=root,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                start_new_session=True,
            )
            replies = queue.Queue()

            def read():
                for line in self.process.stdout:
                    replies.put(line)
                replies.put(None)

            threading.Thread(target=read, daemon=True).start()
            ready = self._read(replies, 120)
            if ready.get("status") != "ready" or ready.get("model_version") != expected:
                raise ValueError("模型就绪身份不符")
            if options is not None and options.profile == "development":
                self._development_check(replies, options, expected)
                return
            gello = (
                Path(options.gello_repository)
                if options
                else Path(os.environ.get("SHARPA_GELLO_REPO", str(Path.home() / "gello_upper_body_teleop")))
            )
            ros_project = gello / "portable_deps/litchi_hardware"
            overlay = (
                Path(options.ros_overlay)
                if options
                else Path(
                    os.environ.get(
                        "SHARPA_ROS_MESSAGES_OVERLAY",
                        str(self.controller.root / ".deployment/sharpa_ros_messages/install"),
                    )
                )
            )
            setup = overlay / "setup.bash"
            if not setup.is_file():
                raise FileNotFoundError(setup)
            roi_args = (' --shared-roi-directory '+shlex.quote(str(self.controller.root / '.deployment/shared_roi'))) if ready.get('requires_local_roi') else ''
            inner = f'source {shlex.quote(str(setup))}; export PYTHONPATH={shlex.quote(str(self.controller.root))}:"${{PYTHONPATH:-}}"; exec python -m deploy.fr3_sharpa_runtime.agent_history --directory {shlex.quote(str(self.directory))}{roi_args}'
            log = (self.directory / "observation.log").open("w")
            logs.append(log)
            ros_env = dict(os.environ)
            domain = next(
                (
                    line.split("=", 1)[1].strip()
                    for line in (gello / "docker/.env").read_text().splitlines()
                    if line.startswith("TELEOP_ROS_DOMAIN_ID=")
                ),
                "",
            )
            if not domain.isdigit():
                raise ValueError("Missing TELEOP_ROS_DOMAIN_ID")
            ros_env.update(
                ROS_DOMAIN_ID=domain,
                ROS_LOCALHOST_ONLY="1",
                RMW_IMPLEMENTATION="rmw_cyclonedds_cpp",
                CYCLONEDDS_URI="file://" + str(gello / "config/cyclonedds.xml"),
            )
            self.collector = subprocess.Popen(
                ["pixi", "run", "--environment", "ros2", "bash", "-lc", inner],
                cwd=ros_project,
                env=ros_env,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            with self.lock:
                self.state.update(
                    status="waiting", model_version=expected, control_eligible=ready.get("control_eligible",True), requires_local_roi=ready.get("requires_local_roi",False), message="模型已加载；等待匹配的相机、关节及所需ROI历史"
                )
            last = None
            while not self.closed.wait(0.025):
                self._poll_strategy()
                if self.process.poll() is not None or self.collector.poll() is not None:
                    raise RuntimeError("Agent 或观测进程退出，请查看日志")
                path = self.directory / "latest.json"
                if not path.exists():
                    continue
                meta = json.loads(path.read_text())
                if not meta.get("valid") or time.time() - meta["captured_unix"] > 2:
                    with self.lock:
                        auto = self.mode == "auto"
                        self.gate.reset()
                        self.state.update(status="waiting", message=meta.get("error", "观测已过期"), confirmations=0)
                    if auto:
                        raise RuntimeError("自动模式实时观测丢失：" + self.state["message"])
                    continue
                if meta["observation_id"] == last:
                    continue
                last = meta["observation_id"]
                if abs(meta["captured_unix"] - meta["ros_now"]) > 0.25:
                    raise ValueError("ROS 与 Unix 时钟域不一致，拒绝改写源时间")
                with self.lock:
                    epoch = self.epoch
                request = dict(
                    protocol="tomato-stage-prediction-v1",
                    observation_npz=meta["observation_npz"],
                    observation_id=last,
                    session_id=self.directory.name,
                    timestamp=meta["timestamp"],
                )
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
                packet = self._read(replies, 3)
                if packet.get("observation_id") != last or packet.get("session_id") != self.directory.name:
                    raise ValueError("Agent 响应与观测身份不符")
                with self.lock:
                    task = self._task()
                    if self.mode == "auto" and (
                        not task
                        or task["execution_state"] in ("STOPPING", "STOPPED", "COMPLETED", "FAULT", "PAUSING", "PAUSED")
                    ):
                        self.set_mode("manual")
                    display_task = task or dict(
                        run_id="observe",
                        stage_version=0,
                        active_stage="pickup_1",
                        requested_stage="pickup_1",
                        execution_state="READY",
                    )
                    if epoch != self.epoch:
                        continue
                    observed_task = display_task
                    if display_task.get("task_profile", "tomato-b") != "tomato-b":
                        observed_task = dict(run_id="observe119", stage_version=0, active_stage="pickup_1", requested_stage="pickup_1", execution_state="READY")
                    if self.settings.get("provider") == "tairan" and self.mode != "auto":
                        observed_task = {**observed_task, "execution_state": "READY"}
                    target = self.gate.update(packet, observed_task, now=time.time(), model_version=expected)
                    self.state.update(
                        status="ready",
                        message=f"实时观测与 {self.settings.get('provider', 'mengxiang')} 正常",
                        received_unix=time.time(),
                        current=dict(zip(LABELS, packet["current_probabilities"])),
                        future=dict(zip(LABELS, packet["future_probabilities"])),
                        latency_ms=packet["latency_ms"],
                        observation_id=last,
                        source_age_s=meta["source_age_s"],
                        confirmations=self.gate.count,
                        suggested_stage=target,
                        active_stage=display_task.get("active_stage"),
                        requested_stage=display_task.get("requested_stage"),
                    )
                    self.event("prediction", packet=packet, task=display_task, target=target, epoch=epoch)
                    self._submit_target(target, task)
                    self._policy_state()
        except InterruptedError:
            pass
        except Exception as error:
            self._fail(error)
        finally:
            for process in (self.collector, self.process):
                if process is not None and process.poll() is None:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            for log in logs:
                log.close()
            with self.lock:
                if self.state["status"] != "failed":
                    self.state.update(status="stopped", message="Agent 已停止")
                self.mode = "manual"
                self.gate.reset()
