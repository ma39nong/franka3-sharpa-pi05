"""Independent Qwen lifecycle and ROS observation owner for the deployment UI."""

import contextlib
from dataclasses import asdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid

from deploy.fr3_sharpa_runtime.task_control import task_rpc

from .qwen_client import KEY_PATH
from .qwen_client import MODELS
from .qwen_client import build_payload
from .qwen_client import recognize
from .qwen_memory import HarvestPlan
from .qwen_memory import TaskMemory
from .qwen_observer import FrameBuffer
from .qwen_replay import atomic_json


@dataclass
class QwenSettings:
    provider: str = "qwen"
    profile: str = "workstation"
    model: str = MODELS[0]
    quotas: str = "2,2"
    python: str = ""
    key_path: str = str(KEY_PATH)
    gello_repository: str = ""
    ros_overlay: str = ""

    @classmethod
    def parse(cls, values, root):
        defaults = asdict(
            cls(
                python=str(Path(root) / ".venv/bin/python"),
                gello_repository=str(Path.home() / "gello_upper_body_teleop"),
                ros_overlay=str(Path(root) / ".deployment/sharpa_ros_messages/install"),
            )
        )
        if (
            not isinstance(values, dict)
            or set(values) - set(defaults)
            or any(not isinstance(v, str) for v in values.values())
        ):
            raise ValueError("Qwen配置字段或类型无效")
        defaults.update(values)
        obj = cls(**defaults)
        if obj.provider != "qwen" or obj.profile != "workstation" or obj.model not in MODELS:
            raise ValueError("请选择Qwen独立工作机观察及支持的视觉模型")
        obj.plan()
        for name in ("python", "key_path", "gello_repository", "ros_overlay"):
            path = Path(getattr(obj, name)).expanduser()
            setattr(obj, name, os.path.abspath(path if path.is_absolute() else Path(root) / path))
        return obj

    def plan(self):
        try:
            return HarvestPlan(tuple(int(n.strip()) for n in self.quotas.split(",")))
        except (TypeError, ValueError):
            raise ValueError("计划填写4或2,2等一轮/两轮正整数") from None

    def check(self):
        checks = []

        def add(name, ok, detail):
            checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

        add("Qwen Python", os.access(self.python, os.X_OK), self.python)
        configured = False
        with contextlib.suppress(OSError):
            configured = bool(Path(self.key_path).read_text().strip())
        add("本机API密钥文件", configured, self.key_path)
        if os.access(self.python, os.X_OK):
            try:
                result = subprocess.run(
                    [self.python, "-c", "import cv2,numpy"], capture_output=True, timeout=15, check=False
                )
                add("图像处理依赖", result.returncode == 0, "cv2/numpy;不加载Torch、编码器或V-JEPA权重")
            except (OSError, subprocess.TimeoutExpired):
                add("图像处理依赖", ok=False, detail="无法运行所选Python")
        gello = Path(self.gello_repository)
        for name, path in (
            ("ROS项目", gello / "portable_deps/litchi_hardware/pixi.toml"),
            ("ROS overlay", Path(self.ros_overlay) / "setup.bash"),
            ("DDS配置", gello / "config/cyclonedds.xml"),
            ("ROS domain", gello / "docker/.env"),
        ):
            add(name, path.is_file(), path)
        add("Pixi", shutil.which("pixi") is not None, shutil.which("pixi") or "未找到Pixi")
        try:
            maximum = int(Path("/proc/sys/net/core/rmem_max").read_text())
        except (OSError, ValueError):
            maximum = 0
        add("工作机DDS接收缓冲", maximum >= 8388608, f"当前{maximum},要求8388608")
        return {
            "provider": "qwen",
            "profile": self.profile,
            "ready": all(c["ok"] for c in checks),
            "checks": checks,
            "settings": asdict(self),
            "note": "Qwen自行订阅已有ROS图像和状态,不启动V-JEPA或硬件。此检查不调用API;任务运行后才发视觉请求。当前仅观察和记忆,不允许自动切换。",
        }


class QwenSupervisor:
    def __init__(self, controller):
        self.controller = controller
        self.lock = threading.RLock()
        self.closed = threading.Event()
        self.thread = self.collector = None
        self.directory = None
        self.config_path = Path(controller.root) / ".deployment/qwen_settings.json"
        values = {}
        with contextlib.suppress(OSError, ValueError):
            values = json.loads(self.config_path.read_text())
        try:
            self.settings = QwenSettings.parse(values, controller.root)
        except ValueError:
            self.settings = QwenSettings.parse({}, controller.root)
        self.state = {
            "provider": "qwen",
            "status": "stopped",
            "mode": "manual",
            "control_eligible": False,
            "message": "Qwen未启动;独立观察,不加载V-JEPA",
        }

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.state, ensure_ascii=False))

    def update(self, **fields):
        with self.lock:
            self.state.update(fields)

    def set_mode(self, mode):
        if mode != "manual":
            raise ValueError("Qwen当前仅支持独立观察、记忆和计数,自动执行尚未验收")
        return self.snapshot()

    def start(self, values):
        settings = QwenSettings.parse(values, self.controller.root)
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError("Qwen仍在运行或停止中")
        report = settings.check()
        if not report["ready"]:
            raise ValueError(
                "Qwen环境检查未通过:" + ";".join(c["name"] + ":" + c["detail"] for c in report["checks"] if not c["ok"])
            )
        with self.lock:
            self.settings = settings
            atomic_json(self.config_path, asdict(settings))
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self.directory = Path(tempfile.mkdtemp(prefix="sharpa-qwen-", dir=self.config_path.parent))
            self.closed.clear()
            self.state = {
                "provider": "qwen",
                "model": settings.model,
                "quotas": list(settings.plan().quotas),
                "status": "loading",
                "mode": "manual",
                "control_eligible": False,
                "message": "启动Qwen独立观测采集",
                "directory": str(self.directory),
            }
            self.thread = threading.Thread(target=self._run, args=(settings,), name="qwen-observer", daemon=True)
            self.thread.start()
        return self.snapshot()

    def stop(self):
        self.closed.set()
        if self.collector is not None and self.collector.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.collector.pid, signal.SIGINT)
        self.update(
            status="stopping" if self.thread and self.thread.is_alive() else "stopped",
            message="停止Qwen;不改变机器人当前执行",
        )
        return self.snapshot()

    def collector_command(self, settings):
        gello = Path(settings.gello_repository)
        domain = next(
            (
                line.split("=", 1)[1].strip()
                for line in (gello / "docker/.env").read_text().splitlines()
                if line.startswith("TELEOP_ROS_DOMAIN_ID=")
            ),
            "",
        )
        if not domain.isdigit():
            raise ValueError("工作机ROS domain无效")
        env = dict(
            os.environ,
            ROS_DOMAIN_ID=domain,
            ROS_LOCALHOST_ONLY="1",
            RMW_IMPLEMENTATION="rmw_cyclonedds_cpp",
            CYCLONEDDS_URI="file://" + str(gello / "config/cyclonedds.xml"),
        )
        inner = (
            f"source {shlex.quote(str(Path(settings.ros_overlay) / 'setup.bash'))}; "
            f'export PYTHONPATH={shlex.quote(str(self.controller.root))}:"${{PYTHONPATH:-}}"; '
            f"exec python -m deploy.fr3_sharpa_runtime.agent_history --native-agent-roi "
            f"--directory {shlex.quote(str(self.directory))}"
        )
        return (
            ["pixi", "run", "--environment", "ros2", "bash", "-lc", inner],
            gello / "portable_deps/litchi_hardware",
            env,
        )

    def task(self):
        c = self.controller
        if not c.task_endpoint or not c.state.get("running") or c.operation != "execute":
            return None
        return task_rpc(c.task_endpoint, c.task_run_id, renew_contact=False)

    def _run(self, settings):
        buffer, memory, attempted_end = FrameBuffer(), None, None
        capture = threading.Thread(target=buffer.collect, args=(self.directory, self.closed), daemon=True)
        try:
            command, cwd, env = self.collector_command(settings)
            with (self.directory / "observation.log").open("w") as log:
                self.collector = subprocess.Popen(
                    command, cwd=cwd, env=env, stdout=log, stderr=log, start_new_session=True
                )
            capture.start()
            self.update(status="waiting", message="Qwen已启动,等待相机数据及运行中的手中调整任务")
            while not self.closed.wait(0.2):
                if self.collector.poll() is not None:
                    raise RuntimeError("Qwen独立观测进程退出,请查看observation.log")
                try:
                    task = self.task()
                    if (
                        not task
                        or task["task_profile"] != "tomato-in-hand119"
                        or task["execution_state"] != "RUNNING"
                        or task["active_stage"] != task["requested_stage"]
                    ):
                        self.update(status="waiting", message="等待稳定运行的手中调整任务;尚未调用API")
                        continue
                    if memory is None or memory.run_id != task["run_id"]:
                        memory_path = (
                            Path(self.controller.root)
                            / ".deployment/qwen_memories"
                            / (hashlib.sha256(task["run_id"].encode()).hexdigest() + ".json")
                        )
                        loaded = (
                            TaskMemory.restore(json.loads(memory_path.read_text()))
                            if memory_path.exists()
                            else TaskMemory(task["run_id"], settings.plan())
                        )
                        if loaded.run_id != task["run_id"] or loaded.plan != settings.plan():
                            raise ValueError("已有会话记忆与所选计划不一致,请核查计划")
                        memory = loaded
                        attempted_end = None
                        self.update(evaluation=None, message="新执行会话:建立独立Qwen任务记忆")
                    if task["active_stage"] != memory.stage:
                        memory.acknowledge(
                            task["active_stage"],
                            run_id=memory.run_id,
                            expected_revision=memory.revision,
                            applied_at=task["active_applied_unix_s"],
                        )
                    elif memory.stage_started_at is None:
                        memory.stage_started_at = task["active_applied_unix_s"]
                    frames, images = buffer.window(memory)
                    if len(frames) < 4 or frames[-1]["timestamp"] - frames[0]["timestamp"] < 1:
                        continue
                    if attempted_end is not None and frames[-1]["timestamp"] <= attempted_end + 0.5:
                        continue
                    attempted_end = frames[-1]["timestamp"]
                    request = {
                        "id": uuid.uuid4().hex,
                        "run_id": memory.run_id,
                        "revision": memory.revision,
                        "active_stage": memory.stage,
                        "start": frames[0]["timestamp"],
                        "end": attempted_end,
                        "frames": frames,
                    }
                    directory = self.directory / "requests" / request["id"]
                    atomic_json(
                        directory / "request.json",
                        {
                            "request": request,
                            "memory": memory.snapshot(),
                            "model": settings.model,
                            "enable_thinking": False,
                            "execution_version": task["stage_version"],
                        },
                    )
                    payload = build_payload(memory.snapshot(), request, images, model=settings.model)
                    for frame in frames:
                        (directory / (hashlib.sha256(frame["id"].encode()).hexdigest() + ".jpg")).write_bytes(
                            images[frame["id"]]
                        )
                    self.update(status="waiting", message="Qwen分析当前画面中;机器人仍由原执行端控制")
                    result = recognize(payload, key_path=Path(settings.key_path))
                    atomic_json(directory / "result.json", result)
                    if self.closed.is_set():
                        break
                    current = self.task()
                    if not current or any(
                        current.get(k) != task.get(k)
                        for k in ("run_id", "stage_version", "active_stage", "execution_state", "requested_stage")
                    ):
                        self.update(status="waiting", message="执行状态变化,Qwen旧回答已丢弃")
                        continue
                    evaluated = memory.evaluate(request, result["answer"], now=time.time())
                    atomic_json(directory / "evaluation.json", evaluated)
                    atomic_json(self.directory / "memories" / (memory.run_id + ".json"), memory.snapshot())
                    atomic_json(memory_path, memory.snapshot())
                    self.update(
                        status="ready",
                        message="Qwen独立观察结果;阶段仍需手动切换",
                        evaluation=evaluated,
                        latency_ms=result["elapsed_s"] * 1000,
                        received_unix=time.time(),
                    )
                except (OSError, ValueError, RuntimeError, KeyError) as error:
                    self.update(status="waiting", message=str(error))
        except Exception as error:
            self.update(status="failed", message=str(error))
        finally:
            self.closed.set()
            if self.collector is not None and self.collector.poll() is None:
                for stopping_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(self.collector.pid, stopping_signal)
                        self.collector.wait(timeout=3)
                        break
                    except ProcessLookupError:
                        break
                    except subprocess.TimeoutExpired:
                        continue
            if capture.is_alive():
                capture.join(timeout=2)
            if self.state["status"] != "failed":
                self.update(status="stopped", message="Qwen已停止")
