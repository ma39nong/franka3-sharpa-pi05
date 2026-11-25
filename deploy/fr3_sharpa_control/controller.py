"""Own deployment subprocesses, replay immutable plans, preserve normal SIGINT cleanup."""
import argparse
from contextlib import suppress
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
import uuid

from .plan import DeploymentSettings, allocate_run, plan_environment
from .recording import mark_stop, rpc, write_json
from deploy.fr3_sharpa_runtime.camera_lease import CameraLease, LEASE_ENV


class Cancelled(Exception):
    pass


def validate_ui_preferences(values):
    if not isinstance(values, dict) or set(values) - {'stage_switch_mode', 'dagger_output'}:
        raise ValueError('不支持的界面默认选项')
    if 'stage_switch_mode' in values and values['stage_switch_mode'] not in ('manual', 'auto'):
        raise ValueError('阶段切换方式必须为manual或auto')
    if 'dagger_output' in values and not isinstance(values['dagger_output'], str):
        raise ValueError('DAgger保存父目录必须为字符串')
    return dict(values)


def interrupt(process):
    if process is not None and process.poll() is None:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGINT)


def command_failure_detail(path):
    """Surface the subprocess cause without loading an unbounded log."""
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 16000))
            lines = stream.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    prefixes = ("TimeoutError:", "RuntimeError:", "ValueError:", "FileNotFoundError:", "PermissionError:")
    causes = [line.strip() for line in lines if line.strip().startswith(prefixes)]
    if causes:
        # Source-clock diagnostics may append a large JSON report to the cause.
        return causes[-1].split("；{", 1)[0][:600]
    return next((line.strip()[:600] for line in reversed(lines) if line.strip()), "")


def port_listening(port):
    """Linux listen-table probe, without bogus HTTP handshakes in model logs."""
    for name in ("tcp", "tcp6"):
        path = Path("/proc/net") / name
        if not path.exists():
            continue
        for row in path.read_text().splitlines()[1:]:
            fields = row.split()
            address, encoded_port = fields[1].split(":")
            if fields[3] == "0A" and int(encoded_port, 16) == port and address in (
                "0100007F", "00000000", "00000000000000000000000000000000"
            ):
                return True
    return False


def probe(mode, port):
    from deploy.fr3_sharpa_runtime.runtime import SharpaRuntime
    metadata = SharpaRuntime(f"ws://127.0.0.1:{port}", **({"train_rtc": True, "paper_rtc": mode == "train-paper-rtc"} if mode in ("train-rtc", "train-paper-rtc") else {})).inspect_metadata(require_rtc=mode in ("rtc", "paper-rtc", "train-rtc", "train-paper-rtc"))
    if mode == "paper-rtc":
        from deploy.fr3_sharpa_runtime.rtc_stream import require_paper_capabilities
        require_paper_capabilities(metadata)
    elif mode == "soft-async":
        from deploy.fr3_sharpa_runtime.async_soft_stream import require_capabilities
        require_capabilities(metadata)
    elif mode == "soft-serial":
        if "paper-rolling-v1" not in metadata.get("inference_capabilities", []) or metadata.get("inpainting_warmed_up") is not True:
            raise ValueError("当前服务不是已预热的串行Soft服务")
    return metadata


class DeploymentController:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.cache = self.root / ".deployment/control_bindings"
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.preferences_path = self.root / '.deployment/control_preferences.json'
        self.ui_preferences = {}
        with suppress(OSError, ValueError):
            self.ui_preferences = validate_ui_preferences(json.loads(self.preferences_path.read_text()))
        self.cancel = threading.Event()
        self.thread = None
        self.runtime = self.model = self.recorder = None
        self.model_key = None
        self.model_ready = False
        self.model_directory = None
        self.execution_directory = None
        self.operation = None
        self.home_thread = self.home_process = None
        self.home_cancel = threading.Event()
        self.home_state = {"running": False, "phase": "idle"}
        self.cleanup_running = False
        self.plan = None
        self.directory = None
        self.socket = None
        self.state = {"phase": "idle", "message": "请选择权重和部署配置", "running": False}
        self.export_mutex = threading.Lock()
        self.exports = []
        self.export_threads = []
        self.task_endpoint = None
        self.task_run_id = None
        self.task_last = None
        self.dagger_endpoint = self.dagger_run_id = self.dagger_last = None
        from .agent import AgentSupervisor
        self.agent = AgentSupervisor(self)

    def update(self, phase, message, **fields):
        with self.lock:
            self.state.update(phase=phase, message=message, **fields)
            if self.directory:
                write_json(self.directory / "ui_status.json", self.state)

    def save_ui_preferences(self, values):
        changes = validate_ui_preferences(values)
        with self.lock, self.preferences_path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            saved = self.ui_preferences
            with suppress(OSError, ValueError):
                saved = validate_ui_preferences(json.loads(self.preferences_path.read_text()))
            preferences = {**saved, **changes}
            write_json(self.preferences_path, preferences)
            self.ui_preferences = preferences
            return dict(preferences)

    def snapshot(self):
        with self.lock:
            state = dict(self.state)
            directory = self.directory
            model_directory = self.model_directory
            execution_directory = self.execution_directory
            alive = self.model is not None and self.model.poll() is None
            state["model_running"] = alive or (self.model_ready and self.model is None)
            state["model_ready"] = self.model_ready and (alive or self.model is None)
            state["model_key"] = self.model_key
            state["model_owned"] = alive
            state["operation"] = self.operation
            state["execution_running"] = bool(state.get("running") and self.operation == "execute")
            state["execution_cleaning"] = bool(state["execution_running"] and
                                                (self.cancel.is_set() or self.cleanup_running))
            state["home"] = dict(self.home_state)
            state["model_loading"] = bool(state.get("running") and self.operation == "model")
            state["directory"] = str(directory) if directory else None
        state["interaction_crop"] = None
        if self.plan and self.plan["model"].get("crop_locator"):
            with suppress(OSError, ValueError):
                status = Path(self.plan["settings"]["repository"]) / ".deployment/crop_live/status.json"
                crop = json.loads(status.read_text())
                crop["age_s"] = max(0, time.time()-crop["updated_wall_time"])
                state["interaction_crop"] = crop
        state["task"] = self.task_status()
        state['dagger'] = self.dagger_status()
        state["agent"] = self.agent.snapshot()
        # Keep detailed files on disk; UI always has the same three terminal panes.
        groups = {
            "hardware": [(execution_directory, "hardware.log"), (execution_directory, "hardware_cleanup.log")],
            "inference": [(model_directory, "model_check.log"), (model_directory, "model.log"),
                          (self.agent.directory, "model.log"), (self.agent.directory, "observation.log")],
            "execution": [(execution_directory, name) for name in ("parameter_check.log", "recording_check.log", "recorder.log", "runtime.log", "video_export.log")],
        }
        if state["home"].get("directory"):
            groups["execution"].append((Path(state["home"]["directory"]), "home.log"))
        logs = {}
        for group, files in groups.items():
            parts = []
            for folder, name in files:
                if folder:
                    path = folder / name
                    with suppress(OSError):
                        with path.open("rb") as stream:
                            stream.seek(max(0, path.stat().st_size - 18000))
                            label = "Agent · " + name if folder == self.agent.directory else name
                            parts.append(f"[{label}]\n" + stream.read().decode(errors="replace"))
            logs[group] = "\n".join(parts)[-18000:]
        state["logs"] = logs
        state["exports"] = []
        for path in list(self.exports)[-12:]:
            with suppress(OSError, ValueError):
                state["exports"].append({"directory": str(path.parent), **json.loads(path.read_text())})
        if directory:
            for name in ("recording.json", "video_export.json"):
                with suppress(OSError, ValueError):
                    state[name] = json.loads((directory / name).read_text())
        return state

    def task_status(self):
        """Read task state without coupling execution to browser polling."""
        with self.lock:
            endpoint, run_id, task_last = self.task_endpoint, self.task_run_id, self.task_last
            executing = self.state.get("running") and self.operation == "execute"
        if endpoint and endpoint.exists():
            try:
                from deploy.fr3_sharpa_runtime.task_control import task_rpc
                task_last = task_rpc(endpoint, run_id, renew_contact=False)
                self.task_last = task_last
            except (OSError, RuntimeError, ValueError) as error:
                task_last = {**(task_last or {}), "communication_error": str(error), "safe_to_resume": False}
        elif task_last and executing:
            task_last = {**task_last, "communication_error": "执行端任务通信未连接", "safe_to_resume": False}
        return task_last

    def dagger_status(self):
        if self.dagger_endpoint and self.dagger_endpoint.exists():
            try:
                from deploy.fr3_sharpa_runtime.dagger_io import rpc as dagger_rpc
                self.dagger_last = dagger_rpc(self.dagger_endpoint, self.dagger_run_id, renew_contact=False)
            except (OSError, RuntimeError, ValueError) as error:
                return {**(self.dagger_last or {}), 'communication_error': str(error)}
        return self.dagger_last

    def dagger_command(self, values):
        if not self.dagger_endpoint or not self.state.get('running') or self.operation != 'execute':
            raise ValueError('没有运行中的DAgger采集，请勾选DAgger并启动执行')
        if values.get('run_id') != self.dagger_run_id:
            raise ValueError('DAgger采集身份已变化，请刷新后再操作')
        # Revokes pending Agent intent without stopping its observation process.
        self.agent.set_mode('manual')
        from deploy.fr3_sharpa_runtime.dagger_io import rpc as dagger_rpc
        result = dagger_rpc(self.dagger_endpoint, self.dagger_run_id,
            command=values['command'], request_id=values['request_id'],
            expected_generation=values['expected_generation'])
        self.dagger_last = result
        return result

    def task_command(self, values):
        # Serialize authority changes with the automatic coordinator.
        with self.agent.lock:
            if values.get("source") not in (None, "operator"):
                raise ValueError("UI 不允许伪造 Agent 来源")
            if values.get("command") in ("stage", "next", "recover") and self.agent.mode == "auto":
                raise ValueError("请先切换为手动控制")
            if values.get("command") in ("pause", "stop", "complete", "recover", "resume"):
                self.agent.set_mode("manual")
            self.agent.gate.reset()
            self.agent.epoch += 1
            return self._operator_task_command(values)

    def _operator_task_command(self, values):
        from deploy.fr3_sharpa_runtime.task_control import task_rpc
        if not self.task_endpoint or not self.state.get("running") or self.operation != "execute":
            raise ValueError("没有运行中的番茄任务；请先核对并启动任务执行端")
        # A separate socket per request; task stop never queues behind another
        # controller command. The execution owner serializes actual authority.
        result = task_rpc(self.task_endpoint, self.task_run_id,
                          **{**values, "http_received_monotonic_ns": time.monotonic_ns()})
        self.task_last = result
        return result

    def preview(self, values):
        return DeploymentSettings(**values).plan(cache=self.cache, machine_root=self.root)

    @staticmethod
    def _model_identity(plan):
        return (plan["settings"]["repository"], plan["commit"], plan["model"]["identity"],
                plan["model"]["profile"], plan["port"], plan["settings"]["mode"])

    def start_model(self, values):
        # Home, playback speed/steps, DAgger and recording are execution settings. Model
        # warmup must work even when no robot or Home files are available.
        original_values = dict(values)
        mode = values.get("mode", "serial")
        values = {**values, "initialization": "current", "record": False, "dagger": False, "rounds": 0,
                  "speed": "fast" if mode in ("rtc", "paper-rtc", "train-rtc", "train-paper-rtc") else "slow", "steps": 26 if mode.startswith("soft-") else 15}
        return self.start(values, operation="model", persisted_values=original_values)

    def home_arms(self, repository):
        from deploy.fr3_sharpa_runtime.home_pose import load_home
        with self.lock:
            executing = self.state.get("running") and self.operation == "execute"
            cleaning = executing and (self.cancel.is_set() or self.cleanup_running)
            if executing and not cleaning:
                return {"ignored": True, "reason": "执行期间B踏板无效"}
            if self.home_thread and self.home_thread.is_alive():
                return {"ignored": True, "reason": "双臂回Home已请求"}
            repo = Path(repository or self.root).expanduser().resolve()
            launcher = repo / "deploy/fr3_sharpa_runtime/home_arms.sh"
            if not launcher.is_file():
                raise ValueError("所选部署仓库缺少双臂独立回Home入口")
            poses = repo / ".deployment/operator_poses.yaml"
            if not poses.is_file():
                poses = self.root / ".deployment/operator_poses.yaml"
            load_home(poses, "powder_weighing")
            self.home_cancel.clear()
            directory = allocate_run(self.root / ".deployment/control_home_sessions")
            self.home_state = {"running": True, "phase": "queued" if cleaning else "starting",
                               "message": "清理完成后双臂回Home" if cleaning else "正在启动双臂回Home",
                               "directory": str(directory)}
            self.home_thread = threading.Thread(target=self._run_home,
                args=(repo, poses, directory, self.thread if cleaning else None,
                      self.execution_directory if cleaning and self.runtime is not None else None),
                daemon=True, name="sharpa-idle-home")
            self.home_thread.start()
            return {"accepted": True, "home": dict(self.home_state)}

    def _run_home(self, repo, poses, directory, cleanup_thread, shutdown_directory):
        failure = None
        try:
            if cleanup_thread is not None:
                cleanup_thread.join()
            if shutdown_directory is not None:
                reports = list(shutdown_directory.glob("logs/shutdown-*.json"))
                if not reports:
                    raise RuntimeError("缺少设备失能确认，双臂回Home已取消")
                for report in reports:
                    if json.loads(report.read_text()).get("status") != "PASS":
                        raise RuntimeError("清理尚未确认设备失能，双臂回Home已取消")
            if self.home_cancel.is_set():
                return
            env = os.environ.copy()
            overlay = self.root / ".deployment/sharpa_ros_messages/install"
            if overlay.is_dir():
                env.setdefault("SHARPA_ROS_MESSAGES_OVERLAY", str(overlay))
            with (directory / "home.log").open("ab", buffering=0) as output:
                with self.lock:
                    if self.home_cancel.is_set():
                        return
                    self.home_state.update(phase="running", message="双臂正在回Home")
                    self.home_process = subprocess.Popen(
                        ["bash", "deploy/fr3_sharpa_runtime/home_arms.sh", "--home-poses", str(poses)],
                        cwd=repo, env=env, stdin=subprocess.DEVNULL,
                        stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                self.home_process.wait()
                if self.home_process.returncode and not self.home_cancel.is_set():
                    detail = command_failure_detail(directory / "home.log")
                    raise RuntimeError("双臂回Home失败；" + (detail or "请查看home.log"))
        except Exception as error:
            failure = str(error)
        finally:
            with self.lock:
                cancelled = self.home_cancel.is_set()
                self.home_process = None
                self.home_state.update(running=False, phase="failed" if failure else "stopped" if cancelled else "finished",
                    message=failure or ("双臂回Home已取消" if cancelled else "双臂已回Home，硬件清理完成"))

    def _require_model(self, plan, env):
        if not self.model_ready or self.model_key != self._model_identity(plan):
            raise ValueError("请先启动与当前仓库、权重及模式一致的推理端，再启动执行")
        if (self.model is not None and self.model.poll() is not None) or not self._probe(env, plan):
            self.model_ready = False
            raise RuntimeError("推理端已退出或身份握手失败，请重新启动推理端")

    def start(self, values, *, expected_approval=None, operation="execute", persisted_values=None):
        self.agent.set_mode("manual")
        if operation not in ("model", "execute"):
            raise ValueError("Unknown deployment operation")
        with self.lock:
            if operation == "execute" and self.home_thread and self.home_thread.is_alive():
                raise ValueError("请等待双臂回Home及硬件清理完成")
            if self.thread and self.thread.is_alive():
                raise ValueError("当前部署尚未停止和清理完毕")
            if self.recorder and self.recorder.poll() is None:
                raise ValueError("上一录制器仍在收尾，请等待或检查recorder.log")
            plan = self.preview(values)
            if expected_approval is not None and expected_approval != plan["approval"]:
                raise ValueError("核对后权重、代码或配置发生变化，请重新核对最终配置")
            if operation == "execute":
                self._require_model(plan, plan_environment(plan))
            self.cancel.clear()
            self.cleanup_running = False
            self.operation = operation
            self.plan = json.loads(json.dumps(plan))
            self.directory = allocate_run(self.root / ".deployment/control_model_sessions" if operation == "model" else plan["settings"]["output"])
            if operation == "execute":
                self.execution_directory = self.directory
            if operation == "model":
                # Reusing the same model must retain its original live log path.
                if self.model_key != self._model_identity(plan) or self.model is None or self.model.poll() is not None:
                    self.model_directory = self.directory
            self.socket = Path(tempfile.mkdtemp(prefix="sharpa-record-")) / "control.sock"
            self.runtime = self.recorder = None
            self.task_endpoint = None
            self.task_run_id = None
            self.task_last = None
            self.dagger_endpoint = self.dagger_run_id = self.dagger_last = None
            if operation == 'execute' and plan['settings'].get('dagger'):
                self.dagger_endpoint = Path(tempfile.mkdtemp(prefix='sharpa-dagger-')) / 'dagger.sock'
                self.dagger_run_id = uuid.uuid4().hex
                self.dagger_last = dict(run_id=self.dagger_run_id, mode='STARTING', generation=0)
            if operation == "execute" and plan["settings"].get("task_mode") in ("tomato-b", "tomato-in-hand119"):
                self.task_endpoint = Path(tempfile.mkdtemp(prefix="sharpa-task-")) / "task.sock"
                self.task_run_id = uuid.uuid4().hex
            self.state = {"phase": "validating", "message": "正在检查模型契约和参数", "running": True,
                          "plan": self.plan, "stop_requested": False}
            try:
                write_json(self.directory / "run.json", self.plan)
                if operation == "execute" and plan["settings"]["record"]:
                    from .quality import begin
                    archive_parent = None
                    if plan["settings"].get("dagger"):
                        archive_parent = plan["settings"].get("dagger_output") or str(Path(plan["settings"]["output"]) / "DAgger")
                    begin(self.directory, archive_parent=archive_parent)
                write_json(self.root / ".deployment/control_last.json", persisted_values if persisted_values is not None else values)
                # Save only evidence bound to both manifest and statistics, never path alone.
                self.cache.mkdir(parents=True, exist_ok=True)
                write_json(self.cache / f"{plan['model']['identity']}.json", {"semantics": plan["model"]["semantics"]})
            except Exception:
                self.state.update(phase="failed", running=False, message="配置或归档目录无法写入；未启动设备")
                self.socket.parent.rmdir()
                if self.task_endpoint:
                    with suppress(OSError):
                        self.task_endpoint.parent.rmdir()
                if self.dagger_endpoint:
                    with suppress(OSError):
                        self.dagger_endpoint.parent.rmdir()
                raise
            self.thread = threading.Thread(target=self._run_model if operation == "model" else self._run, daemon=True, name="sharpa-deployment")
            self.thread.start()
            return self.snapshot()

    def _spawn(self, command, name, env):
        folder = self.model_directory if name in ("model", "model_check") else self.directory
        output = (folder / f"{name}.log").open("ab", buffering=0)
        try:
            process = subprocess.Popen(command, cwd=self.plan["settings"]["repository"], env=env,
                                       stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True,
                                       pass_fds=(int(env[LEASE_ENV]),) if env.get(LEASE_ENV) else ())
        finally:
            output.close()  # Child retains its own regular-file descriptor.
        return process

    def _command(self, command, name, env, timeout=120, cancellable=True):
        process = self._spawn(command, name, env)
        deadline = time.monotonic() + timeout
        while process.poll() is None:
            if time.monotonic() > deadline:
                interrupt(process)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise TimeoutError(f"{name} 超时，请查看对应日志")
            time.sleep(0.1)
        if process.returncode:
            folder = self.model_directory if name in ("model", "model_check") else self.directory
            detail = command_failure_detail(folder / f"{name}.log") if folder else ""
            raise RuntimeError(f"{name} 失败，退出码{process.returncode}；{detail or '请查看对应日志'}")
        if cancellable:
            self._cancelled()

    def _cancelled(self):
        if self.cancel.is_set():
            raise Cancelled()

    def _probe(self, env, plan=None):
        plan = plan or self.plan
        root, profile, mode = plan["settings"]["repository"], plan["model"]["profile"], plan["settings"]["mode"]
        command = ["bash", "-c", 'source "$1/deploy/fr3_sharpa_runtime/model_env.sh" --model "$2"; '
                   'sharpa_python -m deploy.fr3_sharpa_control.controller probe "$3" "$4"',
                   "--", root, profile, mode, str(plan["port"])]
        try:
            result = subprocess.run(command, env=env, cwd=root, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, timeout=5)
            if result.returncode == 0:
                write_json(self.directory / "model_metadata.json", json.loads(result.stdout))
            return result.returncode == 0
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return False

    def _queue_export(self, directory, env):
        """Keep old run paths frozen while a new deployment can start after cleanup."""
        repository = self.plan["settings"]["repository"]
        report = directory / "video_export.json"
        write_json(report, {"status": "QUEUED"})
        self.exports.append(report)
        def work():
            with self.export_mutex:
                try:
                    write_json(report, {"status": "EXPORTING"})
                    export_env = {**env, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
                    with (directory / "video_export.log").open("wb") as log:
                        process = subprocess.Popen(["nice", "-n", "10", "bash", "deploy/fr3_sharpa_control/record_session.sh", "export", str(directory)],
                                                   cwd=repository, env=export_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        try:
                            result = process.wait(timeout=3600)
                        except subprocess.TimeoutExpired:
                            interrupt(process)
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)  # Export-only group.
                                process.wait()
                            raise
                    if result:
                        raise RuntimeError(f"导出退出码{result}，原始bag已保留，详见video_export.log")
                    status = json.loads(report.read_text())
                    if status.get("status") != "COMPLETE":
                        raise RuntimeError("导出进程未提供COMPLETE结果")
                except Exception as error:
                    write_json(report, {"status": "FAILED", "error": str(error)})
                finally:
                    from .quality import archive_when_ready
                    archive_when_ready(directory)
        thread = threading.Thread(target=work, daemon=True, name="sharpa-video-export")
        self.export_threads.append(thread)
        thread.start()

    def wait_exports(self):
        for thread in list(self.export_threads):
            thread.join()

    def _queue_dagger_export(self, directory):
        report = directory / 'dagger/labels_export.json'
        write_json(report, dict(status='QUEUED'))
        self.exports.append(report)
        def work():
            with self.export_mutex:
                try:
                    from deploy.fr3_sharpa_runtime.dagger_export import export
                    export(directory)
                except Exception as error:
                    write_json(report, dict(status='FAILED',error=str(error),raw_preserved=True))
                finally:
                    from .quality import archive_when_ready
                    archive_when_ready(directory)
        thread = threading.Thread(target=work,daemon=True,name='dagger-label-export')
        self.export_threads.append(thread)
        thread.start()

    def _prepare_model(self, env):
        key = self._model_identity(self.plan)
        if self.model and self.model.poll() is None and key != self.model_key:
            self._close_model()
        self._cancelled()
        occupied = port_listening(self.plan["port"])
        if occupied:
            if not self._probe(env):
                raise RuntimeError("服务端口已有不匹配的模型/模式；不会关闭不属于本UI的服务")
            if self.model is not None and self.model.poll() is not None:
                self.model = None
            self.model_key = key
            return
        if self.model is None or self.model.poll() is not None:
            with self.lock:
                self._cancelled()
                self.model = self._spawn(self.plan["service"], "model", env)
                self.model_key = key
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            self._cancelled()
            if self.model.poll() is not None:
                raise RuntimeError("模型服务退出，请查看model.log")
            listening = port_listening(self.plan["port"])
            if listening and self._probe(env):
                return
            time.sleep(0.3)
        raise TimeoutError("GPU模型预热/身份握手超时")

    def _stop_recorder(self):
        if self.socket and self.socket.exists():
            with suppress(OSError, RuntimeError, ValueError):
                mark_stop(self.socket)
                rpc(self.socket, "stop", timeout=3)

    def stop(self):
        self.agent.set_mode("manual")
        with self.lock:
            if self.home_thread and self.home_thread.is_alive():
                self.home_cancel.set()
                interrupt(self.home_process)
                self.home_state.update(phase="stopping", message="正在取消双臂回Home并等待硬件清理")
                return self.snapshot()
            if not self.state.get("running") or self.cancel.is_set():
                return self.snapshot()
            if self.operation == "model":
                raise ValueError("推理端正在加载，请使用停止推理")
            self.cancel.set()
            if self.socket:
                with suppress(OSError):
                    mark_stop(self.socket)
            self.update("stopping", "已发送一次Ctrl-C，等待失能确认及硬件清理", stop_requested=True)
            interrupt(self.runtime)
        threading.Thread(target=self._stop_recorder, daemon=True).start()
        return self.snapshot()

    def _close_model(self):
        if self.model and self.model.poll() is None:
            interrupt(self.model)
            try:
                self.model.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(self.model.pid, signal.SIGKILL)  # Model-only process group, never SDK cleanup.
                self.model.wait()
        self.model = None
        self.model_key = None
        self.model_ready = False

    def close_model(self):
        with self.lock:
            if self.state.get("running"):
                if self.operation == "model":
                    if self.cancel.is_set():
                        return self.snapshot()
                    self.cancel.set()
                    self.model_ready = False
                    interrupt(self.model)
                    self.update("model_stopping", "正在停止推理加载", stop_requested=True)
                    return self.snapshot()
                raise ValueError("先停止执行并等待清理完成，再停止推理")
            if self.model is None and self.model_ready:
                raise ValueError("推理端由外部终端启动，请在原终端停止；本UI不拥有该进程")
            self._close_model()
            self.update("model_off", "推理端已停止；再次执行前请先启动推理")
            return self.snapshot()

    def _run_model(self):
        env, failure = plan_environment(self.plan), None
        try:
            self._command(self.plan["service"] + ["--check-only"], "model_check", env)
            self.update("model_loading", "正在加载和预热推理端；不会启动硬件或执行")
            self._prepare_model(env)
            self._cancelled()
            if self.preview(self.plan["settings"])["approval"] != self.plan["approval"]:
                raise RuntimeError("预热期间代码或权重发生变化，请重新启动推理")
            with self.lock:
                self._cancelled()
                self.model_key = self._model_identity(self.plan)
                self.model_ready = True
        except Cancelled:
            self._close_model()
        except Exception as error:
            failure = str(error)
            self._close_model()
        finally:
            if self.cancel.is_set():
                self._close_model()
            if self.socket:
                self.socket.unlink(missing_ok=True)
                with suppress(OSError):
                    self.socket.parent.rmdir()
            self.update("failed" if failure else "model_off" if self.cancel.is_set() else "model_ready",
                        failure or ("推理端已停止" if self.cancel.is_set() else "推理端已就绪；核对执行配置后可启动执行"), running=False)

    def _run(self):
        env = plan_environment(self.plan)
        directory, failure = self.directory, None
        hardware_up = runtime_started = cleanup_verified = False
        camera_lease = None
        trace = directory / "logs"
        env["SHARPA_SHUTDOWN_REPORT_DIR"] = str(trace)
        if self.dagger_endpoint:
            env.update(SHARPA_DAGGER_SOCKET=str(self.dagger_endpoint),
                       SHARPA_DAGGER_RUN_ID=self.dagger_run_id,
                       SHARPA_DAGGER_DIRECTORY=str(directory))
        if self.task_endpoint:
            env["SHARPA_TASK_SOCKET"] = str(self.task_endpoint)
            env["SHARPA_TASK_RUN_ID"] = self.task_run_id
        if self.plan["settings"]["record"]:
            env["SHARPA_RECORD_SOCKET"] = str(self.socket)
            if self.plan['model'].get('crop_locator'):
                env['SHARPA_FOUR_VIEW_RECORD_DIR'] = str(directory/'policy_views')
        try:
            self._require_model(self.plan, env)
            self._command(self.plan["check"], "parameter_check", env)
            if self.plan["settings"]["record"]:
                self._command(["bash", "deploy/fr3_sharpa_control/record_session.sh", "check"], "recording_check", env)
            camera_lease = CameraLease().acquire()
            env[LEASE_ENV] = str(camera_lease.fd)
            self.update("hardware_up", (
                "正在核对相机占用并启动硬件；随后并行验证视频与执行Home，均成功后才开始任务"
                if "--defer-camera-check" in self.plan["hardware"] else
                "正在按相机策略核对占用、等待释放并验证三路图像；通过后才启动执行端"))
            hardware_up = True  # Includes partially successful compose up.
            self._command(self.plan["hardware"], "hardware", env, timeout=180)
            self._require_model(self.plan, env)
            if self.plan["settings"]["record"]:
                self.update("recording_ready", "正在准备录制订阅；Home/prepare期间不写包")
                self.recorder = self._spawn(["bash", "deploy/fr3_sharpa_control/record_session.sh",
                                            "--directory", str(directory), "--socket", str(self.socket)], "recorder", env)
                deadline = time.monotonic() + 45
                while True:
                    self._cancelled()
                    if self.recorder.poll() is not None:
                        raise RuntimeError("录制器启动失败，请查看recorder.log")
                    if self.socket.exists():
                        rpc(self.socket, "status")
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError("录制器就绪超时")
                    time.sleep(0.1)
            self._cancelled()
            # Never execute a changed checkout or a weight replaced during warmup.
            fresh = self.preview(self.plan["settings"])
            if fresh["approval"] != self.plan["approval"]:
                raise RuntimeError("启动期间代码、权重或Home配置发生变化，请重新确认配置")
            with self.lock:
                self._cancelled()  # stop() cannot fall between this check and process ownership.
                self.runtime = self._spawn(self.plan["execute"] + ["--execution-trace-dir", str(trace)], "runtime", env)
                runtime_started = True
            self.update("running", "执行端已启动；停止按钮等效执行端Ctrl-C")
            next_heartbeat = 0.0
            while self.runtime.poll() is None:
                if self.cancel.is_set():
                    # stop() forwards exactly once. Never kill a cleanup in progress.
                    self.update("stopping", "等待双手失能确认和硬件清理完成")
                if self.recorder and self.recorder.poll() is not None:
                    report = json.loads((directory / "recording.json").read_text()) if (directory / "recording.json").is_file() else {}
                    unexpected_dagger_end = (self.dagger_endpoint is not None and self.dagger_endpoint.exists())
                    if (unexpected_dagger_end or report.get("status") not in ("COMPLETE", "NOT_STARTED")) and not self.cancel.is_set():
                        failure = "录制器异常退出，已停止执行；原始数据和日志保留"
                        self.stop()
                if (self.task_endpoint or self.dagger_endpoint) and not self.cancel.is_set() and time.monotonic() >= next_heartbeat:
                    from deploy.fr3_sharpa_runtime.task_control import task_rpc
                    next_heartbeat = time.monotonic() + 1.0
                    # Renew from the local controller even if every browser tab
                    # is suspended. Loss of the controller still expires it.
                    if self.task_endpoint:
                        with suppress(OSError, RuntimeError, ValueError):
                            task_rpc(self.task_endpoint, self.task_run_id)
                    if self.dagger_endpoint and self.dagger_endpoint.exists():
                        from deploy.fr3_sharpa_runtime.dagger_io import rpc as dagger_rpc
                        with suppress(OSError, RuntimeError, ValueError):
                            self.dagger_last = dagger_rpc(self.dagger_endpoint, self.dagger_run_id)
                time.sleep(0.2)
            if self.runtime.returncode not in (0, 130):
                detail = command_failure_detail(directory / "runtime.log")
                raise RuntimeError(f"执行/清理失败，退出码{self.runtime.returncode}；{detail or '查看runtime.log和独立失能报告'}")
            reports = list(trace.glob("shutdown-*.json"))
            if not reports or any(json.loads(path.read_text()).get("status") != "PASS" for path in reports):
                raise RuntimeError("未取得完整的双手失能PASS报告，不能将进程结束当作失能成功")
            cleanup_verified = True
        except Cancelled:
            pass
        except Exception as error:
            failure = str(error)
            if self.runtime and self.runtime.poll() is None:
                self.stop()
                self.runtime.wait()  # Existing finite disable-confirmation owner controls its timeout.
        finally:
            with self.lock:
                self.cleanup_running = True
            self._stop_recorder()
            if self.recorder:
                try:
                    self.recorder.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    interrupt(self.recorder)
                    failure = failure or "录制器未正常关闭，视频未导出"
            if hardware_up and (not runtime_started or not cleanup_verified):
                try:
                    self._command(["bash", "deploy/fr3_sharpa_runtime/hardware.sh", "stop"], "hardware_cleanup", env, cancellable=False)
                except Exception as error:
                    failure = f"{failure or ''} 硬件清理：{error}"
            if self.dagger_endpoint and (directory/'dagger/frames.jsonl').is_file():
                try:
                    self._queue_dagger_export(directory)
                except Exception as error:
                    failure = f"{failure or ''} 标签导出提交失败：{error}；原始记录保留"
            if self.recorder and self.recorder.poll() is None:
                failure = failure or "录制器仍在收尾"
            elif (directory / "rosbag/metadata.yaml").is_file():
                try:
                    self._queue_export(directory, dict(env))
                except Exception as error:
                    failure = f"{failure or ''} 无法提交视频导出：{error}；原始bag保留"
            self.socket.unlink(missing_ok=True)
            (self.socket.parent / "stop-recording").unlink(missing_ok=True)
            with suppress(OSError):
                self.socket.parent.rmdir()
            if camera_lease is not None:
                camera_lease.close()
            if self.task_endpoint:
                with suppress(OSError, ValueError):
                    self.task_last = json.loads((trace / "task-final.json").read_text())
                if failure:
                    previous_error = (self.task_last or {}).get("last_error")
                    self.task_last = {**(self.task_last or {}), "execution_state": "FAULT", "safe_to_resume": False,
                                      "last_error": f"{previous_error}；{failure}" if previous_error else failure}
                self.task_endpoint.unlink(missing_ok=True)
                with suppress(OSError):
                    self.task_endpoint.parent.rmdir()
            if self.dagger_endpoint:
                self.dagger_last = {**(self.dagger_last or {}), 'mode': 'STOPPED',
                                    'reason': failure or 'execution ended'}
                self.dagger_endpoint.unlink(missing_ok=True)
                with suppress(OSError):
                    self.dagger_endpoint.parent.rmdir()
            if self.plan["settings"]["record"]:
                try:
                    from .quality import finish
                    finish(directory, outcome="FAILED" if failure else "FINISHED",
                           reason=failure or ("operator stop" if self.cancel.is_set() else "execution completed"),
                           returncode=self.runtime.returncode if self.runtime is not None else None,
                           recorder_closed=self.recorder is not None and self.recorder.poll() is not None)
                except Exception as error:
                    failure = f"{failure or ''} 评级记录保存失败：{error}；原始bag保留"
            self.update("failed" if failure else "stopped" if self.cancel.is_set() else "finished",
                        failure or "执行、失能确认和硬件清理已完成；可重新选择Home/Current启动", running=False)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("probe")
    check.add_argument("mode")
    check.add_argument("port", type=int)
    args = parser.parse_args()
    print(json.dumps(probe(args.mode, args.port)))


if __name__ == "__main__":
    main()
