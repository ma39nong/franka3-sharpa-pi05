"""Own deployment subprocesses, replay immutable plans, preserve normal SIGINT cleanup."""
import argparse
from contextlib import suppress
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time

from .plan import DeploymentSettings, allocate_run, plan_environment
from .recording import mark_stop, rpc, write_json
from deploy.fr3_sharpa_runtime.camera_lease import CameraLease, LEASE_ENV


class Cancelled(Exception):
    pass


def interrupt(process):
    if process is not None and process.poll() is None:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGINT)


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
    metadata = SharpaRuntime(f"ws://127.0.0.1:{port}").inspect_metadata(require_rtc=mode in ("rtc", "paper-rtc"))
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
        self.cancel = threading.Event()
        self.thread = None
        self.runtime = self.model = self.recorder = None
        self.model_key = None
        self.model_ready = False
        self.model_directory = None
        self.execution_directory = None
        self.operation = None
        self.plan = None
        self.directory = None
        self.socket = None
        self.state = {"phase": "idle", "message": "请选择权重和部署配置", "running": False}
        self.export_mutex = threading.Lock()
        self.exports = []
        self.export_threads = []

    def update(self, phase, message, **fields):
        with self.lock:
            self.state.update(phase=phase, message=message, **fields)
            if self.directory:
                write_json(self.directory / "ui_status.json", self.state)

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
            state["model_loading"] = bool(state.get("running") and self.operation == "model")
            state["directory"] = str(directory) if directory else None
        # Keep detailed files on disk; UI always has the same three terminal panes.
        groups = {
            "hardware": [(execution_directory, "hardware.log"), (execution_directory, "hardware_cleanup.log")],
            "inference": [(model_directory, "model_check.log"), (model_directory, "model.log")],
            "execution": [(execution_directory, name) for name in ("parameter_check.log", "recording_check.log", "recorder.log", "runtime.log", "video_export.log")],
        }
        logs = {}
        for group, files in groups.items():
            parts = []
            for folder, name in files:
                if folder:
                    path = folder / name
                    with suppress(OSError):
                        with path.open("rb") as stream:
                            stream.seek(max(0, path.stat().st_size - 18000))
                            parts.append(f"[{name}]\n" + stream.read().decode(errors="replace"))
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

    def preview(self, values):
        return DeploymentSettings(**values).plan(cache=self.cache, machine_root=self.root)

    @staticmethod
    def _model_identity(plan):
        return (plan["settings"]["repository"], plan["commit"], plan["model"]["identity"],
                plan["model"]["profile"], plan["port"], plan["settings"]["mode"])

    def start_model(self, values):
        # Home, playback speed/steps and recording are execution settings. Model
        # warmup must work even when no robot or Home files are available.
        original_values = dict(values)
        mode = values.get("mode", "serial")
        values = {**values, "initialization": "current", "record": False, "rounds": 0,
                  "speed": "fast" if mode in ("rtc", "paper-rtc") else "slow", "steps": 26 if mode.startswith("soft-") else 15}
        return self.start(values, operation="model", persisted_values=original_values)

    def _require_model(self, plan, env):
        if not self.model_ready or self.model_key != self._model_identity(plan):
            raise ValueError("请先启动与当前仓库、权重及模式一致的推理端，再启动执行")
        if (self.model is not None and self.model.poll() is not None) or not self._probe(env, plan):
            self.model_ready = False
            raise RuntimeError("推理端已退出或身份握手失败，请重新启动推理端")

    def start(self, values, *, expected_approval=None, operation="execute", persisted_values=None):
        if operation not in ("model", "execute"):
            raise ValueError("Unknown deployment operation")
        with self.lock:
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
            self.state = {"phase": "validating", "message": "正在检查模型契约和参数", "running": True,
                          "plan": self.plan, "stop_requested": False}
            try:
                write_json(self.directory / "run.json", self.plan)
                write_json(self.root / ".deployment/control_last.json", persisted_values if persisted_values is not None else values)
                # Save only evidence bound to both manifest and statistics, never path alone.
                self.cache.mkdir(parents=True, exist_ok=True)
                write_json(self.cache / f"{plan['model']['identity']}.json", {"semantics": plan["model"]["semantics"]})
            except Exception:
                self.state.update(phase="failed", running=False, message="配置或归档目录无法写入；未启动设备")
                self.socket.parent.rmdir()
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
            raise RuntimeError(f"{name} 失败，退出码{process.returncode}，请查看对应日志")
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
        thread = threading.Thread(target=work, daemon=True, name="sharpa-video-export")
        self.export_threads.append(thread)
        thread.start()

    def wait_exports(self):
        for thread in list(self.export_threads):
            thread.join()

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
        with self.lock:
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
        if self.plan["settings"]["record"]:
            env["SHARPA_RECORD_SOCKET"] = str(self.socket)
        try:
            self._require_model(self.plan, env)
            self._command(self.plan["check"], "parameter_check", env)
            if self.plan["settings"]["record"]:
                self._command(["bash", "deploy/fr3_sharpa_control/record_session.sh", "check"], "recording_check", env)
            camera_lease = CameraLease().acquire()
            env[LEASE_ENV] = str(camera_lease.fd)
            self.update("hardware_up", "正在按相机策略核对占用、等待释放并验证三路图像；通过后才启动执行端")
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
            while self.runtime.poll() is None:
                if self.cancel.is_set():
                    # stop() forwards exactly once. Never kill a cleanup in progress.
                    self.update("stopping", "等待双手失能确认和硬件清理完成")
                if self.recorder and self.recorder.poll() is not None:
                    report = json.loads((directory / "recording.json").read_text()) if (directory / "recording.json").is_file() else {}
                    if report.get("status") not in ("COMPLETE", "NOT_STARTED") and not self.cancel.is_set():
                        failure = "录制器异常退出，已停止执行；原始数据和日志保留"
                        self.stop()
                time.sleep(0.2)
            if self.runtime.returncode not in (0, 130):
                raise RuntimeError(f"执行/清理失败，退出码{self.runtime.returncode}；查看runtime.log和独立失能报告")
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
