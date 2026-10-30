"""Own deployment subprocesses, replay immutable plans, preserve normal SIGINT cleanup."""
import argparse
from contextlib import suppress
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import threading
import time

from .plan import DeploymentSettings, allocate_run, plan_environment
from .recording import mark_stop, rpc, write_json


class Cancelled(Exception):
    pass


def interrupt(process):
    if process is not None and process.poll() is None:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGINT)


def probe(mode, port):
    from deploy.fr3_sharpa_runtime.runtime import SharpaRuntime
    metadata = SharpaRuntime(f"ws://127.0.0.1:{port}").inspect_metadata(require_rtc=mode == "rtc")
    if mode == "soft-async":
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
            state["model_running"] = self.model is not None and self.model.poll() is None
            state["directory"] = str(directory) if directory else None
        logs = {}
        if directory:
            for path in sorted(directory.glob("*.log")):
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 18000))
                    logs[path.name] = stream.read().decode(errors="replace")
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

    def start(self, values, *, expected_approval=None):
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError("当前部署尚未停止和清理完毕")
            if self.recorder and self.recorder.poll() is None:
                raise ValueError("上一录制器仍在收尾，请等待或检查recorder.log")
            plan = self.preview(values)
            if expected_approval is not None and expected_approval != plan["approval"]:
                raise ValueError("核对后权重、代码或配置发生变化，请重新核对最终配置")
            self.cancel.clear()
            self.plan = json.loads(json.dumps(plan))
            self.directory = allocate_run(plan["settings"]["output"])
            self.socket = Path(tempfile.mkdtemp(prefix="sharpa-record-")) / "control.sock"
            self.runtime = self.recorder = None
            self.state = {"phase": "validating", "message": "正在检查模型契约和参数", "running": True,
                          "plan": self.plan, "stop_requested": False}
            try:
                write_json(self.directory / "run.json", self.plan)
                write_json(self.root / ".deployment/control_last.json", values)
                # Save only evidence bound to both manifest and statistics, never path alone.
                self.cache.mkdir(parents=True, exist_ok=True)
                write_json(self.cache / f"{plan['model']['identity']}.json", {"semantics": plan["model"]["semantics"]})
            except Exception:
                self.state.update(phase="failed", running=False, message="配置或归档目录无法写入；未启动设备")
                self.socket.parent.rmdir()
                raise
            self.thread = threading.Thread(target=self._run, daemon=True, name="sharpa-deployment")
            self.thread.start()
            return self.snapshot()

    def _spawn(self, command, name, env):
        output = (self.directory / f"{name}.log").open("ab", buffering=0)
        try:
            process = subprocess.Popen(command, cwd=self.plan["settings"]["repository"], env=env,
                                       stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True)
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

    def _probe(self, env):
        root, profile, mode = self.plan["settings"]["repository"], self.plan["model"]["profile"], self.plan["settings"]["mode"]
        command = ["bash", "-c", 'source "$1/deploy/fr3_sharpa_runtime/model_env.sh" --model "$2"; '
                   'sharpa_python -m deploy.fr3_sharpa_control.controller probe "$3" "$4"',
                   "--", root, profile, mode, str(self.plan["port"])]
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
        key = (self.plan["settings"]["repository"], self.plan["commit"], self.plan["model"]["identity"],
               self.plan["model"]["profile"], self.plan["port"], self.plan["settings"]["mode"])
        if self.model and self.model.poll() is None and key != self.model_key:
            self._close_model()
        self._cancelled()
        with socket.socket() as client:
            client.settimeout(0.3)
            occupied = client.connect_ex(("127.0.0.1", self.plan["port"])) == 0
        if occupied:
            if not self._probe(env):
                raise RuntimeError("服务端口已有不匹配的模型/模式；不会关闭不属于本UI的服务")
            return
        if self.model is None or self.model.poll() is not None:
            self.model = self._spawn(self.plan["service"], "model", env)
            self.model_key = key
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            self._cancelled()
            if self.model.poll() is not None:
                raise RuntimeError("模型服务退出，请查看model.log")
            with socket.socket() as client:
                client.settimeout(0.1)
                listening = client.connect_ex(("127.0.0.1", self.plan["port"])) == 0
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

    def close_model(self):
        with self.lock:
            if self.state.get("running"):
                raise ValueError("先停止执行并等待清理完成，再关闭模型")
            self._close_model()

    def _run(self):
        env = plan_environment(self.plan)
        directory, failure = self.directory, None
        hardware_up = runtime_started = cleanup_verified = False
        trace = directory / "logs"
        env["SHARPA_SHUTDOWN_REPORT_DIR"] = str(trace)
        if self.plan["settings"]["record"]:
            env["SHARPA_RECORD_SOCKET"] = str(self.socket)
        try:
            self._command(self.plan["service"] + ["--check-only"], "model_check", env)
            self._command(self.plan["check"], "parameter_check", env)
            if self.plan["settings"]["record"]:
                self._command(["bash", "deploy/fr3_sharpa_control/record_session.sh", "check"], "recording_check", env)
            self.update("hardware_up", "正在确保硬件服务已启动（每次重新执行都包含up）")
            hardware_up = True  # Includes partially successful compose up.
            self._command(self.plan["hardware"], "hardware", env)
            self.update("model_loading", "正在加载模型并核对预热后的服务身份")
            self._prepare_model(env)
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
            self._close_model() if not runtime_started else None
        except Exception as error:
            failure = str(error)
            if not runtime_started:
                self._close_model()
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
