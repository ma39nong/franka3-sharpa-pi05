"""Optional third-terminal recorder wrapper; uses the same first-chunk gate as UI."""
import argparse
from contextlib import suppress
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

from .plan import allocate_run, desktop
from .recording import rpc, write_json
from deploy.fr3_sharpa_runtime.record_console import record_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(os.environ.get("SHARPA_RECORDINGS_DIR", str(desktop() / "Sharpa部署记录"))))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("-- 后需要原执行命令")
    enabled = os.environ.get("SHARPA_RECORD_ENABLED", "1")
    if enabled not in ("0", "1"):
        parser.error("SHARPA_RECORD_ENABLED仅接受0或1")
    if enabled == "0":
        env = dict(os.environ)
        env.pop("SHARPA_RECORD_SOCKET", None)
        os.execvpe(command[0], command, env)
    directory = allocate_run(args.output)
    root = Path(__file__).resolve().parents[2]
    sockdir = Path(tempfile.mkdtemp(prefix="sharpa-record-"))
    path = sockdir / "control.sock"
    env = dict(os.environ, SHARPA_RECORD_SOCKET=str(path), SHARPA_SHUTDOWN_REPORT_DIR=str(directory / "logs"))
    write_json(directory / "run.json", {"command": command, "repository": str(root),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "weights": {k: v for k, v in env.items() if k.endswith("CHECKPOINT") and k.startswith("SHARPA_")}})
    script = str(root / "deploy/fr3_sharpa_control/record_session.sh")
    recorder = None
    try:
        subprocess.run(["bash", script, "check"], cwd=root, env=env, check=True)
        with (directory / "recorder.log").open("wb") as log:
            recorder = subprocess.Popen(["bash", script, "--directory", str(directory), "--socket", str(path)],
                                         cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 45
        while not path.exists():
            if recorder.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"录制器未就绪：{directory / 'recorder.log'}")
            time.sleep(0.1)
        rpc(path, "status")
        # record_command forwards SIGINT once and waits for the existing owner
        # to complete disable/readback and hardware cleanup. No tee pipe.
        status = record_command(command + ["--execution-trace-dir", str(directory / "logs")], directory / "runtime.log", env=env, cwd=root)
    finally:
        if path.exists():
            with suppress(OSError, RuntimeError, ValueError):
                rpc(path, "stop", timeout=3)
        if recorder is not None:
            try:
                recorder.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(recorder.pid, signal.SIGINT)
                raise RuntimeError("录制器尚未关闭，请查看recorder.log；未强制终止写包")
        path.unlink(missing_ok=True)
        (sockdir / "stop-recording").unlink(missing_ok=True)
        sockdir.rmdir()
    print(f"本次归档：{directory}", flush=True)
    if (directory / "rosbag/metadata.yaml").is_file():
        result = subprocess.run(["bash", script, "export", str(directory)], cwd=root, env=env)
        if result.returncode:
            raise RuntimeError(f"视频导出失败，原始bag保留：{directory}")
    raise SystemExit(status)


if __name__ == "__main__":
    main()
