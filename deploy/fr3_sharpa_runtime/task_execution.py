"""Wire the existing ROS/SDK owner, task IPC, and ordinary RTC coordinator."""

import json
import os
from pathlib import Path
import shlex
import signal
import threading

from .task_control import TaskControl, TaskLease, TaskServer
from .task_rtc import SDKTaskBackend, TaskRTC
from .task_stages import stage


def execute_task(args):
    from . import run
    from .motion_worker import MotionWorker
    from .runtime import SharpaRuntime
    from .shutdown import protect_disable_confirmation

    directory = Path(args.execution_trace_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path, run_id = os.environ.get("SHARPA_TASK_SOCKET"), os.environ.get("SHARPA_TASK_RUN_ID")
    if not path or not run_id:
        raise ValueError("番茄任务必须由部署控制台创建独立任务 socket 和运行身份")
    log_lock = threading.Lock()

    def emit(**event):
        with log_lock, (directory / "task-events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

    task = TaskControl(run_id, args.initial_stage, emit=emit)
    server = TaskServer(path, task)
    lease = TaskLease(task, lambda: os.kill(os.getpid(), signal.SIGINT))
    lease.start()
    worker = backend = engine = None
    try:
        runtime = SharpaRuntime(args.policy_url)
        metadata = runtime.inspect_metadata(require_rtc=True)
        if (
            not metadata.get("tomato_model")
            or metadata.get("train_rtc")
            or metadata.get("paper_algorithm")
            or metadata.get("model_profile") != "full-delta"
        ):
            raise ValueError("任务服务未匹配已核对的 B Coarse Full Delta 普通 RTC 模型")
        setup = args.ros_overlay / "setup.bash"
        if not setup.is_file():
            raise FileNotFoundError(setup)
        root = Path(__file__).resolve().parents[2]
        core = run.GELLO_REPO / "ros_ws/src/teleop_core"
        inner = (
            f"source {shlex.quote(str(setup))}; "
            f"export SHARPA_SHUTDOWN_REPORT_DIR={shlex.quote(str(directory))}; "
            f"export PYTHONPATH={shlex.quote(str(root))}:{shlex.quote(str(run.GELLO_REPO))}:"
            f'{shlex.quote(str(core))}:"${{PYTHONPATH:-}}"; '
            "exec python -m deploy.fr3_sharpa_runtime.sdk_direct_motion_ros --enable-hardware --online-task-handoff"
        )
        worker = MotionWorker(
            ["pixi", "run", "-e", "ros2", "bash", "-lc", inner], cwd=run.ROS_PROJECT, protocol_fd=True
        )
        backend = SDKTaskBackend(worker, args)
        worker.request({"operation": "ready"}, timeout=20)
        run._start_observation_worker(args.ros_overlay)
        if args.initialization == "home":
            from .home_pose import home_target, load_home

            arms, metadata = load_home(args.home_poses, args.home_task, hand_path=args.hand_home)
            feedback = worker.request({"operation": "feedback"}, timeout=6)
            target = home_target(arms, feedback["state"])
            report = worker.request(
                dict(
                    operation="prepare",
                    target=target.tolist(),
                    speed_rad_s=args.prepare_speed_rad_s,
                    tolerance_rad=0.02,
                    home_recovery=True,
                ),
                timeout=180,
            )
            task.event("home_complete", report=report, home=metadata)

        def observe(**options):
            snapshot = task.snapshot()
            return run.live_observation(args.ros_overlay, stage(snapshot["requested_stage"]).prompt, **options)

        worker.request({"operation": "feedback"}, timeout=6)
        observe(timeout_s=2)  # Validate all three live images before reporting READY.

        engine = TaskRTC(task, runtime, backend, observe, args, directory)
        if task.data["execution_state"] != "IDLE":
            raise RuntimeError("任务在初始化期间已停止")
        engine.run()
        return task.snapshot()
    except BaseException as error:
        task.fault(error)
        raise
    finally:
        # The lease interrupt can race with startup failure. Protect the entire
        # cleanup entry, including joining the monitor, before closing devices.
        with protect_disable_confirmation():
            lease.close()
            try:
                if backend is not None:
                    backend.stop()
                elif worker is not None:
                    worker.close()
            finally:
                # Status is archival evidence only, never a runtime control channel.
                try:
                    (directory / "task-final.json").write_text(
                        json.dumps(task.snapshot(), ensure_ascii=False, indent=2)
                    )
                finally:
                    server.close()
