# ruff: noqa: RUF001 -- Chinese terminal messages use Chinese punctuation.
"""Finite receding-horizon execution with an independent motion process.

Model inference, spline planning and artifact writes stay in the parent. The
motion process keeps the same gateway session/sequence across all chunks.
"""

from collections import deque
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import time
import traceback

import numpy as np

from .player import Admission
from .player import OneShot
from .runtime.ipc import RemoteDevices
from .runtime.ipc import wire_feedback
from .live import LiveConsumer
from .profile import ARM_SPEED_RAD_S
from .prefetch import endpoint_ready, accept_prefetch
from .profile import slow_motion_config

CONTROL_TICK_SECONDS = 0.01
CONTROL_OVERRUN_GRACE_SECONDS = 0.005
CONTROL_OVERRUN_CONSECUTIVE_LIMIT = 3
CONTROL_OVERRUN_WINDOW_SECONDS = 1.0
CONTROL_OVERRUN_WINDOW_LIMIT = 5


@dataclass(frozen=True)
class PlanAlternatives:
    """Both profiles are planned before the child reaches a switching boundary."""

    medium: OneShot
    slow: OneShot


def choose_next_player(candidate, now, started, slow_after_seconds, already_slow):
    if not isinstance(candidate, PlanAlternatives):
        return candidate
    if slow_after_seconds is None:
        raise ValueError("A timed plan requires an active switch threshold")
    return candidate.slow if already_slow or now - started >= slow_after_seconds else candidate.medium


class StopFlag:
    """Shared cancellation byte; no condition lock can be stranded on child exit."""

    def __init__(self, context):
        self.value = context.RawValue("b", 0)

    def set(self):
        self.value.value = 1

    def is_set(self):
        return bool(self.value.value)

    def wait(self, seconds):
        end = time.monotonic() + seconds
        while not self.is_set() and time.monotonic() < end:
            time.sleep(min(0.01, max(0.0, end - time.monotonic())))
        return self.is_set()


def control_loop(
    devices,
    first,
    incoming,
    notify,
    emit,
    stopping,
    rounds,
    *,
    clock=time.monotonic,
    sleep=time.sleep,
    idle_timeout=10.0,
    slow_after_seconds=None,
):
    if slow_after_seconds is not None and slow_after_seconds <= 0:
        raise ValueError("Slow switch threshold must be positive or disabled")
    player, sequence, round_number = first, 0, 1
    slow_active = False
    notify({"event": "stage", "stage": "通信预热：连续检查四设备反馈 1 秒"})
    warm_until = clock() + 1.0
    while clock() < warm_until:
        if stopping():
            raise RuntimeError("Feedback warmup cancelled")
        devices.feedback(clock())
        sleep(0.01)
    fb = devices.feedback(clock())
    player.check_start(fb, clock())
    notify({"event": "stage", "stage": "设备准备：确认手部使能并刷新两臂反馈"})
    devices.prepare(player, fb, clock())
    fb = devices.feedback(clock())
    player.start(fb, clock())
    notify({"event": "stage", "stage": "第 1 轮归位：中速移到模型第一帧"})
    due, last, waiting_since, last_status = clock(), None, None, 0.0
    recent_overruns, consecutive_overruns = deque(), 0
    prefetched_round = None
    while not stopping():
        now = clock()
        if now < due:
            sleep(due - now)
        now = clock()
        if now - due > 0.01:
            raise RuntimeError(f"Control wake deadline missed: lateness_ms={(now - due) * 1000:.3f}")
        # Reuse the measured sample from the preceding device submit. Its source
        # timestamps are preserved and checked, never refreshed by the client.
        fb.check(now, fb.epoch)
        if waiting_since is not None:
            if now - waiting_since > idle_timeout:
                raise RuntimeError("Next fresh plan did not arrive within 10 seconds")
            try:
                candidate = incoming.get_nowait()
            except queue.Empty:
                candidate = None
            if candidate is not None:
                candidate = choose_next_player(candidate, now, first.started, slow_after_seconds, slow_active)
                accept_prefetch(candidate, now)
                candidate.run_id = player.run_id
                candidate.check_start(fb, now)
                result = devices.call(
                    "next_plan", run_id=player.run_id, plan_hash=candidate.digest,
                    start=candidate.plan.start.tolist(),
                    slider_speed_rad_s=candidate.plan.config.get("slider_speed_rad_s", 2.0),
                )
                if result["sequence"] != sequence:
                    raise RuntimeError("Next-plan sequence mismatch")
                selected_profile = candidate.plan.config.get("speed_profile", "medium")
                if selected_profile == "slow" and not slow_active:
                    slow_active = True
                    notify({"event": "speed_profile_changed", "round": round_number + 1,
                            "elapsed_seconds": now - first.started, "profile": "slow",
                            "plan_hash": candidate.digest})
                player = candidate
                player.sequence = sequence
                player.start(fb, clock())
                waiting_since = None
                round_number += 1
                notify({"event": "round_started", "round": round_number,
                        "profile": selected_profile, "plan_hash": player.digest})
                notify({"event": "stage", "stage": f"第 {round_number} 轮：平滑衔接新动作段"})
                now = clock()
        if waiting_since is None:
            previous_phase = player.state
            frame = player.tick(fb, now)
            if player.state != previous_phase and player.state not in {"fault", "stopping", "complete"}:
                descriptions = {
                    "settle_start": "等待起始姿态到位",
                    "playback": "慢速播放动作段" if slow_active else "中速播放动作段",
                    "settle_end": "等待本轮终点到位",
                }
                notify(
                    {"event": "stage", "stage": f"第 {round_number} 轮：{descriptions.get(player.state, player.state)}"}
                )
            if player.state == "complete":
                notify(
                    {
                        "event": "round_complete",
                        "round": round_number,
                        "transitions": player.transitions,
                        "target": last.positions.tolist(),
                    }
                )
                if round_number == rounds:
                    devices.finish()
                    notify({"event": "complete", "rounds": rounds})
                    return
                waiting_since = now
                notify({"event": "stage", "stage": "轮间保持：等待下一轮计划就绪"})
                frame = None
            elif player.state in {"fault", "stopping"}:
                raise RuntimeError(f"Round {round_number}: {player.reason}")
        else:
            frame = None
        if frame is None:
            # Hold frames share the active plan identity and global wire sequence.
            frame = replace(
                last,
                sequence=sequence,
                created=now,
                valid_until=now + 0.02,
                velocities=np.zeros(54),
                phase="settle_end",
            )
        submitted = devices.submit(frame, clock())
        for warning in getattr(devices, "last_submission", {}).get("warnings", []):
            notify({"event": "stage", "stage": "【手部警告】" + warning})
        if submitted is None:
            raise RuntimeError("Device bridge lacks submission feedback support")
        fb, last, sequence = submitted, frame, frame.sequence + 1
        emit(frame, fb)
        # Ask for a fresh observation as soon as the endpoint is measured ready.
        # Inference/planning overlap the remaining settle dwell in the parent;
        # the child cannot activate the queued plan before round_complete.
        if (round_number < rounds and prefetched_round != round_number
                and player.state == "settle_end" and endpoint_ready(player, fb)):
            notify({"event": "prefetch", "round": round_number,
                    "target": last.positions.tolist(), "feedback": wire_feedback(fb)})
            prefetched_round = round_number
        if waiting_since is not None and now - last_status >= 0.05:
            notify(
                {
                    "event": "between",
                    "round": round_number,
                    "target": last.positions.tolist(),
                    "feedback": wire_feedback(fb),
                }
            )
            last_status = now
        due += CONTROL_TICK_SECONDS
        finished = clock()
        overrun = finished - due
        if overrun > 1e-9:
            # submit must have returned successfully. Never turn an expired
            # command or unhealthy feedback into a scheduling-only warning.
            if finished >= frame.valid_until:
                raise RuntimeError("Control submission exceeded frame deadline")
            fb.check(finished, fb.epoch)
            if overrun > CONTROL_OVERRUN_GRACE_SECONDS + 1e-9:
                raise RuntimeError(f"Control submission exceeded 100 Hz tick: overrun_ms={overrun * 1000:.3f}; "
                                   f"grace_ms={CONTROL_OVERRUN_GRACE_SECONDS * 1000:.3f}")
            consecutive_overruns += 1
            while recent_overruns and finished - recent_overruns[0] >= CONTROL_OVERRUN_WINDOW_SECONDS:
                recent_overruns.popleft()
            recent_overruns.append(finished)
            if (consecutive_overruns >= CONTROL_OVERRUN_CONSECUTIVE_LIMIT
                    or len(recent_overruns) > CONTROL_OVERRUN_WINDOW_LIMIT):
                raise RuntimeError("Repeated control tick overruns: "
                                   f"consecutive={consecutive_overruns}, in_last_second={len(recent_overruns)}")
            # Rebase the schedule: do not replay missed ticks or shorten the
            # next start-to-start interval to catch up. Frame expiry is unchanged.
            due = max(finished, frame.created + CONTROL_TICK_SECONDS)
            notify({"event": "control_tick_overrun", "round": round_number,
                    "sequence": frame.sequence, "overrun_ms": overrun * 1000,
                    "consecutive": consecutive_overruns, "in_last_second": len(recent_overruns),
                    "next_due": due})
        else:
            consecutive_overruns = 0
    raise RuntimeError("Continuous execution cancelled")


def _worker(sock, counter, finish_policy, first, incoming, status, stop_event, output, rounds,
            slow_after_seconds=None):
    devices = RemoteDevices(None, execute=True, finish_policy=finish_policy, sock=sock, arm_speed_rad_s=ARM_SPEED_RAD_S)
    devices.counter = counter
    recorder = None
    local = queue.Queue(maxsize=1)
    parent_pid = os.getppid()

    def watch_parent():
        while not stop_event.wait(0.05):
            if os.getppid() != parent_pid:
                stop_event.set()
                return

    def load_plans():
        # Unpickle spline data outside the control thread.
        while not stop_event.is_set():
            try:
                item = incoming.get(timeout=0.1)
                local.put(item, timeout=0.1)
            except queue.Empty:
                continue
            except BaseException:
                stop_event.set()
                return

    def notify(event):
        status.put_nowait(event)

    try:
        from experiments.weight_motion_eval.reference import deployment_module

        recorder = deployment_module("executor.async_recorder").AsyncRecorder(Path(output) / "control")
        emitter = LiveConsumer.__new__(LiveConsumer)
        emitter.devices, emitter.recorder = devices, recorder
        threading.Thread(target=load_plans, daemon=True).start()
        threading.Thread(target=watch_parent, daemon=True).start()
        control_loop(devices, first, local, notify, emitter.emit, stop_event.is_set, rounds,
                     slow_after_seconds=slow_after_seconds)
        while finish_policy == "hold" and not stop_event.wait(0.05):
            if devices.hold_error:
                raise RuntimeError(devices.hold_error)
    except BaseException as error:
        # Stop before traceback formatting, reporting, or draining the recorder.
        try:
            devices.stop()
        finally:
            notify(
                {"event": "failed", "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
            )
    finally:
        try:
            devices.stop()
        finally:
            devices.close()
            if recorder is not None:
                recorder.close()


class ContinuousConsumer(LiveConsumer):
    def __init__(self, *args, rounds=50, replan_steps=20, slow_after_seconds=None, **kwargs):
        super().__init__(*args, **kwargs)
        if slow_after_seconds is not None and slow_after_seconds <= 0:
            raise ValueError("Slow switch threshold must be positive or disabled")
        self.config = dict(self.config, execution_steps=replan_steps)
        self.config["speed_profile"] = "medium"
        self.config["project_hand_predictions"] = self.config.get("hand_control") == "slider"
        self.rounds, self.rounds_completed = rounds, 0
        self.process = None
        self.between = None
        self.pending = False
        self.rounds_submitted = 0
        self.slow_after_seconds = slow_after_seconds
        self.slow_active = False
        self.context = mp.get_context("spawn")
        self.incoming = self.context.Queue(1)
        self.status = self.context.Queue(256)
        self.stop_event = StopFlag(self.context)
        self.report.update(mode="medium_continuous", rounds=rounds, replan_steps=replan_steps,
                           slow_after_seconds=slow_after_seconds)

    def poll(self, store=None, output=None):
        while True:
            try:
                event = self.status.get_nowait()
            except queue.Empty:
                break
            kind = event["event"]
            if kind == "control_tick_overrun":
                self.recorder.event(event)
                self.report["control_tick_overrun_count"] = self.report.get("control_tick_overrun_count", 0) + 1
                print(f"【阶段】【周期警告】第 {event['round']} 轮，帧 {event['sequence']}："
                      f"已确认提交，周期超时 {event['overrun_ms']:.3f} ms，调整调度后继续", flush=True)
                continue
            if kind == "stage":
                self.report["stage"] = event["stage"]
                print("【阶段】" + event["stage"], flush=True)
                continue
            if kind == "failed":
                print("【阶段】执行异常，停止设备：" + event["error"], flush=True)
                self.report.update(state="failed", error=event["error"], traceback=event.get("traceback"))
                self.save_report()
                raise RuntimeError(event["error"])
            if kind == "round_complete":
                self.rounds_completed = event["round"]
                self.recorder.event(event)
                print(f"【阶段】第 {self.rounds_completed}/{self.rounds} 轮已完成", flush=True)
            elif kind == "speed_profile_changed":
                self.slow_active = True
                self.report["speed_profile_change"] = event
                self.recorder.event(event)
                print(f"【阶段】第 {event['round']} 轮起切换慢速；动作开始后 "
                      f"{event['elapsed_seconds']:.3f} 秒", flush=True)
            elif kind in {"prefetch", "between"} and event["round"] == self.rounds_submitted:
                self.between = event
                self.pending = False
            elif kind == "round_started":
                self.pending = False
                self.between = None
            elif kind == "complete":
                print("【阶段】全部推理轮次完成，执行结束策略", flush=True)
                self.completed = 1
                self.report.update(state="complete", rounds_completed=self.rounds_completed)
                self.save_report()
        if self.process is not None and not self.process.is_alive() and not self.completed:
            raise RuntimeError("Continuous control process exited before completion")

    def save_report(self):
        (self.output / "continuous-report.json").write_text(json.dumps(self.report, indent=2) + "\n")

    def __call__(self, store, obs, metadata, directory):
        try:
            return self.consume(store, obs, metadata, directory)
        except BaseException as error:
            self.report.update(state="failed", error=f"{type(error).__name__}: {error}")
            self.close()
            raise

    def consume(self, store, obs, metadata, directory):
        self.poll()
        if self.completed or self.pending or self.rounds_submitted >= self.rounds or (self.process is not None and self.between is None):
            return
        round_number = self.rounds_submitted + 1
        start_target = None if self.process is None else np.asarray(self.between["target"])
        packet = self.codec.packb(obs)
        sent, wall = time.monotonic(), time.time()
        ages = [wall - value["stamp"] for value in metadata["samples"].values()]
        if not ages or any(not 0 <= age <= 0.07 for age in ages):
            return
        print(f"【阶段】第 {round_number}/{self.rounds} 轮：使用实时观测进行模型推理", flush=True)
        self.requested = True
        self.ws.send(packet)
        result = self.ws.recv(timeout=1.0 - max(ages))
        received = time.monotonic()
        if abs((time.time() - wall) - (received - sent)) > 0.05:
            raise ValueError("Host clock changed during inference")
        if isinstance(result, str):
            raise RuntimeError("Policy server returned an error: " + result)
        import uuid

        admission = Admission(
            sent - max(ages), sent, received, uuid.uuid4().hex, self.model["checkpoint_manifest_sha256"], 0
        )
        admission.check()
        actions = np.asarray(self.codec.unpackb(result)["actions"])
        self.poll()
        from .planner import build_plan
        from .planner import save_plan

        if self.process is None:
            start = self.devices.feedback(time.monotonic()).positions
        else:
            start = start_target
        config = dict(self.config, continuation=self.process is not None, inference_request_id=admission.request_id,
                      final_chunk=round_number == self.rounds)
        print("【阶段】推理返回，规划动作段", flush=True)
        round_dir = self.output / f"round-{round_number:04d}"
        round_dir.mkdir()
        # Preserve the full prediction even when planning rejects its prefix.
        np.savez_compressed(round_dir / "inference.npz", actions=actions,
                            state=obs["observation/state"], recorded_start=start)
        (round_dir / "admission.json").write_text(json.dumps(asdict(admission), indent=2) + "\n")
        try:
            if self.slow_active:
                plan = build_plan(actions, start, slow_motion_config(config), self.limits, self.names)
                player = OneShot(plan, admission, time.monotonic())
                slow_player = None
            else:
                plan = build_plan(actions, start, config, self.limits, self.names)
                player = OneShot(plan, admission, time.monotonic())
                slow_player = None
                if self.process is not None and self.slow_after_seconds is not None:
                    slow_plan = build_plan(actions, start, slow_motion_config(config), self.limits, self.names)
                    slow_player = OneShot(slow_plan, admission, time.monotonic())
        except ValueError as error:
            (round_dir / "planning-error.json").write_text(
                json.dumps({"error": str(error), "execution_steps": config.get("execution_steps", 50)}, indent=2) + "\n"
            )
            raise
        save_plan(plan, round_dir / "plan")
        if slow_player is not None:
            save_plan(slow_player.plan, round_dir / "plan-slow")
        projection = plan.report["hand_prediction_projection"]
        if projection["adjustment_count"]:
            print(f"【阶段】【手部预测警告】{projection['adjustment_count']} 个预测角度超出关节范围，"
                  f"已限制到合法边界；最大调整 {projection['max_adjustment_rad']:.4f} rad，继续执行", flush=True)
            self.recorder.event({"event": "hand_prediction_projected",
                                 "round": round_number, **projection})
        self.recorder.event(
            {
                "event": "round_admitted",
                "round": round_number,
                "plan_hash": player.digest,
                "alternative_slow_plan_hash": slow_player.digest if slow_player is not None else None,
                "admission": asdict(admission),
            }
        )
        if self.process is None:
            self.process = self.context.Process(
                target=_worker,
                args=(
                    self.devices.sock,
                    self.devices.counter,
                    self.devices.finish_policy,
                    player,
                    self.incoming,
                    self.status,
                    self.stop_event,
                    str(self.output),
                    self.rounds,
                    self.slow_after_seconds,
                ),
            )
            self.process.start()
            # Transfer the sole IPC connection to the motion process.
            self.devices.sock.close()
            self.devices.closed = True
        else:
            self.incoming.put_nowait(PlanAlternatives(player, slow_player) if slow_player is not None else player)
        self.pending = True
        self.rounds_submitted = round_number
        self.between = None
        print(f"【阶段】第 {round_number} 轮计划就绪：预测 50 步，执行前 {len(plan.raw)} 步", flush=True)

    def close(self):
        self.stop_event.set()
        if self.process is not None:
            self.process.join(5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2)
        if self.report.get("state") not in {"complete", "failed"}:
            self.report["state"] = "stopped"
        self.save_report()
