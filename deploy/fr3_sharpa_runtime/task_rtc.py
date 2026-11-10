"""Seven-stage task coordinator using the existing ordinary RTC wire protocol.

No cross-stage prefix is reused. The production backend deliberately refuses
automatic rearming: the current hardware contract has no whole-robot hold ACK.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import time
import uuid

import numpy as np

from .protocol import as_finite_action_chunk
from .rtc_stream import RTCSettings


def safe_stop_proof(proof):
    return (
        isinstance(proof, dict)
        and proof.get("status") == "PASS"
        and proof.get("hands_disabled") is True
        and proof.get("arms_safe") is True
        and proof.get("old_stream_invalidated") is True
    )


class TaskRTC:
    def __init__(self, control, runtime, backend, observe, args, directory, *, clock=time.monotonic):
        self.control, self.runtime, self.backend = control, runtime, backend
        self.observe, self.args, self.clock = observe, args, clock
        self.directory = Path(directory)
        self.settings = RTCSettings(
            args.rtc_initial_delay_steps,
            args.rtc_delay_buffer_size,
            args.rtc_prefix_attention_schedule,
            args.rtc_max_guidance_weight,
        )
        self.delays = self.settings.history()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="task-policy")
        self.future = self.ticket = None
        self.streaming = False
        self.context = None
        self.chunk_count = 0
        self.first_deadline = None

    def record(self, event, **fields):
        self.control.event(event, **fields)

    def predict(self, ticket, context, timeout, observation=None):
        started = self.clock()

        def check():
            if not self.control.valid(ticket):
                raise RuntimeError("superseded before observation completed")

        if observation is None:
            observation = self.observe(timeout_s=min(timeout, 2.0), on_poll=check)
        observation = dict(observation, prompt=ticket["prompt"])
        check()
        age = time.monotonic_ns() - observation["_captured_monotonic_ns"]
        if not 0 <= age <= 200_000_000:
            raise ValueError("Task observation stale before inference")
        self.record(
            "inference_start",
            inference_request_id=ticket["request_id"],
            stage_id=ticket["stage_id"],
            request_stage_version=ticket["stage_version"],
            prompt=ticket["prompt"],
            observation_state=np.asarray(observation["observation/state"]).tolist(),
            captured_monotonic_ns=observation["_captured_monotonic_ns"],
            rtc_guidance=context is not None,
            prefix_length=len(context["previous_actions"]) if context else 0,
        )
        try:
            result = self.runtime.infer_chunk(observation, rtc_context=context, response_timeout=timeout)
        except Exception as error:
            self.record(
                "inference_failed",
                inference_request_id=ticket["request_id"],
                request_stage_version=ticket["stage_version"],
                reason=str(error),
                duration_s=self.clock() - started,
            )
            raise
        actions = as_finite_action_chunk(result[1])
        if context is None and not 0 <= time.monotonic_ns() - observation["_captured_monotonic_ns"] <= 200_000_000:
            raise TimeoutError("Fresh first chunk expired during inference; execution blocked")
        self.record(
            "inference_complete",
            inference_request_id=ticket["request_id"],
            request_stage_version=ticket["stage_version"],
            duration_s=self.clock() - started,
            actions=actions.tolist(),
        )
        return result[0], actions, self.runtime.last_receipt

    def submit(self, context=None, timeout=5.0):
        self.ticket = self.control.ticket()
        self.context = context
        self.first_deadline = self.clock() + timeout + 2.0
        self.control.data["inference_status"] = "guided_prediction" if context else "fresh_first_prediction"
        self.future = self.pool.submit(self.predict, self.ticket, context, timeout)

    def stop_boundary(self):
        # close() interrupts motion and waits for disable/readback. Never equate
        # stream_stop (only a software timeline operation) with physical holding.
        proof = self.backend.stop()
        self.streaming = False
        self.context = None
        self.delays = self.settings.history()
        self.control.data.update(active_chunk_id=None, chunk_progress=None)
        self.record("safe_stop_result", proof=proof)
        if not safe_stop_proof(proof):
            raise RuntimeError(
                "安全边界未确认：双手失能与双臂安全停止均需确认；禁止自动恢复。"
                "请等待完整清理后，人工检查设备并重新核对启动。"
            )
        return proof

    def tick(self):
        c = self.control
        snapshot = c.snapshot()
        state = snapshot["execution_state"]
        if state in ("STOPPED", "COMPLETED", "FAULT"):
            return False
        if self.clock() - c.last_contact > 5 and state != "STOPPING":
            with c.lock:
                c.data.update(stage_version=c.data["stage_version"] + 1, safe_to_resume=False)
                c.transition("STOPPING", "UI/controller lease expired; stopping hardware")
            state = "STOPPING"
        if state in ("SWITCH_REQUESTED", "SWITCHING", "PAUSING", "STOPPING"):
            with c.lock:
                if c.data["execution_state"] == "SWITCH_REQUESTED":
                    c.transition("SWITCHING", "invalidate stream and all outstanding contexts")
            self.stop_boundary()
            with c.lock:
                # Stop has priority even if it arrived while disable/readback blocked.
                if c.data["execution_state"] == "STOPPING":
                    c.data["cleanup_confirmed"] = True
                    c.transition("COMPLETED" if c.data.get("finish_requested") else "STOPPED", "verified stop")
                    return False
                c.data.update(safe_to_resume=True, request_status="awaiting_reauthorization")
                c.transition("PAUSED", "verified boundary; operator must explicitly authorize fresh execution")
            return True
        # An old running GPU call may finish later. Always reap before reusing
        # the sole inference slot, discard errors as well as outputs if stale.
        if self.future is not None and self.future.done():
            future, ticket, context = self.future, self.ticket, self.context
            self.future = None
            with c.installation(ticket) as valid:
                if not valid:
                    self.record(
                        "inference_discarded",
                        inference_request_id=ticket["request_id"],
                        request_stage_version=ticket["stage_version"],
                        reason="stage/state invalidated",
                    )
                    c.data["inference_status"] = "discarded"
                    return True
                metadata, actions, receipt = future.result()
                if context is None:
                    installed = self.backend.start(
                        actions, ticket, self.directory / f"control-{uuid.uuid4().hex}.jsonl"
                    )
                    self.streaming = True
                else:
                    installed = self.backend.install(actions, receipt, ticket)
                    self.delays.append(installed["skipped"])
                self.chunk_count += 1
                c.activate(ticket, f"{ticket['stage_version']}:{installed['generation']}")
                c.data["inference_status"] = "idle"
                self.record(
                    "inference_accepted",
                    inference_request_id=ticket["request_id"],
                    installation=installed,
                    checkpoint=metadata["checkpoint"],
                    rtc_guidance=context is not None,
                )
            return True
        if self.future is not None and self.clock() > self.first_deadline:
            raise TimeoutError("Task inference/observation deadline exceeded")
        if state == "RESUMING" and self.future is None:
            self.backend.reauthorize()
            self.submit()
        elif state == "RUNNING":
            status = self.backend.status()
            with c.lock:
                c.data["chunk_progress"] = status
            if self.future is None and status["consumed_since_install"] >= self.args.execute_steps:
                if self.args.num_chunks and self.chunk_count >= self.args.num_chunks:
                    c.command("pause", uuid.uuid4().hex, expected_version=c.data["stage_version"])
                    return True
                remaining = (50 - status["phase"]) / 30 - max(0, self.clock() - status["sampled_monotonic_s"])
                budget = remaining - max(1, max(self.delays)) / 30 - 0.05
                if budget <= 0:
                    raise TimeoutError("RTC old chunk has insufficient observation/inference budget")
                # Observe first, then snapshot, exactly as the ordinary RTC path.
                # Snapshot reflects the timeline at observation completion.
                # Only the owner thread can read/modify the hardware timeline.
                observed_version = c.snapshot()["stage_version"]

                def check_motion():
                    self.backend.status()
                    if c.snapshot()["stage_version"] != observed_version:
                        raise InterruptedError("stage request during observation")

                try:
                    observation = self.observe(timeout_s=budget, on_poll=check_motion)
                    check_motion()
                except InterruptedError:
                    return True
                context = self.backend.snapshot(self.settings, max(self.delays))
                # Preserve NEW observation as Delta reference. Only one submit
                # can consume it; no shared args.prompt mutation.
                self.submit_observed(observation, context)
        return True

    def submit_observed(self, observation, context):
        # The prediction job receives its own acquisition closure. No global
        # observation callback is changed while another Future could use it.
        ticket = self.control.ticket()
        self.ticket, self.context = ticket, context
        self.first_deadline = self.clock() + context["prefix_attention_horizon"] / 30 + 0.1
        self.control.data["inference_status"] = "guided_prediction"
        self.future = self.pool.submit(
            self.predict, ticket, context, max(0.01, context["prefix_attention_horizon"] / 30), observation
        )

    def run(self):
        try:
            self.control.ready()
            while self.tick():
                time.sleep(0.01)
        except KeyboardInterrupt:
            with self.control.lock:
                self.control.data.update(stage_version=self.control.data["stage_version"] + 1, safe_to_resume=False)
                if self.control.data["execution_state"] not in ("STOPPING", "STOPPED", "COMPLETED"):
                    self.control.transition("STOPPING", "execution stop signal; awaiting verified cleanup")
            raise
        except BaseException as error:
            self.control.fault(error)
            raise
        finally:
            if self.future is not None:
                ticket = dict(self.ticket)

                def discarded_on_exit(done):
                    try:
                        self.record(
                            "inference_discarded",
                            inference_request_id=ticket["request_id"],
                            request_stage_version=ticket["stage_version"],
                            reason="owner exited; no installation authority",
                            cancelled=done.cancelled(),
                        )
                    except Exception:
                        pass  # Audit failure must never prevent device disable/readback below.

                self.future.add_done_callback(discarded_on_exit)
            try:
                if self.control.data["execution_state"] not in ("PAUSED", "STOPPED", "COMPLETED"):
                    self.stop_boundary()
            except Exception as error:
                self.control.fault(error)
            self.pool.shutdown(wait=False, cancel_futures=True)
            report = {
                "status": self.control.snapshot(),
                "settings": asdict(self.settings),
                "accepted_chunks": self.chunk_count,
                "safety_boundary": "requires hands_disabled AND arms_safe acknowledgement",
            }
            (self.directory / "task-run.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


class SDKTaskBackend:
    """Only the task coordinator touches this worker. No IPC thread sends motion."""

    def __init__(self, worker, args):
        self.worker, self.args = worker, args
        self.stopped = False
        self.started = False
        self.proof = None

    def reauthorize(self):
        if self.stopped:
            raise RuntimeError("SDK 已失能；必须结束本次运行并重新核对完整部署方案")

    def start(self, actions, ticket, events_path):
        result = self.worker.request(
            dict(
                operation="stream_start",
                actions=actions.tolist(),
                prediction_clip_rad=self.args.prediction_clip_rad,
                execute_speed_scale=self.args.execute_speed_scale,
                approach_speed_rad_s=self.args.prepare_speed_rad_s,
                events_path=str(events_path),
                task_epoch=ticket["stage_version"],
                task_request_id=ticket["request_id"],
                strict_start=True,
            ),
            timeout=2,
        )
        self.started = True
        return result

    def status(self):
        return self.worker.request(dict(operation="stream_status"), timeout=1)

    def snapshot(self, settings, delay):
        return self.worker.request(
            dict(
                operation="stream_snapshot",
                delay_steps=delay,
                schedule=settings.prefix_attention_schedule,
                weight=settings.max_guidance_weight,
            ),
            timeout=1,
        )["context"]

    def install(self, actions, receipt, ticket):
        return self.worker.request(
            dict(
                operation="stream_install",
                actions=actions.tolist(),
                receipt=receipt,
                prediction_clip_rad=self.args.prediction_clip_rad,
                task_epoch=ticket["stage_version"],
                task_request_id=ticket["request_id"],
            ),
            timeout=1,
        )

    def stop(self):
        if not self.stopped:
            self.stopped = True
            self.worker.close()
            report = getattr(self.worker, "shutdown_report", None)
            self.proof = dict(
                status="BLOCKED",
                hands_disabled=bool(report and report.get("status") == "PASS"),
                arms_safe=False,
                old_stream_invalidated=True,
                hand_shutdown=report,
                reason="FR3 gateway has no verified physical hold/stop acknowledgement",
            )
        return self.proof
