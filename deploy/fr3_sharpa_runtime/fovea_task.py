"""Execution-owned five-stage Fovea coordinator. No RTC or Agent motion authority.

Requests arriving during a chunk are applied after its confirmed 15 steps.
New stages reset the crop session; stale inference never gains dispatch authority.
"""

from contextlib import contextmanager
import json
from pathlib import Path
import time

from .execution_trace import ExecutionTraceWriter
from .shutdown import protect_disable_confirmation
from .task_rtc import SDKTaskBackend
from .task_rtc import safe_stop_proof


class SupersededPrediction(Exception):
    pass


class FoveaTaskBackend(SDKTaskBackend):
    def execute(self, actions, *, admission, on_poll):
        self.reauthorize()
        return self.worker.request(
            dict(operation="chunk", actions=actions.tolist(), execute_steps=15,
                 prediction_clip_rad=self.args.prediction_clip_rad,
                 execute_speed_scale=1.0, approach_speed_rad_s=self.args.prepare_speed_rad_s),
            timeout=180, on_poll=on_poll, admission=admission,
        )


class FoveaTask:
    def __init__(self, control, runtime, backend, observe, args, directory):
        if (control.profile != "tomato-in-hand119" or args.execute_steps != 15
                or args.execute_speed_scale != 1.0 or args.num_chunks != 0):
            raise ValueError("Fovea task requires five stages, continuous rounds and native 30Hz/15-step execution")
        self.control, self.runtime, self.backend = control, runtime, backend
        self.observe, self.args = observe, args
        self.directory = Path(directory)
        self.trace = ExecutionTraceWriter(self.directory)
        self.stage_key = None
        self.completed = None
        self.chunk_count = 0

    def check_stop(self):
        status = self.control.snapshot()["execution_state"]
        if status in ("PAUSING", "STOPPING", "FAULT", "STOPPED", "COMPLETED"):
            raise InterruptedError("Fovea task interrupted by stop/pause/fault")
        if time.monotonic() - self.control.last_contact > 5:
            raise TimeoutError("Fovea controller lease expired")

    def capture(self, ticket):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            self.check_stop()
            observation = self.observe(timeout_s=min(2, deadline - time.monotonic()), on_poll=self.check_stop)
            if not self.control.valid(ticket):
                raise SupersededPrediction()
            if self.completed is not None:
                stamps = observation.get("_topic_stamp_ns", {})
                if (len(stamps) != 7 or min(stamps.values()) < self.completed["completed_ros_ns"]
                        or observation["_captured_monotonic_ns"] < self.completed["completed_monotonic_ns"]):
                    continue
            return dict(observation, prompt=ticket["prompt"], _task_record=ticket)
        raise TimeoutError("No fresh post-chunk Fovea observation arrived")

    @staticmethod
    def require_fresh(observation):
        if not 0 <= time.monotonic_ns() - observation["_captured_monotonic_ns"] <= 200_000_000:
            raise TimeoutError("Fovea observation expired; chunk dispatch blocked")

    def tick(self):
        c = self.control
        snapshot = c.snapshot()
        state = snapshot["execution_state"]
        if state in ("STOPPED", "COMPLETED", "FAULT"):
            return False
        if state in ("PAUSING", "STOPPING"):
            proof = self.backend.stop()
            c.event("safe_stop_result", proof=proof)
            if not safe_stop_proof(proof):
                raise RuntimeError("Fovea停止未取得双臂安全停止确认；禁止自动恢复，请结束运行并人工检查")
            with c.lock:
                if c.data["execution_state"] == "STOPPING":
                    c.data["cleanup_confirmed"] = True
                    c.transition("COMPLETED" if c.data.get("finish_requested") else "STOPPED", "verified stop")
                    return False
                c.data["safe_to_resume"] = True
                c.transition("PAUSED", "verified stop; explicit reauthorization required")
            return True
        if state in ("IDLE", "READY", "PAUSED"):
            return True
        self.check_stop()
        with c.lock:
            if c.data["execution_state"] == "SWITCH_REQUESTED":
                c.transition("SWITCHING", "previous fifteen-step chunk finished; applying new stage")
            if c.data["execution_state"] not in ("RUNNING", "RESUMING", "SWITCHING"):
                return True
            key = (c.data["requested_stage"], c.data["stage_version"])
            if key != self.stage_key:
                self.runtime.reset_stage()
                self.stage_key = key
                c.event("fovea_stage_reset", requested_stage=key[0], crop_session=self.runtime.session)
            ticket = c.ticket()
            c.data["inference_status"] = "fovea_prediction"
        path = None
        try:
            observation = self.capture(ticket)
            self.require_fresh(observation)
            metadata, actions, chunk = self.runtime.infer_chunk(observation, response_timeout=1)
            c.event("fovea_inference", inference_request_id=ticket["request_id"],
                    request_stage_version=ticket["stage_version"],
                    stage_id=ticket["stage_id"], prompt=ticket["prompt"],
                    foveation=metadata["foveation_query"])
            self.require_fresh(observation)

            @contextmanager
            def admission():
                with c.installation(ticket) as valid:
                    if not valid:
                        raise SupersededPrediction()
                    self.check_stop()
                    self.require_fresh(observation)
                    yield

            path = self.trace.begin_chunk(chunk_index=self.chunk_count, metadata=metadata,
                                          observation_before=observation, actions=actions,
                                          adapted_chunk=chunk, execute_steps=15)
            started = time.monotonic()
            report = self.backend.execute(actions, admission=admission, on_poll=self.check_stop)
            if (report.get("status") != "PASS" or report.get("execute_steps") != 15
                    or report.get("policy_frames") != 15
                    or not isinstance(report.get("completed_ros_ns"), int)
                    or not isinstance(report.get("completed_monotonic_ns"), int)):
                raise ValueError("Fovea requires a confirmed complete fifteen-step chunk")
            # This acknowledges physical completion even when a newer stage
            # request arrived during motion. The following stage then resets it.
            self.runtime.acknowledge_execution(15)
            self.completed = report
            self.chunk_count += 1
            self.trace.finish_chunk(path, observation_after=None,
                                    command_duration_s=time.monotonic()-started,
                                    end_to_end_duration_s=time.monotonic()-started, execution_report=report)
            with c.installation(ticket) as valid:
                if valid:
                    # Conservative activation: only after the first 15 commands
                    # of this prompt have actually completed, never at request time.
                    c.activate(ticket, f"fovea-{self.chunk_count}")
                    c.data["inference_status"] = "idle"
            c.event("fovea_chunk_complete", inference_request_id=ticket["request_id"],
                    request_stage_version=ticket["stage_version"],
                    source_stage=ticket["stage_id"], completed_steps=15)
        except SupersededPrediction:
            self.runtime.pending = None
            c.event("inference_discarded", inference_request_id=ticket["request_id"],
                    request_stage_version=ticket["stage_version"], reason="stage superseded before dispatch")
            if path is not None:
                self.trace.fail_chunk(path, error="stage superseded", stage="discarded_before_dispatch")
        except BaseException as error:
            if path is not None:
                self.trace.fail_chunk(path, error=str(error), stage="fovea_task")
            raise
        return True

    def run(self):
        try:
            self.control.data["rtc_mode"] = "fovea_serial_15"
            self.control.ready()
            while True:
                self.control.wakeup.clear()
                if not self.tick():
                    break
                self.control.wakeup.wait(0.01)
        except BaseException as error:
            self.control.fault(error)
            raise
        finally:
            # One SDK owner; no Agent/IPC callback may send motion or resume it.
            with protect_disable_confirmation():
                proof = self.backend.stop()
                self.trace.finish_run({"status": self.control.snapshot(), "accepted_chunks": self.chunk_count,
                                       "shutdown": proof})
                (self.directory / "task-run.json").write_text(json.dumps(
                    {"status": self.control.snapshot(), "accepted_chunks": self.chunk_count,
                     "mode": "fovea_serial_15", "shutdown": proof}, ensure_ascii=False, indent=2))
