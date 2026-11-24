"""Fovea on the existing SDK stream, with observation-clock crop scheduling.

Only the owner installs actions/boxes. A background prediction has no authority
until its stage ticket and the stream receipt are accepted. This is inference-
time RTC guidance for the existing checkpoint, not an RTC-trained checkpoint.
"""

from contextlib import contextmanager
import math
import time

from .fovea_runtime import FoveaRuntime
from .rtc_stream import require_capabilities
from .task_rtc import TaskRTC

PROTOCOL = "fovea-observation-clock-v1"
PERIOD = 15 / 30
BOX_TOLERANCE = 2 / 30


def source_time(observation):
    """Map the actual head-camera stamp to the local monotonic clock."""
    age = observation.get("_alignment", {}).get("source_age_ms", {}).get("cam0")
    if not isinstance(age, (int, float)) or not math.isfinite(age) or not 0 <= age <= 100:
        raise ValueError("Fovea async requires a causal head-camera age within 100ms")
    captured = observation["_captured_monotonic_ns"] / 1e9
    return captured - age / 1000


class FoveaAsyncRuntime(FoveaRuntime):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_key = None
        self.next_crop = None
        self.prediction_source = None
        self.used_target = None

    def validate_metadata(self, metadata):
        super().validate_metadata(metadata)
        require_capabilities(metadata)
        if metadata.get("fovea_async_protocol") != PROTOCOL:
            raise ValueError("Matching warmed Fovea async service required")
        return metadata

    def reset_stage(self):
        super().reset_stage()
        self.next_crop = None
        self.prediction_source = None
        self.used_target = None

    def prepare_observation(self, observation):
        task = observation["_task_record"]
        key = (task["run_id"], task["stage_version"])
        if key != self.stage_key:
            self.reset_stage()
            self.stage_key = key
        source = source_time(observation)
        self.used_target = None
        if self.next_crop is not None:
            self.used_target = self.next_crop["target"]
            if abs(source - self.next_crop["target"]) > BOX_TOLERANCE:
                raise TimeoutError("Fovea future box missed its target image time; stop required")
            self.box = self.next_crop["box"].copy()
            self.query_index += 1
            self.next_crop = None
        elif self.query_index != 0:
            raise ValueError("Fovea async has no accepted crop for this observation")
        self.prediction_source = source

    def fovea_context(self):
        context = super().fovea_context()
        context.pop("executed_steps")
        return dict(context, execution_mode=PROTOCOL, nominal_observation_steps=15,
                    source_monotonic_s=self.prediction_source, used_target_monotonic_s=self.used_target)

    def accept_result(self, metadata, result, *, started=None):
        receipt = result["foveation"]
        if (receipt.get("execution_mode") != PROTOCOL
                or receipt.get("source_monotonic_s") != self.prediction_source
                or receipt.get("used_target_monotonic_s") != self.used_target):
            raise ValueError("Fovea async observation-clock receipt mismatch")
        return super().accept_result(metadata, result, started=started)

    def infer_chunk(self, observation, *, rtc_context=None, response_timeout=120):
        self.prepare_observation(observation)
        metadata, actions, chunk = super().infer_chunk(
            observation, rtc_context=rtc_context, response_timeout=response_timeout,
        )
        receipt = dict(metadata["foveation_query"], source_monotonic_s=self.prediction_source,
                       nominal_target_monotonic_s=self.prediction_source + PERIOD,
                       async_protocol=PROTOCOL)
        return {**metadata, "foveation_query": receipt}, actions, chunk

    def commit_installation(self, installed, *, initial_start):
        if self.pending is None or self.prediction_source is None:
            raise ValueError("No Fovea prediction to commit")
        target = self.prediction_source + PERIOD
        if initial_start:
            # The robot waited for the first inference. The t+15 training
            # target starts at motion start, not at the preceding held image.
            target = installed["sampled_monotonic_s"] - installed["phase"] / 30 + PERIOD
        if target <= time.monotonic() + 1 / 30:
            raise TimeoutError("Fovea prediction arrived too late for the next crop")
        self.next_crop = dict(target=target, box=self.pending.copy())
        self.pending = None
        return target


class FoveaAsyncTask(TaskRTC):
    """Reuse main's five-stage owner, streaming, handover, lease and shutdown."""

    def __init__(self, control, runtime, backend, observe, args, directory, **kwargs):
        if (control.profile != "tomato-in-hand119" or args.execute_steps != 15
                or args.execute_speed_scale != 1.0 or args.num_chunks != 0):
            raise ValueError("Async Fovea requires five stages, 30Hz, 15-step observations and continuous rounds")
        def fresh_observation(**options):
            timeout = min(float(options.get("timeout_s", 2.0)), 0.3)
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("No head-camera observation within 100ms arrived before the Fovea acquisition deadline")
                observation = observe(**{**options, "timeout_s": remaining})
                age = observation.get("_alignment", {}).get("source_age_ms", {}).get("cam0")
                if not isinstance(age, (int, float)) or not math.isfinite(age) or age < 0:
                    source_time(observation)
                if age <= 100:
                    return observation
                if options.get("on_poll"):
                    options["on_poll"]()

        super().__init__(control, runtime, backend, fresh_observation, args, directory, **kwargs)
        self.crop_target = None
        self.camera_lag = 0.0

    def start_prediction(self, actions, ticket, path, on_poll):
        @contextmanager
        def admission():
            with self.control.installation(ticket) as valid:
                if not valid:
                    raise InterruptedError("Fovea start superseded before worker dispatch")
                yield
        return self.backend.start(actions, ticket, path, on_poll=on_poll, admission=admission)

    def replan_due(self, status):
        # Independent of installation delay: resetting install+15 every time
        # would gradually move the camera queries away from the trained t+15.
        return self.crop_target is None or self.clock() >= self.crop_target + self.camera_lag

    def snapshot_for_observation(self, observation):
        return self.backend.worker.request(dict(
            operation="stream_snapshot", delay_steps=max(self.delays),
            schedule=self.settings.prefix_attention_schedule,
            weight=self.settings.max_guidance_weight,
            observed_at=source_time(observation),
        ), timeout=1)["context"]

    def submit_observed(self, observation, context):
        super().submit_observed(observation, context)
        self.first_deadline = min(self.first_deadline, source_time(observation) + PERIOD - 1 / 30)

    def predict(self, ticket, context, timeout, observation=None):
        if context is not None:
            timeout = min(timeout, source_time(observation) + PERIOD - self.clock() - 1 / 30)
        result = super().predict(ticket, context, timeout, observation)
        if context is not None and self.runtime.prediction_source + PERIOD <= self.clock() + 1 / 30:
            raise TimeoutError("Fovea inference missed the next observation deadline")
        return result

    def prediction_installed(self, metadata, installed, captured, ticket, *, initial_start):
        self.camera_lag = max(0.0, captured - self.runtime.prediction_source)
        startup_approach = initial_start and installed.get("startup_approach_steps", 0) > 0
        if startup_approach or installed.get("entry_bridge_steps", 0):
            # The bounded worker bridge altered the predicted trajectory. Reset
            # the crop session on the next observation rather than asserting
            # that the old t+15 prediction still describes that trajectory.
            self.runtime.reset_stage()
            self.crop_target = self.clock() + PERIOD
            self.record("fovea_crop_reset", reason=(
                "initial slow approach changed trajectory" if startup_approach
                else "RTC entry bridge changed trajectory"))
            return
        self.crop_target = self.runtime.commit_installation(installed, initial_start=initial_start)
        self.record("fovea_async_accepted", inference_request_id=ticket["request_id"],
                    request_stage_version=ticket["stage_version"],
                    foveation=metadata["foveation_query"], crop_target_monotonic_s=self.crop_target,
                    installation=installed)

    def prediction_rejected(self, installed):
        raise RuntimeError("Fovea async trajectory rejected: " + installed["reason"])

    def switch_applied(self):
        # The new-stage block traverses main's bounded trajectory bridge.
        # Its box was predicted before that bridge, so reacquire with the fixed
        # initial box and a fresh post-handover image, then resume guided RTC.
        self.runtime.reset_stage()
        self.crop_target = self.clock() + 0.1
        self.record("fovea_crop_reset", reason="new stage bridge applied")

    def run(self):
        self.control.data["rtc_mode"] = PROTOCOL
        return super().run()
