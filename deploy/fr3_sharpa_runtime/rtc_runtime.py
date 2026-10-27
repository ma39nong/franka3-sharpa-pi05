"""RTC coordinator. One inference may overlap a worker-owned 30 Hz stream."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import time
import uuid

import numpy as np

from .rtc_stream import RTCSettings


def run_rtc(runtime, observe, worker, first, initial_observation, args, *, clock=time.monotonic, sleep=time.sleep):
    settings = RTCSettings(
        args.rtc_initial_delay_steps,
        args.rtc_delay_buffer_size,
        args.rtc_prefix_attention_schedule,
        args.rtc_max_guidance_weight,
    )
    delays = settings.history()
    directory = Path(args.execution_trace_dir).expanduser().resolve() / ("rtc-" + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    records = directory / "requests.jsonl"
    run_path = directory / "run.json"
    accepted = 0
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sharpa-rtc-inference")
    future = None
    context = None
    started = clock()
    stage = "initial_prediction"
    last_diagnostic = None

    def record(**fields):
        with records.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"monotonic_s": clock(), **fields}, allow_nan=False) + "\n")

    def save(report):
        report.update(
            broker_mode="rtc",
            accepted_chunks=accepted,
            settings=asdict(settings),
            elapsed_s=clock() - started,
            events=str(directory / "control.jsonl"),
            stage=stage,
            last_observation_diagnostic=last_diagnostic,
        )
        run_path.write_text(json.dumps(report, indent=2) + "\n")
        return {**report, "run_report": str(run_path)}

    def predict(observation, context):
        # Bounded by the amount of OLD motion that is still valid. New actions
        # are not issued based on an arbitrarily old observation on completion.
        effective_hz = 30.0 / np.ceil(1.0 / args.execute_speed_scale)
        timeout = max(0.01, context["prefix_attention_horizon"] / effective_hz)
        result = runtime.infer_chunk(observation, rtc_context=context, response_timeout=timeout)
        return result, runtime.last_receipt

    try:
        metadata, actions, _ = first
        record(
            event="initial_prediction",
            metadata=metadata,
            observation_state=np.asarray(initial_observation["observation/state"]).tolist(),
            actions=actions.tolist(),
        )
        worker.request(
            {
                "operation": "stream_start",
                "actions": actions.tolist(),
                "prediction_clip_rad": args.prediction_clip_rad,
                "execute_speed_scale": args.execute_speed_scale,
                "approach_speed_rad_s": args.prepare_speed_rad_s,
                "events_path": str(directory / "control.jsonl"),
            }
        )
        accepted = 1
        stage = "playing"
        while True:
            status = worker.request({"operation": "stream_status"})
            if future is not None and future.done():
                (metadata, actions, _), receipt = future.result()
                installed = worker.request(
                    {
                        "operation": "stream_install",
                        "actions": actions.tolist(),
                        "prediction_clip_rad": args.prediction_clip_rad,
                        "receipt": receipt,
                    }
                )
                accepted += 1
                delays.append(installed["skipped"])
                record(event="accepted", actions=actions.tolist(), receipt=receipt, installation=installed)
                future = context = None
                stage = "playing"
                continue
            if future is None and status["consumed_since_install"] >= args.execute_steps:
                if args.num_chunks and accepted >= args.num_chunks:
                    stopped = worker.request({"operation": "stream_stop"})
                    return save(
                        {
                            "status": "COMPLETE",
                            "completion": "commands_sent",
                            "final_stream": stopped,
                            "checkpoint": metadata["checkpoint"],
                        }
                    )
                # Acquisition, encoding and IPC must fit inside OLD motion,
                # reserving the delay estimate plus one polling/install margin.
                # This is a stop deadline, not permission to relax freshness.
                effective_hz = 30.0 / np.ceil(1.0 / args.execute_speed_scale)
                remaining_s = ((50 - status["phase"]) / effective_hz
                               - max(0, clock() - status["sampled_monotonic_s"]))
                reserve_s = max(1, max(delays)) / effective_hz + 0.05
                observation_budget_s = remaining_s - reserve_s
                observation_started = clock()
                observation_deadline = observation_started + observation_budget_s
                stage = "observing"
                last_diagnostic = None
                record(event="observation_start", stream=status, remaining_s=remaining_s,
                       observation_budget_s=observation_budget_s, reserved_s=reserve_s)
                if observation_budget_s <= 0:
                    raise TimeoutError("RTC old chunk has insufficient time for a new observation and inference")
                motion_error = None

                def check_motion():
                    nonlocal motion_error
                    # A worker FAIL (e.g. exhausted chunk or feedback fault)
                    # takes priority over a later observation timeout.
                    try:
                        worker.request({"operation": "stream_status"})
                    except Exception as error:
                        motion_error = error
                        raise
                    if clock() >= observation_deadline:
                        raise TimeoutError("RTC observation deadline reached before old chunk exhaustion")

                def diagnostic(data):
                    nonlocal last_diagnostic
                    last_diagnostic = data
                    record(**data)

                try:
                    check_motion()
                    observation = observe(timeout_s=observation_deadline - clock(),
                                          on_poll=check_motion, on_diagnostic=diagnostic)
                except Exception:
                    # Check once more if a final ROS error and device failure
                    # arrived together; retain the device's original report.
                    if motion_error is None:
                        worker.request({"operation": "stream_status"})
                    raise
                check_motion()  # Includes decoding time after the IPC reply.
                record(event="observation_ready", elapsed_s=clock() - observation_started)
                age = time.monotonic_ns() - observation["_captured_monotonic_ns"]
                if not 0 <= age <= 200_000_000:
                    raise ValueError("RTC request observation is stale; stopped")
                context = worker.request(
                    {
                        "operation": "stream_snapshot",
                        "delay_steps": max(delays),
                        "schedule": settings.prefix_attention_schedule,
                        "weight": settings.max_guidance_weight,
                    }
                )["context"]
                record(
                    event="request",
                    context=context,
                    observation_state=np.asarray(observation["observation/state"]).tolist(),
                    captured_monotonic_ns=observation["_captured_monotonic_ns"],
                )
                future = pool.submit(predict, observation, context)
                stage = "guided_prediction"
            sleep(0.01)
    except BaseException as error:
        save(
            {
                "status": "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                "error": str(error) or type(error).__name__,
            }
        )
        raise
    finally:
        # The caller closes the SDK worker using existing disable/readback.
        # Socket response timeouts bound any already-running inference.
        pool.shutdown(wait=False, cancel_futures=True)
