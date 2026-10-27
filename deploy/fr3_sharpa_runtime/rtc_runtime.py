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
                observation = observe()
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
