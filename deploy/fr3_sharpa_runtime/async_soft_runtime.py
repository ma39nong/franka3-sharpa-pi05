"""Pipeline fixed 20-target segments; inference overlaps the current segment."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import uuid

import numpy as np

from .async_soft_stream import handover_window


def run_async_soft(
    client, observe, worker, first, initial_observation, args, *, clock=time.monotonic, sleep=time.sleep
):
    directory = Path(args.execution_trace_dir).expanduser().resolve() / ("async-soft-" + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sharpa-async-soft")
    future = context = None
    accepted, stage = 1, "initial_prediction"
    started = clock()
    requested_generation = None

    def log(event, **fields):
        with (directory / "requests.jsonl").open("a") as stream:
            stream.write(json.dumps({"event": event, "monotonic_s": clock(), **fields}, allow_nan=False) + "\n")

    def save(status, **fields):
        result = {
            "status": status,
            "broker_mode": "async-soft",
            "accepted_chunks": accepted,
            "stage": stage,
            "elapsed_s": clock() - started,
            "compression": "cubic",
            "effective_horizon": 30,
            "execute_steps": 26,
            "execution_target_count": 20,
            "soft_reference_count": 4,
            "alignment": "trajectory_phase",
            "correlation": False,
            "rtc_guidance": False,
            **fields,
        }
        path = directory / "run.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        return {**result, "run_report": str(path)}

    def predict(observation, ctx):
        result = client.infer_chunk(observation, context=ctx, response_timeout=handover_window(ctx)[1] - clock())
        return result, client.last_receipt

    try:
        metadata, actions, diagnostic = first
        log(
            "initial_prediction",
            metadata=metadata,
            actions=actions[:30].tolist(),
            diagnostic=diagnostic,
            observation_state=np.asarray(initial_observation["observation/state"]).tolist(),
        )
        worker.request(
            {
                "operation": "stream_start",
                "actions": actions.tolist(),
                "prediction_clip_rad": args.prediction_clip_rad,
                "execute_speed_scale": args.execute_speed_scale,
                "approach_speed_rad_s": args.prepare_speed_rad_s,
                "terminal": args.num_chunks == 1,
                "events_path": str(directory / "control.jsonl"),
            }
        )
        while True:
            status = worker.request({"operation": "stream_status"})
            if status["finished"]:
                stopped = worker.request({"operation": "stream_stop"})
                return save("COMPLETE", completion="commands_sent", final_stream=stopped)
            if future is not None:
                if clock() >= context["handover_monotonic_s"]:
                    raise TimeoutError("Async Soft prediction missed fixed handover")
                if future.done():
                    (metadata, actions, diagnostic), acknowledgement = future.result()
                    installed = worker.request(
                        {
                            "operation": "stream_install",
                            "actions": actions.tolist(),
                            "prediction_clip_rad": args.prediction_clip_rad,
                            "receipt": acknowledgement,
                            "terminal": bool(args.num_chunks and accepted + 1 >= args.num_chunks),
                        }
                    )
                    accepted += 1
                    log(
                        "queued",
                        actions=actions[:30].tolist(),
                        receipt=acknowledgement,
                        installation=installed,
                        diagnostic=diagnostic,
                    )
                    future = context = None
                    stage = "playing"
            # Launch once per activated segment, immediately; never wait for 15 source nodes.
            if (
                future is None
                and not status["queued"]
                and status["generation"] != requested_generation
                and not (args.num_chunks and accepted >= args.num_chunks)
            ):
                stage = "observing"
                deadline = status["handover_monotonic_s"] - 0.01

                def poll(deadline=deadline):
                    worker.request({"operation": "stream_status"})
                    if clock() >= deadline:
                        raise TimeoutError("Async Soft observation missed fixed handover")

                poll()
                observation = observe(
                    timeout_s=deadline - clock(),
                    on_poll=poll,
                    on_diagnostic=lambda data: log("observation_diagnostic", data=data),
                )
                poll()
                captured = observation["_captured_monotonic_ns"] / 1e9
                if not 0 <= clock() - captured <= 0.2:
                    raise ValueError("Async Soft request observation is stale")
                context = worker.request({"operation": "stream_snapshot", "observation_monotonic_s": captured})[
                    "context"
                ]
                requested_generation = status["generation"]
                log("request", context=context, observation_state=np.asarray(observation["observation/state"]).tolist())
                future = pool.submit(predict, observation, context)
                stage = "soft_prediction"
            sleep(0.005)
    except BaseException as error:
        save(
            "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
            error=str(error) or type(error).__name__,
        )
        raise
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
