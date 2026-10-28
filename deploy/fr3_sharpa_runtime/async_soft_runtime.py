"""Independent coordinator: old trajectory plays while Soft inference runs."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np

from .async_soft_stream import handover_window


def run_async_soft(
    client, observe, worker, first, initial_observation, args, *, clock=time.monotonic, sleep=time.sleep
):
    delays = deque([args.soft_initial_delay_steps], maxlen=args.soft_delay_buffer_size)
    directory = Path(args.execution_trace_dir).expanduser().resolve() / ("async-soft-" + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sharpa-async-soft")
    future = ready = context = None
    accepted, stage = 0, "initial_prediction"
    started = clock()
    hz = 30.0 / math.ceil(1.0 / args.execute_speed_scale)

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
            "node_hz": hz,
            "correlation": False,
            "compression": False,
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
            actions=actions.tolist(),
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
                "events_path": str(directory / "control.jsonl"),
            }
        )
        accepted, stage = 1, "playing"
        while True:
            status = worker.request({"operation": "stream_status"})
            if future is not None:
                earliest, latest = handover_window(context)
                if clock() > latest:
                    raise TimeoutError("Async Soft prediction missed handover window")
                if ready is None and future.done():
                    ready = future.result()
                    inference_nodes = math.ceil(max(0, clock() - context["snapshot_monotonic_s"]) * hz)
                    log("prediction_ready", inference_nodes=inference_nodes, diagnostic=ready[0][2])
                if ready is not None and clock() >= earliest:
                    (metadata, actions, diagnostic), acknowledgement = ready
                    installed = worker.request(
                        {
                            "operation": "stream_install",
                            "actions": actions.tolist(),
                            "prediction_clip_rad": args.prediction_clip_rad,
                            "receipt": acknowledgement,
                        }
                    )
                    accepted += 1
                    delays.append(max(1, min(20, inference_nodes)))
                    log("accepted", actions=actions.tolist(), receipt=acknowledgement, installation=installed)
                    future = ready = context = None
                    stage = "playing"
                    continue
            if future is None and status["consumed_since_install"] >= args.execute_steps:
                if args.num_chunks and accepted >= args.num_chunks:
                    stopped = worker.request({"operation": "stream_stop"})
                    return save("COMPLETE", completion="commands_sent", final_stream=stopped)
                stage = "observing"
                # Reserve enough original-model nodes for delay + 4-point overlap + IPC margin.
                remaining = (49 - status["phase"]) / hz - max(0, clock() - status["sampled_monotonic_s"])
                deadline = clock() + remaining - (max(delays) + 4) / hz - 0.05

                def poll(deadline=deadline):
                    worker.request({"operation": "stream_status"})
                    if clock() >= deadline:
                        raise TimeoutError("Async Soft observation deadline reached")

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
                context = worker.request(
                    {
                        "operation": "stream_snapshot",
                        "delay_steps": max(delays),
                        "observation_monotonic_s": captured,
                    }
                )["context"]
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
        # The execution owner always closes the unchanged disable/readback lifecycle.
        pool.shutdown(wait=False, cancel_futures=True)
