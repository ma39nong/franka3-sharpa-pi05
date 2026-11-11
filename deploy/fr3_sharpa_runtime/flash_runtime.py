"""Single in-flight FLASH inference overlapping the SDK worker's own clock."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import time
import uuid

import numpy as np

from .flash_contract import validate_flash_observation
from .flash_stream import FlashSettings, observation_anchor


def settings_from_args(args):
    return FlashSettings(args.flash_seam_position_rad, args.flash_seam_velocity_rad_s,
                         args.flash_feedback_error_rad, trigger_nodes=args.execute_steps)


def run_flash_async(runtime, observe, worker, first, initial_observation, args,
                    *, clock=time.monotonic, sleep=time.sleep):
    settings = settings_from_args(args)
    session = uuid.uuid4().hex
    directory = Path(args.execution_trace_dir).expanduser().resolve() / ("flashRTC-" + session)
    directory.mkdir(parents=True)
    accepted = 0
    future = None
    request = None
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flash-inference")
    stage = "prepare"
    started = clock()
    pipeline_ms = []
    rejected = 0
    consecutive_rejections = 0

    def record(event, **fields):
        with (directory / "requests.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps(dict(event=event, monotonic_s=clock(), **fields), allow_nan=False) + "\n")

    def save(status, **fields):
        report = dict(status=status, broker_mode="flashRTC", run_session_id=session,
                      accepted_chunks=accepted, stage=stage, elapsed_s=clock() - started,
                      settings=asdict(settings), **fields)
        report["rejected_candidates"] = rejected
        report["pipeline_latency_ms"] = ({str(p): float(np.percentile(pipeline_ms, p)) for p in (50, 95, 99)}
                                         if pipeline_ms else {})
        path = directory / "run.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        return {**report, "run_report": str(path)}

    def candidate(observation, deadline):
        # This path intentionally does NOT use Serial's response-age 200ms gate.
        # Fresh sensors are checked at capture; the worker validates future phase.
        if clock() >= deadline:
            raise TimeoutError("FLASH inference deadline expired before submission")
        result = runtime.infer_chunk(observation, response_timeout=deadline - clock(), deadline_monotonic=deadline)
        return result, observation

    def plan(observation, metadata):
        validate_flash_observation(observation)
        anchor = observation_anchor(observation, clock())
        return dict(anchor_s=anchor, model_identity=metadata["flash_manifest_sha256"],
                    history_session=observation["flash_context"]["session_id"],
                    request_id=observation["flash_context"]["request_id"],
                    flash_context=observation["flash_context"])

    try:
        metadata, actions, _ = first
        # Slow initial approach and recorder setup consume time. Use that initial
        # prediction only for approach, then discard it and reacquire EVERYTHING.
        prepared = worker.request(dict(operation="flash_prepare", actions=actions.tolist(),
            prediction_clip_rad=args.prediction_clip_rad, approach_speed_rad_s=args.prepare_speed_rad_s,
            plan=dict(anchor_s=(initial_observation["flash_context"]["captured_monotonic_ns"]
                                - initial_observation["flash_context"]["anchor_age_at_capture_ns"]) / 1e9,
                model_identity=metadata["flash_manifest_sha256"],
                history_session=initial_observation["flash_context"]["session_id"],
                request_id=initial_observation["flash_context"]["request_id"],
                flash_context=initial_observation["flash_context"], receipt=initial_observation["_flash_receipt"])))
        observation = observe()
        if observation["_captured_monotonic_ns"] < prepared["completed_monotonic_ns"]:
            raise ValueError("FLASH initial observation predates approach completion")
        identity = metadata["flash_manifest_sha256"]
        history_session = observation["flash_context"]["session_id"]
        initial_anchor = observation_anchor(observation, clock())
        (metadata, actions, _), observation = candidate(observation,
            initial_anchor + (49 - settings.min_remaining_nodes) / 30)
        if metadata["flash_manifest_sha256"] != identity:
            raise ValueError("FLASH model changed during prepare")
        # Freshness at response is phase-based; observation_anchor is only called
        # before submission. Never re-date the original observation here.
        initial_plan = dict(anchor_s=initial_anchor, model_identity=identity, history_session=history_session,
                            request_id=observation["flash_context"]["request_id"],
                            flash_context=observation["flash_context"], receipt=observation["_flash_receipt"])
        worker.request(dict(operation="flash_start", actions=actions.tolist(), plan=initial_plan,
            run_session_id=session, settings=asdict(settings), prediction_clip_rad=args.prediction_clip_rad,
            events_path=str(directory / "control.jsonl")))
        accepted = 1
        record("started", actions=actions.tolist(), plan=initial_plan,
               history=np.asarray(observation["observation/history"]).tolist(),
               state=np.asarray(observation["observation/state"]).tolist())
        stage = "playing"
        last_watchdog = 0.0
        installation = None
        observation = None
        while True:
            status = worker.request({"operation": "stream_status"})
            now = clock()
            remaining = status["remaining_s"] - (now - status["sampled_monotonic_s"])
            if remaining <= settings.reserve_s:
                raise TimeoutError("FLASH old trajectory reached stop reserve")

            def check_motion():
                current = worker.request({"operation": "stream_status"})
                if current["remaining_s"] <= settings.reserve_s:
                    raise TimeoutError("FLASH observation reached old trajectory stop reserve")

            # Independent camera/joint/history watchdog also runs while GPU waits.
            if now - last_watchdog >= 0.1:
                observation = observe(timeout_s=min(0.2, remaining - settings.reserve_s), on_poll=check_motion)
                observation_anchor(observation, clock())
                if observation["flash_context"]["session_id"] != history_session:
                    raise ValueError("FLASH history session changed during motion")
                last_watchdog = clock()
            if installation is not None and not status["pending_install"]:
                accepted += 1
                consecutive_rejections = 0
                record("accepted", installation=installation)
                installation = None
                stage = "playing"
            if future is not None and future.done():
                (new_metadata, actions, _), used = future.result()
                if new_metadata["flash_manifest_sha256"] != identity:
                    raise ValueError("FLASH model changed in flight")
                record("candidate", request=request, actions=actions.tolist(),
                       receipt=used["_flash_receipt"], timing=getattr(runtime, "last_timing", {}))
                installed = worker.request(dict(operation="flash_install", actions=actions.tolist(),
                    request=request, plan={**initial_plan, "anchor_s": request["anchor_s"],
                        "request_id": request["request_id"], "flash_context": used["flash_context"],
                        "receipt": used["_flash_receipt"]}, prediction_clip_rad=args.prediction_clip_rad))
                future = None
                request = None
                if installed.get("accepted"):
                    installation = installed["installation"]
                    pipeline_ms.append((installation["switch_s"] - request_started) * 1000)
                    stage = "pending_install"
                else:
                    rejected += 1
                    consecutive_rejections += 1
                    record("rejected", reason=installed["reason"])
                    if consecutive_rejections >= 2:
                        raise ValueError("FLASH candidate rejected twice; stopped")
                continue
            if future is not None and clock() >= request["deadline_s"]:
                raise TimeoutError("FLASH inference exceeded old trajectory deadline")
            if future is None and installation is None and status["consumed_since_install"] >= args.execute_steps:
                if args.num_chunks and accepted >= args.num_chunks:
                    worker.request({"operation": "stream_stop"})
                    return save("COMPLETE", completion="commands_sent")
                # Always a fresh immutable snapshot, never queue observations.
                observation = observe(timeout_s=min(0.2, remaining - settings.reserve_s), on_poll=check_motion)
                info = plan(observation, metadata)
                if info["history_session"] != history_session:
                    raise ValueError("FLASH history session changed")
                request = dict(run_session_id=session, source_generation=status["generation"],
                    source_sha256=status["source_sha256"], model_identity=identity, history_session=history_session,
                    request_id=info["request_id"], anchor_s=info["anchor_s"],
                    deadline_s=status["sampled_monotonic_s"] + status["remaining_s"] - settings.reserve_s)
                request_started = observation["flash_context"]["captured_monotonic_ns"] / 1e9
                worker.request(dict(operation="flash_snapshot", request=request))
                record("request", request=request, flash_context=observation["flash_context"],
                       history=np.asarray(observation["observation/history"]).tolist(),
                       state=np.asarray(observation["observation/state"]).tolist())
                future = pool.submit(candidate, observation, request["deadline_s"])
                stage = "infer_in_flight"
            sleep(0.005)
    except BaseException as error:
        save("INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", error=str(error))
        raise
    finally:
        # Caller owns SDK stop/disable/readback. A late Future has no worker access.
        if future is not None:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
