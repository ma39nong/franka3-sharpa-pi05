"""Read-only seven-stage experiment analysis. No ROS, SDK, policy or robot imports.

Missing evidence is None, never zero. Clock domains are explicit host+boot
assertions, not inferred by comparing numeric timestamps or file locations.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .task_stages import STAGES

GROUPS = {"left_arm": (0, 7), "left_hand": (7, 29), "right_arm": (29, 36), "right_hand": (36, 58)}
STAGE_MAP = {s.id: s for s in STAGES}
METRICS = ("click_to_registered_ms", "registered_to_observation_ms", "old_future_wait_ms",
           "new_prompt_inference_ms", "inference_to_armed_ms", "bridge_start_to_applied_ms",
           "click_to_applied_ms", "registered_to_applied_ms", "observation_age_at_inference_ms",
           "observation_age_at_result_ms", "observation_age_at_armed_ms")
TERMINAL = {"FAULT", "STOPPING", "STOPPED", "COMPLETED"}
KEEP = {"event", "run_id", "stage_version", "request_stage_version", "request_id", "inference_request_id",
        "old_inference_request_id", "target_stage", "previous_stage", "requested_stage", "active_stage",
        "operator_action", "execution_state", "previous_state", "reason", "cancelled_before_start", "prompt",
        "rtc_guidance", "monotonic_ns", "monotonic_s", "unix_ns", "client_clicked_unix_ms",
        "ipc_received_monotonic_ns", "http_received_monotonic_ns", "captured_monotonic_ns",
        "observation_age_ms", "policy_duration_s", "observation_timing", "alignment", "topic_stamp_ns",
        "boundary", "installation", "stream", "token", "applied_token", "generation", "phase",
        "task_epoch", "bridge_start", "bridge_end", "observation_age_s", "skipped", "handoff_state",
        "handoff_token", "command", "measured", "source_kind", "simulation_only", "checkpoint",
        "clock_domain", "proof", "trajectory_metrics"}


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def stat(values):
    a = np.asarray([x for x in values if number(x)], dtype=float)
    if not a.size:
        return dict(n=0, mean=None, rmse=None, max=None, p50=None, p95=None, p99=None)
    scale = float(np.max(np.abs(a))) or 1.0
    normalized = a/scale
    return dict(n=int(a.size), mean=float(normalized.mean()*scale), rmse=float(np.sqrt(np.mean(normalized**2))*scale),
                max=float(a.max()), **{f"p{p}": float(np.percentile(normalized, p)*scale) for p in (50, 95, 99)})


def issue(issues, code, source, line=None, detail=None):
    issues.append(dict(code=code, source=str(source), line=line, detail=detail))


def declared_domain(manifest, relative):
    matches = {v for k, v in manifest.get("clock_domains", {}).items()
               if fnmatch.fnmatch(relative, k) and isinstance(v, str) and v}
    return next(iter(matches)) if len(matches) == 1 else None


def read_jsonl(path, root, manifest, issues, inventory):
    relative = path.relative_to(root).as_posix()
    digest, result, previous, segment = hashlib.sha256(), [], None, 0
    domain = declared_domain(manifest, relative)
    matches = {v for k, v in manifest.get("clock_domains", {}).items()
               if fnmatch.fnmatch(relative, k) and isinstance(v, str) and v}
    if len(matches) > 1:
        issue(issues, "ambiguous_clock_declaration", relative)
    previous_domain, previous_offset, wall_epoch = domain, None, 0
    with path.open("rb") as stream:
        for line, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                e = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
                if not isinstance(e, dict) or not isinstance(e.get("event"), str):
                    raise ValueError("event object/name missing")
            except (ValueError, UnicodeError) as error:
                issue(issues, "malformed_jsonl", relative, line, str(error))
                segment += 1  # Do not differentiate across a missing/corrupt frame.
                previous = None
                continue
            e = {k: v for k, v in e.items() if k in KEEP}
            for key in ("boundary", "installation", "stream", "alignment", "observation_timing"):
                if key in e and not isinstance(e[key], dict):
                    issue(issues, "invalid_field_type", relative, line, key)
                    e[key] = {}
            for key in ("run_id", "request_id", "inference_request_id", "old_inference_request_id", "token", "handoff_token", "target_stage", "active_stage", "execution_state", "previous_state", "previous_stage", "requested_stage", "operator_action"):
                if e.get(key) is not None and not isinstance(e[key], str):
                    issue(issues, "invalid_identity_field", relative, line, key)
                    e[key] = None
            for key in ("stage_version", "request_stage_version", "task_epoch", "generation"):
                if e.get(key) is not None and (not isinstance(e[key], int) or isinstance(e[key], bool)):
                    issue(issues, "invalid_identity_field", relative, line, key)
                    e[key] = None
            t = e.get("monotonic_s")
            if not number(t) and number(e.get("monotonic_ns")):
                t = e["monotonic_ns"] / 1e9
            if not number(t) or t < 0:
                issue(issues, "missing_or_invalid_monotonic", relative, line)
                t = None
                segment += 1
            elif previous is not None and t < previous:
                segment += 1
                issue(issues, "clock_regression_or_out_of_order", relative, line)
            previous = t
            embedded_domain = e.get("clock_domain")
            effective = domain
            if embedded_domain is not None:
                if not isinstance(embedded_domain, str) or (domain and embedded_domain != domain):
                    issue(issues, "clock_domain_conflict", relative, line)
                    effective = None
                    segment += 1
                else:
                    effective = embedded_domain
            if effective != previous_domain:
                segment += 1
                issue(issues, "clock_domain_changed", relative, line)
            previous_domain = effective
            if t is not None and number(e.get("unix_ns")):
                offset = e["unix_ns"]/1e9 - t
                if previous_offset is not None and abs(offset-previous_offset) > .1:
                    segment += 1
                    wall_epoch += 1
                    issue(issues, "wall_or_monotonic_discontinuity", relative, line)
                previous_offset = offset
            e.update(_file=relative, _line=line, _t=t, _segment=segment, _domain=effective, _wall_epoch=wall_epoch)
            result.append(e)
    inventory.append(dict(path=relative, sha256=digest.hexdigest(), events=len(result), clock_domain=domain))
    return result


def read_object(path, issues):
    try:
        obj = json.loads(path.read_text(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
        if not isinstance(obj, dict):
            raise ValueError("expected object")
        return obj
    except (OSError, ValueError) as error:
        issue(issues, "invalid_json_object", path.name, detail=str(error))
        return {}


def delta(a, b, issues, label):
    """Only compare same-source monotonic segments or verified host/boot domains."""
    if not a or not b or a.get("_t") is None or b.get("_t") is None:
        return None
    same_file = a["_file"] == b["_file"]
    comparable = (a["_segment"] == b["_segment"]) if same_file else (
        bool(a.get("_domain")) and a["_domain"] == b.get("_domain")
        and a["_segment"] == 0 and b["_segment"] == 0)
    if not comparable:
        issue(issues, "incomparable_clock", label)
        return None
    # Preserve integer-ns precision for original task events (legacy output too).
    ans, bns = a.get("monotonic_ns"), b.get("monotonic_ns")
    if number(ans) and number(bns) and ans/1e9 == a["_t"] and bns/1e9 == b["_t"]:
        value = (bns-ans)/1e6
    else:
        value = (b["_t"] - a["_t"]) * 1000
    if not math.isfinite(value):
        issue(issues, "nonfinite_interval", label)
        return None
    if value < -1e-5:
        issue(issues, "negative_interval", label, detail=value)
        return None
    return max(0.0, value)


def embedded_stamp(event, value, producer, manifest):
    """A copied worker/ROS timestamp retains its PRODUCER clock, not its carrier's."""
    domain = manifest.get("embedded_clock_domains", {}).get(producer)
    if not event or not number(value) or value < 0 or not isinstance(domain, str) or not domain:
        return None
    return {**event, "_file": event["_file"]+"#"+producer, "_t": float(value), "_domain": domain}


def first(events, name):
    return next((e for e in events if e["event"] == name), None)


def metric_map(frames):
    result = {}
    for name, (a, b) in GROUPS.items():
        result[name] = {}
        for key in ("error", "command_velocity", "command_acceleration", "command_jerk",
                    "measured_velocity", "measured_acceleration", "measured_jerk"):
            arrays = [f[key][a:b] for f in frames if f.get(key) is not None]
            values = np.abs(np.concatenate(arrays)).tolist() if arrays else []
            result[name][key] = stat(values)
    return result


def prepare_frames(events, gap, issues):
    """Irregular-time Newton differences, independently for command/feedback.

    Missing channels break their own derivative history. No interpolation,
    filtering, fixed-30Hz substitution or bridging a clock/gap anomaly.
    """
    frames, history, previous_segment = [], {"command": [], "measured": []}, None
    previous_t = None
    for e in events:
        if e["event"] != "frame":
            continue
        if e["_t"] is None:
            history, previous_t = {"command": [], "measured": []}, None
            continue
        channels = {}
        for key in ("command", "measured"):
            try:
                values = np.asarray(e.get(key), float)
                if values.shape != (58,) or not np.isfinite(values).all():
                    raise ValueError(f"{key} requires finite 58D")
                channels[key] = values
            except (TypeError, ValueError) as error:
                issue(issues, "invalid_frame", e["_file"], e["_line"], str(error))
                channels[key] = None
        dt = e["_t"]-previous_t if previous_t is not None else None
        if previous_segment != e["_segment"] or (dt is not None and (dt <= 0 or dt > gap)):
            if dt is not None and (dt <= 0 or dt > gap):
                issue(issues, "frame_gap_or_nonpositive_dt", e["_file"], e["_line"], dt)
            history = {"command": [], "measured": []}
        f = {**e, **channels, "error": None, "dt_s": dt}
        if all(channels[k] is not None for k in channels):
            with np.errstate(over="ignore", invalid="ignore"):
                error = channels["command"]-channels["measured"]
            if np.isfinite(error).all():
                f["error"] = error
            else:
                issue(issues, "nonfinite_derived_error", e["_file"], e["_line"])
        for key in channels:
            if channels[key] is None:
                history[key] = []
                continue
            history[key] = (history[key]+[f])[-4:]
            for order, label in enumerate(("velocity", "acceleration", "jerk"), 1):
                if len(history[key]) <= order:
                    break
                points = history[key][-order-1:]
                times = np.asarray([p["_t"] for p in points])
                divided = np.stack([p[key] for p in points])
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    for level in range(1, order+1):
                        divided = (divided[1:]-divided[:-1])/(times[level:]-times[:-level])[:, None]
                    derived = divided[0]*math.factorial(order)
                if not np.isfinite(derived).all():
                    issue(issues, "nonfinite_derived_kinematics", e["_file"], e["_line"], key+"_"+label)
                    break
                f[f"{key}_{label}"] = derived
        frames.append(f)
        previous_t, previous_segment = e["_t"], e["_segment"]
    return frames


def reason_kind(reason):
    text = str(reason).lower()
    if any(s in text for s in ("timeout", "timed out", "deadline", "expired", "超时", "budget", "耗尽")):
        return "timeout"
    if any(s in text for s in ("reject", "blocked", "exceeds", "insufficient", "越界", "拒绝", "限位")):
        return "rejected"
    return "failed"


def analyze_session(path, manifest=None):
    """Analyze a task-events file and adjacent control/snapshot archives."""
    path = Path(path).resolve()
    root = path.parent
    manifest = manifest or {}
    if not isinstance(manifest.get("clock_domains", {}), dict) or not isinstance(manifest.get("browser_wall_clock", {}), dict) or not isinstance(manifest.get("embedded_clock_domains", {}), dict):
        raise ValueError("clock_domains and browser_wall_clock must be objects")
    issues, inventory = [], []
    events = read_jsonl(path, root, manifest, issues, inventory)
    controls = []
    gap = manifest.get("max_frame_gap_s", .1)
    if not number(gap) or gap <= 0:
        raise ValueError("max_frame_gap_s must be finite and positive")
    for file in sorted(root.glob("control-*.jsonl")):
        ce = read_jsonl(file, root, manifest, issues, inventory)
        frames = prepare_frames(ce, gap, issues)
        for e in ce:
            e.pop("command", None)
            e.pop("measured", None)
        controls.append(dict(file=file.name, events=ce, frames=frames))
    if not controls:
        issue(issues, "missing_control_logs", root.name)
    snapshots = []
    for name in ("task-run.json", "task-final.json"):
        candidates = [root / name, root.parent / name]
        file = next((p for p in candidates if p.is_file()), None)
        if file:
            obj = read_object(file, issues)
            status = obj.get("status", obj)
            if isinstance(status, dict):
                snapshots.append(dict(file=name, status=status, accepted_chunks=obj.get("accepted_chunks"),
                                      settings=obj.get("settings"), safety_boundary=obj.get("safety_boundary")))
            inventory.append(dict(path=str(file.relative_to(root)) if file.parent == root else "../"+name,
                                  sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        else:
            issue(issues, "missing_snapshot", name)
    run_ids = sorted({str(e.get("run_id")) for e in events if e.get("run_id")}) or ["unknown"]
    kind = manifest.get("source_kind", "unknown")
    if kind not in ("synthetic", "hardware_recording", "unknown"):
        raise ValueError("source_kind must be synthetic, hardware_recording or unknown")
    synthetic = any(e.get("simulation_only") is True or e.get("source_kind") == "synthetic"
                    or any(x in str(e.get("checkpoint", "")).lower() for x in ("simulation", "synthetic", "fake"))
                    for e in events)
    if synthetic:
        if kind == "hardware_recording":
            issue(issues, "provenance_conflict", path.name, detail="synthetic evidence overrides declared hardware")
        kind = "synthetic"
    session_id = inventory[0]["sha256"][:12]
    job_versions = {(e.get("run_id", "unknown"), e["inference_request_id"]): e.get("request_stage_version", e.get("stage_version"))
                    for e in events if e["event"] in ("observation_start", "inference_start") and e.get("inference_request_id")}
    def version_of(e):
        return e.get("request_stage_version", job_versions.get((e.get("run_id", "unknown"), e.get("inference_request_id")), e.get("stage_version")))
    requests = [e for e in events if e["event"] == "stage_requested"]
    switches = []
    seen = set()
    for request in requests:
        rid, run, version = request.get("request_id"), request.get("run_id", "unknown"), request.get("stage_version")
        identity = (run, rid, version)
        if identity in seen and rid is not None:
            issue(issues, "duplicate_stage_request", path.name, request["_line"], rid)
            continue
        seen.add(identity)
        if not rid or not isinstance(version, int) or isinstance(version, bool):
            issue(issues, "missing_request_identity", path.name, request["_line"])
        same_run = [e for e in events if e.get("run_id", "unknown") == run]
        later = [e for e in same_run if e["_line"] >= request["_line"]]
        next_request = next((e for e in later if e["event"] == "stage_requested" and e["_line"] > request["_line"]
                             and (e.get("request_id"), e.get("stage_version")) != (rid, version)), None)
        next_line = next_request["_line"] if next_request else math.inf
        relevant = [e for e in later if version_of(e) == version
                    and e["_line"] < next_line]
        boundary = first(relevant, "online_switch_boundary")
        b = boundary.get("boundary") or {} if boundary else {}
        token = b.get("token")
        applied = first(relevant, "online_switch_applied")
        armed = first(relevant, "online_switch_armed")
        if not token:
            token = (armed or {}).get("installation", {}).get("token") or (applied or {}).get("token")
        matching = []
        if token:
            for control in controls:
                found = [e for e in control["events"] if (e.get("boundary", {}).get("token") == token
                         or e.get("token") == token) and e.get("task_epoch") == version]
                if found:
                    matching.append((control, found))
        child, child_events = matching[0] if len(matching) == 1 else (None, [])
        if len(matching) > 1:
            issue(issues, "ambiguous_control_token", rid)
        wb = first(child_events, "switch_requested")
        wa = first(child_events, "handoff_armed")
        wp = first(child_events, "handoff_applied")
        # A token/version mismatch must not be interpreted as an applied switch.
        logged_applied = applied
        conflict = applied and (not token or applied.get("token") != token
                    or applied.get("stream", {}).get("task_epoch") != version
                    or applied.get("stream", {}).get("applied_token") != token
                    or (number(b.get("source_generation")) and (not number(applied.get("stream", {}).get("generation"))
                        or applied["stream"]["generation"] <= b["source_generation"])))
        if conflict:
            issue(issues, "applied_identity_mismatch", rid)
            applied = None
        accepted_id = (armed or {}).get("inference_request_id")
        starts = [e for e in relevant if e["event"] == "inference_start" and e.get("rtc_guidance") is False]
        start = next((e for e in starts if e.get("inference_request_id") == accepted_id), None) if accepted_id else None
        start = start or (starts[-1] if starts else None)
        iid = (start or {}).get("inference_request_id") or accepted_id
        job = [e for e in relevant if iid and e.get("inference_request_id") == iid]
        obs_start, obs_end, infer_end = (first(job, n) for n in ("observation_start", "observation_complete", "inference_complete"))
        infer_failure = next((e for e in job if e["event"] in ("inference_failed", "inference_expired")), None)
        result_event = infer_end or infer_failure
        if not iid:
            obs_end = next((e for e in reversed(relevant) if e["event"] == "observation_complete"), None)
            obs_start = next((e for e in relevant if obs_end and e["event"] == "observation_start"
                              and e.get("inference_request_id") == obs_end.get("inference_request_id")), None)
        full_relevant = relevant
        if applied:
            relevant = [e for e in relevant if e["_line"] <= applied["_line"]]
        failures = [e for e in relevant if e["event"] in ("inference_failed", "inference_expired")
                    and (not iid or e.get("inference_request_id") == iid)]
        terminal = next((e for e in later if e["_line"] < next_line and e["event"] == "execution_transition"
                         and e.get("execution_state") in TERMINAL), None)
        if applied and (failures or (terminal and terminal["_line"] < applied["_line"])):
            issue(issues, "terminal_or_failure_before_applied", rid)
            conflict, applied = True, None
        reason = (failures[0].get("reason") or (terminal or {}).get("reason") or failures[0]["event"]) if failures else (terminal or {}).get("reason")
        final = next((s["status"] for s in reversed(snapshots) if s["status"].get("run_id") == run
                      and s["status"].get("pending_request_id") == rid), {})
        reason = reason or final.get("last_error")
        if reason is not None:
            reason = str(reason)
        before = [e for e in same_run if e["_line"] < request["_line"]]
        state = next((e.get("execution_state") for e in reversed(before) if e["event"] == "execution_transition"), None)
        active = next((e.get("active_stage") for e in reversed(before) if e.get("active_stage")), None)
        scope = "online" if boundary or state in ("RUNNING", "SWITCHING", "SWITCH_REQUESTED", "RESUMING") else (
            "selection" if state in ("READY", "PAUSED") or request.get("operator_action") == "recover" else "unknown")
        if applied:
            outcome, reason = "applied", None
        elif conflict:
            outcome, reason = "evidence_conflict", reason or "application acknowledgement identity/state conflict"
        elif failures or final.get("last_error") or (terminal or {}).get("execution_state") == "FAULT":
            outcome = reason_kind(reason)
        elif wp:
            outcome = "worker_applied_unconfirmed"
        elif next_request:
            outcome, reason = "superseded", "newer operator request"
        elif terminal or final.get("execution_state") in TERMINAL:
            outcome, reason = "interrupted", (terminal or {}).get("reason") or final.get("execution_state")
        elif scope == "selection":
            outcome = "selection_only"
        else:
            outcome = "incomplete"
        waits = [e for e in relevant if e["event"] == "switch_policy_wait"]
        wait = waits[0] if waits else None
        old_id = (wait or {}).get("old_inference_request_id")
        discarded = next((e for e in later if e["event"] == "inference_discarded"
                          and e.get("inference_request_id") == old_id), None) if old_id else None
        metrics = {key: None for key in METRICS}
        metrics["registered_to_observation_ms"] = delta(request, obs_end, issues, "request_to_observation")
        wait_end = discarded
        if next_request and wait and (discarded is None or next_request["_line"] < discarded["_line"]):
            wait_end = next_request
        metrics["old_future_wait_ms"] = 0.0 if wait and wait.get("cancelled_before_start") is True else delta(wait, wait_end, issues, "old_future_wait_until_reclaim_or_supersede")
        metrics["new_prompt_inference_ms"] = delta(start, result_event, issues, "new_prompt_inference_until_complete_or_failure")
        metrics["inference_to_armed_ms"] = delta(infer_end, armed, issues, "inference_to_parent_armed")
        # Parent stores an actual post-send worker timestamp; never substitute its later poll time.
        worker_time = (applied or {}).get("stream", {}).get("applied_monotonic_s")
        precise_actual = False
        actual = None
        if applied and number(worker_time):
            if wp:
                if wp["_t"] is not None and 0 <= wp["_t"]-worker_time <= gap:
                    # Same token and same worker: field produced immediately before its log.
                    actual, precise_actual = {**wp, "_t": worker_time}, True
                else:
                    issue(issues, "worker_applied_field_outside_log_envelope", rid)
            else:
                actual = embedded_stamp(applied, worker_time, "worker", manifest)
                precise_actual = actual is not None
        if actual is None and wp:
            actual = wp  # Explicitly labeled post-send worker log timestamp proxy.
        metrics["registered_to_applied_ms"] = delta(request, actual, issues, "request_to_worker_applied") if applied or wp else None
        bridge = wa.get("bridge_start") if wa else (armed or {}).get("installation", {}).get("bridge_start")
        bridge_event = wa or armed
        bridge_stamp = ({**wa, "_t": bridge} if wa and number(bridge) else embedded_stamp(bridge_event, bridge, "worker", manifest))
        metrics["bridge_start_to_applied_ms"] = delta(bridge_stamp, actual, issues, "scheduled_bridge_to_applied")
        wall = manifest.get("browser_wall_clock", {})
        click, unix = request.get("client_clicked_unix_ms"), request.get("unix_ns")
        wall_ok = (wall.get("verified") is True and number(wall.get("offset_to_task_ms"))
                   and number(wall.get("uncertainty_ms")) and wall["uncertainty_ms"] >= 0)
        if wall_ok and number(click) and number(unix) and request["_wall_epoch"] == 0:
            elapsed = unix / 1e6 - click - wall["offset_to_task_ms"]
            if elapsed >= 0:
                metrics["click_to_registered_ms"] = elapsed
                if metrics["registered_to_applied_ms"] is not None:
                    metrics["click_to_applied_ms"] = elapsed + metrics["registered_to_applied_ms"]
            else:
                issue(issues, "negative_click_interval", rid, detail=elapsed)
        elif number(click):
            issue(issues, "unverified_browser_wall_clock", rid)
        captured = (start or {}).get("captured_monotonic_ns")
        cap = embedded_stamp(start, captured/1e9, "observation", manifest) if number(captured) else None
        metrics["observation_age_at_inference_ms"] = delta(cap, start, issues, "snapshot_age_inference")
        metrics["observation_age_at_result_ms"] = (result_event or {}).get("observation_age_ms")
        if not number(metrics["observation_age_at_result_ms"]) or metrics["observation_age_at_result_ms"] < 0:
            metrics["observation_age_at_result_ms"] = delta(cap, result_event, issues, "snapshot_age_result")
        reported_age = wa.get("observation_age_s") if wa else (armed or {}).get("installation", {}).get("observation_age_s")
        metrics["observation_age_at_armed_ms"] = reported_age*1000 if number(reported_age) and reported_age >= 0 else None
        # Phase is a recorded prior frame, not an extrapolated exact phase.
        prior, phase_age, bridge_frames = None, None, []
        if child:
            pivot = wb or wa
            preceding = [f for f in child["frames"] if pivot and f["_segment"] == pivot["_segment"]
                         and f["_line"] < pivot["_line"] and f["_t"] is not None]
            prior = preceding[-1] if preceding else None
            phase_age = delta(prior, pivot, issues, "phase_sample_age")
            if phase_age is not None and phase_age > gap*1000:
                issue(issues, "stale_phase_sample", rid, detail=phase_age)
            end_stamp = wp or actual
            if number(bridge) and bridge_event and end_stamp and end_stamp.get("_t") is not None:
                # Only local worker time (or explicitly shared domain) may select frames.
                local_start = embedded_stamp(bridge_event, bridge, "worker", manifest)
                if wa:  # bridge_start is produced by this same worker even without a domain assertion.
                    local_start = {**wa, "_t": bridge}
                if local_start and delta(local_start, end_stamp, issues, "bridge_frame_window") is not None:
                    limit = end_stamp["_t"]
                    bridge_frames = [f for f in child["frames"] if f["_segment"] == local_start["_segment"]
                                     and local_start["_t"] <= f["_t"] <= limit
                                     and f.get("task_epoch") == version
                                     and f.get("handoff_token") in (None, token)]
                    # The final endpoint frame is logged immediately AFTER handoff_applied.
                    endpoint = next((f for f in child["frames"] if wp and f["_line"] > wp["_line"]
                                     and f["_segment"] == wp["_segment"] and f.get("task_epoch") == version
                                     and 0 <= f["_t"] - wp["_t"] <= gap), None)
                    if endpoint and not any(f is endpoint for f in bridge_frames):
                        bridge_frames.append(endpoint)
        row = dict(session_id=session_id, run_id=run, source_kind=kind, request_id=rid, stage_version=version,
                   from_stage=active, previous_requested_stage=request.get("previous_stage"),
                   target_stage=request.get("target_stage"), scope=scope, outcome=outcome, reason=reason,
                   token=token, inference_request_id=iid, metrics=metrics,
                   click_uncertainty_ms=wall.get("uncertainty_ms") if wall_ok else None,
                   applied_time_source="worker_post_send_field" if precise_actual else (
                       "worker_post_send_log_proxy" if wp else "missing"),
                   parent_ack_present=bool(logged_applied), parent_ack_valid=bool(applied), worker_ack_present=bool(wp),
                   old_future_discarded=bool(discarded), old_inference_request_id=old_id,
                   old_future_slot_reclaim_ms=delta(wait, discarded, issues, "old_future_slot_reclaim"),
                   old_future_wait_end="cancelled_before_start" if wait and wait.get("cancelled_before_start") is True else (
                       "superseded" if wait_end is next_request and wait_end else "slot_reclaimed" if discarded else "missing"),
                   old_phase=(prior or {}).get("phase") if phase_age is not None and phase_age <= gap*1000 else None, old_phase_source="prior_worker_frame" if prior else "missing",
                   old_phase_sample_age_ms=phase_age,
                   bridge_frame_count=len(bridge_frames), bridge_joint_metrics=metric_map(bridge_frames),
                   observation_alignment=(obs_end or {}).get("alignment"),
                   inference_timing_end="complete" if infer_end else "failed" if infer_failure else "missing",
                   policy_call_ms=(infer_end or {}).get("policy_duration_s", 0)*1000
                       if number((infer_end or {}).get("policy_duration_s")) else None,
                   timeline=[], inference_attempts=[], missing_metrics=[k for k, v in metrics.items() if v is None])
        job_ids = dict.fromkeys(e.get("inference_request_id") for e in full_relevant if e.get("inference_request_id"))
        for jid in job_ids:
            history = [e for e in full_relevant if e.get("inference_request_id") == jid]
            row["inference_attempts"].append(dict(inference_request_id=jid,
                events=[dict(event=e["event"], source=e["_file"], line=e["_line"], monotonic_s=e["_t"],
                             clock_domain=e["_domain"], reason=e.get("reason")) for e in history]))
        for label, e in (("registered", request), ("boundary", boundary), ("observation_start", obs_start),
                         ("observation_complete", obs_end), ("inference_start", start), ("inference_complete", infer_end), ("inference_failure", infer_failure),
                         ("armed", armed), ("scheduled_bridge_start", bridge_stamp), ("worker_applied", actual),
                         ("parent_ack", logged_applied), ("ordinary_rtc_resumed", first(full_relevant, "ordinary_rtc_resumed"))):
            row["timeline"].append(dict(milestone=label, monotonic_s=e.get("_t") if e else None,
                                        clock_domain=e.get("_domain") if e else None,
                                        source=e.get("_file") if e else None, line=e.get("_line") if e else None,
                                        since_registered_ms=delta(request, e, issues, label)))
        switches.append(row)
    all_frames = [f for c in controls for f in c["frames"]]
    stage_records = []
    for run in run_ids:
        for stage in STAGES:
            activations = [e for e in events if e.get("run_id", "unknown") == run and e["event"] == "chunk_active"
                           and e.get("active_stage") == stage.id]
            relevant_rows = [r for r in switches if r["run_id"] == run and r["target_stage"] == stage.id]
            stage_records.append(dict(run_id=run, stage_id=stage.id, label=stage.label, prompt_class=stage.action,
                                      prompt=stage.prompt, request_count=len(relevant_rows),
                                      request_ids=[r["request_id"] for r in relevant_rows],
                                      activation_count=len(activations),
                                      activation_times=[dict(monotonic_s=e["_t"], clock_domain=e["_domain"], line=e["_line"])
                                                        for e in activations],
                                      software_stage_evidence="active" if activations else "snapshot_only" if any(
                                          ss["status"].get("run_id") == run and ss["status"].get("active_stage") == stage.id
                                          for ss in snapshots) else "missing",
                                      picking_success=None, grasp_stability=None, contact_force=None))
    per_joint = []
    for joint in range(58):
        group = next(name for name, (a, b) in GROUPS.items() if a <= joint < b)
        per_joint.append(dict(joint_index=joint, group=group,
                              absolute_error_rad=stat([abs(f["error"][joint]) for f in all_frames if f["error"] is not None])))
    return dict(session_id=session_id, source=str(path), source_kind=kind, run_ids=run_ids, inventory=inventory,
                switches=switches, stages=stage_records, joint_metrics=metric_map(all_frames), per_joint=per_joint,
                snapshots=snapshots, quality_issues=issues, manifest_assertions=manifest, _frames=all_frames)


def summarize_rows(rows):
    attempts = [r for r in rows if r["scope"] != "selection"]
    successes = [r for r in attempts if r["outcome"] == "applied"]
    unresolved = {"incomplete", "evidence_conflict", "worker_applied_unconfirmed"}
    resolved = [r for r in attempts if r["outcome"] not in unresolved]
    return dict(request_count=len(rows), attempt_count=len(attempts), applied_count=len(successes),
                outcome_counts=dict(Counter(r["outcome"] for r in rows)),
                applied_fraction_all_attempts=len(successes)/len(attempts) if attempts else None,
                applied_fraction_resolved=len(successes)/len(resolved) if resolved else None,
                resolved_denominator=len(resolved),
                latency_population="confirmed applied switches only; missing metrics excluded, n shown",
                metrics={name: stat([r["metrics"][name] for r in successes]) for name in METRICS},
                all_attempt_metrics={name: stat([r["metrics"][name] for r in attempts]) for name in METRICS})


def grouped(rows, key):
    buckets = defaultdict(list)
    for row in rows:
        buckets[key(row)].append(row)
    return {k: summarize_rows(v) for k, v in sorted(buckets.items())}


def phase_bin(row):
    phase = row["old_phase"]
    if not number(phase) or not 0 <= phase < 50:
        return "missing"
    low = int(phase//5)*5
    return f"[{low:02d},{low+5:02d})"


def analyze(inputs, manifest=None):
    """Batch durations may be pooled; absolute clocks never cross sessions."""
    paths = set()
    for value in inputs:
        p = Path(value).resolve()
        if p.is_file():
            paths.add(p)
        elif p.is_dir():
            paths.update(p.rglob("task-events.jsonl"))
        else:
            raise FileNotFoundError(p)
    if not paths:
        raise ValueError("No task-events.jsonl found")
    sessions, seen, batch_issues = [], set(), []
    for path in sorted(paths):
        local = manifest
        if local is None:
            mpath = next((p for p in (path.parent / "analysis-manifest.json", path.parent.parent / "analysis-manifest.json")
                          if p.is_file()), None)
            local = read_object(mpath, batch_issues) if mpath else {}
        session = analyze_session(path, local)
        identity = session["inventory"][0]["sha256"]
        if identity in seen:
            issue(batch_issues, "duplicate_task_archive_excluded", path)
            continue
        seen.add(identity)
        sessions.append(session)
    rows = [r for s in sessions for r in s["switches"]]
    kinds = sorted({s["source_kind"] for s in sessions})
    report = dict(schema_version=1, generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  source_kind=kinds[0] if len(kinds) == 1 else "mixed",
                  population="registered stage_requested only; UI/HTTP rejects before registration are not observed",
                  evidence_notice="source_kind is declared evidence, not certification; synthetic != hardware",
                  physical_outcomes=dict(grasp_stability=None, contact_force=None, picking_success_rate=None,
                                         reason="No independent physical outcome annotations analyzed"),
                  definitions=dict(clock_policy="same file segment or explicitly shared host/boot domain only",
                                   click_policy="verified browser->task wall offset plus server-local duration; no wall/mono subtraction",
                                   percentiles="numpy linear empirical percentile; small n is not tail certification",
                                   tracking="same logged frame command minus measured feedback; not delay-aligned tracking error",
                                   derivatives="Newton divided differences times order factorial on trailing irregular-time windows; last frame attribution",
                                   bridge_start="scheduled start, not inferred exact actuator transition onset",
                                   frame_time="post-send logging time proxy, not a hardware sample timestamp",
                                   units="angles rad, times ms/s as named; derivatives rad/s, rad/s^2, rad/s^3"),
                  summary=summarize_rows(rows),
                  by_source_kind=grouped(rows, lambda r: r["source_kind"]),
                  by_transition=grouped(rows, lambda r: f"{r['from_stage'] or 'unknown'}->{r['target_stage'] or 'unknown'}"),
                  by_phase=grouped(rows, phase_bin),
                  by_prompt_transition=grouped(rows, lambda r: (
                      (STAGE_MAP[r["from_stage"]].action if r["from_stage"] in STAGE_MAP else "unknown")+"->"+
                      (STAGE_MAP[r["target_stage"]].action if r["target_stage"] in STAGE_MAP else "unknown"))),
                  reason_distribution=dict(Counter(r["reason"] for r in rows if r["reason"])),
                  quality_issue_distribution=dict(Counter(e["code"] for s in sessions for e in s["quality_issues"]) +
                                                  Counter(e["code"] for e in batch_issues)),
                  batch_issues=batch_issues, sessions=sessions)
    return report


def cell(value):
    if value is None:
        return "缺失"
    return f"{value:.3f}" if isinstance(value, float) else str(value).replace("|", "\\|").replace("\n", " ")


def markdown(report):
    rows = [r for s in report["sessions"] for r in s["switches"]]
    summary = report["summary"]
    lines = ["# 七阶段在线切换实验报告", "", f"数据来源：**{report['source_kind']}**。生成时间：{report['generated_at_utc']}。",
             "", "synthetic 为模拟数据；hardware_recording 仅表示实验清单声明的实机记录，未经本工具认证。未知或混合来源不可作为纯实机结论。",
             "", "本报告的成功仅指软件命令流接入。抓握稳定性、接触力、采摘成功率：**缺失**，不从阶段按钮、COMPLETED或位置插值推断。",
             "", f"归档数：{len(report['sessions'])}；在线/未知范围请求：{summary['attempt_count']}；确认接入：{summary['applied_count']}。",
             f"全部请求接入比例：{cell(summary['applied_fraction_all_attempts'])}；有明确结果的接入比例：{cell(summary['applied_fraction_resolved'])}（分母{summary['resolved_denominator']}）。",
             "", "## 延迟统计", "", "仅统计已确认接入的切换；每个指标单独排除缺失值，不能把缺失当0。",
             "", "| 指标 | n | P50 ms | P95 ms | P99 ms |", "|---|---:|---:|---:|---:|"]
    for name, values in summary["metrics"].items():
        lines.append("| " + " | ".join(cell(x) for x in (name, values["n"], values["p50"], values["p95"], values["p99"])) + " |")
    lines += ["", "小样本分位数是经验插值，不构成尾延迟保证。来源混合时请使用JSON中by_source_kind分组。",
              "", "## 每次切换", "", "| run / request_id / version | 源→目标阶段 | 结果 | Click→Applied ms | 登记→Applied ms | 旧phase | 原因 |",
              "|---|---|---|---:|---:|---:|---|"]
    for row in rows:
        lines.append("| " + " | ".join(cell(x) for x in (
            f"{row['run_id']} / {row['request_id']} / {row['stage_version']}",
            f"{row['from_stage']}→{row['target_stage']}", row["outcome"], row["metrics"]["click_to_applied_ms"],
            row["metrics"]["registered_to_applied_ms"], row["old_phase"], row["reason"])) + " |")
    lines += ["", "## 详细时间线", "", "绝对monotonic时间只在各自文件段或声明的同一host/boot域内有意义，不跨域拼接。"]
    for row in rows:
        lines += ["", f"### {cell(row['request_id'])}（{cell(row['target_stage'])}，v{cell(row['stage_version'])}）", "",
                  "| 时间点 | monotonic s | 时钟域 | 距登记 ms | 文件:行 |", "|---|---:|---|---:|---|"]
        for t in row["timeline"]:
            lines.append("| " + " | ".join(cell(x) for x in (t["milestone"], t["monotonic_s"], t["clock_domain"],
                         t["since_registered_ms"], f"{t['source']}:{t['line']}" if t["source"] else None)) + " |")
        lines += ["", "缺失指标：" + ("、".join(row["missing_metrics"]) or "无"),
                  f"实际接入时间证据：{row['applied_time_source']}；桥接有效帧：{row['bridge_frame_count']}；旧phase证据：{row['old_phase_source']}。"]
    lines += ["", "## 七阶段记录", "", "| run | 阶段 | Prompt类别 | 请求数 | 已接纳Chunk数 | 软件生效证据 | 采摘/抓握 |", "|---|---|---|---:|---:|---|---|"]
    for session in report["sessions"]:
        for stage in session["stages"]:
            lines.append("| " + " | ".join(cell(x) for x in (stage["run_id"], stage["label"], stage["prompt_class"],
                         stage["request_count"], stage["activation_count"], stage["software_stage_evidence"], "缺失")) + " |")
    for title, key in (("各类阶段切换", "by_transition"), ("四类Prompt切换", "by_prompt_transition"), ("不同旧Chunk相位", "by_phase")):
        lines += ["", "## " + title, "", "| 分组 | 请求数 | 接入数 | 全部请求接入比例 | 登记→Applied P50 ms |", "|---|---:|---:|---:|---:|"]
        for group, values in report[key].items():
            lines.append("| " + " | ".join(cell(x) for x in (group, values["attempt_count"], values["applied_count"],
                         values["applied_fraction_all_attempts"], values["metrics"]["registered_to_applied_ms"]["p50"])) + " |")
    lines += ["", "## 关节误差与桥接动态", "", "误差为同一日志帧内command与measured的差，未校正反馈延迟。导数使用实际日志间隔；不是硬件编码器同步采样或动态安全认证。",
              "", "| session / 范围 | 关节组 | 误差RMSE rad | 误差P95 rad | 命令速度峰值 rad/s | 命令加速度峰值 rad/s² | 命令jerk峰值 rad/s³ |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for s in report["sessions"]:
        scopes = [(s["session_id"]+" / 全程", s["joint_metrics"])] + [
            (f"{s['session_id']} / {r['request_id']}", r["bridge_joint_metrics"]) for r in s["switches"]]
        for name, groups in scopes:
            for group, metrics in groups.items():
                lines.append("| " + " | ".join(cell(x) for x in (name, group, metrics["error"]["rmse"], metrics["error"]["p95"],
                    metrics["command_velocity"]["max"], metrics["command_acceleration"]["max"], metrics["command_jerk"]["max"])) + " |")
    lines += ["", "## 错误与数据质量", "", "| 类型 | 次数 |", "|---|---:|"]
    for k, v in report["reason_distribution"].items():
        lines.append(f"| {cell(k)} | {v} |")
    for k, v in report["quality_issue_distribution"].items():
        lines.append(f"| 数据质量：{cell(k)} | {v} |")
    lines += ["", "缺失或截断的日志不能自动判成超时/安全停止。快照仅作终态证据，不补造时间点。",
              "浏览器wall clock没有已验证偏移时，Click相关指标保持缺失；跨文件没有同一host/boot声明时，不计算跨文件monotonic差值。",
              "所有详细数据、字段定义、来源校验摘要和质量问题见summary.json及CSV。", ""]
    return "\n".join(lines)


def csv_write(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [r for s in report["sessions"] for r in s["switches"]]
    fields = ["session_id", "source_kind", "run_id", "request_id", "stage_version", "from_stage", "target_stage",
              "scope", "outcome", "reason", "inference_timing_end", "old_phase", "old_phase_sample_age_ms", "applied_time_source", "click_uncertainty_ms"]
    csv_write(output / "switches.csv", [{**r, **r["metrics"]} for r in rows], fields+list(METRICS))
    csv_write(output / "timeline.csv", [dict(session_id=r["session_id"], source_kind=r["source_kind"], run_id=r["run_id"], request_id=r["request_id"],
                stage_version=r["stage_version"], **t) for r in rows for t in r["timeline"]],
              ["session_id", "source_kind", "run_id", "request_id", "stage_version", "milestone", "monotonic_s", "clock_domain", "since_registered_ms", "source", "line"])
    csv_write(output / "inference_attempts.csv", [dict(session_id=r["session_id"], source_kind=r["source_kind"], request_id=r["request_id"],
              stage_version=r["stage_version"], inference_request_id=j["inference_request_id"], **e)
              for r in rows for j in r["inference_attempts"] for e in j["events"]],
              ["session_id", "source_kind", "request_id", "stage_version", "inference_request_id", "event", "source", "line", "monotonic_s", "clock_domain", "reason"])
    csv_write(output / "stages.csv", [dict(session_id=s["session_id"], source_kind=s["source_kind"], **r) for s in report["sessions"] for r in s["stages"]],
              ["session_id", "source_kind", "run_id", "stage_id", "label", "prompt_class", "request_count", "activation_count",
               "software_stage_evidence", "picking_success", "grasp_stability", "contact_force"])
    groups = []
    for key in ("by_source_kind", "by_transition", "by_prompt_transition", "by_phase"):
        for name, values in report[key].items():
            for metric, distribution in values["metrics"].items():
                groups.append(dict(group_type=key, source_kind=name if key == "by_source_kind" else report["source_kind"], group=name, metric=metric, attempt_count=values["attempt_count"],
                                   applied_count=values["applied_count"], applied_fraction=values["applied_fraction_all_attempts"], **distribution))
    csv_write(output / "groups.csv", groups, ["group_type", "source_kind", "group", "metric", "attempt_count", "applied_count", "applied_fraction", "n", "mean", "p50", "p95", "p99"])
    csv_write(output / "joint_errors.csv", [dict(session_id=s["session_id"], source_kind=s["source_kind"], joint_index=j["joint_index"], group=j["group"],
                **j["absolute_error_rad"]) for s in report["sessions"] for j in s["per_joint"]],
              ["session_id", "source_kind", "joint_index", "group", "n", "mean", "rmse", "max", "p50", "p95", "p99"])
    dynamics = []
    for s in report["sessions"]:
        for scope, metrics in [("session", s["joint_metrics"])] + [(r["request_id"], r["bridge_joint_metrics"]) for r in s["switches"]]:
            for group, values in metrics.items():
                for quantity, distribution in values.items():
                    dynamics.append(dict(session_id=s["session_id"], source_kind=s["source_kind"], scope=scope, group=group, quantity=quantity, **distribution))
    csv_write(output / "joint_dynamics.csv", dynamics, ["session_id", "source_kind", "scope", "group", "quantity", "n", "mean", "rmse", "max", "p50", "p95", "p99"])
    # Wide per-frame group measurements are sufficient for paper plots; raw 58D remains in source logs.
    frame_path = output / "frames.csv"
    frame_fields = ["session_id", "source_kind", "source", "line", "monotonic_s", "clock_domain", "task_epoch", "handoff_token", "phase", "group", "dt_s",
                    "max_abs_error_rad", "rmse_error_rad", "command_velocity_max_rad_s", "command_acceleration_max_rad_s2", "command_jerk_max_rad_s3",
                    "measured_velocity_max_rad_s", "measured_acceleration_max_rad_s2", "measured_jerk_max_rad_s3"]
    def frame_rows():
        for s in report["sessions"]:
            for f in s["_frames"]:
                for name, (a, b) in GROUPS.items():
                    r = dict(session_id=s["session_id"], source_kind=s["source_kind"], source=f["_file"], line=f["_line"], monotonic_s=f["_t"], clock_domain=f["_domain"],
                             task_epoch=f.get("task_epoch"), handoff_token=f.get("handoff_token"), phase=f.get("phase"), group=name, dt_s=f["dt_s"],
                             max_abs_error_rad=float(np.max(np.abs(f["error"][a:b]))) if f["error"] is not None else None,
                             rmse_error_rad=stat(np.abs(f["error"][a:b]))["rmse"] if f["error"] is not None else None)
                    for side in ("command", "measured"):
                        for derivative, unit in (("velocity", "rad_s"), ("acceleration", "rad_s2"), ("jerk", "rad_s3")):
                            value = f.get(f"{side}_{derivative}")
                            r[f"{side}_{derivative}_max_{unit}"] = float(np.max(np.abs(value[a:b]))) if value is not None else None
                    yield r
    csv_write(frame_path, frame_rows(), frame_fields)
    # Remove internal ndarray frame storage from JSON without mutating report.
    public = {**report, "sessions": [{k: v for k, v in s.items() if not k.startswith("_")} for s in report["sessions"]]}
    (output / "summary.json").write_text(json.dumps(public, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    (output / "report.md").write_text(markdown(report))
    return output


def load_manifest(path):
    problems = []
    obj = read_object(Path(path), problems)
    if problems:
        raise ValueError(f"Invalid analysis manifest: {problems[0]['detail']}")
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="archive directories or task-events.jsonl; recursive batch discovery")
    parser.add_argument("--manifest", type=Path, help="explicit analysis manifest; otherwise auto-detect per archive")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest) if args.manifest else None
    report = analyze(args.inputs, manifest)
    export(report, args.out)
    print(f"{args.out / 'report.md'} ({len(report['sessions'])} archives; source_kind={report['source_kind']})")


if __name__ == "__main__":
    main()
