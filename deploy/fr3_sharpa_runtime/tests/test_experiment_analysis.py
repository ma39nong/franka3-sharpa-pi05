"""Read-only analysis correctness: identities, clocks, irregular derivatives and exports."""

import csv
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.experiment_analysis import analyze, export, prepare_frames, stat
from deploy.fr3_sharpa_runtime.experiment_example import create_example


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path, records):
    path.write_text("".join(json.dumps(e)+"\n" for e in records))


@pytest.fixture
def archive(tmp_path):
    root = create_example(tmp_path / "synthetic")
    return root / "logs"


def one(archive):
    write(archive / "task-events.jsonl", [e for e in read(archive / "task-events.jsonl") if e["monotonic_ns"] < 13e9])
    write(archive / "control-synthetic.jsonl", [e for e in read(archive / "control-synthetic.jsonl") if e["monotonic_s"] < 13])
    return archive


def patch_manifest(archive, **fields):
    p = archive / "analysis-manifest.json"
    m = json.loads(p.read_text())
    m.update(fields)
    p.write_text(json.dumps(m))


def test_full_seven_stage_and_every_requested_measurement(archive):
    report = analyze([archive])
    s = report["sessions"][0]
    assert report["source_kind"] == "synthetic"
    assert report["summary"]["attempt_count"] == report["summary"]["applied_count"] == 6
    assert len(s["stages"]) == 7 and all(r["activation_count"] == 1 for r in s["stages"])
    assert s["stages"][0]["prompt_class"] == s["stages"][4]["prompt_class"] == "pickup"
    for r in s["switches"]:
        assert r["reason"] is None and r["outcome"] == "applied"
        assert r["bridge_frame_count"] == 7
        assert r["old_future_discarded"]
        expected = dict(click_to_registered_ms=5, registered_to_observation_ms=24, old_future_wait_ms=14.5,
                        new_prompt_inference_ms=60, inference_to_armed_ms=6, bridge_start_to_applied_ms=200,
                        click_to_applied_ms=305, registered_to_applied_ms=300,
                        observation_age_at_inference_ms=3, observation_age_at_result_ms=63, observation_age_at_armed_ms=68)
        assert r["metrics"] == pytest.approx(expected)
        assert r["missing_metrics"] == []
        assert next(t for t in r["timeline"] if t["milestone"] == "ordinary_rtc_resumed")["since_registered_ms"] == pytest.approx(600)
        assert r["bridge_joint_metrics"]["right_hand"]["error"]["rmse"] == pytest.approx(.003)
    assert all(v is None for k, v in report["physical_outcomes"].items() if k != "reason")
    assert report["summary"]["metrics"]["click_to_applied_ms"]["p95"] == pytest.approx(305)
    assert len(s["per_joint"]) == 58


def test_no_clock_declaration_means_missing_not_cross_file_subtraction(archive):
    (archive / "analysis-manifest.json").unlink()
    r = analyze([archive])
    row = r["sessions"][0]["switches"][0]
    assert row["metrics"]["new_prompt_inference_ms"] == pytest.approx(60)
    assert row["metrics"]["click_to_registered_ms"] is None
    assert row["metrics"]["registered_to_applied_ms"] is None
    assert row["metrics"]["click_to_applied_ms"] is None
    assert row["metrics"]["observation_age_at_inference_ms"] is None
    # Worker-local bridge timing IS comparable without a cross-process declaration.
    assert row["metrics"]["bridge_start_to_applied_ms"] == pytest.approx(200)
    assert "incomparable_clock" in r["quality_issue_distribution"]


def test_worker_timestamp_in_parent_rpc_keeps_worker_clock(archive):
    patch_manifest(archive, clock_domains={"task-events.jsonl": "parent/bootA", "control-*.jsonl": "worker/bootB"},
                   embedded_clock_domains={"worker": "worker/bootB", "observation": "sensor/bootC"})
    row = analyze([archive])["sessions"][0]["switches"][0]
    assert row["metrics"]["registered_to_applied_ms"] is None
    assert row["metrics"]["click_to_applied_ms"] is None
    assert row["metrics"]["observation_age_at_inference_ms"] is None
    assert row["metrics"]["bridge_start_to_applied_ms"] == pytest.approx(200)


def test_browser_wall_offset_is_explicit_with_uncertainty(archive):
    patch_manifest(archive, browser_wall_clock=dict(verified=True, offset_to_task_ms=2, uncertainty_ms=.5))
    row = analyze([archive])["sessions"][0]["switches"][0]
    assert row["metrics"]["click_to_registered_ms"] == 3
    assert row["metrics"]["click_to_applied_ms"] == pytest.approx(303)
    assert row["click_uncertainty_ms"] == .5
    patch_manifest(archive, browser_wall_clock=dict(verified=True, offset_to_task_ms=10, uncertainty_ms=.5))
    r = analyze([archive])
    assert r["sessions"][0]["switches"][0]["metrics"]["click_to_applied_ms"] is None
    assert "negative_click_interval" in r["quality_issue_distribution"]


@pytest.mark.parametrize("reason,outcome", [("policy timed out", "timeout"), ("actions exceeds gateway speed", "rejected"),
                                            ("feedback unavailable", "failed")])
def test_failures_and_reasons_are_not_lost_to_final_cleanup(archive, reason, outcome):
    one(archive)
    path = archive / "task-events.jsonl"
    events = [e for e in read(path) if e["monotonic_ns"] < 12.08e9]
    base = events[-1]
    events += [{**base, "event": "inference_failed", "inference_request_id": "new-1", "request_stage_version": 1,
                "monotonic_ns": 12_090_000_000, "unix_ns": 1_800_000_012_090_000_000, "reason": reason},
               {**base, "event": "execution_transition", "execution_state": "FAULT", "request_id": "demo-1",
                "monotonic_ns": 12_091_000_000, "unix_ns": 1_800_000_012_091_000_000, "reason": reason}]
    write(path, events)
    write(archive / "control-synthetic.jsonl", [e for e in read(archive / "control-synthetic.jsonl") if e["monotonic_s"] < 12.08])
    report = analyze([archive])
    row = report["sessions"][0]["switches"][0]
    assert row["outcome"] == outcome and row["reason"] == reason
    assert row["inference_timing_end"] == "failed"
    assert row["metrics"]["new_prompt_inference_ms"] == pytest.approx(65)
    assert row["metrics"]["click_to_applied_ms"] is None
    assert report["reason_distribution"][reason] == 1
    assert report["summary"]["applied_fraction_all_attempts"] == 0


def test_superseded_and_late_old_future_dont_activate_new_stage(archive):
    path = archive / "task-events.jsonl"
    events = read(path)
    # Supersede before demo-1 has an accepted result. Late old inference is stamped
    # with the current version but binds its ORIGINAL request_stage_version=1.
    events = [e for e in events if not (e.get("stage_version") == 1 and e["event"] in ("online_switch_applied", "chunk_active", "online_switch_armed"))]
    second = next(e for e in events if e["event"] == "stage_requested" and e["stage_version"] == 2)
    old = next(e for e in events if e["event"] == "inference_complete" and e.get("inference_request_id") == "new-1")
    events.remove(old)
    events.append({**old, "stage_version": 2, "request_stage_version": 1,
                   "monotonic_ns": second["monotonic_ns"]+50_000_000, "unix_ns": second["unix_ns"]+50_000_000})
    write(path, sorted(events, key=lambda e: e["monotonic_ns"]))
    write(archive / "control-synthetic.jsonl", [e for e in read(archive / "control-synthetic.jsonl") if e.get("token") != "synthetic-token-1"])
    report = analyze([archive])
    a, b = report["sessions"][0]["switches"][:2]
    assert a["outcome"] == "superseded" and a["metrics"]["new_prompt_inference_ms"] is None
    assert b["outcome"] == "applied" and b["metrics"]["new_prompt_inference_ms"] == pytest.approx(60)


def test_only_worker_ack_is_not_parent_confirmed_success(archive):
    one(archive)
    p = archive / "task-events.jsonl"
    write(p, [e for e in read(p) if e["event"] != "online_switch_applied"])
    r = analyze([archive])
    row = r["sessions"][0]["switches"][0]
    assert row["outcome"] == "worker_applied_unconfirmed"
    assert r["summary"]["applied_count"] == 0


def test_incomplete_is_not_automatically_timeout(archive):
    one(archive)
    for name in ("task-events.jsonl", "control-synthetic.jsonl"):
        records = read(archive / name)
        write(archive / name, [e for e in records if e.get("monotonic_s", e.get("monotonic_ns", 0)/1e9) < 12.09])
    row = analyze([archive])["sessions"][0]["switches"][0]
    assert row["outcome"] == "incomplete" and row["reason"] is None


def test_independent_stop_is_interrupted_not_inference_failure(archive):
    one(archive)
    path = archive / "task-events.jsonl"
    records = [e for e in read(path) if e["monotonic_ns"] < 12.08e9]
    records.append({**records[-1], "event": "execution_transition", "execution_state": "STOPPING", "reason": "operator stop",
                    "monotonic_ns": 12_080_000_000, "unix_ns": 1_800_000_012_080_000_000})
    write(path, records)
    write(archive / "control-synthetic.jsonl", [e for e in read(archive / "control-synthetic.jsonl") if e["monotonic_s"] < 12.08])
    assert analyze([archive])["sessions"][0]["switches"][0]["outcome"] == "interrupted"


@pytest.mark.parametrize("bad", ["token", "epoch"])
def test_ack_identity_mismatch_is_never_success(archive, bad):
    p = archive / "task-events.jsonl"
    events = read(p)
    event = next(e for e in events if e["event"] == "online_switch_applied")
    if bad == "token":
        event["stream"]["applied_token"] = "wrong"
    else:
        event["stream"]["task_epoch"] = 100
    write(p, events)
    r = analyze([archive])
    assert r["sessions"][0]["switches"][0]["outcome"] == "evidence_conflict"
    assert r["summary"]["applied_count"] == 5


def test_corrupt_partial_and_missing_fields_do_not_crash(archive):
    p = archive / "task-events.jsonl"
    events = read(p)
    for e in events:
        e.pop("client_clicked_unix_ms", None)
        e.pop("captured_monotonic_ns", None)
    events.append(dict(event="stage_requested", request_id=["wrong"], boundary="wrong", stage_version=None))
    write(p, events)
    with p.open("a") as stream:
        stream.write('{"event":')
    (archive / "task-run.json").write_text('{')
    r = analyze([archive])
    assert "malformed_jsonl" in r["quality_issue_distribution"]
    assert "invalid_json_object" in r["quality_issue_distribution"]
    assert "invalid_identity_field" in r["quality_issue_distribution"]
    assert r["sessions"][0]["switches"][0]["metrics"]["click_to_registered_ms"] is None


def test_regressed_clock_and_wall_jump_are_flagged_without_sort_repair(archive):
    one(archive)
    p = archive / "task-events.jsonl"
    events = read(p)
    # Keep append order, put a backward timestamp at inference_complete.
    end = next(e for e in events if e["event"] == "inference_complete" and e.get("inference_request_id") == "new-1")
    end["monotonic_ns"] = 11_000_000_000
    write(p, events)
    r = analyze([archive])
    assert "clock_regression_or_out_of_order" in r["quality_issue_distribution"]
    assert r["sessions"][0]["switches"][0]["metrics"]["new_prompt_inference_ms"] is None


def test_source_provenance_cannot_promote_synthetic_to_hardware(archive):
    patch_manifest(archive, source_kind="hardware_recording")
    r = analyze([archive])
    assert r["source_kind"] == "synthetic"
    assert "provenance_conflict" in r["quality_issue_distribution"]


def test_repeated_archive_is_not_counted_twice(archive, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(archive, copy)
    r = analyze([archive, copy])
    assert r["summary"]["attempt_count"] == 6 and len(r["sessions"]) == 1
    assert r["quality_issue_distribution"]["duplicate_task_archive_excluded"] == 1


def test_irregular_time_derivatives_are_exact_for_cubic_and_groups():
    times = [1, 1.02, 1.055, 1.08, 1.12]
    events = [dict(event="frame", _file="control.jsonl", _line=i+1, _segment=0, _domain=None, _t=t,
                   command=np.full(58, (t-1)**3).tolist(), measured=np.full(58, (t-1)**3-.001).tolist())
              for i, t in enumerate(times)]
    f = prepare_frames(events, .1, [])
    np.testing.assert_allclose(f[-1]["command_jerk"], 6, atol=1e-8)
    assert np.max(np.abs(f[-1]["error"])) == pytest.approx(.001)
    assert f[0].get("command_velocity") is None and f[1].get("command_acceleration") is None


def test_gap_and_invalid_frame_break_derivative_history():
    events = [dict(event="frame", _file="control.jsonl", _line=i+1, _segment=0, _domain=None, _t=t,
                   command=[t]*58, measured=[t]*58) for i, t in enumerate([1, 1.03, 1.07, 2, 2.03, 2.04, 2.05])]
    events[-2]["command"] = [0]*57
    issues = []
    f = prepare_frames(events, .1, issues)
    assert f[3].get("command_velocity") is None
    assert f[-1].get("command_velocity") is None
    assert {e["code"] for e in issues} == {"invalid_frame", "frame_gap_or_nonpositive_dt"}


def test_exports_are_machine_readable_and_do_not_claim_physical_success(archive, tmp_path):
    r = analyze([archive])
    output = export(r, tmp_path / "report")
    loaded = json.loads((output / "summary.json").read_text())
    assert loaded["sessions"][0]["switches"][0]["metrics"]["click_to_applied_ms"] == pytest.approx(305)
    assert "缺失" in (output / "report.md").read_text() and "synthetic" in (output / "report.md").read_text()
    for name in ("switches", "stages", "timeline", "groups", "joint_errors", "joint_dynamics", "frames", "inference_attempts"):
        with (output / (name+".csv")).open(encoding="utf-8-sig") as stream:
            assert list(csv.DictReader(stream)), name
    assert (output / "frames.csv").stat().st_size > 0
    assert b"\r\n" not in (output / "frames.csv").read_bytes()


def test_empty_discovery_and_nonfinite_json_are_explicit(tmp_path):
    with pytest.raises(ValueError, match="No task-events"):
        analyze([tmp_path])
    p = tmp_path / "task-events.jsonl"
    p.write_text('{"event":"frame","monotonic_s":NaN}\n')
    r = analyze([p])
    assert r["quality_issue_distribution"]["malformed_jsonl"] == 1
    assert stat([])["p95"] is None and stat([1, 2, 3])["p50"] == 2


def test_analysis_cli_and_legacy_cli_do_not_require_hardware(archive, tmp_path):
    result = subprocess.run([sys.executable, "-m", "deploy.fr3_sharpa_runtime.experiment_analysis", str(archive),
                             "--out", str(tmp_path / "cli")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, "-m", "deploy.fr3_sharpa_runtime.switch_audit", "--experiment", str(archive),
                             "--output", str(tmp_path / "compat")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "compat/report.md").exists()


def test_existing_coordinator_simulator_events_are_accepted_without_fabricated_frames(tmp_path):
    from deploy.fr3_sharpa_runtime.switch_audit import simulate
    from deploy.fr3_sharpa_runtime.task_rtc import TaskRTC
    from deploy.fr3_sharpa_runtime.stage_handoff import StageTimeline
    result = simulate(TaskRTC, StageTimeline, observation_ms=1, infer_ms=5)
    assert result["status"] == "PASS"
    path = tmp_path / "task-events.jsonl"
    write(path, result["events"])
    r = analyze([path])
    assert r["source_kind"] == "synthetic"
    assert r["summary"]["applied_count"] == 1
    assert r["summary"]["metrics"]["click_to_applied_ms"]["n"] == 0
    assert r["sessions"][0]["joint_metrics"]["left_arm"]["error"]["n"] == 0


def test_final_snapshot_failure_is_evidence_without_fabricated_timing(archive):
    one(archive)
    p = archive / "task-events.jsonl"
    write(p, [e for e in read(p) if e["monotonic_ns"] < 12.08e9])
    write(archive / "control-synthetic.jsonl", [])
    (archive / "task-final.json").write_text(json.dumps(dict(run_id="synthetic-seven-stage", pending_request_id="demo-1",
                          execution_state="FAULT", last_error="RTC deadline exceeded", stage_version=1)))
    row = analyze([archive])["sessions"][0]["switches"][0]
    assert row["outcome"] == "timeout" and row["reason"] == "RTC deadline exceeded"
    assert row["metrics"]["registered_to_applied_ms"] is None


def test_wall_jump_does_not_silently_reuse_browser_offset(archive):
    p = archive / "task-events.jsonl"
    events = read(p)
    for e in events:
        if e["monotonic_ns"] > 13e9:
            e["unix_ns"] += 1_000_000_000
    write(p, events)
    r = analyze([archive])
    assert "wall_or_monotonic_discontinuity" in r["quality_issue_distribution"]
    assert r["sessions"][0]["switches"][1]["metrics"]["click_to_registered_ms"] is None


def test_version_resets_are_partitioned_by_run_id(archive):
    p = archive / "task-events.jsonl"
    events = read(p)
    second = []
    for e in events:
        new = json.loads(json.dumps(e))
        new["run_id"] = "synthetic-run-2"
        new["monotonic_ns"] += 100_000_000_000
        new["unix_ns"] += 100_000_000_000
        if new["event"] in ("online_switch_boundary", "online_switch_armed", "online_switch_applied"):
            continue  # No worker evidence for run 2: must not borrow run 1's same version.
        second.append(new)
    write(p, events+second)
    r = analyze([archive])
    rows = r["sessions"][0]["switches"]
    assert len(rows) == 12 and len(r["sessions"][0]["stages"]) == 14
    assert sum(row["outcome"] == "applied" for row in rows) == 6


def test_queued_old_future_wait_is_explicit_zero_not_missing(archive):
    p = archive / "task-events.jsonl"
    events = read(p)
    for e in events:
        if e["event"] == "switch_policy_wait":
            e["cancelled_before_start"] = True
    write(p, events)
    row = analyze([archive])["sessions"][0]["switches"][0]
    assert row["metrics"]["old_future_wait_ms"] == 0


def test_ambiguous_domains_and_duplicate_requests_are_reported(archive):
    patch_manifest(archive, clock_domains={"*": "boot-A", "task-events.jsonl": "boot-B"})
    p = archive / "task-events.jsonl"
    events = read(p)
    idx = next(i for i, e in enumerate(events) if e["event"] == "stage_requested")
    events.insert(idx+1, events[idx].copy())
    write(p, events)
    r = analyze([archive])
    assert r["summary"]["attempt_count"] == 6 and r["summary"]["applied_count"] == 6
    assert "duplicate_stage_request" in r["quality_issue_distribution"]
    assert "ambiguous_clock_declaration" in r["quality_issue_distribution"]


def test_fault_before_applied_is_conflicting_evidence_not_success(archive):
    p = archive / "task-events.jsonl"
    records = read(p)
    request = next(e for e in records if e["event"] == "stage_requested")
    records.append({**request, "event": "execution_transition", "execution_state": "FAULT", "reason": "feedback fault",
                    "monotonic_ns": 12_200_000_000, "unix_ns": 1_800_000_012_200_000_000})
    write(p, sorted(records, key=lambda e: e["monotonic_ns"]))
    r = analyze([archive])
    assert r["sessions"][0]["switches"][0]["outcome"] == "evidence_conflict"
    assert "terminal_or_failure_before_applied" in r["quality_issue_distribution"]


def test_missing_source_generation_ack_is_not_accepted(archive):
    p = archive / "task-events.jsonl"
    records = read(p)
    applied = next(e for e in records if e["event"] == "online_switch_applied")
    applied["stream"].pop("generation")
    write(p, records)
    assert analyze([archive])["sessions"][0]["switches"][0]["outcome"] == "evidence_conflict"


def test_bad_explicit_manifest_is_an_error(tmp_path):
    from deploy.fr3_sharpa_runtime.experiment_analysis import load_manifest
    p = tmp_path / "bad.json"
    p.write_text('{')
    with pytest.raises(ValueError, match="Invalid analysis manifest"):
        load_manifest(p)


def test_missing_feedback_preserves_available_command_dynamics():
    events = [dict(event="frame", _file="x", _line=i+1, _segment=0, _domain=None, _t=1+i*.03,
                   command=[i*.001]*58) for i in range(5)]
    issues = []
    frames = prepare_frames(events, .1, issues)
    assert frames[-1]["error"] is None
    assert frames[-1]["command_velocity"][0] == pytest.approx(.001/.03)
    assert frames[-1].get("measured_velocity") is None
    assert len(issues) == 5
