"""Create deterministic SYNTHETIC archival fixtures and their analysis report.

This module writes files only. It never runs TaskRTC, ROS, a policy or hardware.
"""

import argparse
import json
from pathlib import Path

from .experiment_analysis import analyze, export
from .task_stages import STAGES


def create_example(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"
    logs.mkdir()
    run_id = "synthetic-seven-stage"
    parent, worker = [], []
    wall_base = 1_800_000_000

    def event(name, t, version, **fields):
        return dict(event=name, monotonic_ns=round(t*1e9), unix_ns=round((wall_base+t)*1e9),
                    run_id=run_id, stage_version=version, simulation_only=True, **fields)

    parent += [event("execution_transition", 10, 0, execution_state="RUNNING", active_stage="pickup_1"),
               event("chunk_active", 10.001, 0, active_stage="pickup_1", inference_request_id="initial")]
    worker.append(dict(event="start", monotonic_s=10.001, task_epoch=0, task_request_id="initial"))
    for version, stage in enumerate(STAGES[1:], 1):
        t = 10+2*version
        rid, iid, old = f"demo-{version}", f"new-{version}", f"old-{version}"
        token = f"synthetic-token-{version}"
        parent += [event("stage_requested", t, version, request_id=rid, target_stage=stage.id,
                         previous_stage=STAGES[version-1].id, client_clicked_unix_ms=(wall_base+t-.005)*1000,
                         http_received_monotonic_ns=round((t-.002)*1e9), ipc_received_monotonic_ns=round((t-.001)*1e9)),
                   event("switch_policy_wait", t+.0025, version, cancelled_before_start=False, old_inference_request_id=old)]
        boundary = dict(token=token, task_epoch=version, request_id=rid, source_generation=version*3,
                        source_sha256="synthetic-source", requested_at=t+.002, deadline=t+.752)
        parent += [event("online_switch_boundary", t+.003, version, requested_stage=stage.id, boundary=boundary),
                   event("observation_start", t+.004, version, inference_request_id=iid, request_stage_version=version),
                   event("inference_complete", t+.015, version, inference_request_id=old, request_stage_version=version-1),
                   event("inference_discarded", t+.017, version, inference_request_id=old, request_stage_version=version-1, reason="stage/state invalidated"),
                   event("observation_complete", t+.024, version, inference_request_id=iid, request_stage_version=version,
                         captured_monotonic_ns=round((t+.022)*1e9), alignment=dict(source_span_ms=10, source_age_ms=dict(cam0=25, cam1=10, cam2=12))),
                   event("inference_start", t+.025, version, inference_request_id=iid, request_stage_version=version,
                         captured_monotonic_ns=round((t+.022)*1e9), rtc_guidance=False, prompt=stage.prompt),
                   event("inference_complete", t+.085, version, inference_request_id=iid, request_stage_version=version,
                         observation_age_ms=63, policy_duration_s=.06)]
        installation = dict(token=token, bridge_start=t+.1, bridge_end=t+.3, observation_age_s=.068,
                            generation=version*3, accepted=True, applied=False)
        parent += [event("online_switch_armed", t+.091, version, inference_request_id=iid, installation=installation),
                   event("online_switch_applied", t+.305, version, token=token, installation=installation,
                         stream=dict(applied_token=token, task_epoch=version, generation=version*3+1,
                                     applied_monotonic_s=t+.3)),
                   event("chunk_active", t+.3051, version, active_stage=stage.id, inference_request_id=iid),
                   event("ordinary_rtc_resumed", t+.6, version, inference_request_id=f"guided-{version}")]
        worker += [dict(event="switch_requested", monotonic_s=t+.0021, task_epoch=version, boundary=boundary),
                   dict(event="handoff_armed", monotonic_s=t+.09, task_epoch=version, **installation),
                   dict(event="handoff_applied", monotonic_s=t+.3001, task_epoch=version, token=token,
                        generation=version*3+1)]
        for i in range(14):
            tick = t-.1+i/30
            phase = (10, 15, 25, 20, 15, 25)[version-1]+(tick-t)*30
            epoch = version if tick > t+.002 else version-1
            blend = t+.1-1e-8 <= tick < t+.3-1e-8
            # Post-send timestamps include a fixed 0.2ms logging delay.
            command = [round(version*.001 + .003*(tick-t) + joint*.00001, 9) for joint in range(58)]
            measured = [value-(.001 if joint < 7 or 29 <= joint < 36 else .003) for joint, value in enumerate(command)]
            worker.append(dict(event="frame", monotonic_s=tick+.0002, task_epoch=epoch, phase=phase,
                               handoff_state="blend" if blend else "stage" if tick >= t+.3-1e-8 else "bounded_previous",
                               handoff_token=token if blend else None, command=command, measured=measured))
    parent.append(event("execution_transition", 23, 7, execution_state="COMPLETED", reason="synthetic operator complete", active_stage="return_stem"))
    for filename, records, key in (("task-events.jsonl", parent, "monotonic_ns"), ("control-synthetic.jsonl", worker, "monotonic_s")):
        (logs / filename).write_text("".join(json.dumps(e, ensure_ascii=False)+"\n" for e in sorted(records, key=lambda e: e[key])))
    status = dict(run_id=run_id, execution_state="COMPLETED", pending_request_id="synthetic-complete",
                  stage_version=7, active_stage="return_stem", safe_to_resume=False, simulation_only=True)
    (logs / "task-run.json").write_text(json.dumps(dict(status=status, accepted_chunks=7)))
    (logs / "task-final.json").write_text(json.dumps(status))
    manifest = dict(schema_version=1, source_kind="synthetic",
                    clock_domains={"task-events.jsonl": "synthetic-host/boot-demo", "control-*.jsonl": "synthetic-host/boot-demo"},
                    embedded_clock_domains=dict(worker="synthetic-host/boot-demo", observation="synthetic-host/boot-demo"),
                    browser_wall_clock=dict(verified=True, offset_to_task_ms=0, uncertainty_ms=1),
                    max_frame_gap_s=.1, notes="DETERMINISTIC SYNTHETIC EXCERPTS; no robot, camera or GPU used")
    (logs / "analysis-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="must be a new directory")
    args = parser.parse_args()
    root = create_example(args.out)
    export(analyze([root]), root / "report")
    print(root / "report/report.md")


if __name__ == "__main__":
    main()
