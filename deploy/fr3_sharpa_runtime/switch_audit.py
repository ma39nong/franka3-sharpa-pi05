"""Repeatable hardware-free coordinator benchmark and task-event summarizer.

All timings from ``benchmark`` are synthetic workloads on the real coordinator
and timeline, NOT robot/camera/GPU measurements. No ROS, SDK or weights loaded.
"""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace
import uuid

import numpy as np
from openpi_client import rtc

from .stage_handoff import StageTimeline, DT
from .task_control import TaskControl
from .task_rtc import TaskRTC


def load_baseline(ref, filename):
    source = subprocess.run(["git", "show", f"{ref}:deploy/fr3_sharpa_runtime/{filename}.py"],
                            check=True, text=True, capture_output=True).stdout
    module = ModuleType(f"deploy.fr3_sharpa_runtime._audit_{filename}")
    module.__package__ = "deploy.fr3_sharpa_runtime"
    exec(compile(source, f"{ref}/{filename}.py", "exec"), module.__dict__)
    return module


class SimBackend:
    def __init__(self, timeline, phase):
        self.plan = timeline(np.zeros((50, 58), np.float32), time.monotonic() - phase * DT,
                             1.0, task_epoch=0)
        self.next_tick = time.monotonic()
        self.last_command = self.plan.sample(self.next_tick)
        self.applied_at = None
        self.installs = []

    def status(self):
        now = time.monotonic()
        while self.next_tick <= now:
            previous = self.plan.applied_token
            self.last_command = self.plan.sample(self.next_tick)
            if self.plan.applied_token != previous:
                self.applied_at = self.next_tick
            self.next_tick += DT
        return self.plan.status(now)

    def begin_switch(self, epoch, request_id):
        return self.plan.begin_switch(epoch, request_id, time.monotonic())

    def snapshot(self, settings, delay):
        return self.plan.snapshot(time.monotonic(), delay_steps=delay,
                                  schedule=settings.prefix_attention_schedule, weight=settings.max_guidance_weight)

    def install(self, actions, receipt, ticket):
        self.installs.append(ticket["stage_version"])
        return self.plan.install(actions, receipt, time.monotonic(), task_epoch=ticket["stage_version"])

    def handoff(self, actions, boundary, captured):
        return self.plan.prepare_handoff(actions, {**boundary, "captured_monotonic_s": captured},
                                         time.monotonic(), self.next_tick, self.last_command, 0.7, self.validate)

    @staticmethod
    def validate(actions):
        if np.any(np.abs(actions) > 1):
            raise ValueError("simulated range violation")

    def stop(self):
        self.plan.invalidate()
        return dict(status="PASS", hands_disabled=True, arms_safe=True, old_stream_invalidated=True,
                    simulation_only=True)


class SimPolicy:
    def __init__(self, old_ms, infer_ms, offset):
        self.old_ms, self.infer_ms, self.offset = old_ms, infer_ms, offset
        self.entered = threading.Event()
        self.last_receipt = None
        self.active = self.max_active = 0

    def infer_chunk(self, observation, *, rtc_context=None, response_timeout=5):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.entered.set()
        try:
            delay = (self.old_ms if rtc_context else self.infer_ms) / 1000
            time.sleep(min(delay, response_timeout))
            if delay > response_timeout:
                raise TimeoutError("simulated policy timeout")
            self.last_receipt = rtc.make_receipt(rtc_context) if rtc_context else None
            offset = 0 if rtc_context else self.offset
            return {"checkpoint": "synthetic-no-weights"}, np.full((50, 58), offset, np.float32), None
        finally:
            self.active -= 1


def simulate(engine_class, timeline_class, *, phase=15, old_ms=0, observation_ms=20, infer_ms=60,
             offset=0.002, rapid=False):
    events = []
    c = TaskControl(emit=lambda **event: events.append(event))
    c.ready()
    c.command("start", uuid.uuid4().hex, expected_version=0)
    c.activate(c.ticket(), "0:1")
    b = SimBackend(timeline_class, phase)
    policy = SimPolicy(old_ms, infer_ms, offset)

    def observe(**options):
        until = time.monotonic() + observation_ms / 1000
        while time.monotonic() < until:
            options.get("on_poll", lambda: None)()
            time.sleep(0.001)
        return {"observation/state": np.zeros(58), "_captured_monotonic_ns": time.monotonic_ns()}

    args = SimpleNamespace(rtc_initial_delay_steps=5, rtc_delay_buffer_size=10,
                           rtc_prefix_attention_schedule="exp", rtc_max_guidance_weight=5,
                           execute_steps=15, num_chunks=0)
    with tempfile.TemporaryDirectory(prefix="switch-audit-") as directory:
        e = engine_class(c, policy, b, observe, args, directory)
        e.streaming = True
        try:
            if old_ms:
                context = b.snapshot(e.settings, 5)
                e.submit_observed({"observation/state": np.zeros(58),
                                   "_captured_monotonic_ns": time.monotonic_ns()}, context)
                if not policy.entered.wait(1):
                    raise RuntimeError("Synthetic old policy did not start")
            clicked = time.monotonic()
            c.command("next", uuid.uuid4().hex, expected_version=0)
            reason = None
            did_rapid = False
            while c.snapshot()["execution_state"] != "RUNNING":
                if time.monotonic() - clicked > 1.5:
                    raise TimeoutError("simulation harness timeout")
                if rapid and not did_rapid and time.monotonic() - clicked > 0.035:
                    c.command("next", uuid.uuid4().hex, expected_version=1)
                    did_rapid = True
                try:
                    e.tick()
                except Exception as error:
                    reason = str(error)
                    c.fault(error)
                    b.stop()
                    break
                # Reproduce production scheduling; optimized completions/intent wake immediately.
                if hasattr(c, "wakeup") and hasattr(e, "observation_pool"):
                    c.wakeup.wait(0.01)
                    c.wakeup.clear()
                else:
                    time.sleep(0.01)
            elapsed = time.monotonic() - clicked
            start_events = [x for x in events if x["event"] == "inference_start" and not x["rtc_guidance"]]
            return dict(status="PASS" if reason is None else "BLOCKED", reason=reason,
                        observed_applied_ms=elapsed * 1000 if reason is None else None,
                        applied_tick_ms=(b.applied_at - clicked) * 1000 if b.applied_at else None,
                        new_policy_start_ms=(start_events[0]["monotonic_ns"] / 1e9 - clicked) * 1000
                        if start_events else None,
                        max_concurrent_policy=policy.max_active,
                        old_stage_installs=b.installs,
                        events=events)
        finally:
            e.pool.shutdown(wait=True, cancel_futures=True)
            if hasattr(e, "observation_pool"):
                e.observation_pool.shutdown(wait=True, cancel_futures=True)


def benchmark(baseline="edeb45f", repeats=3):
    old_engine = load_baseline(baseline, "task_rtc").TaskRTC
    old_timeline = load_baseline(baseline, "stage_handoff").StageTimeline
    cases = [(f"phase_{phase}", dict(phase=phase)) for phase in (10, 15, 25, 35, 40, 45)]
    cases += [("overlap", dict(phase=25, old_ms=120, observation_ms=80)),
              ("late_overlap", dict(phase=35, old_ms=100, observation_ms=80)),
              ("stale_prefetch", dict(phase=25, old_ms=220, observation_ms=20)),
              ("slow_images", dict(phase=15, observation_ms=150)),
              ("slow_policy", dict(phase=15, infer_ms=250)),
              ("old_policy_timeout", dict(phase=15, old_ms=900)),
              ("rapid", dict(phase=15, old_ms=100, observation_ms=40, rapid=True)),
              ("large_difference", dict(phase=15, offset=0.5)),
              ("late_fast", dict(phase=40, observation_ms=1, infer_ms=5))]
    source_cost = {}
    for name, cls in (("before", old_timeline), ("after", StageTimeline)):
        samples = []
        for _ in range(3):
            timeline = cls(np.zeros((50, 58)), 1, 1, task_epoch=0)
            timeline.begin_switch(1, "microbenchmark", 1.3)
            entered = time.perf_counter()
            for _ in range(1000):
                timeline.source_curve().sample(1.4)
            samples.append((time.perf_counter() - entered) * 1e6 / 1000)
        source_cost[name] = statistics.median(samples)
    report = dict(kind="synthetic-coordinator-benchmark-NOT-hardware", baseline=baseline,
                  baseline_scope="git TaskRTC + StageTimeline; common simulated backend/control",
                  repeats=repeats, source_sample_median_us=source_cost, rows=[])
    for name, parameters in cases:
        row = dict(case=name, parameters=parameters)
        for label, engine, timeline in (("before", old_engine, old_timeline), ("after", TaskRTC, StageTimeline)):
            runs = [simulate(engine, timeline, **parameters) for _ in range(repeats)]
            good = [r for r in runs if r["status"] == "PASS"]
            row[label] = dict(passed=len(good), total=len(runs),
                              median_applied_ms=statistics.median(r["applied_tick_ms"] for r in good) if good else None,
                              median_policy_start_ms=statistics.median(r["new_policy_start_ms"] for r in good) if good else None,
                              max_concurrent_policy=max(r["max_concurrent_policy"] for r in runs),
                              reasons=[r["reason"] for r in runs if r["reason"]])
        report["rows"].append(row)
    return report


def summarize(path, manifest=None):
    """Legacy single-file view backed by the clock-aware experiment analyzer."""
    from .experiment_analysis import analyze
    report = analyze([path], manifest)
    names = {"boundary": "online_switch_boundary", "observation_start": "observation_start",
             "observation_complete": "observation_complete", "inference_start": "inference_start",
             "inference_complete": "inference_complete", "armed": "online_switch_armed",
             "parent_ack": "online_switch_applied", "ordinary_rtc_resumed": "ordinary_rtc_resumed"}
    return dict(source=str(path), kind="recorded-events-source-not-certified",
                switches=[dict(stage_version=r["stage_version"], request_id=r["request_id"],
                               milestones_ms={names[t["milestone"]]: t["since_registered_ms"]
                                              for t in r["timeline"] if t["milestone"] in names},
                               worker_command_applied_ms=r["metrics"]["registered_to_applied_ms"],
                               http_to_registered_ms=None,  # Controller clock not present in legacy schema.
                               scheduled_bridge_start_ms=next(t["since_registered_ms"] for t in r["timeline"]
                                                              if t["milestone"] == "scheduled_bridge_start"))
                          for session in report["sessions"] for r in session["switches"]],
                quality_issue_distribution=report["quality_issue_distribution"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="edeb45f")
    parser.add_argument("--repeats", type=int, default=3)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--events", type=Path)
    group.add_argument("--experiment", type=Path, help="archive/directory; --output becomes report directory")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.experiment:
        from .experiment_analysis import analyze, export, load_manifest
        manifest = load_manifest(args.manifest) if args.manifest else None
        export(analyze([args.experiment], manifest), args.output)
        print(args.output / "report.md")
        return
    if not 1 <= args.repeats <= 20:
        parser.error("repeats must be in [1,20]")
    if args.events:
        from .experiment_analysis import load_manifest
        report = summarize(args.events, load_manifest(args.manifest) if args.manifest else None)
    else:
        report = benchmark(args.baseline, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
