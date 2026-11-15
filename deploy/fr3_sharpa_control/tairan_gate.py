"""Pinned Tairan handshake bridged to execution-owned seven-position progress.

Strategy commit is a request; perception ACK windows begin at actual stage
installation. This explicit live adaptation is distinct from offline replay.
"""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

from .agent import DecisionGate, LABELS, CLASSES
from deploy.fr3_sharpa_runtime.task_stages import PHASES


class TairanGate:
    def __init__(self, repository, policy_path):
        root = Path(repository)
        inventory = json.loads((root / "configs/tairan_deployment.json").read_text())
        path = root / "tools/tomato_review_ui/handshake.py"
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != inventory["code"]["source_sha256"]["tools/tomato_review_ui/handshake.py"]:
            raise ValueError("tairan 握手源码身份不符")
        if hashlib.sha256(Path(policy_path).read_bytes()).hexdigest() != inventory["handshake"]["sha256"]:
            raise ValueError("tairan 握手配置身份不符")
        name = "_tairan_handshake_" + sha[:16]
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        self.module = sys.modules[name]
        self.config = self.module.validate(json.loads(Path(policy_path).read_text()))
        self.reset()

    def reset(self):
        self.validator = DecisionGate(horizon=1.0)
        self.engine = None
        self.bound = None
        self.pending = None
        self.proposal = None
        self.next_tick = None
        self.count = 0
        self.observation_barrier = None
        self.trace = []

    def _new(self, task, now):
        self.engine = self.module.Handshake(self.config)
        self.engine.active = LABELS[CLASSES[PHASES.index(task["active_stage"])]]
        self.bound = (task["run_id"], task["stage_version"], task["active_stage"])
        self.pending = None
        self.proposal = None
        self.observation_barrier = now
        self.next_tick = self._next_tick(now)

    def _next_tick(self, now):
        period = self.config["plan_period_s"]
        offset = self.config["plan_offset_s"]
        return (math.floor((now - offset) / period) + 1) * period + offset

    def _sync(self, task, now):
        key = (task["run_id"], task["stage_version"], task.get("active_stage"))
        if self.pending:
            p = self.pending
            if (
                task["run_id"] != p["run_id"]
                or task["stage_version"] != p["stage_version"]
                or task["requested_stage"] != p["target"]
            ):
                raise ValueError("tairan 在途阶段与执行端版本不一致")
            if task["execution_state"] == "RUNNING" and task["active_stage"] == p["target"]:
                applied = task.get("active_applied_unix_s")
                if (
                    not isinstance(applied, (int, float))
                    or not math.isfinite(applied)
                    or not p["requested_at"] <= applied <= now
                ):
                    raise ValueError("缺少可信的阶段实际生效时间")
                self.bound = key
                self.pending = None
                self.proposal = None
                self.engine.commit_at = applied
                self.engine.deadline = applied + self.config["ack_timeout_s"]
                self.engine.hard_deadline = applied + self.config["late_ack_timeout_s"]
                self.engine.last = None
                self.observation_barrier = applied
                self.trace.append(dict(kind="execution_applied", time=applied, target=task["active_stage"]))
            elif now - p["requested_at"] > 5:
                raise TimeoutError("tairan 执行接入反馈超时")
            else:
                return False
        if not self.engine or key != self.bound:
            if (
                task["active_stage"] not in PHASES
                or task["requested_stage"] != task["active_stage"]
                or task["execution_state"] != "RUNNING"
            ):
                return False
            self._new(task, now)
        return task["execution_state"] == "RUNNING" and task["active_stage"] == task["requested_stage"]

    def update(self, packet, task, *, now, model_version):
        self.validator.update(packet, {**task, "execution_state": "READY"}, now=now, model_version=model_version)
        if not self._sync(task, now):
            return None
        if packet["input_timestamp"] <= self.observation_barrier:
            return None
        before = len(self.engine.events)
        self.engine.prediction(
            self.module.Evidence(
                packet["input_timestamp"],
                now,
                tuple(packet["current_probabilities"]),
                tuple(packet["future_probabilities"]),
            )
        )
        self.trace.extend(self.engine.events[before:])
        self.engine.events.clear()
        self.count = (
            self.engine.late_count
            if self.engine.state == "WAIT_LATE_ACK"
            else (self.engine.pending or {}).get("count", 0)
        )
        if self.engine.state == "RECOVERY_REQUIRED":
            raise RuntimeError("tairan ACK硬截止，需要人工恢复")
        return self.poll(task, now=now)

    def poll(self, task, *, now):
        if not self._sync(task, now):
            return None
        if self.proposal:
            return None
        before = len(self.engine.events)
        self.engine.timer(now)
        phase = PHASES.index(task["active_stage"])
        if phase == 6:
            self.engine.cancel(now, "final_return_requires_operator_completion")
        elif now >= self.next_tick:
            self.next_tick = self._next_tick(now)
            if self.engine.last and now - self.engine.last.observed_at <= self.config["prediction_ttl_s"]:
                previous = self.engine.version
                self.engine.planning(now)
                if self.engine.version != previous:
                    target = PHASES[phase + 1]
                    if self.engine.active != LABELS[CLASSES[phase + 1]]:
                        raise ValueError("tairan 非法七阶段后继")
                    self.proposal = dict(target=target, requested_at=now)
        self.trace.extend(self.engine.events[before:])
        self.engine.events.clear()
        if self.engine.state == "RECOVERY_REQUIRED":
            raise RuntimeError("tairan ACK硬截止，需要人工恢复")
        return self.proposal["target"] if self.proposal else None

    def submitted(self, task):
        if not self.proposal:
            raise ValueError("缺少 tairan 策略提议")
        self.pending = dict(run_id=task["run_id"], stage_version=task["stage_version"], **self.proposal)
        self.proposal = None

    def take_events(self):
        events = self.trace
        self.trace = []
        return events

    def describe(self):
        return dict(
            policy_state="WAIT_EXECUTION_ACK" if self.pending else self.engine.state if self.engine else "OBSERVING",
            perception_ack_deadline=self.engine.deadline if self.engine else None,
            hard_ack_deadline=self.engine.hard_deadline if self.engine else None,
        )
