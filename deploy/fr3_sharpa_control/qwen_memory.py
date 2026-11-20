"""Plan and evidence ledger, independent of models, ROS and hardware."""

import copy
from dataclasses import dataclass
import itertools
import math


@dataclass(frozen=True)
class HarvestPlan:
    quotas: tuple[int, ...] = (2, 2)

    def __post_init__(self):
        if (
            not isinstance(self.quotas, tuple)
            or len(self.quotas) not in (1, 2)
            or any(type(n) is not int or not 1 <= n <= 50 for n in self.quotas)
        ):
            raise ValueError("计划须为一轮或两轮正整数,例如 (4,) 或 (2, 2)")

    @property
    def stages(self):
        return (
            "pickup",
            "harvest_1",
            *(() if len(self.quotas) == 1 else ("adjust_in_hand", "harvest_2")),
            "return_stem",
        )


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


class TaskMemory:
    def __init__(self, run_id, plan=None):
        plan = HarvestPlan() if plan is None else plan
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("执行会话不能为空")
        self.run_id, self.plan = run_id, plan
        self.stage_index = self.revision = 0
        self.events = []
        self.last_end = None
        self.stage_started_at = None
        self.blockers = []
        self.processed = set()

    @property
    def stage(self):
        return self.plan.stages[self.stage_index]

    def snapshot(self):
        counts = {name: sum(e["stage"] == name for e in self.events) for name in ("harvest_1", "harvest_2")}
        quota = self.plan.quotas[int(self.stage[-1]) - 1] if self.stage.startswith("harvest_") else None
        return copy.deepcopy(
            {
                "run_id": self.run_id,
                "revision": self.revision,
                "quotas": list(self.plan.quotas),
                "stages": list(self.plan.stages),
                "active_stage": self.stage,
                "counts": counts,
                "quota": quota,
                "events": self.events,
                "last_end": self.last_end,
                "blockers": self.blockers,
                "stage_started_at": self.stage_started_at,
            }
        )

    def evaluate(self, request, answer, *, now, max_age_s=2.0):
        if (
            request["run_id"] != self.run_id
            or request["revision"] != self.revision
            or request["active_stage"] != self.stage
        ):
            raise ValueError("会话或步骤已变化,旧回答未采用")
        if request["id"] in self.processed:
            return {"status": "duplicate", "proposed_stage": None, "memory": self.snapshot()}
        frames = {f["id"]: f["timestamp"] for f in request["frames"]}
        start, end = request["start"], request["end"]
        if (
            not finite(now)
            or not finite(start)
            or not finite(end)
            or not start < end <= now
            or not frames
            or len(frames) != len(request["frames"])
            or any(not finite(t) or not start <= t <= end for t in frames.values())
            or max_age_s <= 0
        ):
            raise ValueError("观测时间或画面身份无效")
        if self.last_end is not None and end <= self.last_end:
            raise ValueError("观测乱序")
        if self.stage_started_at is not None and start < self.stage_started_at:
            raise ValueError("观测包含当前步骤实际生效之前的画面")
        for key in ("ready", "left_empty"):
            if answer.get(key) is not None and type(answer[key]) is not bool:
                raise ValueError("观察状态必须为布尔或null")
        uncertainties = answer.get("uncertainties")
        if type(answer.get("counting_complete")) is not bool:
            raise ValueError("缺少投放观察覆盖状态")
        if not isinstance(uncertainties, list) or any(not isinstance(s, str) for s in uncertainties):
            raise ValueError("缺少疑点列表")
        evidence = answer.get("ready_evidence")
        events = answer.get("events")
        if (
            not isinstance(evidence, list)
            or any(not isinstance(k, str) or k not in frames for k in evidence)
            or not isinstance(events, list)
            or len(events) > 20
        ):
            raise ValueError("画面证据或事件列表无效")
        candidates = []
        for event in events:
            try:
                a, b, c = (frames[event[k]] for k in ("held", "released", "empty"))
                if not a < b < c or not isinstance(event["evidence"], str) or not event["evidence"].strip():
                    raise ValueError("投放事件必须有先持果、后释放、再空手的证据")
            except (KeyError, TypeError) as error:
                raise ValueError("事件引用了未知画面") from error
            candidates.append(
                {"held": a, "released": b, "empty": c, "evidence": event["evidence"], "stage": self.stage}
            )
        if candidates and not self.stage.startswith("harvest_"):
            raise ValueError("非采摘步骤出现投放事件,需要核查实际阶段")
        updated = copy.deepcopy(self.events)
        blockers = list(self.blockers)
        if not answer["counting_complete"] and self.stage.startswith("harvest_"):
            blockers.append("采摘观察存在未确认信息,需要核查后恢复精确计数")
        if self.last_end is not None and start > self.last_end:
            blockers.append("观测存在未覆盖时段,累计计数可能漏失")
        if (
            self.last_end is None
            and self.stage_started_at is not None
            and self.stage.startswith("harvest_")
            and start > self.stage_started_at + 1
        ):
            blockers.append("未覆盖本轮采摘开始后的完整历史,不能从零猜测累计数")
        for camera in {frame["camera"] for frame in request["frames"]}:
            stamps = sorted({frame["timestamp"] for frame in request["frames"] if frame["camera"] == camera})
            if any(b - a > 1.0 for a, b in itertools.pairwise(stamps)):
                blockers.append("相机采样存在超过一秒的缺口,需要核查是否漏计")
        for event in sorted(candidates, key=lambda e: e["released"]):
            overlaps = [e for e in updated if event["held"] <= e["empty"] and event["empty"] >= e["held"]]
            if overlaps:
                if any(abs(e["released"] - event["released"]) > 0.75 for e in overlaps):
                    blockers.append("投放证据与已有事件重叠但释放时间不同,需要核查")
                continue
            updated.append(event)
        current_count = sum(e["stage"] == self.stage for e in updated)
        quota = self.snapshot()["quota"]
        if quota is not None and current_count > quota:
            blockers.append("已观察投放数超过本轮目标,不能宣称精确完成计划")
        ready_times = {frames[k] for k in evidence}
        ready = (
            answer.get("ready") is True
            and answer.get("left_empty") is True
            and len(ready_times) >= 2
            and max(ready_times) >= end - 0.5
            and not uncertainties
            and not blockers
            and now - end <= max_age_s
        )
        if quota is not None:
            ready = ready and current_count == quota
        self.events, self.blockers = updated, list(dict.fromkeys(blockers))
        self.last_end = end
        self.processed.add(request["id"])
        target = (
            self.plan.stages[self.stage_index + 1] if ready and self.stage_index + 1 < len(self.plan.stages) else None
        )
        return {
            "status": "fresh" if now - end <= max_age_s else "historical_only",
            "proposed_stage": target,
            "requires_execution_ack": target is not None,
            "manual_completion_required": self.stage == "return_stem",
            "memory": self.snapshot(),
        }

    def acknowledge(self, target, *, run_id, expected_revision, applied_at=None):
        if (
            run_id != self.run_id
            or expected_revision != self.revision
            or self.stage_index + 1 >= len(self.plan.stages)
            or target != self.plan.stages[self.stage_index + 1]
            or self.blockers
        ):
            raise ValueError("只能根据同会话执行回执确认下一个计划步骤")
        if self.stage.startswith("harvest_"):
            state = self.snapshot()
            if state["counts"][self.stage] != state["quota"]:
                raise ValueError("实际阶段已经推进,但本轮投放计数尚未与计划一致,需要人工核查")
        if applied_at is not None and (
            not finite(applied_at) or (self.last_end is not None and applied_at < self.last_end)
        ):
            raise ValueError("步骤实际生效时间无效")
        self.stage_index += 1
        self.revision += 1
        self.last_end = None
        self.stage_started_at = applied_at
        self.processed.clear()

    @classmethod
    def restore(cls, data):
        memory = cls(data["run_id"], HarvestPlan(tuple(data["quotas"])))
        if data["active_stage"] not in memory.plan.stages:
            raise ValueError("恢复的阶段不在计划内")
        memory.stage_index = memory.plan.stages.index(data["active_stage"])
        memory.revision = data["revision"]
        memory.events = copy.deepcopy(data["events"])
        memory.last_end = data["last_end"]
        memory.stage_started_at = data.get("stage_started_at")
        memory.blockers = list(data["blockers"])
        return memory
