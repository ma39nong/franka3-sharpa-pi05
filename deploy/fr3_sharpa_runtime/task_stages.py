"""Independent frozen seven/five-stage contracts. Progress IDs never enter prompts."""

from dataclasses import dataclass

ACTION_PROMPTS = {
    "pickup": "Right hand grasps and lifts the truss to the center",
    "harvest": "Right hand holds truss while left hand picks tomatoes into left basket",
    "return": "Right hand returns the held object to the right plate",
    "adjust": "Right fingers reposition the truss on the right plate",
}


@dataclass(frozen=True)
class Stage:
    id: str
    label: str
    action: str

    @property
    def prompt(self):
        return ACTION_PROMPTS[self.action]


STAGES = tuple(
    Stage(*row)
    for row in (
        ("pickup_1", "首次拿起", "pickup"),
        ("harvest_1", "首次采摘", "harvest"),
        ("return_truss", "放回整串", "return"),
        ("adjust", "调整", "adjust"),
        ("pickup_2", "再次拿起", "pickup"),
        ("harvest_2", "再次采摘", "harvest"),
        ("return_stem", "放回裸杆", "return"),
    )
)
PHASES = tuple(s.id for s in STAGES)


IN_HAND_PROMPTS = {
    "pickup": "Right hand grasps and lifts the truss to the center",
    "harvest": "Left hand picks tomatoes into left basket while right hand holds truss",
    "adjust": "Right hand adjusts tomato truss position and orientation at the center",
    "return": "Right hand returns the held object to the right plate",
}


@dataclass(frozen=True)
class InHandStage(Stage):
    @property
    def prompt(self):
        return IN_HAND_PROMPTS[self.action]


IN_HAND_STAGES = tuple(InHandStage(*row) for row in (
    ("pickup", "拿起番茄串", "pickup"),
    ("harvest_1", "首次采摘", "harvest"),
    ("adjust_in_hand", "手中调整", "adjust"),
    ("harvest_2", "再次采摘", "harvest"),
    ("return_stem", "最终放回", "return"),
))
TASK_CONTRACTS = {"tomato-b": "tomato-coarse-four-prompts-en-v4",
                  "tomato-in-hand119": "tomato-in-hand-five-phases-v3"}


def stages(profile="tomato-b"):
    if profile == "tomato-b":
        return STAGES
    if profile == "tomato-in-hand119":
        return IN_HAND_STAGES
    raise ValueError(f"未知任务模式: {profile!r}")


def phase_ids(profile="tomato-b"):
    return tuple(s.id for s in stages(profile))


def stage(value, profile="tomato-b"):
    for item in stages(profile):
        if item.id == value:
            # Preserve old Stage.prompt semantics; only the new contract uses
            # its own verbatim training prompts.
            return item
    raise ValueError(f"未知任务阶段: {value!r} ({profile})")


def catalog(profile="tomato-b"):
    return [dict(id=s.id, label=s.label, action=s.action, prompt=stage(s.id, profile).prompt)
            for s in stages(profile)]
