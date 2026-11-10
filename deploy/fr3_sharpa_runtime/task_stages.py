"""Frozen B-coarse instructions. Progress labels never enter the model prompt."""

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


def stage(value):
    if value not in PHASES:
        raise ValueError(f"未知任务阶段: {value!r}")
    return STAGES[PHASES.index(value)]


def catalog():
    return [dict(id=s.id, label=s.label, action=s.action, prompt=s.prompt) for s in STAGES]
