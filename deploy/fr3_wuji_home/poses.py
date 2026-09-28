"""Load Pi05-owned named dual-arm/dual-hand Home poses."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

SIDES = ("left", "right")
ARM_COUNT = 7
HAND_COUNT = 20
DEFAULT_POSES = Path(__file__).with_name("poses.yaml")


def _vector(value, count, label):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain {count} finite positions")
    return result.copy()


@dataclass(frozen=True)
class HomePose:
    name: str
    description: str
    target: np.ndarray
    source: Path
    provenance: dict


def load_home(name=None, path=DEFAULT_POSES):
    path = Path(path).resolve()
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("Unsupported Home pose document")
    homes = document.get("homes")
    selected = document.get("default_home") if name is None else name
    if not isinstance(homes, dict) or not isinstance(selected, str) or selected not in homes:
        raise ValueError(f"Unknown Home pose: {selected!r}")
    entry = homes[selected]
    if (
        not isinstance(entry, dict)
        or set(entry.get("arms", {})) != set(SIDES)
        or set(entry.get("hands", {})) != set(SIDES)
    ):
        raise ValueError(f"Home {selected!r} must contain both arms and both hands")
    sections = {
        "left_arm": _vector(entry["arms"]["left"], ARM_COUNT, f"{selected} left arm"),
        "left_hand": _vector(entry["hands"]["left"], HAND_COUNT, f"{selected} left hand"),
        "right_arm": _vector(entry["arms"]["right"], ARM_COUNT, f"{selected} right arm"),
        "right_hand": _vector(entry["hands"]["right"], HAND_COUNT, f"{selected} right hand"),
    }
    target = np.concatenate(tuple(sections.values()))
    target.flags.writeable = False
    return HomePose(
        name=selected,
        description=str(entry.get("description", "")),
        target=target,
        source=path,
        provenance=dict(entry.get("provenance", {})),
    )
