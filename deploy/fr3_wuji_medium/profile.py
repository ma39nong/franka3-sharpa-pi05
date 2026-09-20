"""Medium-only settings and model contracts; no global mutation of slow modules."""

import copy
from pathlib import Path

import numpy as np

from .planner import read_config
from .planner import validate_config

ARM_SPEED_RAD_S = 2.0


def planning_config(model="30000", *, minimum_time_scale=0.625):
    from deploy.fr3_wuji_models.registry import profile

    model_spec = profile(model)
    config = read_config(Path(__file__).with_name("config.yaml"))
    config.update(model_profile=model, minimum_time_scale=minimum_time_scale, final_settle_seconds=0.5)
    config["source_hz"] = model_spec.source_hz
    if model != "19999":
        config["arm_raw_initial_delta_rad"] = 1.5
        config["max_smoothing_delta_rad"]["arm"] = 0.2
    return validate_config(config)


def slow_motion_config(medium_config):
    """Use the existing slow trajectory bounds without changing the device session."""
    from experiments.weight_motion_eval import planner as slow_planner

    slow = slow_planner.read_config(Path(slow_planner.__file__).with_name("config.yaml"))
    config = copy.deepcopy(medium_config)
    config["speed_profile"] = "slow"
    config["minimum_time_scale"] = slow["minimum_time_scale"]
    config["settle_seconds"] = slow["settle_seconds"]
    config["initial_min_approach_seconds"] = 1.0
    config["continuation_min_approach_seconds"] = 1.0
    config["slider_speed_rad_s"] = 1.0
    for phase in ("approach", "playback"):
        config[phase] = copy.deepcopy(slow[phase])
    return validate_config(config)


def session_limits(reference):
    # The reference config records the slow gateway's software speed setting.
    # Medium uses the existing explicit per-session speed argument at ALL layers.
    limits = dict(reference)
    limits["speed"] = np.array(reference["speed"], copy=True)
    limits["speed"][np.r_[0:7, 27:34]] = ARM_SPEED_RAD_S
    return limits


def checkpoint_contract(model, checkpoint):
    from deploy.fr3_wuji_models.registry import profile

    return profile(model).contract(checkpoint)
