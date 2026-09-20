"""Medium-only settings and model contracts; no global mutation of slow modules."""
from pathlib import Path
import copy
import numpy as np
from .planner import read_config, validate_config

ARM_SPEED_RAD_S = 2.0


def planning_config(model="30000", *, minimum_time_scale=0.625):
    if model not in {"19999", "30000", "30000v2", "25000", "25000-single"}:
        raise ValueError("Unknown model profile")
    config = read_config(Path(__file__).with_name("config.yaml"))
    config.update(model_profile=model, minimum_time_scale=minimum_time_scale,
                  final_settle_seconds=0.5)
    if model in {"25000", "25000-single"}:
        # Both checkpoints have 15hz normalization assets. Keep their
        # action timebase separate from the older 30 Hz model profiles.
        config["source_hz"] = 15
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
    if model == "19999":
        from experiments.weight_motion_eval.oneshot.policy_server import checkpoint_contract as contract
    elif model == "30000":
        from deploy.fr3_wuji_30000.serve import checkpoint_contract as contract
    elif model == "30000v2":
        from deploy.fr3_wuji_30000v2.contract import checkpoint_contract as contract
    elif model == "25000":
        from deploy.fr3_wuji_25000.contract import checkpoint_contract as contract
    elif model == "25000-single":
        from deploy.fr3_wuji_25000_single.contract import checkpoint_contract as contract
    else:
        raise ValueError("Unknown model profile")
    return contract(checkpoint)
