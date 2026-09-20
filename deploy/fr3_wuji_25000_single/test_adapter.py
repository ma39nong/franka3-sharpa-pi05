"""25000-single metadata is isolated from the tomato_AB model."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.fr3_wuji_25000 import serve as paired_service
from deploy.fr3_wuji_25000.contract import checkpoint_contract as paired_contract
from deploy.fr3_wuji_25000_single import serve
from deploy.fr3_wuji_25000_single.contract import DEFAULT_CHECKPOINT
from deploy.fr3_wuji_25000_single.contract import DEFAULT_URI
from deploy.fr3_wuji_25000_single.contract import checkpoint_contract
from deploy.fr3_wuji_medium.deploy import parse_args
from deploy.fr3_wuji_medium.deploy import write_runtime
from deploy.fr3_wuji_medium.profile import planning_config
from deploy.fr3_wuji_medium.profile import slow_motion_config
from deploy.fr3_wuji_medium.runtime.bridge import configured_hand_kp
from deploy.fr3_wuji_medium.runtime.hand import HandOwner


def test_single_contract_has_own_weights_and_stats():
    single, stats_file = checkpoint_contract()
    paired, _ = paired_contract()
    assert single["checkpoint"] == str(DEFAULT_CHECKPOINT.resolve())
    assert stats_file.parent.name == "tomato_A_15hz"
    assert single["model_action_dim"] == 64
    assert single["action_dim"] == 54
    assert single["model_finetune_mode"] == "full"
    assert single["source_hz"] == 15
    assert single["model_state_order"] == paired["model_state_order"]
    assert single["model_action_order"] == paired["model_action_order"]
    assert single["checkpoint_manifest_sha256"] != paired["checkpoint_manifest_sha256"]
    assert single["normalization_sha256"] != paired["normalization_sha256"]
    assert single["config"] != paired["config"]
    with pytest.raises(ValueError, match="tomato_AB_15hz"):
        paired_contract(DEFAULT_CHECKPOINT)
    with pytest.raises(ValueError, match="tomato_A_15hz"):
        checkpoint_contract(paired["checkpoint"])


def test_single_stats_support_the_same_state_and_action_maps():
    stats_file = checkpoint_contract()[1]
    stats = json.loads(stats_file.read_text())["norm_stats"]
    for group in ("state", "actions"):
        for field in ("mean", "std", "q01", "q99"):
            values = np.asarray(stats[group][field])
            assert values.shape == (64,)
            assert np.count_nonzero(values[54:]) == 0
    state = np.asarray(stats["state"]["mean"])
    actions = np.asarray(stats["actions"]["mean"])
    reordered = np.r_[0:7, 14:34, 7:14, 34:54]
    assert np.mean(np.abs(state[:54] - actions[reordered])) < 0.05
    assert np.mean(np.abs(state[:54] - actions[:54])) > 0.3
    assert serve.base.PhysicalOrderedState is paired_service.PhysicalOrderedState


def test_single_medium_defaults_and_timing_are_isolated():
    args = parse_args(["--model", "25000-single"])
    assert args.checkpoint == DEFAULT_CHECKPOINT
    assert args.uri == DEFAULT_URI
    assert args.slow_after_seconds == 25.0
    single = planning_config("25000-single")
    assert single["source_hz"] == 15
    assert slow_motion_config(single)["source_hz"] == 15
    assert planning_config("25000")["source_hz"] == 15
    assert planning_config("30000v2")["source_hz"] == 30


def test_only_single_profile_requests_lower_left_hand_kp(tmp_path):
    single, _, _ = write_runtime(parse_args(["--model", "25000-single"]), tmp_path)
    assert single["hand_kp"] == {"left": 5.0, "right": 8.0}
    assert configured_hand_kp(single) == single["hand_kp"]
    for model in ("19999", "30000", "30000v2", "25000"):
        old = {"checkpoint": {"config": model}, "hand_kp": {"left": 8.0, "right": 8.0}}
        assert configured_hand_kp(old) == old["hand_kp"]
    with pytest.raises(ValueError, match="Hand Kp"):
        configured_hand_kp({**single, "hand_kp": {"left": 8.0, "right": 8.0}})
    with pytest.raises(ValueError, match="Hand Kp"):
        configured_hand_kp({"checkpoint": {"config": "30000"}, "hand_kp": single["hand_kp"]})


@pytest.mark.parametrize("kp", [5.0, 8.0])
def test_disabled_hand_receives_and_reads_back_only_its_selected_kp(kp):
    owner = HandOwner.__new__(HandOwner)
    owner.hand_control, owner.owns_enable, owner.faulted = "slider", False, False
    owner.deployment_kp = kp
    owner.poll = lambda *_: (np.zeros(20), np.zeros(20), 0.0)
    owner.diagnostics = ({index: SimpleNamespace(status_word=SimpleNamespace(ext_state=1)) for index in range(20)}, 0.0)
    gains = [[8.0, 0.1] for _ in range(20)]
    currents = [2.0] * 20
    events = []
    owner.trace = SimpleNamespace(add=lambda *args, **kwargs: events.append((args, kwargs)))

    def set_gains(values):
        gains[:] = [list(value) for value in values]

    def set_current(value):
        currents[:] = [value] * 20

    owner.hand = SimpleNamespace(
        effort_limit=lambda: SimpleNamespace(set=set_current),
        mit_params=lambda: SimpleNamespace(set=set_gains),
    )
    owner.identity = lambda: {
        "mit_gains": [{"kp": pair[0], "kd": pair[1]} for pair in gains],
        "effort_limits": list(currents),
    }
    actual = owner.configure_deployment()
    assert actual["mit_gains"] == [{"kp": kp, "kd": 0.1}] * 20
    assert actual["effort_limits"] == [2.0] * 20
    assert owner.trace_identity == actual
    assert any(args == ("deployment_parameters_write_attempt",) and kwargs["kp"] == kp for args, kwargs in events)
