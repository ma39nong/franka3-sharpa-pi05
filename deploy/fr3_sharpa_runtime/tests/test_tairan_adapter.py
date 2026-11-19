"""Live bridge tests use pinned policy source and synthetic probabilities only."""

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from deploy.fr3_sharpa_control.tairan_gate import TairanGate
from deploy.fr3_sharpa_control.agent_settings import AgentSettings
from test_agent_integration import packet, task


@pytest.fixture
def gate(tmp_path):
    vendor = Path(__file__).resolve().parents[2] / "fr3_sharpa_control/vendor"
    source = tmp_path / "tools/tomato_review_ui/handshake.py"
    source.parent.mkdir(parents=True)
    shutil.copy2(vendor / "tairan_handshake_90ac57e.py", source)
    policy = tmp_path / "configs/tairan_handshake.json"
    policy.parent.mkdir()
    shutil.copy2(vendor / "tairan_handshake.json", policy)
    inventory = dict(
        code=dict(
            source_sha256={"tools/tomato_review_ui/handshake.py": hashlib.sha256(source.read_bytes()).hexdigest()}
        ),
        handshake=dict(sha256=hashlib.sha256(policy.read_bytes()).hexdigest()),
    )
    (tmp_path / "configs/tairan_deployment.json").write_text(json.dumps(inventory))
    return TairanGate(tmp_path, policy)


def prediction(t, current=0, future=1):
    r = packet(t, label=current)
    f = packet(t, label=future)
    r.update(future_horizon_s=1.0, future_probabilities=f["future_probabilities"], future_logits=f["future_logits"])
    return r


def propose(g):
    s = task()
    g.poll(s, now=10.0)
    assert g.update(prediction(10.1), s, now=10.19, model_version="b0") is None
    assert g.poll(s, now=10.5) == "harvest_1"
    return s


def installed(g):
    s = propose(g)
    s.update(stage_version=1, requested_stage="harvest_1", execution_state="SWITCH_REQUESTED")
    g.submitted(s)
    assert g.poll(s, now=10.6) is None
    s.update(active_stage="harvest_1", execution_state="RUNNING", active_applied_unix_s=10.8)
    g.poll(s, now=10.9)
    return s


def test_future_strategy_and_actual_execution_ack(gate):
    s = installed(gate)
    assert gate.engine.commit_at == 10.8 and gate.engine.deadline == 12.3 and gate.engine.hard_deadline == 15.8
    gate.update(prediction(10.7, current=1, future=1), s, now=10.99, model_version="b0")
    assert gate.engine.state == "WAIT_CURRENT_ACK"
    gate.update(prediction(11.0, current=1, future=1), s, now=11.09, model_version="b0")
    assert gate.engine.state == "READY"


def test_pending_stage_never_moves_until_owner_ack(gate):
    s = propose(gate)
    s.update(stage_version=1, requested_stage="harvest_1", execution_state="SWITCHING")
    gate.submitted(s)
    for t in [10.6, 11.0, 12.0]:
        assert gate.poll(s, now=t) is None
    assert gate.engine.commit_at == 10.5 and gate.pending is not None
    with pytest.raises(TimeoutError):
        gate.poll(s, now=15.6)


def test_late_ack_requires_two_new_current_observations(gate):
    s = installed(gate)
    gate.poll(s, now=12.3)
    assert gate.engine.state == "WAIT_LATE_ACK"
    gate.update(prediction(12.4, 1, 1), s, now=12.49, model_version="b0")
    assert gate.engine.state == "WAIT_LATE_ACK"
    gate.update(prediction(12.9, 1, 1), s, now=12.99, model_version="b0")
    assert gate.engine.state == "READY"
    assert any(e["kind"] == "late_ack" for e in gate.take_events())


def test_timer_runs_hard_deadline_without_new_messages(gate):
    s = installed(gate)
    gate.poll(s, now=12.3)
    with pytest.raises(RuntimeError, match="硬截止"):
        gate.poll(s, now=15.8)


def test_final_return_never_requests_adjust(gate):
    s = task()
    s.update(active_stage="return_stem", requested_stage="return_stem", stage_version=6)
    gate.poll(s, now=10.0)
    gate.update(prediction(10.1, 2, 3), s, now=10.19, model_version="b0")
    assert gate.poll(s, now=10.5) is None and gate.engine.active == "return"


def test_stale_future_cannot_use_fresh_result_clock(gate):
    s = task()
    gate.poll(s, now=10.0)
    gate.update(prediction(10.1), s, now=10.7, model_version="b0")
    assert gate.poll(s, now=11.0) is None


def test_wrong_horizon_or_inconsistent_execution_version_rejected(gate):
    s = task()
    with pytest.raises(ValueError):
        gate.update(packet(10.0), s, now=10.1, model_version="b0")
    s = propose(gate)
    s.update(stage_version=1, requested_stage="harvest_1", execution_state="SWITCHING")
    gate.submitted(s)
    s["stage_version"] = 2
    with pytest.raises(ValueError, match="版本"):
        gate.poll(s, now=10.6)


def test_provider_defaults_are_separate(tmp_path):
    a = AgentSettings.parse({}, tmp_path)
    b = AgentSettings.parse(dict(provider="tairan"), tmp_path)
    assert a.provider == "mengxiang" and b.provider == "tairan"
    assert a.checkpoint != b.checkpoint and b.repository.endswith("/agents/inhand119_four")
    assert b.checkpoint.endswith("/head/best_head.pt") and b.pretrained.endswith("/encoder")
    assert b.profile == "workstation"
    with pytest.raises(ValueError):
        AgentSettings.parse(dict(provider="unknown"), tmp_path)


def test_saved_four_view_and_previous_tairan_use_new_tairan_provider(tmp_path):
    four = AgentSettings.parse(dict(provider="inhand119_four", gello_repository="custom-gello"), tmp_path)
    old = AgentSettings.parse(dict(provider="tairan", handshake_config="configs/tairan_handshake.json",
                                   repository="old-source", checkpoint="old.pt", python="old-python",
                                   pretrained="old-encoder", gello_repository="custom-gello"), tmp_path)
    assert old == four
    assert four.provider == "tairan"
    assert four.gello_repository.endswith("/custom-gello")


def test_activation_timestamp_only_changes_on_stage_version():
    from deploy.fr3_sharpa_runtime.task_control import TaskControl

    c = TaskControl()
    c.ready()
    c.command("start", "s", expected_version=0)
    c.activate(c.ticket(), "one")
    first = c.snapshot()["active_applied_unix_s"]
    c.activate(c.ticket(), "two")
    assert c.snapshot()["active_applied_unix_s"] == first
