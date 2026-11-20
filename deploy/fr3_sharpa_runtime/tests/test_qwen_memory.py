import copy

import pytest

from deploy.fr3_sharpa_control.qwen_client import build_payload
from deploy.fr3_sharpa_control.qwen_memory import HarvestPlan
from deploy.fr3_sharpa_control.qwen_memory import TaskMemory


def window(memory, start=10, end=15, name="one"):
    return {
        "id": name,
        "run_id": memory.run_id,
        "revision": memory.revision,
        "active_stage": memory.stage,
        "start": start,
        "end": end,
        "frames": [{"id": str(t), "timestamp": t, "camera": "cam0"} for t in range(start, end + 1)],
    }


def answer(events=(), **kwargs):
    return dict(
        events=[
            {"held": str(a), "released": str(b), "empty": str(c), "evidence": "持果、盘内释放、空手离开"}
            for a, b, c in events
        ],
        ready=True,
        ready_evidence=["14", "15"],
        left_empty=True,
        counting_complete=True,
        uncertainties=[],
        **kwargs,
    )


def harvest(quotas=(2, 2)):
    memory = TaskMemory("run", HarvestPlan(quotas))
    memory.acknowledge("harvest_1", run_id="run", expected_revision=0)
    return memory


def test_plan_sequences():
    assert HarvestPlan((4,)).stages == ("pickup", "harvest_1", "return_stem")
    assert HarvestPlan((2, 2)).stages == ("pickup", "harvest_1", "adjust_in_hand", "harvest_2", "return_stem")
    for invalid in ((), (0,), (True,), (1, 2, 3), [4]):
        with pytest.raises(ValueError, match=".+"):
            HarvestPlan(invalid)


def test_quota_then_ack_preserves_ledger_and_requires_second_round():
    memory = harvest()
    result = memory.evaluate(window(memory), answer([(10, 11, 12), (13, 14, 15)]), now=15.2)
    assert result["proposed_stage"] == "adjust_in_hand"
    assert memory.stage == "harvest_1"
    memory.acknowledge("adjust_in_hand", run_id="run", expected_revision=1)
    memory.acknowledge("harvest_2", run_id="run", expected_revision=2)
    result = memory.evaluate(
        window(memory, 20, 25, "two"), {**answer([(20, 21, 22)]), "ready_evidence": ["24", "25"]}, now=25.2
    )
    assert result["proposed_stage"] is None
    assert result["memory"]["counts"] == {"harvest_1": 2, "harvest_2": 1}


def test_four_without_adjustment():
    memory = harvest((4,))
    result = memory.evaluate(
        window(memory, 10, 21),
        {**answer([(10, 11, 12), (13, 14, 15), (16, 17, 18), (19, 20, 21)]), "ready_evidence": ["20", "21"]},
        now=21.1,
    )
    assert result["proposed_stage"] == "return_stem"


def test_duplicate_cross_window_and_restart_never_recounts():
    memory = harvest()
    memory.evaluate(window(memory), answer([(10, 11, 12)]), now=15.1)
    assert memory.evaluate(window(memory), answer([(10, 11, 12)]), now=15.2)["status"] == "duplicate"
    memory = TaskMemory.restore(memory.snapshot())
    result = memory.evaluate(window(memory, 10, 16, "overlap"), answer([(10, 11, 12)]), now=16.2)
    assert result["memory"]["counts"]["harvest_1"] == 1


def test_historical_results_count_but_cannot_authorize_motion():
    memory = harvest((1,))
    result = memory.evaluate(window(memory), answer([(10, 11, 12)]), now=26)
    assert result["status"] == "historical_only"
    assert result["proposed_stage"] is None
    assert result["memory"]["counts"]["harvest_1"] == 1


def test_gap_overquota_or_ambiguity_block_recommendation():
    for kind in ("gap", "overquota", "uncertain"):
        memory = harvest((1,))
        if kind == "gap":
            memory.evaluate(window(memory, 1, 5), {**answer(), "ready_evidence": ["4", "5"]}, now=5.2)
        result = memory.evaluate(
            window(memory),
            {
                **answer([(10, 11, 12)] + ([(13, 14, 15)] if kind == "overquota" else [])),
                "uncertainties": ["遮挡"] if kind == "uncertain" else [],
            },
            now=15.2,
        )
        assert result["proposed_stage"] is None


def test_bad_output_or_stale_session_is_atomic():
    memory = harvest()
    for kind in ("frame", "order", "bool", "session", "revision"):
        req, response = window(memory), answer([(10, 11, 12)])
        if kind == "frame":
            response["events"][0]["released"] = "unknown"
        elif kind == "order":
            response["events"][0]["released"] = "10"
        elif kind == "bool":
            response["ready"] = 1
        elif kind == "session":
            req["run_id"] = "old"
        else:
            req["revision"] -= 1
        before = memory.snapshot()
        with pytest.raises(ValueError, match=".+"):
            memory.evaluate(req, response, now=15.1)
        assert memory.snapshot() == before


def test_prompt_carries_plan_and_ledger_and_forces_thinking_off():
    memory = harvest()
    request = window(memory)
    images = {f["id"]: b"\xff\xd8test" for f in request["frames"]}
    for model in ("qwen3.8-flash", "qwen3.8-max", "qwen3.6-flash"):
        payload = build_payload(memory.snapshot(), request, images, model=model)
        assert payload["enable_thinking"] is False
        assert payload["stream"] is False
        assert '"quotas": [2, 2]' in payload["messages"][1]["content"][0]["text"]
    wrong = copy.deepcopy(request)
    wrong["revision"] += 1
    with pytest.raises(ValueError, match=".+"):
        build_payload(memory.snapshot(), wrong, images)


def test_handoff_rejects_old_phase_images():
    memory = TaskMemory("run")
    memory.acknowledge("harvest_1", run_id="run", expected_revision=0, applied_at=20)
    with pytest.raises(ValueError, match="生效之前"):
        memory.evaluate(window(memory), answer([(10, 11, 12)]), now=21)


def test_execution_advancing_before_quota_cannot_hide_shortfall():
    memory = harvest()
    with pytest.raises(ValueError, match="计数尚未"):
        memory.acknowledge("adjust_in_hand", run_id="run", expected_revision=1)


def test_later_clear_view_does_not_silently_erase_unobserved_interval():
    memory = harvest((1,))
    memory.evaluate(window(memory), {**answer(), "counting_complete": False}, now=15.1)
    result = memory.evaluate(
        window(memory, 14, 19, "later"), {**answer([(16, 17, 18)]), "ready_evidence": ["18", "19"]}, now=19.1
    )
    assert result["proposed_stage"] is None
    assert result["memory"]["blockers"]
