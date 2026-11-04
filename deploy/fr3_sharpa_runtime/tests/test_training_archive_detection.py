"""New effective archives and explicit RTC declarations, with no model/device startup."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from deploy.fr3_sharpa_control.plan import DeploymentSettings
from deploy.fr3_sharpa_control.plan import inspect_checkpoint
from deploy.fr3_sharpa_control.plan import plan_environment
from deploy.fr3_sharpa_runtime.tests import test_train_paper_rtc as paper_fixtures
from deploy.fr3_sharpa_runtime.tests import test_train_rtc as plain_fixtures
from deploy.fr3_sharpa_runtime.train_rtc_contract import inspect_train_rtc


@pytest.fixture(params=[False, True], ids=["plain", "paper"])
def archive(request, tmp_path):
    if request.param:
        base = paper_fixtures.paper_checkpoint.__wrapped__(tmp_path)
        root, assets, record_path, record = paper_fixtures.trained_paper.__wrapped__(base)
        stats = assets / "norm_stats.json"
    else:
        root, stats, record_path, record = plain_fixtures.trained_checkpoint.__wrapped__(tmp_path)
    return root, stats, record_path, record, request.param


def settings(root, is_paper, **kwargs):
    return DeploymentSettings(
        str(Path(__file__).resolve().parents[3]),
        str(root),
        speed="fast",
        mode="train-paper-rtc" if is_paper else "train-rtc",
        **kwargs,
    )


@pytest.mark.parametrize("location", ["step", "step_repro", "run_repro", "explicit"])
def test_effective_config_archive_auto_detection_and_explicit_selection(archive, location):
    root, stats, old, record, is_paper = archive
    old.unlink()
    destination = {
        "step": root,
        "step_repro": root / "reproducibility",
        "run_repro": root.parent / "reproducibility",
        "explicit": root.parent / "selected",
    }[location]
    destination.mkdir(exist_ok=True)
    config = destination / "effective_config.json"
    config.write_text(json.dumps(record["config"]))
    (destination / "norm_stats.json").write_text(json.dumps(json.loads(stats.read_text())["norm_stats"]))
    opts = {"training_record": str(config)} if location == "explicit" else {}
    info = inspect_checkpoint(root, **opts)
    assert info["train_rtc"]["max_delay"] == 10
    assert info["train_rtc"]["record"] == str(config)
    plan = settings(root, is_paper, **opts).plan()
    assert plan_environment(plan)["SHARPA_TRAIN_RTC_RECORD"] == str(config)


def test_manual_fallback_plan_and_runtime_contract_agree(archive, monkeypatch):
    root, stats, old, _, is_paper = archive
    old.unlink()
    assert inspect_checkpoint(root)["train_rtc"] is None
    config = settings(root, is_paper, train_rtc_max_delay=10, semantics="delta")
    plan = config.plan()
    trained = plan["model"]["train_rtc"]
    assert trained["manual_confirmation"] is True
    assert trained["record"] is None
    assert plan["model"]["supported_modes"] == ["serial", config.mode]
    monkeypatch.setenv("SHARPA_TRAIN_RTC_RECORD", "/stale/record.json")
    env = plan_environment(plan)
    assert "SHARPA_TRAIN_RTC_RECORD" not in env
    assert env["SHARPA_TRAIN_RTC_MAX_DELAY"] == "10"
    monkeypatch.delenv("SHARPA_TRAIN_RTC_RECORD")
    monkeypatch.setenv("SHARPA_TRAIN_RTC_MAX_DELAY", env["SHARPA_TRAIN_RTC_MAX_DELAY"])
    monkeypatch.setenv("SHARPA_TRAIN_RTC_SEMANTICS", env["SHARPA_TRAIN_RTC_SEMANTICS"])
    assert inspect_train_rtc(stats, root, paper_algorithm=plan["model"]["paper_algorithm"]) == trained
    # Even without proof, selected algorithms cannot be swapped silently.
    wrong = replace(config, mode="rtc" if is_paper else "paper-rtc")
    with pytest.raises(ValueError, match="不支持所选"):
        wrong.plan()


def test_manual_never_overrides_conflicting_training_evidence(archive):
    root, _, _, _, is_paper = archive
    with pytest.raises(ValueError, match="冲突"):
        settings(root, is_paper, train_rtc_max_delay=9, semantics="delta").plan()


@pytest.mark.parametrize("delay", [-1, 50, True, 1.5, "10"])
def test_manual_delay_requires_valid_integer(archive, delay):
    root, _, old, _, is_paper = archive
    old.unlink()
    with pytest.raises(ValueError, match="最大延迟"):
        settings(root, is_paper, train_rtc_max_delay=delay, semantics="delta").plan()


def test_raw_archive_mismatched_assets_or_stats_rejected(archive):
    root, stats, old, record, is_paper = archive
    old.unlink()
    config = root / "effective_config.json"
    (root / "norm_stats.json").write_text("{}")
    if is_paper:
        record["config"]["fields"]["model"]["fields"]["paper_asset_sha256"] = "wrong"
    config.write_text(json.dumps(record["config"]))
    with pytest.raises(ValueError, match="不一致"):
        inspect_checkpoint(root, semantics="delta")


def test_manual_plain_requires_action_semantics(tmp_path):
    root, _, old, _ = plain_fixtures.trained_checkpoint.__wrapped__(tmp_path)
    old.unlink()
    with pytest.raises(ValueError, match="动作类型"):
        inspect_checkpoint(root, train_rtc_max_delay=10)


def test_plain_environment_does_not_inherit_old_manual_override(archive, monkeypatch):
    root, _, _, _, is_paper = archive
    plan = settings(root, is_paper).plan()
    monkeypatch.setenv("SHARPA_TRAIN_RTC_MAX_DELAY", "25")
    monkeypatch.setenv("SHARPA_TRAIN_RTC_SEMANTICS", "absolute")
    env = plan_environment(plan)
    assert "SHARPA_TRAIN_RTC_MAX_DELAY" not in env
    assert "SHARPA_TRAIN_RTC_SEMANTICS" not in env
