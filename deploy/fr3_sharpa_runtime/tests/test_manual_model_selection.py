"""Explicit model configuration works without copied training contracts."""
from dataclasses import replace

import pytest

from deploy.fr3_sharpa_control.plan import inspect_checkpoint
from deploy.fr3_sharpa_control.plan import plan_environment
from deploy.fr3_sharpa_runtime.tests import test_training_archive_detection as archive_tests
from deploy.fr3_sharpa_runtime.train_rtc_contract import inspect_train_rtc

archive = archive_tests.archive
settings = archive_tests.settings


def test_manual_train_settings_ignore_automatic_archives_and_match_child(archive, monkeypatch):
    root, stats, record, _, paper = archive
    record.write_text('{"stale": "not a training config"}')
    selected = "train-paper-rtc" if paper else "train-rtc-absolute"
    options = settings(root, paper, model_type=selected, train_rtc_max_delay=9, rtc_initial_delay_steps=4)
    before = sorted(str(p) for p in root.rglob("*"))
    plan = options.plan()
    assert plan["model"]["train_rtc"]["max_delay"] == 9
    assert plan["model"]["train_rtc"]["record"] is None
    assert sorted(str(p) for p in root.rglob("*")) == before
    assert plan["execute"][plan["execute"].index("--rtc-initial-delay-steps") + 1] == "4"
    env = plan_environment(plan)
    for key in ("SHARPA_DEPLOY_MODEL_TYPE", "SHARPA_TRAIN_RTC_MAX_DELAY", "SHARPA_TRAIN_RTC_SEMANTICS"):
        monkeypatch.setenv(key, env[key])
    monkeypatch.setenv("SHARPA_TRAIN_RTC_RECORD", "/nonexistent/stale.json")
    assert inspect_train_rtc(stats, root, paper_algorithm=plan["model"]["paper_algorithm"]) == plan["model"]["train_rtc"]
    with pytest.raises(ValueError, match="初始延迟"):
        replace(options, rtc_initial_delay_steps=10).plan()
    with pytest.raises(ValueError, match="最大延迟"):
        replace(options, train_rtc_max_delay=0).plan()


def test_manual_ordinary_type_does_not_become_train_rtc(archive, monkeypatch):
    root, stats, _, _, paper = archive
    selected = "paper-delta" if paper else "full"
    info = inspect_checkpoint(root, model_type=selected)
    assert info["train_rtc"] is None
    assert ("paper-rtc" if paper else "rtc") in info["supported_modes"]
    monkeypatch.setenv("SHARPA_DEPLOY_MODEL_TYPE", selected)
    assert inspect_train_rtc(stats, root, paper_algorithm=info["paper_algorithm"]) is None
    with pytest.raises(ValueError, match="训练记录"):
        inspect_checkpoint(root, model_type=selected, training_record=root / "train_rtc_record.json")


def test_manual_paper_choice_requires_actual_paper_assets(archive):
    root, _, _, _, paper = archive
    with pytest.raises(ValueError, match="Paper"):
        inspect_checkpoint(root, model_type="full-delta" if paper else "paper-delta")


def test_optional_explicit_record_is_used_by_service_too(archive, monkeypatch):
    root, stats, record, _, paper = archive
    selected = "train-paper-rtc" if paper else "train-rtc-delta"
    plan = settings(root, paper, model_type=selected, train_rtc_max_delay=10, training_record=str(record)).plan()
    env = plan_environment(plan)
    for key in ("SHARPA_DEPLOY_MODEL_TYPE", "SHARPA_DEPLOY_TRAINING_RECORD", "SHARPA_TRAIN_RTC_MAX_DELAY"):
        monkeypatch.setenv(key, env[key])
    assert inspect_train_rtc(stats, root, paper_algorithm=plan["model"]["paper_algorithm"]) == plan["model"]["train_rtc"]
