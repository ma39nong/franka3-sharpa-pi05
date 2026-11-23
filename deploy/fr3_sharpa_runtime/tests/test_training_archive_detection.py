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
        **{"image_input": "three", **kwargs},
    )


@pytest.mark.parametrize("location", ["step", "step_repro", "run_repro", "explicit"])
def test_effective_archive_supplies_hints_without_norm_archive(archive, location):
    root, stats, old, record, is_paper = archive
    old.unlink()
    destination = {"step": root, "step_repro": root / "reproducibility",
                   "run_repro": root.parent / "reproducibility", "explicit": root.parent / "selected"}[location]
    destination.mkdir(exist_ok=True)
    path = destination / "effective_config.json"
    path.write_text(json.dumps(record["config"]))
    opts = {"training_record": str(path)} if location == "explicit" else {}
    info = inspect_checkpoint(root, **opts)
    assert info["train_rtc"]["max_delay"] == 10
    assert info["suggestions"]["source"] == str(path)
    plan = settings(root, is_paper, **opts).plan()
    assert plan_environment(plan)["SHARPA_TRAIN_RTC_MAX_DELAY"] == "10"
    assert "SHARPA_TRAIN_RTC_RECORD" not in plan_environment(plan)


def test_manual_values_override_optional_archive(archive):
    root, _, _, _, paper = archive
    plan = settings(root, paper, train_rtc_max_delay=9, semantics="delta").plan()
    assert plan["model"]["train_rtc"]["max_delay"] == 9


@pytest.mark.parametrize("delay", [-1, 50, True, 1.5, "10"])
def test_delay_requires_valid_integer(archive, delay):
    root, _, _, _, paper = archive
    with pytest.raises(ValueError, match="最大延迟"):
        settings(root, paper, train_rtc_max_delay=delay).plan()


def test_missing_semantics_returns_partial_result(archive):
    root, _, old, _, paper = archive
    old.unlink()
    info = inspect_checkpoint(root, train_rtc_max_delay=10)
    assert info["profile"] == ("full-delta" if paper else None)
    assert info["input_views"] is None


def test_child_environment_replaces_stale_hints(archive, monkeypatch):
    root, _, _, _, paper = archive
    monkeypatch.setenv("SHARPA_TRAIN_RTC_RECORD", "/stale")
    monkeypatch.setenv("SHARPA_DEPLOY_IMAGE_INPUT", "four")
    plan = settings(root, paper).plan()
    env = plan_environment(plan)
    assert "SHARPA_TRAIN_RTC_RECORD" not in env
    assert env["SHARPA_DEPLOY_IMAGE_INPUT"] == "three"
