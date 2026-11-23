"""Manual configuration is authoritative; optional archives never gate execution."""
from dataclasses import replace
import pytest
from deploy.fr3_sharpa_control.plan import inspect_checkpoint, plan_environment
from deploy.fr3_sharpa_runtime.tests.test_training_archive_detection import archive, settings


def test_manual_train_ignores_stale_archives(archive):
    root, _, record, _, paper = archive
    record.write_text('{broken')
    selected = 'train-paper-rtc' if paper else 'train-rtc-absolute'
    opts = settings(root, paper, model_type=selected, train_rtc_max_delay=9, rtc_initial_delay_steps=4)
    before = sorted(str(p) for p in root.rglob('*'))
    plan = opts.plan()
    assert plan['model']['train_rtc']['max_delay'] == 9
    assert sorted(str(p) for p in root.rglob('*')) == before
    assert plan_environment(plan)['SHARPA_TRAIN_RTC_MAX_DELAY'] == '9'
    with pytest.raises(ValueError, match='初始延迟'):
        replace(opts, rtc_initial_delay_steps=10).plan()
    with pytest.raises(ValueError, match='最大延迟'):
        replace(opts, train_rtc_max_delay=0).plan()


def test_ordinary_selection_overrides_rtc_archive(archive):
    root, _, record, _, paper = archive
    info = inspect_checkpoint(root, model_type='paper-delta' if paper else 'full', training_record=record)
    assert info['train_rtc'] is None
    assert ('paper-rtc' if paper else 'rtc') in info['supported_modes']


def test_paper_selection_still_requires_numerical_assets(archive):
    root, _, _, _, paper = archive
    with pytest.raises(ValueError, match='Paper'):
        inspect_checkpoint(root, model_type='full-delta' if paper else 'paper-delta')
