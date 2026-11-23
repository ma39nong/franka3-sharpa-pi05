"""Optional hints, complete manual fallback and matching child configuration."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from deploy.fr3_sharpa_control.plan import inspect_checkpoint, DeploymentSettings, plan_environment
from deploy.fr3_sharpa_runtime.tests.test_deployment_center import checkpoint
from deploy.fr3_sharpa_runtime.tests.test_training_archive_detection import archive


def test_unknown_checkpoint_returns_partial_configuration(checkpoint):
    found = inspect_checkpoint(checkpoint)
    assert found['profile'] is None and found['input_views'] is None
    assert found['warnings']
    manual = inspect_checkpoint(checkpoint, model_type='full-delta', image_input='four', task_mode='tomato-in-hand119')
    assert manual['input_views'] == 4
    assert manual['tomato_model']['task_profile'] == 'tomato-in-hand119'
    assert manual['supported_modes'] == ['serial', 'rtc']


def test_optional_archive_detects_views_and_prompts(checkpoint):
    path = checkpoint / 'effective_config.json'
    path.write_text(json.dumps({'fields': {'model': {'fields': {'rtc_max_delay': 10,
        'image_keys': ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb', 'interaction_0_rgb']}},
        'data': {'fields': {'extra_delta_transform': True}},
        'policy_metadata': {'prompts': ['Right hand adjusts tomato truss position and orientation at the center']}}}))
    found = inspect_checkpoint(checkpoint)
    assert found['input_views'] == 4 and found['train_rtc']['max_delay'] == 10
    assert found['suggestions']['task_mode'] == 'tomato-in-hand119'
    manual = inspect_checkpoint(checkpoint, model_type='full', image_input='three')
    assert manual['input_views'] == 3 and manual['train_rtc'] is None and manual['semantics'] == 'absolute'
    path.write_text('{broken')
    assert inspect_checkpoint(checkpoint, model_type='full', image_input='three')['profile'] == 'full'


def test_manual_train_four_view_plan_and_service_agree(archive, monkeypatch):
    root, stats, record, _, paper = archive
    record.write_text('{stale')
    from deploy.fr3_sharpa_runtime import interaction_crop, config, model_loader
    from deploy.fr3_sharpa_runtime.model_profiles import get_model_profile
    monkeypatch.setattr(interaction_crop, 'locator_identity', lambda *a: ('/tmp/locator', 'test-locator'))
    selected = 'train-paper-rtc' if paper else 'train-rtc-delta'
    opts = DeploymentSettings(str(Path(__file__).resolve().parents[3]), str(root), model_type=selected,
        train_rtc_max_delay=10, image_input='four', task_mode='tomato-in-hand119', initial_stage='pickup',
        mode='train-paper-rtc' if paper else 'train-rtc', speed='fast')
    plan = opts.plan()
    for k, v in plan_environment(plan).items():
        if k.startswith('SHARPA_'):
            monkeypatch.setenv(k, v)
    monkeypatch.setattr(config, 'CHECKPOINT', root)
    monkeypatch.setattr(config, 'NORM_STATS', stats)
    monkeypatch.setattr(config, 'MODEL_PROFILE', get_model_profile('full-delta'))
    info = model_loader.checkpoint_contract()
    assert info['train_rtc'] == plan['model']['train_rtc']
    assert info['tomato_model'] == plan['model']['tomato_model']
    training = model_loader.build_config()
    assert len(training.model.image_keys) == 4
    if paper:
        from deploy.fr3_sharpa_runtime.crop_input import PaperCropRuntimeData
        assert isinstance(training.data, PaperCropRuntimeData)
    with pytest.raises(ValueError, match='初始延迟'):
        replace(opts, rtc_initial_delay_steps=11).plan()


def test_task_backend_requests_hard_prefix_for_trained_rtc():
    from types import SimpleNamespace
    from deploy.fr3_sharpa_runtime.task_rtc import SDKTaskBackend
    from deploy.fr3_sharpa_runtime.rtc_stream import RTCSettings
    requests = []
    class Worker:
        def request(self, payload, **kwargs):
            requests.append(payload)
            return {'context': payload}
    for trained in (False, True):
        backend = SDKTaskBackend(Worker(), SimpleNamespace(train_rtc=trained))
        backend.snapshot(RTCSettings(initial_delay_steps=10), 10)
        assert requests[-1]['hard_prefix'] is trained
        assert requests[-1]['delay_steps'] == 10
