# ruff: noqa: SLF001
"""Adversarial mode-construction and CLI review; no hardware or trained weights."""

import sys
from types import SimpleNamespace

import pytest

from deploy.fr3_sharpa_runtime import serve
from deploy.fr3_sharpa_runtime.mamba_model import MambaPaperSoftRTCPolicy
from deploy.fr3_sharpa_runtime.mamba_model import MambaPolicy
from deploy.fr3_sharpa_runtime.mamba_model import MambaSoftRTCPolicy
from deploy.fr3_sharpa_runtime.mamba_model import MambaTrainPaperRTCPolicy
from deploy.fr3_sharpa_runtime.mamba_model import MambaTrainRTCPolicy
from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import checkpoint
from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import select

CLASSES = [MambaPolicy, MambaSoftRTCPolicy, MambaPaperSoftRTCPolicy, MambaTrainRTCPolicy, MambaTrainPaperRTCPolicy]


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize(
    "key", ["rtc_actions", "rtc_weights", "rtc_max_guidance_weight", "prefix_actions", "prefix_length"]
)
def test_raw_sample_kwargs_cannot_override_any_policy_mode(cls, key):
    # No NNX/JIT initialization is available: rejection must happen on the host.
    paper = cls in (MambaPaperSoftRTCPolicy, MambaTrainPaperRTCPolicy)
    model = SimpleNamespace(
        rtc_max_delay=3 if "Train" in cls.__name__ else 0,
        config=SimpleNamespace(normalization="paper" if paper else "ordinary_per_dimension"),
    )
    with pytest.raises(ValueError, match="validated request context"):
        cls(model, sample_kwargs={key: None})


@pytest.mark.parametrize(
    ("cls", "normalization"), [(MambaTrainRTCPolicy, "paper"), (MambaTrainPaperRTCPolicy, "ordinary_per_dimension")]
)
def test_hard_plain_paper_class_mismatch_rejected_before_jit(cls, normalization):
    model = SimpleNamespace(rtc_max_delay=3, config=SimpleNamespace(normalization=normalization))
    with pytest.raises(ValueError, match="normalization"):
        cls(model)


FLAGS = ["--enable-rtc", "--enable-paper-rtc", "--enable-train-rtc", "--enable-train-paper-rtc"]
FLAG_SETS = [tuple(f for i, f in enumerate(FLAGS) if mask & (1 << i)) for mask in range(16)]


@pytest.mark.parametrize("paper", [False, True])
@pytest.mark.parametrize("delay", [0, 3])
def test_every_mamba_service_flag_combination(tmp_path, monkeypatch, capsys, paper, delay):
    root, assets, _ = checkpoint(tmp_path, paper=paper, delay=delay)
    select(monkeypatch, root, assets, "mamba-delta" if paper else "mamba")
    for flags in FLAG_SETS:
        # Both soft flags are redundant aliases for Paper Soft, never Hard.
        legal = not flags or (
            flags == (("--enable-train-paper-rtc",) if paper else ("--enable-train-rtc",))
            if delay
            else flags
            in ([("--enable-paper-rtc",), ("--enable-rtc", "--enable-paper-rtc")] if paper else [("--enable-rtc",)])
        )
        monkeypatch.setattr(sys, "argv", ["serve", "--check-only", *flags])
        if legal:
            serve.main()
            capsys.readouterr()
        else:
            with pytest.raises(SystemExit):
                serve.main()
            capsys.readouterr()


@pytest.mark.parametrize(
    ("profile", "paper", "trained"),
    [
        ("full", False, False),
        ("full-delta", False, False),
        ("full-delta", True, False),
        ("full", False, True),
        ("full-delta", False, True),
        ("full-delta", True, True),
        ("lora", False, False),
    ],
)
def test_transformer_flag_behavior_matches_original_train_rtc_commit(monkeypatch, capsys, profile, paper, trained):
    import copy
    import json
    from pathlib import Path
    import subprocess
    import types

    from deploy.fr3_sharpa_runtime import config
    from deploy.fr3_sharpa_runtime import model_loader
    from deploy.fr3_sharpa_runtime.model_profiles import get_model_profile

    source = subprocess.check_output(
        ["git", "show", "e04dade1707af6d6f574e8b4890c54705c855980:deploy/fr3_sharpa_runtime/serve.py"],
        cwd=Path(__file__).resolve().parents[3],
        text=True,
    )
    legacy = types.ModuleType("deploy.fr3_sharpa_runtime.review_legacy_serve")
    legacy.__package__ = "deploy.fr3_sharpa_runtime"
    exec(compile(source, "original_transformer_serve", "exec"), legacy.__dict__)
    monkeypatch.setattr(config, "MODEL_PROFILE", get_model_profile(profile))
    metadata = {
        "paper_algorithm": {"mode": "timestamp_zscore_arms", "beta": 0.5} if paper else None,
        "train_rtc": {"max_delay": 3} if trained else None,
    }
    monkeypatch.setattr(model_loader, "checkpoint_contract", lambda: copy.deepcopy(metadata))

    def result(main, flags):
        monkeypatch.setattr(sys, "argv", ["serve", "--check-only", *flags])
        try:
            main()
        except SystemExit as error:
            capsys.readouterr()
            return error.code, None
        return 0, json.loads(capsys.readouterr().out)

    for flags in FLAG_SETS:
        assert result(serve.main, flags) == result(legacy.main, flags), (profile, paper, trained, flags)


@pytest.mark.parametrize("mamba", [False, True])
def test_original_restore_dtype_and_strict_mamba_restore_remain_separate(tmp_path, monkeypatch, mamba):
    import jax.numpy as jnp

    from openpi import transforms
    from openpi.policies import policy_config

    calls = []
    sentinel = object()

    def restore(path, **kwargs):
        calls.append(("restore", kwargs))
        return sentinel

    def load(params, **kwargs):
        assert params is sentinel
        calls.append(("load", kwargs))
        return sentinel

    model = SimpleNamespace(load=load)
    if mamba:
        model.architecture = "pi05_mamba3_jax"
    data = SimpleNamespace(
        use_quantile_norm=False, data_transforms=transforms.Group(), model_transforms=transforms.Group()
    )
    training = SimpleNamespace(
        model=model, data=SimpleNamespace(create=lambda *args: data), assets_dirs=tmp_path, policy_metadata={}
    )
    monkeypatch.setattr(policy_config._model, "restore_params", restore)
    actual = policy_config.create_trained_policy(
        training, tmp_path, norm_stats={}, policy_class=lambda model, **kwargs: model
    )
    assert actual is sentinel
    assert calls == [
        ("restore", {} if mamba else {"dtype": jnp.bfloat16}),
        ("load", {"remove_extra_params": False} if mamba else {}),
    ]
