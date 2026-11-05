# ruff: noqa: SLF001
"""Hard-prefix invariants, current state anchoring and new time-axis references."""

from dataclasses import fields
from dataclasses import replace
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import rtc
import pytest

from deploy.fr3_sharpa_runtime.mamba_model import MambaTrainPaperRTCPolicy
from deploy.fr3_sharpa_runtime.paper_model import normalization_bridge
from deploy.fr3_sharpa_runtime.rtc_stream import canonical_prefix
from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import checkpoint
from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import observation
from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import tiny_config
from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import tiny_network
from openpi import transforms
from openpi.models import model as model_lib
from openpi.models.pi0_mamba import MambaPi0
from openpi.models.pi0_mamba_config import MambaPi0Config
from openpi.models.pi0_mamba_paper import PaperMambaPi0Config
from openpi.policies.fr3_sharpa_policy import Fr3SharpaInputs
from openpi.policies.fr3_sharpa_policy import Fr3SharpaOutputs
from openpi.shared import delta_paper as paper
from openpi.shared import normalize


def test_prefix_is_fixed_before_first_and_every_euler_velocity(monkeypatch):
    calls = []
    with tiny_network():
        cfg = tiny_config(rtc_max_delay=3)
        model = cfg.create(jax.random.key(42))
        prefix = jnp.arange(50 * 58, dtype=jnp.float32).reshape(1, 50, 58) * 0.001
        condition_calls = []
        monkeypatch.setattr(MambaPi0, "encode_condition", lambda self, obs: condition_calls.append(1))

        def velocity(self, x, time, condition):
            calls.append((np.asarray(x).copy(), np.asarray(time).copy()))
            return jnp.ones_like(x) * 0.2

        monkeypatch.setattr(MambaPi0, "velocity", velocity)
        with jax.disable_jit():
            result = model.sample_actions(
                jax.random.key(1), cfg.fake_obs(), prefix_actions=prefix, prefix_length=3, num_steps=4
            )
        assert len(condition_calls) == 1
        assert len(calls) == 4
        for x, time in calls:
            np.testing.assert_array_equal(x[:, :3], prefix[:, :3])
            np.testing.assert_array_equal(time[:, :3], 0.0)
        np.testing.assert_array_equal(result[:, :3], prefix[:, :3])


def _paper_config(assets):
    base = tiny_config(rtc_max_delay=3)
    values = {
        f.name: getattr(base, f.name)
        for f in fields(MambaPi0Config)
        if f.name not in ("stage", "action_representation", "normalization")
    }
    meta = json.loads((assets / "delta_paper.json").read_text())
    return PaperMambaPi0Config(
        **values,
        paper_asset_dir=str(assets),
        paper_asset_sha256=meta["bundle_sha256"],
        paper_mode=meta["mode"],
        paper_beta=meta["beta"],
        paper_std_floor=meta["std_floor"],
    )


class Tokens:
    def __call__(self, data):
        return {
            k: v
            for k, v in {
                **data,
                "tokenized_prompt": np.arange(8, dtype=np.int32),
                "tokenized_prompt_mask": np.ones(8, bool),
            }.items()
            if k != "prompt"
        }


def test_paper_reanchors_short_physical_references_and_preserves_prefix(tmp_path):
    _, assets, _ = checkpoint(tmp_path, paper=True, delay=3)
    stats = normalize.load(assets)
    mask = transforms.make_bool_mask(7, -22, 7, -22)
    with tiny_network():
        cfg = _paper_config(assets)
        model = cfg.create(jax.random.key(42))
        bridge = normalization_bridge(cfg, stats)
        inputs = [
            Fr3SharpaInputs(model_lib.ModelType.PI05),
            transforms.DeltaActions(mask),
            transforms.Normalize(stats, use_quantiles=True),
            bridge,
            Tokens(),
        ]
        outputs = [
            replace(bridge, inverse=True),
            transforms.Unnormalize(stats, use_quantiles=True),
            transforms.AbsoluteActions(mask),
            Fr3SharpaOutputs(),
        ]
        policy = MambaTrainPaperRTCPolicy(
            model,
            transforms=inputs,
            output_transforms=outputs,
            rtc_action_repack=canonical_prefix,
            sample_kwargs={"num_steps": 2},
        )
        physical = np.random.default_rng(4).normal(size=(50, 58)).astype(np.float32)
        noise = np.random.default_rng(5).normal(size=(50, 58)).astype(np.float32)
        for length in (50, 35, 1):
            for delay in range(min(3, length) + 1):
                context = rtc.validate_context(
                    {
                        "protocol_version": 1,
                        "token": f"paper-{length}-{delay}",
                        "source_number": 1,
                        "source_sha256": hashlib.sha256(physical.tobytes()).hexdigest(),
                        "previous_actions": physical[:length],
                        "inference_delay_steps": delay,
                        "prefix_attention_horizon": max(1, delay),
                        "prefix_attention_schedule": "exp",
                        "max_guidance_weight": 5.0,
                    }
                )
                for anchor in (-0.75, 0.9):
                    obs = {**observation(), "observation/state": np.full(58, anchor, np.float32)}
                    # Verify the same physical reference re-encodes on NEW state
                    # and index zero, including time-varying arm stats.
                    actual = policy._input_transform({**obs, "actions": physical[:length].copy()})["actions"]
                    padded = np.zeros((50, 58), np.float32)
                    padded[:length] = physical[:length]
                    padded[:, paper.ARMS] -= anchor
                    expected = bridge(transforms.Normalize(stats, use_quantiles=True)({"actions": padded}))["actions"][
                        :length
                    ]
                    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
                    result = policy.infer(obs, noise=noise, rtc_context=context)
                    np.testing.assert_array_equal(result["actions"][:delay], physical[:delay])
                    rtc.validate_receipt(result["rtc"], context)
                    assert np.isfinite(result["actions"]).all()
        with pytest.raises(ValueError, match="training max delay"):
            policy.infer(
                observation(),
                rtc_context={
                    **context.to_dict(),
                    "previous_actions": physical,
                    "inference_delay_steps": 4,
                    "prefix_attention_horizon": 10,
                },
            )
        np.testing.assert_array_equal(context.previous_actions, physical[: len(context.previous_actions)])


def test_paper_correlated_prior_and_restored_noise_binding(tmp_path):
    """Synthetic assets exercise beta>0; this is not a trained covariance claim."""
    from flax import nnx

    from deploy.fr3_sharpa_runtime.mamba_contract import inspect_mamba

    root, assets, record = checkpoint(tmp_path, paper=True, delay=3)
    metadata = json.loads((assets / paper.MANIFEST).read_text())
    with np.load(assets / paper.BUNDLE) as bundle:
        means, stds = bundle["arm_mean"], bundle["arm_std"]
    size = 50 * 58
    chol = np.eye(size, dtype=np.float32)
    chol[np.arange(1, size), np.arange(size - 1)] = 0.1
    np.savez(assets / paper.BUNDLE, arm_mean=means, arm_std=stds, noise_cholesky=chol)
    metadata["beta"] = 0.5
    metadata["source_contract"]["vector_transforms"]["delta_paper"]["beta"] = 0.5
    metadata["bundle_sha256"] = paper.sha256(assets / paper.BUNDLE)
    (assets / paper.MANIFEST).write_text(json.dumps(metadata))
    audit = json.loads((assets / "norm_metadata.json").read_text())
    audit["contract"] = metadata["source_contract"]
    (assets / "norm_metadata.json").write_text(json.dumps(audit))
    record["model"].update(paper_beta=0.5, paper_asset_sha256=metadata["bundle_sha256"])
    record["paper_files_sha256"] = {
        n: paper.sha256(assets / n) for n in (paper.BUNDLE, paper.MANIFEST, "norm_metadata.json")
    }
    (assets / "mamba_architecture.json").write_text(json.dumps(record))
    tree_path = root / "params/_METADATA"
    tree = json.loads(tree_path.read_text())
    tree["tree_metadata"]["paper_noise_cholesky"] = {
        "key_metadata": [{"key": n} for n in ("params", "paper_noise_cholesky", "value")],
        "value_metadata": {"value_type": "jax.Array", "write_shape": [100, 2900]},
    }
    tree_path.write_text(json.dumps(tree))
    assert inspect_mamba(root, assets / "norm_stats.json")["paper_algorithm"]["beta"] == 0.5
    with tiny_network():
        cfg = _paper_config(assets)
        model = cfg.create(jax.random.key(42))
        shape = (1, 50, 58)
        z = jax.random.normal(jax.random.key(6), shape)
        expected = (z.reshape(1, -1) @ jnp.asarray(chol).T).reshape(shape)
        np.testing.assert_allclose(model.noise(jax.random.key(6), shape), expected, rtol=1e-6, atol=1e-6)
        assert not np.array_equal(expected, z)
        values = nnx.state(model).to_pure_dict()
        cfg.load(values)
        with pytest.raises(ValueError, match="noise factor"):
            cfg.load({**values, "paper_noise_cholesky": jnp.eye(size)})
