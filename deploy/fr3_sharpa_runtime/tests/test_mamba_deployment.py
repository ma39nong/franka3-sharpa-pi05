# ruff: noqa: PT011, SLF001
"""Mode admission, canonical transforms, and hardware-free Mamba policy transport."""

import json
import os
from pathlib import Path
import subprocess
import sys

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import rtc
import pytest

from deploy.fr3_sharpa_control.plan import DeploymentSettings
from deploy.fr3_sharpa_control.plan import inspect_checkpoint
from deploy.fr3_sharpa_control.plan import plan_environment
from deploy.fr3_sharpa_runtime import config
from deploy.fr3_sharpa_runtime import model_loader
from deploy.fr3_sharpa_runtime import serve
from deploy.fr3_sharpa_runtime.mamba_model import MambaPolicy
from deploy.fr3_sharpa_runtime.mamba_model import MambaTrainRTCPolicy
from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import checkpoint
from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import observation
from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import select
from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import tiny_config
from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import tiny_network
from openpi import transforms


@pytest.mark.parametrize("paper", [False, True])
@pytest.mark.parametrize("delay", [0, 3])
def test_mode_matrix_check_and_ui_plan(tmp_path, monkeypatch, capsys, paper, delay):
    root, assets, _ = checkpoint(tmp_path, paper=paper, delay=delay)
    name = "mamba-delta" if paper else "mamba"
    select(monkeypatch, root, assets, name)
    model = model_loader.build_config()
    config.validate_training_contract(model)
    data = model.data.create(model.assets_dirs, model.model)
    assert data.use_quantile_norm == paper
    metadata = inspect_checkpoint(root)
    assert metadata["profile"] == name
    assert metadata["supported_modes"] == ["serial", ("train-paper-rtc" if paper else "train-rtc") if delay else ("paper-rtc" if paper else "rtc")]
    repository = Path(__file__).resolve().parents[3]
    plan = DeploymentSettings(str(repository), str(root)).plan()
    env = plan_environment(plan)
    assert env[config.MODEL_PROFILE.checkpoint_env] == str(root)
    assert "SHARPA_TRAIN_RTC_RECORD" not in env
    assert "SHARPA_TRAIN_RTC_MAX_DELAY" not in env
    assert plan["port"] == (18769 if paper else 18768)
    flags = ["--enable-rtc", "--enable-paper-rtc", "--enable-train-rtc", "--enable-train-paper-rtc"]
    for flag in [None, *flags]:
        monkeypatch.setattr(sys, "argv", ["serve", "--check-only", *([flag] if flag else [])])
        legal = flag is None or flag == (("--enable-train-paper-rtc" if paper else "--enable-train-rtc") if delay else ("--enable-paper-rtc" if paper else "--enable-rtc"))
        if legal:
            serve.main()
            assert json.loads(capsys.readouterr().out)["model_profile"] == name
        else:
            with pytest.raises(SystemExit):
                serve.main()
            capsys.readouterr()
    for mode in ("rtc", "paper-rtc", "soft-serial", "soft-async"):
        settings = DeploymentSettings(str(repository), str(root), mode=mode, speed="fast", steps=20)
        if mode in metadata["supported_modes"]:
            assert ("--enable-paper-rtc" if paper else "--enable-rtc") in settings.plan()["service"]
        else:
            with pytest.raises(ValueError):
                settings.plan()


@pytest.mark.parametrize("paper", [False, True])
def test_physical_inverse_once_and_camera_order(tmp_path, monkeypatch, paper):
    root, assets, _ = checkpoint(tmp_path, paper=paper)
    select(monkeypatch, root, assets, "mamba-delta" if paper else "mamba")
    training = model_loader.build_config()
    data = training.data.create(training.assets_dirs, training.model)
    vector = transforms.compose(data.data_transforms.inputs)(observation())
    assert tuple(vector["image"]) == ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    assert [int(vector["image"][key][0, 0, 0]) for key in vector["image"]] == [0, 70, 140]
    rng = np.random.default_rng(8)
    physical = rng.normal(size=(50, 58)).astype(np.float32)
    vector_with_actions = transforms.compose(data.data_transforms.inputs)({**observation(), "actions": physical.copy()})
    stats = data.norm_stats
    normalized = transforms.Normalize(stats, use_quantiles=paper)(vector_with_actions)
    # Exclude tokenizer/image transforms: this checks the physical action ABI itself.
    bridge = data.model_transforms.inputs[0] if paper else lambda x: x
    inverse = data.model_transforms.outputs[-1] if paper else lambda x: x
    encoded = bridge(normalized)
    decoded = inverse({"actions": encoded["actions"], "state": normalized["state"]})
    decoded = transforms.Unnormalize(stats, use_quantiles=paper)(decoded)
    actual = transforms.compose(data.data_transforms.outputs)(decoded)["actions"]
    np.testing.assert_allclose(actual, physical, atol=2e-5, rtol=2e-5)


def test_mamba_full_tree_shape_dtype_and_extra_branch_rejected():
    with tiny_network():
        cfg = tiny_config()
        model = cfg.create(jax.random.key(14))
        params = nnx.state(model).to_pure_dict()
        cfg.load(params)
        with pytest.raises(ValueError):
            cfg.load({**params, "old_expert": {"weight": jnp.ones(4)}})
        broken = {
            **params,
            "action_expert": {
                **params["action_expert"],
                "action_in_proj": {**params["action_expert"]["action_in_proj"], "kernel": jnp.ones((57, 16))},
            },
        }
        with pytest.raises(ValueError):
            cfg.load(broken)
        broken = jax.tree.map(lambda x: x.astype(jnp.bfloat16) if hasattr(x, "dtype") else x, params)
        with pytest.raises(ValueError):
            cfg.load(broken)


def test_policy_noise_contract_state_isolation_and_static_rtc():
    with tiny_network():
        cfg = tiny_config(rtc_max_delay=3)
        model = cfg.create(jax.random.key(42))
        obs = jax.tree.map(lambda x: np.asarray(x)[0], cfg.fake_obs().to_dict())
        policy = MambaPolicy(model, sample_kwargs={"num_steps": 2})
        noise = np.random.default_rng(1).normal(size=(50, 58)).astype(np.float32)
        a = policy.infer(obs, noise=noise)["actions"]
        policy.infer(obs)
        np.testing.assert_array_equal(a, policy.infer(obs, noise=noise[None])["actions"])
        for bad in (np.zeros((49, 58)), np.full((50, 58), np.nan), np.zeros((2, 50, 58))):
            with pytest.raises(ValueError, match="noise"):
                policy.infer(obs, noise=bad)
        context = rtc.validate_context(
            {
                "protocol_version": 1,
                "token": "mamba-test",
                "source_number": 1,
                "source_sha256": "a" * 64,
                "previous_actions": np.ones((50, 58), np.float32),
                "inference_delay_steps": 3,
                "prefix_attention_horizon": 10,
                "prefix_attention_schedule": "exp",
                "max_guidance_weight": 5.0,
            }
        )
        with pytest.raises(ValueError, match="serial"):
            policy.infer(obs, rtc_context=context)
        trained = MambaTrainRTCPolicy(model, rtc_action_repack=lambda a: a, sample_kwargs={"num_steps": 2})
        for delay in range(4):
            current = rtc.validate_context({**context.to_dict(), "inference_delay_steps": delay})
            result = trained.infer(obs, noise=noise, rtc_context=current)
            np.testing.assert_array_equal(result["actions"][:delay], current.previous_actions[:delay])
            rtc.validate_receipt(result["rtc"], current)


def test_pre_hardware_gate_catches_bad_mamba_before_ros(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SHARPA_")}
    env.update(SHARPA_PYTHON=sys.executable, SHARPA_GELLO_REPO="/should-never-be-touched")
    for mode in (["--execute"], ["--mode", "execute"], ["--mode=execute"], ["--check"]):
        result = subprocess.run(
            ["bash", "deploy/fr3_sharpa_runtime/runtime.sh", "--model=mamba", *mode],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode != 0
        assert "SHARPA_MAMBA_CHECKPOINT" in result.stderr
        assert "should-never-be-touched" not in result.stderr


def test_transformer_mamba_profile_switching_is_isolated(tmp_path, monkeypatch):
    from deploy.fr3_sharpa_runtime.tests.test_full_finetune import checkpoint as full_fixture

    full_dir = tmp_path / "old"
    full_dir.mkdir()
    _, full_stats = full_fixture.__wrapped__(monkeypatch, full_dir)
    model_loader.checkpoint_contract()
    root, assets, _ = checkpoint(tmp_path / "new")
    for _ in range(2):
        select(monkeypatch, root, assets, "mamba")
        metadata = model_loader.checkpoint_contract()
        metadata["warmed_up"] = True
        config.validate_policy_metadata(metadata)
        select(monkeypatch, root, assets, "full")
        with pytest.raises(ValueError, match="explicit"):
            model_loader.checkpoint_contract()
        select(monkeypatch, full_dir, full_stats.parent, "mamba")
        with pytest.raises(ValueError):
            model_loader.checkpoint_contract()
        select(monkeypatch, full_dir, full_stats.parent, "full")
        metadata = model_loader.checkpoint_contract()
        metadata["warmed_up"] = True
        config.validate_policy_metadata(metadata)


def test_real_websocket_msgpack_with_synthetic_mamba_policy():
    """Actual TCP/WebSocket with a tiny model, not a production checkpoint claim."""
    import asyncio

    from openpi_client import msgpack_numpy
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve as websocket_serve

    from deploy.fr3_sharpa_runtime.action_adapter import SharpaActionAdapter
    from deploy.fr3_sharpa_runtime.timing import TimedPolicy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    with tiny_network():
        cfg = tiny_config(rtc_max_delay=3)
        model = cfg.create(jax.random.key(42))
        policy = MambaTrainRTCPolicy(model, rtc_action_repack=lambda a: a, sample_kwargs={"num_steps": 2})
        obs = jax.tree.map(lambda x: np.asarray(x)[0], cfg.fake_obs().to_dict())
        metadata = {"model_architecture": "pi05_mamba3_jax", "synthetic_test": True}
        server = WebsocketPolicyServer(TimedPolicy(policy), metadata=metadata)

        async def run():
            async with websocket_serve(server._handler, "127.0.0.1", 0, compression=None, max_size=None) as listener:
                port = listener.sockets[0].getsockname()[1]
                async with connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None) as ws:
                    assert msgpack_numpy.unpackb(await ws.recv()) == metadata
                    await ws.send(msgpack_numpy.packb(obs))
                    first = msgpack_numpy.unpackb(await ws.recv())
                    assert first["actions"].shape == (50, 58)
                    context = rtc.validate_context(
                        {
                            "protocol_version": 1,
                            "token": "websocket-mamba",
                            "source_number": 1,
                            "source_sha256": "a" * 64,
                            "previous_actions": first["actions"],
                            "inference_delay_steps": 3,
                            "prefix_attention_horizon": 10,
                            "prefix_attention_schedule": "exp",
                            "max_guidance_weight": 5.0,
                        }
                    )
                    await ws.send(msgpack_numpy.packb(rtc.make_request(obs, context)))
                    second = msgpack_numpy.unpackb(await ws.recv())
                    rtc.validate_receipt(second["rtc"], context)
                    np.testing.assert_array_equal(second["actions"][:3], first["actions"][:3])
                    return second["actions"]

        actions = asyncio.run(run())
        adapter = SharpaActionAdapter()
        chunk = adapter.adapt_chunk(actions)
        assert len(chunk.frames) == 50


def test_orbax_restore_preserves_full_precision_and_full_tree(tmp_path):
    from types import SimpleNamespace

    import orbax.checkpoint as ocp

    from openpi.policies.policy_config import create_trained_policy

    with tiny_network():
        cfg = tiny_config()
        model = cfg.create(jax.random.key(42))
        values = nnx.state(model).to_pure_dict()
        with ocp.PyTreeCheckpointer() as checkpointer:
            checkpointer.save(tmp_path / "params", {"params": values})
        data = SimpleNamespace(
            use_quantile_norm=False, data_transforms=transforms.Group(), model_transforms=transforms.Group()
        )
        training = SimpleNamespace(
            model=cfg, data=SimpleNamespace(create=lambda *args: data), assets_dirs=tmp_path, policy_metadata={}
        )
        policy = create_trained_policy(
            training, tmp_path, norm_stats={}, policy_class=MambaPolicy, sample_kwargs={"num_steps": 2}
        )
        restored = nnx.state(policy._model).to_pure_dict()
        for actual, expected in zip(jax.tree.leaves(restored), jax.tree.leaves(values), strict=True):
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)
        bad = {
            **values,
            "action_expert": {
                **values["action_expert"],
                "action_in_proj": {**values["action_expert"]["action_in_proj"], "kernel": jnp.full((58, 16), jnp.nan)},
            },
        }
        with pytest.raises(ValueError, match="non-finite"):
            cfg.load(bad)
        obs = jax.tree.map(lambda x: np.asarray(x)[0], cfg.fake_obs().to_dict())
        noise = np.random.default_rng(3).normal(size=(1, 50, 58)).astype(np.float32)
        actual = policy.infer(obs, noise=noise)["actions"]
        expected = model.sample_actions(jax.random.key(2), cfg.fake_obs(), noise=jnp.asarray(noise), num_steps=2)[0]
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
