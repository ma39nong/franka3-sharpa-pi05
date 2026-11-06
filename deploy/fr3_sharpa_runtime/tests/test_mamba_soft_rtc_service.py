"""Offline admission and actual WebSocket transport with explicit synthetic models."""

# ruff: noqa: SLF001
import asyncio
from dataclasses import replace
import json
import os
import subprocess
import sys

import jax
import numpy as np
from openpi_client import msgpack_numpy
from openpi_client import rtc
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve as websocket_serve

from deploy.fr3_sharpa_runtime import run
from deploy.fr3_sharpa_runtime.mamba_contract import is_full_t3
from deploy.fr3_sharpa_runtime.mamba_model import MambaPaperSoftRTCPolicy
from deploy.fr3_sharpa_runtime.mamba_model import MambaSoftRTCPolicy
from deploy.fr3_sharpa_runtime.mamba_soft_rtc import warmup_soft_rtc
from deploy.fr3_sharpa_runtime.rtc_stream import require_capabilities
from deploy.fr3_sharpa_runtime.rtc_stream import require_paper_capabilities
from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import checkpoint
from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import select
from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import tiny_config
from deploy.fr3_sharpa_runtime.tests.mamba_fixtures import tiny_network
from deploy.fr3_sharpa_runtime.tests.test_mamba_rtc import _paper_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer


@pytest.mark.parametrize("paper", [False, True])
@pytest.mark.parametrize("trained", [False, True])
def test_runtime_check_uses_same_soft_hard_matrix(tmp_path, monkeypatch, capsys, paper, trained):
    root, assets, _ = checkpoint(tmp_path, paper=paper, delay=3 if trained else 0)
    select(monkeypatch, root, assets, "mamba-delta" if paper else "mamba")
    base = [
        "runtime",
        "--check",
        "--broker-mode",
        "rtc",
        "--speed",
        "fast",
        "--execute-steps",
        "15",
        "--num-chunks",
        "2",
        "--sharpa-backend",
        "sdk_direct",
        "--rtc-initial-delay-steps",
        "3",
    ]
    flags = [[], ["--paper-rtc"], ["--train-rtc"], ["--train-paper-rtc"]]
    expected = ["--train-paper-rtc" if paper else "--train-rtc"] if trained else (["--paper-rtc"] if paper else [])
    for chosen in flags:
        monkeypatch.setattr(sys, "argv", base + chosen)
        if chosen == expected:
            run.main()
            assert json.loads(capsys.readouterr().out)["broker_mode"] == "rtc"
        else:
            with pytest.raises(SystemExit):
                run.main()
            capsys.readouterr()


@pytest.mark.parametrize("paper", [False, True])
def test_shell_soft_checks_do_not_enter_ros_or_hardware(tmp_path, paper):
    root, assets, _ = checkpoint(tmp_path, paper=paper)
    prefix = "SHARPA_MAMBA_DELTA" if paper else "SHARPA_MAMBA"
    env = {k: v for k, v in os.environ.items() if not k.startswith("SHARPA_")}
    env.update(
        {
            prefix + "_CHECKPOINT": str(root),
            "SHARPA_PYTHON": sys.executable,
            "SHARPA_GELLO_REPO": "/must-never-open-ros",
        }
    )
    base = [
        "bash",
        "deploy/fr3_sharpa_runtime/runtime.sh",
        "--model",
        "mamba-delta" if paper else "mamba",
        "--broker-mode",
        "rtc",
        "--speed",
        "fast",
        "--execute-steps",
        "15",
        "--num-chunks",
        "2",
    ]
    selected = ["--paper-rtc"] if paper else []
    result = subprocess.run([*base, "--check", *selected], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["paper_rtc"] == paper
    for execute in (["--execute"], ["--mode", "execute"], ["--mode=execute"]):
        invalid = ["--train-rtc"]  # No clean-prefix training evidence.
        result = subprocess.run(base + execute + invalid, env=env, text=True, capture_output=True, check=False)
        assert result.returncode != 0
        assert "must-never-open-ros" not in result.stderr
        assert "conflicts with verified" in result.stderr


@pytest.mark.parametrize("paper", [False, True])
def test_actual_vjp_warmup_receipts_and_websocket(tmp_path, paper):
    with tiny_network():
        cfg = replace(_paper_config(checkpoint(tmp_path, paper=True)[1]), rtc_max_delay=0) if paper else tiny_config()
        model = cfg.create(jax.random.key(42))
        policy_class = MambaPaperSoftRTCPolicy if paper else MambaSoftRTCPolicy
        policy = policy_class(model, rtc_action_repack=lambda a: a, sample_kwargs={"num_steps": 2})
        obs = jax.tree.map(lambda v: np.asarray(v)[0], cfg.fake_obs().to_dict())
        ordinary = policy.infer(obs)
        assert "rtc" not in ordinary
        metadata = warmup_soft_rtc(policy, obs, ordinary["actions"], paper=paper)
        metadata["paper_algorithm"] = {"mode": cfg.paper_mode, "beta": cfg.paper_beta} if paper else None
        require_capabilities(metadata)
        if paper:
            require_paper_capabilities(metadata)
            assert metadata["rtc_smoke_prefix_lengths"] == [50, 35, 1]
        assert all(v > 0 for v in metadata["rtc_soft_warmup"]["wall_ms"])
        assert metadata["rtc_soft_warmup"]["performance_admission"] == "NOT_MEASURED"
        server = WebsocketPolicyServer(policy, metadata={**metadata, "synthetic_model": True})

        async def request():
            async with websocket_serve(server._handler, "127.0.0.1", 0, compression=None, max_size=None) as listener:
                port = listener.sockets[0].getsockname()[1]
                async with connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None) as ws:
                    hello = msgpack_numpy.unpackb(await ws.recv())
                    require_capabilities(hello)
                    await ws.send(msgpack_numpy.packb(obs))
                    first = msgpack_numpy.unpackb(await ws.recv())
                    assert "rtc" not in first
                    context = rtc.validate_context(
                        {
                            "protocol_version": 1,
                            "token": "soft-ws",
                            "source_number": 1,
                            "source_sha256": "b" * 64,
                            "previous_actions": first["actions"][: 1 if paper else 50],
                            "inference_delay_steps": 1,
                            "prefix_attention_horizon": 1,
                            "prefix_attention_schedule": "exp",
                            "max_guidance_weight": 0.2,
                        }
                    )
                    await ws.send(msgpack_numpy.packb(rtc.make_request(obs, context)))
                    second = msgpack_numpy.unpackb(await ws.recv())
                    rtc.validate_receipt(second["rtc"], context)
                    assert second["actions"].shape == (50, 58)
                    assert np.isfinite(second["actions"]).all()
                    assert not np.array_equal(second["actions"][:1], context.previous_actions[:1])

        asyncio.run(request())


@pytest.mark.parametrize("failure", ["nan", "receipt"])
def test_warmup_failure_never_advertises_ready(failure):
    class Broken:
        def infer(self, obs, *, rtc_context):
            return {"actions": np.full((50, 58), np.nan if failure == "nan" else 0), "rtc": {}}

    with pytest.raises(ValueError, match="NaN or Inf|RTC"):
        warmup_soft_rtc(Broken(), {}, np.zeros((50, 58)))


@pytest.mark.parametrize("mode", ["global_quantile", "global_zscore_arms", "timestamp_zscore_arms"])
@pytest.mark.parametrize("beta", [0.0, 0.5])
@pytest.mark.parametrize("delay", [0, 3])
def test_full_t3_requires_all_three_training_properties(mode, beta, delay):
    meta = {"paper_algorithm": {"mode": mode, "beta": beta}, "train_rtc": {"max_delay": delay} if delay else None}
    assert is_full_t3(meta) == (mode == "timestamp_zscore_arms" and beta > 0 and delay > 0)


@pytest.mark.parametrize("paper", [False, True])
def test_soft_check_only_does_not_import_jax(tmp_path, paper):
    root, _, _ = checkpoint(tmp_path, paper=paper)
    prefix = "SHARPA_MAMBA_DELTA" if paper else "SHARPA_MAMBA"
    env = {k: v for k, v in os.environ.items() if not k.startswith("SHARPA_")}
    env.update({"SHARPA_MODEL_PROFILE": "mamba-delta" if paper else "mamba", prefix + "_CHECKPOINT": str(root)})
    flag = "--enable-paper-rtc" if paper else "--enable-rtc"
    code = (
        "import sys; from deploy.fr3_sharpa_runtime import serve; "
        f"sys.argv=['serve','--check-only',{flag!r}]; serve.main(); "
        "assert 'jax' not in sys.modules; assert 'rclpy' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], env=env, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["rtc_warmed_up"] is False
