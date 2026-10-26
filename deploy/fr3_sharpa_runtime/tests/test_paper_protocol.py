"""Client protocol tests: fake socket, real msgpack, no ROS or device imports."""
# ruff: noqa: PLC0415
# Lazy imports keep pure helpers hardware/framework-free and test patches local.

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.rolling import CONTEXT_KEY
from deploy.fr3_sharpa_runtime.rolling import RollingOptions
from deploy.fr3_sharpa_runtime.rolling import RollingReference
from deploy.fr3_sharpa_runtime.tests.test_paper_rolling import Clock
from deploy.fr3_sharpa_runtime.tests.test_paper_rolling import actions
from deploy.fr3_sharpa_runtime.tests.test_paper_rolling import metadata

ROOT = Path(__file__).resolve().parents[3]


def observation():
    result = {"observation/state": np.zeros(58, np.float32), "prompt": "test"}
    for key in ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image"):
        result[key] = np.zeros((8, 8, 3), np.uint8)
    return result


@pytest.mark.parametrize(
    ("server_restart", "delay", "expect_error"), [(False, 0, False), (True, 0, False), (False, 1.01, True)]
)
def test_client_new_socket_restart_and_late_response(monkeypatch, server_restart, delay, expect_error):
    from openpi_client import msgpack_numpy
    import websockets.sync.client

    from deploy.fr3_sharpa_runtime import runtime

    clock = Clock()
    monkeypatch.setattr(runtime.time, "monotonic_ns", clock)
    monkeypatch.setattr(runtime.config, "require_checkpoint", lambda: Path("synthetic"))
    monkeypatch.setattr(runtime.config, "validate_policy_metadata", lambda m: None)
    cache = RollingReference(RollingOptions(inpainting="soft"), clock_ns=clock)
    meta = {**metadata(), "inpainting_modes": ["off", "soft"], "inpainting_warmed_up": True}
    cache.commit(
        actions(),
        meta,
        {
            "status": "PASS",
            "policy_frames": 26,
            "completion": "commands_sent",
            "execution_target_count": 26,
            "frames": 26,
            "completed_monotonic_ns": clock(),
        },
        0,
    )
    request = observation()
    request[CONTEXT_KEY] = cache.context(1)
    if server_restart:
        meta = {**meta, "server_instance_id": "boot2"}
    sent = []

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def send(self, payload):
            sent.append(msgpack_numpy.unpackb(payload))

        def recv(self, **kwargs):
            if not sent:
                return msgpack_numpy.Packer().pack(meta)
            clock.advance(delay)
            return msgpack_numpy.Packer().pack(
                {
                    "actions": actions(),
                    "inference_experiment": {"run_id": request[CONTEXT_KEY]["run_id"], "chunk_index": 1},
                }
            )

    monkeypatch.setattr(websockets.sync.client, "connect", lambda *a, **kw: Socket())
    client = runtime.SharpaRuntime()
    if expect_error:
        with pytest.raises(RuntimeError, match="response expired"):
            client.infer_chunk(request)
    else:
        _, result, _ = client.infer_chunk(request)
        np.testing.assert_array_equal(result, actions())
    reference = sent[0][CONTEXT_KEY]["reference"]
    assert (reference is None) == server_restart


@pytest.mark.parametrize(
    ("extra", "expected_steps"),
    [
        ([], 15),
        (["--inference-preset", "paper-rolling"], 26),
        (["--inference-preset", "paper-rolling", "--inpainting", "soft", "--compression", "cubic"], 26),
    ],
)
def test_cli_check_default_and_experiments_without_hardware(extra, expected_steps):
    env = {key: value for key, value in os.environ.items() if not key.startswith("SHARPA_")}
    env.update(SHARPA_PYTHON=sys.executable, PYTHONPATH=f"{ROOT}:{ROOT / 'src'}")
    process = subprocess.run(
        ["bash", "deploy/fr3_sharpa_full/runtime_delta.sh", "--check", *extra],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(process.stdout)
    assert report["execute_steps"] == expected_steps
    assert report["initialization"] == "current"
    assert not report["hardware_output"]


def test_cli_rejects_experiment_without_preset_and_conflicting_prefix():
    for extra in (["--compression", "cubic"], ["--inference-preset", "paper-rolling", "--execute-steps", "15"]):
        process = subprocess.run(
            [sys.executable, "-m", "deploy.fr3_sharpa_runtime.run", "--check", *extra],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert process.returncode != 0
        assert "paper-rolling" in process.stderr
