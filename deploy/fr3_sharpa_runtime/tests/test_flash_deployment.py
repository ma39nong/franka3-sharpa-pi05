"""FLASH-specific contracts and old-profile isolation; no real weights/hardware."""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.fr3_sharpa_flash.export import export
from deploy.fr3_sharpa_flash.serve import warmup_observation
from deploy.fr3_sharpa_runtime import flash_contract as fc
from deploy.fr3_sharpa_runtime.flash_history import FlashHistoryBuffer


@pytest.fixture
def package(tmp_path):
    repo = tmp_path / "training"
    files = set(fc.BACKEND_FILES) | {"scripts/train_flash.py", "src/openpi/training/optimizer.py", "scripts/train.py"}
    for name in files:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic backend fixture; never a model\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "synthetic backend",
        ],
        check=True,
    )
    root = tmp_path / "experiment" / "2000"
    assets = root / "assets" / "flash-test"
    (root / "params" / "d").mkdir(parents=True)
    assets.mkdir(parents=True)
    (root / "_CHECKPOINT_METADATA").write_text(json.dumps({"commit_timestamp_nsecs": 1}))
    (root / "params/manifest.ocdbt").write_bytes(b"SYNTHETIC")
    (root / "params/d/blob").write_bytes(b"SYNTHETIC NOT WEIGHTS")
    entries = {}
    for key, names, shape in [
        ("in", ["params", "action_in_proj", "kernel", "value"], [58, 1024]),
        ("out", ["params", "action_out_proj", "bias", "value"], [58]),
    ]:
        entries[key] = {"key_metadata": [{"key": name} for name in names], "value_metadata": {"write_shape": shape}}
    (root / "params/_METADATA").write_text(json.dumps({"tree_metadata": entries}))
    stats = {
        "state": {"mean": [0.0] * 58, "std": [1.0] * 58, "q01": [-2.0] * 58, "q99": [3.0] * 58},
        "actions": {"mean": [0.0] * 58, "std": [1.0] * 58, "q01": [-4.0] * 58, "q99": [6.0] * 58},
    }
    (assets / "norm_stats.json").write_text(json.dumps({"norm_stats": stats}))
    data = {
        "contract": {
            "action_semantics": "absolute",
            "horizon": 50,
            "history": 8,
            "target": 54,
            "K": 6,
            "stride": 1,
            "joint_names": list(fc.ORDER),
        }
    }
    data["sha256"] = fc.digest_json(data)
    (assets / "flash_manifest.json").write_text(json.dumps(data))
    (assets / "flash_validation.json").write_text("[]")
    (assets / "flash_seal.json").write_text(
        json.dumps(
            {
                "manifest_sha256": data["sha256"],
                "norm_sha256": fc.sha(assets / "norm_stats.json"),
                "validation_sha256": fc.digest_json([]),
            }
        )
    )
    options = {
        "pi05": True,
        "dtype": "bfloat16",
        "action_dim": 58,
        "action_horizon": 50,
        "max_token_len": 256,
        "discrete_state_input": True,
        "freeze_vlm": False,
        "poly_degree": 6,
        "history_steps": 8,
        "rtc_max_delay": 0,
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "history_reg_lambda": 0.1,
        "boundary_constraint": True,
        "history_noise_std": 0.5,
        "consistency_weight": 1.0,
        "action_q01": tuple(stats["actions"]["q01"]),
        "action_q99": tuple(stats["actions"]["q99"]),
    }
    model_repr = "Pi0FlashConfig(" + ", ".join(f"{k}={v!r}" for k, v in options.items()) + ")"
    source_files = (
        "scripts/train_flash.py",
        "src/openpi/training/flash_native.py",
        "src/openpi/models/pi0_flash.py",
        "src/openpi/shared/flash_polynomial.py",
        "src/openpi/training/optimizer.py",
        "src/openpi/models/pi0.py",
        "src/openpi/models/gemma.py",
        "src/openpi/models/siglip.py",
        "scripts/train.py",
    )
    contract = root.parent / "flash_training_contract.json"
    contract.write_text(
        json.dumps(
            {
                "model": model_repr,
                "dataset": {
                    "normalization_sha256": fc.sha(assets / "norm_stats.json"),
                    "manifest_sha256": data["sha256"],
                },
                "training_code_sha256": fc.digest_json({name: (repo / name).read_text() for name in source_files}),
            }
        )
    )
    tokenizer = tmp_path / "tokenizer.model"
    tokenizer.write_bytes(b"SYNTHETIC TOKENIZER")
    export(root, repo, tokenizer, contract)
    return root, repo


def test_sealed_package_inspection_never_imports_model_or_jax(package):
    root, repo = package
    code = (
        "import sys; from deploy.fr3_sharpa_runtime.flash_contract import inspect_flash; "
        "m=inspect_flash(sys.argv[1],backend=sys.argv[2]); "
        'assert m["algorithm"]=="flash"; assert "jax" not in sys.modules; assert "openpi" not in sys.modules'
    )
    subprocess.run([sys.executable, "-c", code, str(root), str(repo)], check=True)
    assert fc.execution_budget(root) is None


@pytest.mark.parametrize("damage", ["stats", "params", "backend", "manifest", "missing_blob"])
def test_asset_and_code_mutations_fail_closed(package, damage):
    root, repo = package
    target = {
        "stats": root / "assets/flash-test/norm_stats.json",
        "params": root / "params/d/blob",
        "backend": repo / "src/openpi/models/pi0_flash.py",
    }
    if damage in target:
        target[damage].write_text("corrupt")
    elif damage == "missing_blob":
        (root / "params/d/blob").unlink()
    else:
        path = root / "flash_deployment.json"
        data = json.loads(path.read_text())
        data["algorithm"] = "pi05"
        path.write_text(json.dumps(data))
    with pytest.raises((ValueError, json.JSONDecodeError)):
        fc.inspect_flash(root, backend=repo)


def test_flash_and_ordinary_profiles_are_mutually_exclusive(package, monkeypatch):
    from deploy.fr3_sharpa_control.plan import inspect_checkpoint
    from deploy.fr3_sharpa_runtime import config
    from deploy.fr3_sharpa_runtime import model_loader

    root, repo = package
    found = inspect_checkpoint(root)
    assert found["profile"] == "flash-absolute"
    assert found["supported_modes"] == ["serial"]
    with pytest.raises(ValueError, match="Absolute"):
        inspect_checkpoint(root, semantics="delta")
    monkeypatch.setattr(config, "CHECKPOINT", root)
    monkeypatch.setattr(config, "NORM_STATS", root / "assets/flash-test/norm_stats.json")
    with pytest.raises(ValueError, match="requires flash-absolute"):
        model_loader.checkpoint_contract()
    (root / "flash_deployment.json").unlink()
    assert inspect_checkpoint(root)["profile"] == "flash-absolute"  # Original training step works without export.


def test_model_repr_parser_rejects_code_execution():
    with pytest.raises((ValueError, SyntaxError)):
        fc.parse_model_repr("Pi0FlashConfig(dtype=__import__('os').system('false'))")


@pytest.mark.parametrize("damage", ["state", "shape", "grid", "future", "old", "session", "noise", "rtc", "target"])
def test_wire_rejects_inconsistent_or_nonflash_inputs(damage):
    obs = warmup_observation()
    if damage == "state":
        obs["observation/state"][0] = 1
    if damage == "shape":
        obs["observation/history"] = np.zeros((7, 58))
    if damage == "grid":
        obs["flash_context"]["history_grid_source_ns"][0] += 1
    if damage == "future":
        obs["flash_context"]["history_source_ns"][0][0] += 1
    if damage == "old":
        obs["flash_context"]["history_source_ns"][0][0] -= 100_000_000
    if damage == "session":
        obs["flash_context"]["session_id"] = ""
    if damage in ("noise", "rtc", "target"):
        obs[{"noise": "noise", "rtc": "rtc", "target": "target_ext"}[damage]] = {}
    with pytest.raises(ValueError, match="FLASH"):
        fc.validate_flash_observation(obs)


def feed_history(buffer, until=1_000_000_000, *, shift=0):
    # Four asynchronously sampled groups; hands 20Hz, arms 100Hz.
    for relative in range(0, until + 1, 10_000_000):
        for index, group in enumerate(fc.GROUPS):
            if group.endswith("hand") and relative % 50_000_000:
                continue
            stamp = 2_000_000_000 + relative + shift
            width = 7 if group.endswith("arm") else 22
            buffer.observe(
                group, np.full(width, index + relative / 1e9), source_ns=stamp, received_ns=stamp, now_ros_ns=stamp
            )


def test_20hz_hands_zoh_reuses_real_sources_and_keeps_atomic_state():
    buffer = FlashHistoryBuffer()
    feed_history(buffer)
    result = buffer.snapshot(now_ros_ns=3_000_000_000, now_mono_ns=3_000_000_000)
    history = np.array(result["history"])
    source = np.array(result["flash_context"]["history_source_ns"])
    np.testing.assert_array_equal(history[-1], result["state"])
    assert np.any(np.diff(source[:, 1]) == 0)  # real bounded reuse, not eight fabricated messages.
    assert source[0, 1] < 3_000_000_000 - 200_000_000
    assert source.shape == (8, 4)
    assert all(len(samples) <= 101 for samples in buffer.samples.values())
    obs = warmup_observation()
    obs.update(
        {
            "observation/state": np.array(result["state"], np.float32),
            "observation/history": np.array(result["history"], np.float32),
            "flash_context": result["flash_context"],
        }
    )
    fc.validate_flash_observation(obs)


def test_200ms_is_not_enough_for_history_and_stale_sources_fail():
    buffer = FlashHistoryBuffer()
    feed_history(buffer, until=200_000_000)
    with pytest.raises(ValueError, match="coverage"):
        buffer.snapshot(now_ros_ns=2_200_000_000, now_mono_ns=2_200_000_000)
    feed_history(buffer)  # duplicate/backward first update invalidates old session, then new real coverage accumulates.
    with pytest.raises(ValueError, match="stale"):
        buffer.snapshot(now_ros_ns=3_300_000_000, now_mono_ns=3_300_000_000)


@pytest.mark.parametrize("fault", ["duplicate", "backward", "gap", "nonfinite", "clock"])
def test_history_invalidates_generation_on_fault(fault):
    buffer = FlashHistoryBuffer()
    feed_history(buffer)
    old_session = buffer.session_id
    stamp = 3_000_000_000
    values = np.zeros(7)
    received, ros = stamp + 1, stamp + 1
    if fault == "backward":
        stamp -= 1
    if fault == "gap":
        stamp += 200_000_000
        received = ros = stamp
    if fault == "nonfinite":
        values[0] = np.nan
    if fault == "clock":
        received += 500_000_000
    buffer.observe("left_arm", values, source_ns=stamp, received_ns=received, now_ros_ns=ros)
    assert buffer.session_id != old_session
    with pytest.raises(ValueError, match="FLASH"):
        buffer.snapshot(now_ros_ns=ros, now_mono_ns=received)


def test_expired_plan_rejected_before_any_hardware_output():
    plan = {
        "protocol_version": 1,
        "request_id": "request",
        "session_id": "session",
        "max_plan_age_ms": 200,
        "captured_monotonic_ns": 1_000_000_000,
        "anchor_age_at_capture_ns": 50_000_000,
    }
    assert (
        fc.check_plan_age(plan, bridge_steps=1, speed_scale=1, now_ns=1_050_000_000)["predicted_first_target_age_ms"]
        < 200
    )
    with pytest.raises(ValueError, match="expired before any frame"):
        fc.check_plan_age(plan, bridge_steps=100, speed_scale=1, now_ns=1_050_000_000)
    with pytest.raises(ValueError, match="budget"):
        fc.check_plan_age({**plan, "max_plan_age_ms": None}, bridge_steps=0, speed_scale=1, now_ns=1_050_000_000)


def test_codec_compatible_with_existing_client():
    from openpi_client import msgpack_numpy

    from deploy.fr3_sharpa_flash import codec

    obs = warmup_observation()
    decoded = codec.unpackb(msgpack_numpy.packb(obs))
    np.testing.assert_array_equal(decoded["observation/history"], obs["observation/history"])
    decoded = msgpack_numpy.unpackb(codec.packb(obs))
    np.testing.assert_array_equal(decoded["observation/state"], obs["observation/state"])


def test_service_check_only_is_gpu_free_and_validates_backend(package):
    root, repo = package
    env = {**os.environ, "SHARPA_FLASH_CHECKPOINT": str(root), "SHARPA_FLASH_REPO": str(repo)}
    result = subprocess.run(
        [sys.executable, "-m", "deploy.fr3_sharpa_flash.serve", "--check-only"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["algorithm"] == "flash"


def test_preprocessing_matches_pinned_native_numpy_order_without_loading_dataset():
    # Extract only the two reference methods (no dataset initialization/parquet scan).
    training = Path(__file__).parent / "fixtures/flash_native_reference.py"
    if not training.is_file():
        pytest.fail("Pinned training source required for preprocessing parity test; set up the documented checkout")
    import torch
    import torch.nn.functional as F  # noqa: N812 -- reference training namespace

    from deploy.fr3_sharpa_flash.policy import preprocess

    tree = ast.parse(training.read_text())
    native = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NativeDataset")
    native.body = [
        node
        for node in native.body
        if isinstance(node, ast.FunctionDef) and node.name in ("resize_image", "observation")
    ]
    namespace = {
        "np": np,
        "torch": torch,
        "F": F,
        "CAMERAS": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
        "model_lib": SimpleNamespace(Observation=lambda **kw: kw),
    }
    exec(compile(ast.Module(body=[native], type_ignores=[]), str(training), "exec"), namespace)
    tokenizer = SimpleNamespace(
        tokenize=lambda prompt, state: (np.digitize(state, np.linspace(-1, 1, 257)[:-1]), np.ones(58, bool))
    )
    rng = np.random.default_rng(2)
    obs = warmup_observation()
    obs["observation/state"] = rng.uniform(-2, 2, 58).astype(np.float32)
    images = {key: rng.integers(0, 256, (231, 315, 3), dtype=np.uint8) for key in namespace["CAMERAS"]}
    from deploy.fr3_sharpa_flash.policy import IMAGE_KEYS

    for key, field in IMAGE_KEYS.items():
        obs[field] = images[key]
    lo = np.linspace(-2.001, -1.999, 58)
    hi = np.linspace(1.999, 2.001, 58)
    stats = {"state": {"q01": lo, "q99": hi}}
    reference = namespace["NativeDataset"]()
    reference.stats = {"state": SimpleNamespace(q01=lo, q99=hi)}
    reference.tokenizer = tokenizer
    reference.window = lambda index: (np.zeros((8, 58)), np.zeros((54, 58)), obs["observation/state"], obs["prompt"])
    reference.image = lambda index, key: images[key]
    expected, _, _ = reference.observation(None)
    actual = preprocess(obs, stats, tokenizer)
    for key in ("state", "tokenized_prompt", "tokenized_prompt_mask"):
        np.testing.assert_array_equal(actual[key], expected[key])
    for key in images:
        np.testing.assert_array_equal(actual["images"][key], expected["images"][key])


@pytest.mark.parametrize("tamper", [False, True])
def test_client_preserves_history_checks_receipt_and_does_not_unormalize(package, monkeypatch, tamper):
    from openpi_client import msgpack_numpy
    import websockets.sync.client

    from deploy.fr3_sharpa_runtime import config
    from deploy.fr3_sharpa_runtime.model_profiles import get_model_profile
    from deploy.fr3_sharpa_runtime.runtime import SharpaRuntime

    root, repo = package
    monkeypatch.setattr(config, "MODEL_PROFILE", get_model_profile("flash-absolute"))
    monkeypatch.setattr(config, "CONFIG_NAME", fc.CONFIG_NAME)
    monkeypatch.setattr(config, "CHECKPOINT", root)
    monkeypatch.setattr(config, "NORM_STATS", root / "assets/flash-test/norm_stats.json")
    monkeypatch.setattr(config, "_validation_cache", None)
    metadata = {**fc.inspect_flash(root), "warmed_up": True}
    obs = warmup_observation()
    obs["observation/history"][:] = np.arange(58, dtype=np.float32)
    obs["observation/state"] = obs["observation/history"][-1].copy()
    actions = np.repeat(obs["observation/state"][None], 50, axis=0)

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def recv(self, **kw):
            if not hasattr(self, "request"):
                return msgpack_numpy.packb(metadata)
            ctx = self.request["flash_context"]
            receipt = {
                key: ctx[key]
                for key in ("protocol_version", "request_id", "session_id", "observation_anchor_source_ns")
            }
            receipt["model_identity"] = metadata["flash_manifest_sha256"]
            if tamper:
                receipt["session_id"] = "another-session"
            return msgpack_numpy.packb({"actions": actions, "flash_receipt": receipt})

        def send(self, payload):
            self.request = msgpack_numpy.unpackb(payload)
            np.testing.assert_array_equal(self.request["observation/history"], obs["observation/history"])

    monkeypatch.setattr(websockets.sync.client, "connect", lambda *a, **kw: Connection())
    client = SharpaRuntime("ws://127.0.0.1:18768")
    if tamper:
        with pytest.raises(ValueError, match="receipt"):
            client.infer_chunk(obs)
    else:
        _, actual, _ = client.infer_chunk(obs)
        np.testing.assert_array_equal(actual, actions)
        assert obs["_flash_receipt"]["session_id"] == obs["flash_context"]["session_id"]
    with pytest.raises(ValueError, match="RTC"):
        client.infer_chunk(obs, rtc_context={})


def test_flash_cli_checks_and_other_profiles_remain_independent(package):
    root, repo = package
    env = {**os.environ, "SHARPA_FLASH_CHECKPOINT": str(root), "SHARPA_MODEL_PROFILE": "flash-absolute"}
    result = subprocess.run(
        ["bash", "deploy/fr3_sharpa_runtime/runtime.sh", "--model", "flash-absolute", "--check"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["model_profile"] == "flash-absolute"
    result = subprocess.run(
        [
            "bash",
            "deploy/fr3_sharpa_runtime/runtime.sh",
            "--model",
            "flash-absolute",
            "--check",
            "--broker-mode",
            "rtc",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "FLASH v1" in result.stderr


def test_flash_history_trace_records_values_and_distinct_decoding(package, tmp_path):
    from deploy.fr3_sharpa_runtime.action_adapter import SharpaActionAdapter
    from deploy.fr3_sharpa_runtime.execution_trace import ExecutionTraceWriter

    obs = warmup_observation()
    obs["observation/history"][:] = 0.125
    obs["observation/state"][:] = 0.125
    actions = np.full((50, 58), 0.25, dtype=np.float32)
    chunk = SharpaActionAdapter().adapt_chunk(actions, start_timestamp_ns=1_000_000_000, session_id="test")
    writer = ExecutionTraceWriter(tmp_path / "trace")
    path = writer.begin_chunk(chunk_index=0, metadata={"algorithm": "flash"}, observation_before=obs, actions=actions, adapted_chunk=chunk)
    report = json.loads(Path(path).read_text())
    assert report["flash"]["history_rad"] == obs["observation/history"].tolist()
    assert report["normalization_stage"] == "flash_internal_denormalize"


def test_publisher_reconnect_invalidates_old_flash_history():
    from deploy.fr3_sharpa_runtime.flash_observation import FlashObservationProvider

    provider = object.__new__(FlashObservationProvider)
    provider.flash_history = FlashHistoryBuffer()
    provider.publisher_ids = {}
    provider.remember = lambda *a: None
    callback = provider.subscription_callback("left_hand")
    callback(object(), SimpleNamespace(publisher_gid=b"first"))
    previous = provider.flash_history.session_id
    callback(object(), SimpleNamespace(publisher_gid=b"second"))
    assert provider.flash_history.session_id != previous


def test_real_tokenizer_matches_native_state_bucketing():
    from deploy.fr3_sharpa_flash.policy import load_tokenizer
    from deploy.fr3_sharpa_flash.policy import preprocess

    path = Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model"
    # Asset-independent tests above cover the pipeline everywhere. This local
    # optional paired check is collected only when the real tokenizer is present.
    if not path.is_file():
        pytest.skip("Local real tokenizer not installed; mock preprocessing is tested separately")
    tokenizer = load_tokenizer(path)
    obs = warmup_observation()
    obs["observation/state"] = np.linspace(-2, 2, 58, dtype=np.float32)
    lo, hi = np.linspace(-2.001, -1.999, 58), np.linspace(1.999, 2.001, 58)
    expected_state = (obs["observation/state"] - lo) / (hi - lo + 1e-6) * 2 - 1
    tokens, mask = tokenizer.tokenize(obs["prompt"], expected_state)
    actual = preprocess(obs, {"state": {"q01": lo, "q99": hi}}, tokenizer)
    np.testing.assert_array_equal(actual["tokenized_prompt"][0], tokens)
    np.testing.assert_array_equal(actual["tokenized_prompt_mask"][0], mask)


def test_joint_history_current_row_keeps_original_200ms_freshness():
    buffer = FlashHistoryBuffer()
    for relative in range(0, 1_000_000_001, 5_000_000):
        for index, group in enumerate(fc.GROUPS):
            if group == "left_hand" and relative % 50_000_000 != 25_000_000:
                continue
            if group == "right_hand" and (relative % 50_000_000 or relative > 800_000_000):
                continue
            if group.endswith("arm") and relative % 10_000_000:
                continue
            stamp = 2_000_000_000 + relative
            buffer.observe(
                group,
                np.full(7 if group.endswith("arm") else 22, index),
                source_ns=stamp,
                received_ns=stamp,
                now_ros_ns=stamp,
            )
    # Latest messages all meet 200ms, but at right_hand's common anchor 2.8s,
    # left_hand must use its 2.775s sample, already 225ms old at capture.
    with pytest.raises(ValueError, match="current state stale"):
        buffer.snapshot(now_ros_ns=3_000_000_000, now_mono_ns=3_000_000_000)


def test_policy_calls_history_nfe1_and_returns_physical_actions_without_second_transform(monkeypatch):
    # Exercise the actual FlashPolicy.infer method with a local model sink; no
    # checkpoint restore or GPU result is implied by this interface test.
    import jax

    from deploy.fr3_sharpa_flash import policy as flash_policy

    policy = object.__new__(flash_policy.FlashPolicy)
    policy.stats = {"state": {"q01": [-100.0] * 58, "q99": [100.0] * 58}}
    policy.tokenizer = SimpleNamespace(tokenize=lambda *args: (np.zeros(256, int), np.ones(256, bool)))
    policy.rng = jax.random.key(0)
    policy.metadata = {"flash_manifest_sha256": "fake-manifest"}
    obs = warmup_observation()
    obs["observation/history"][:] = np.arange(58, dtype=np.float32)
    obs["observation/state"] = obs["observation/history"][-1].copy()
    output = np.full((1, 50, 58), 0.375, dtype=np.float32)

    def sample(rng, observation, *, history, num_steps):
        assert type(num_steps) is int
        assert num_steps == 1
        np.testing.assert_array_equal(np.asarray(history)[0], obs["observation/history"])
        return jax.numpy.asarray(output)

    policy.sample = sample
    result = policy.infer(obs)
    np.testing.assert_array_equal(result["actions"], output[0])
    assert result["flash_receipt"]["model_identity"] == "fake-manifest"


def test_historical_report_is_optional_and_does_not_set_execution_parameters(package, monkeypatch):
    root, repo = package
    manifest_path = root / "flash_deployment.json"
    manifest = json.loads(manifest_path.read_text())
    metadata = fc.inspect_flash(root)
    report = {
        "status": "PASS",
        "resampler_version": fc.RESAMPLER,
        "history_pairing_verified": True,
        "model_equivalence_verified": True,
        "max_plan_age_ms": 200,
        **{
            key: metadata[key]
            for key in ("checkpoint_manifest_sha256", "normalization_sha256", "flash_backend_identity")
        },
    }
    path = root / "flash_assets/execution_validation.json"
    path.write_text(json.dumps(report))
    relative = str(path.relative_to(root))
    manifest["execution_validation"] = relative
    manifest["files"][relative] = {"sha256": fc.sha(path), "size": path.stat().st_size}
    manifest_path.write_text(json.dumps(manifest))
    fc.inspect_flash(root)
    assert fc.execution_budget(root) is None
    monkeypatch.setenv("SHARPA_FLASH_MAX_PLAN_AGE_MS", "350")
    assert fc.execution_budget(root) == 350
    report["normalization_sha256"] = "wrong-package"
    path.write_text(json.dumps(report))
    manifest["files"][relative] = {"sha256": fc.sha(path), "size": path.stat().st_size}
    manifest_path.write_text(json.dumps(manifest))
    fc.inspect_flash(root)  # Status/claims in a historical report are not startup gates.
    path.unlink()
    fc.inspect_flash(root)  # Even the old report copy itself is optional.


def test_flash_ros_snapshot_uses_causal_images_at_same_history_anchor(monkeypatch):
    import base64
    from collections import deque

    from deploy.fr3_sharpa_runtime import flash_observation as module

    provider = object.__new__(module.FlashObservationProvider)
    provider.flash_history = FlashHistoryBuffer()
    feed_history(provider.flash_history)
    provider.node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=3_000_000_000))
    )
    provider.rclpy = SimpleNamespace(spin_once=lambda *a, **kw: None)
    provider.cameras_ready = lambda now: True
    provider.histories = {}
    for index, key in enumerate(("cam0", "cam1", "cam2")):

        def message(stamp, pixel):
            return SimpleNamespace(
                header=SimpleNamespace(
                    stamp=SimpleNamespace(sec=stamp // 1_000_000_000, nanosec=stamp % 1_000_000_000)
                ),
                height=1,
                width=1,
                step=3,
                encoding="rgb8",
                data=bytes([pixel] * 3),
            )

        provider.histories[key] = deque(
            [(message(2_980_000_000, index + 1), 2_985_000_000), (message(3_001_000_000, 99), 2_999_000_000)]
        )
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: 3_000_000_000)
    result = provider.snapshot(0.1)
    assert result["flash_context"]["observation_anchor_source_ns"] == 3_000_000_000
    assert result["history"][-1] == result["state"]
    assert result["topic_stamp_ns"]["cam0"] == 2_980_000_000
    for index, key in enumerate(("cam0", "cam1", "cam2")):
        assert base64.b64decode(result["images"][key]["rgb_b64"]) == bytes([index + 1] * 3)


def test_ordinary_runtime_never_imports_flash_history_or_requires_its_environment():
    code = (
        "from deploy.fr3_sharpa_runtime.runtime import validate_observation; import sys; "
        'assert "deploy.fr3_sharpa_runtime.flash_history" not in sys.modules; '
        'assert "deploy.fr3_sharpa_flash.policy" not in sys.modules'
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith("SHARPA_FLASH_")}
    env["SHARPA_MODEL_PROFILE"] = "full"
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_gui_capability_rejects_flash_rtc_and_preserves_serial(package):
    from deploy.fr3_sharpa_control.plan import DeploymentSettings

    root, repo = package
    deployment = Path(__file__).resolve().parents[3]
    settings = DeploymentSettings(
        repository=str(deployment),
        checkpoint=str(root),
        semantics="absolute",
        mode="serial",
        initialization="current",
        speed="fast",
        rounds=0,
    )
    plan = settings.plan()
    assert plan["service"][1] == "deploy/fr3_sharpa_runtime/serve_flash.sh"
    assert plan["port"] == 18768
    assert plan["model"]["supported_modes"] == ["serial"]
    from dataclasses import replace

    with pytest.raises(ValueError, match="支持"):
        replace(settings, mode="rtc").plan()


def test_cli_without_report_reaches_existing_ros_preconditions_without_importing_sdk(package):
    root, repo = package
    env = {**os.environ, "SHARPA_MODEL_PROFILE": "flash-absolute", "SHARPA_FLASH_CHECKPOINT": str(root)}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "deploy.fr3_sharpa_runtime.run",
            "--execute",
            "--enable-hardware",
            "--sharpa-backend",
            "sdk_direct",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "ROS modes require --ros-overlay" in result.stderr
    assert "rclpy" not in result.stderr


def test_real_loopback_websocket_with_contract_model_sink(package, monkeypatch, tmp_path):
    import socket

    from deploy.fr3_sharpa_runtime import config
    from deploy.fr3_sharpa_runtime.model_profiles import get_model_profile
    from deploy.fr3_sharpa_runtime.runtime import SharpaRuntime

    root, repo = package
    monkeypatch.setattr(config, "MODEL_PROFILE", get_model_profile("flash-absolute"))
    monkeypatch.setattr(config, "CONFIG_NAME", fc.CONFIG_NAME)
    monkeypatch.setattr(config, "CHECKPOINT", root)
    monkeypatch.setattr(config, "NORM_STATS", root / "assets/flash-test/norm_stats.json")
    monkeypatch.setattr(config, "_validation_cache", None)
    metadata = {**fc.inspect_flash(root), "warmed_up": True}
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    script = """
import json, sys, numpy as np
from deploy.fr3_sharpa_flash.serve import serve
from deploy.fr3_sharpa_runtime.flash_contract import validate_flash_observation
metadata=json.load(open(sys.argv[1]))
class Sink:
    def infer(self, observation):
        clean=validate_flash_observation(observation)
        ctx=clean['flash_context']
        receipt={key:ctx[key] for key in ('protocol_version','request_id','session_id','observation_anchor_source_ns')}
        receipt['model_identity']=metadata['flash_manifest_sha256']
        return {'actions':np.full((50,58),.125,np.float32),'flash_receipt':receipt}
serve(Sink(), metadata, int(sys.argv[2]))
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(path), str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        client = SharpaRuntime(f"ws://127.0.0.1:{port}")
        deadline = time.monotonic() + 5
        while True:
            try:
                client.inspect_metadata()
                break
            except OSError:
                if child.poll() is not None or time.monotonic() > deadline:
                    raise
                time.sleep(0.02)
        _, actions, _ = client.infer_chunk(warmup_observation())
        np.testing.assert_array_equal(actions, np.full((50, 58), 0.125, np.float32))
    finally:
        child.terminate()
        child.communicate(timeout=5)


@pytest.fixture
def native_step(package):
    import shutil

    root, repo = package
    (root / "flash_deployment.json").unlink()
    shutil.rmtree(root / "flash_assets")
    return root, repo


@pytest.mark.parametrize("original_config", [True, False])
def test_native_step_and_manual_plan_need_no_generated_copies(native_step, original_config, monkeypatch):
    from deploy.fr3_sharpa_control.plan import DeploymentSettings
    from deploy.fr3_sharpa_control.plan import plan_environment
    from deploy.fr3_sharpa_runtime import config
    from deploy.fr3_sharpa_runtime.model_profiles import get_model_profile

    root, repo = native_step
    if not original_config:
        (root.parent / "flash_training_contract.json").unlink()
        # Manual selection also works when only native parameters/stats were downloaded.
        for name in ("flash_manifest.json", "flash_seal.json", "flash_validation.json"):
            (root / "assets/flash-test" / name).unlink()
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    settings = DeploymentSettings(str(Path(__file__).resolve().parents[3]), str(root),
                                  model_type="flash-absolute", mode="serial", steps=15, rounds=0)
    plan = settings.plan()
    assert plan["model"]["supported_modes"] == ["serial"]
    assert plan["settings"]["rounds"] == 0
    assert plan_environment(plan)["SHARPA_FLASH_MAX_PLAN_AGE_MS"] == "0"
    assert fc.flash_settings(root)["model_config"]["action_q01"] == [-4.0] * 58
    monkeypatch.setattr(config, "MODEL_PROFILE", get_model_profile("flash-absolute"))
    monkeypatch.setattr(config, "CONFIG_NAME", fc.CONFIG_NAME)
    monkeypatch.setattr(config, "CHECKPOINT", root)
    monkeypatch.setattr(config, "NORM_STATS", root / "assets/flash-test/norm_stats.json")
    monkeypatch.setattr(config, "_validation_cache", None)
    monkeypatch.delenv("SHARPA_FLASH_REPO", raising=False)
    expected = config.prepare_policy_validation()
    # Server knows the backend path; client does not need another contract copy.
    server = fc.inspect_flash(root, backend=repo)
    server["warmed_up"] = True
    config.validate_policy_metadata(server)
    assert expected["flash_manifest_sha256"] == server["flash_manifest_sha256"]
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    # Replacements still invalidate the lightweight client identity cache.
    (root / "params/manifest.ocdbt").write_bytes(b"REPLACED PARAMETERS INDEX")
    with pytest.raises(ValueError, match="does not match"):
        config.validate_policy_metadata(server)


def test_native_check_only_does_not_import_gpu_or_write_assets(native_step):
    root, repo = native_step
    code = (
        "import sys; from deploy.fr3_sharpa_flash.serve import main; main(); "
        "assert 'jax' not in sys.modules; assert 'openpi' not in sys.modules"
    )
    env = {**os.environ, "SHARPA_FLASH_CHECKPOINT": str(root), "SHARPA_FLASH_REPO": str(repo)}
    result = subprocess.run([sys.executable, "-c", code, "--check-only"],
                            env=env, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["algorithm"] == "flash"
    assert not (root / "flash_deployment.json").exists()
    assert not (root / "flash_assets").exists()


def test_native_metadata_only_download_is_not_a_complete_checkpoint(native_step):
    root, _ = native_step
    (root / "params/d/blob").unlink()
    with pytest.raises(ValueError, match="full parameter data"):
        fc.inspect_flash(root)


def test_original_training_quantiles_must_match_selected_checkpoint(native_step):
    root, _ = native_step
    path = root.parent / "flash_training_contract.json"
    training = json.loads(path.read_text())
    training["model"] = training["model"].replace("-4.0", "-5.0")
    path.write_text(json.dumps(training))
    with pytest.raises(ValueError, match="quantiles"):
        fc.inspect_flash(root)


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "garbage"])
def test_optional_age_budget_rejects_invalid_values(value, monkeypatch):
    monkeypatch.setenv("SHARPA_FLASH_MAX_PLAN_AGE_MS", value)
    with pytest.raises(ValueError, match="nonnegative|could not convert"):
        fc.execution_budget()


def test_gui_flash_settings_override_old_environment_file(native_step, tmp_path):
    root, _ = native_step
    env_file = tmp_path / "flash.env"
    env_file.write_text("export SHARPA_FLASH_CHECKPOINT=/stale/step\n"
                        "export SHARPA_FLASH_NORM_STATS=/stale/stats.json\n"
                        "export SHARPA_FLASH_MAX_PLAN_AGE_MS=999\n")
    stats = root / "assets/flash-test/norm_stats.json"
    env = {**os.environ, "SHARPA_FLASH_ENV_FILE": str(env_file), "SHARPA_FLASH_CHECKPOINT": str(root),
           "SHARPA_FLASH_NORM_STATS": str(stats), "SHARPA_FLASH_MAX_PLAN_AGE_MS": "0"}
    result = subprocess.run(["bash", "-c", 'source deploy/fr3_sharpa_runtime/model_env.sh --model flash-absolute; '
                             'test "$SHARPA_FLASH_CHECKPOINT" = "$1" && '
                             'test "$SHARPA_FLASH_NORM_STATS" = "$2" && '
                             'test "$SHARPA_FLASH_MAX_PLAN_AGE_MS" = 0', "--", str(root), str(stats)],
                            env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
