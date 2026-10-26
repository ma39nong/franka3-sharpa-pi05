"""Hardware-free tests of phase progress, strict planning, and stateless requests."""
# ruff: noqa: PLC0415
# Lazy imports keep pure helpers hardware/framework-free and test patches local.

import copy
import json
from pathlib import Path
import time

import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.action_adapter import SharpaActionAdapter
from deploy.fr3_sharpa_runtime.compression import compression_candidate
from deploy.fr3_sharpa_runtime.compression import plan_execution
from deploy.fr3_sharpa_runtime.correlation import correction_matrix
from deploy.fr3_sharpa_runtime.execution_trace import ExecutionTraceWriter
from deploy.fr3_sharpa_runtime.model_service import ExperimentalPolicy
from deploy.fr3_sharpa_runtime.model_service import encode_reference
from deploy.fr3_sharpa_runtime.policy_prepare import prepare_and_run_chunks
from deploy.fr3_sharpa_runtime.rolling import CAPABILITY
from deploy.fr3_sharpa_runtime.rolling import CONTEXT_KEY
from deploy.fr3_sharpa_runtime.rolling import RollingOptions
from deploy.fr3_sharpa_runtime.rolling import RollingReference
from deploy.fr3_sharpa_runtime.rolling import model_identity
from deploy.fr3_sharpa_runtime.rolling import prepare_context


def metadata():
    return {
        "checkpoint": "synthetic",
        "norm_stats": "synthetic",
        "model_profile": "full-delta",
        "checkpoint_manifest_sha256": "checkpoint",
        "normalization_sha256": "stats",
        "server_instance_id": "boot1",
        "inference_capabilities": [CAPABILITY],
    }


def actions():
    return np.broadcast_to(np.linspace(0, 0.02, 50)[:, None], (50, 58)).astype(np.float32).copy()


class Clock:
    def __init__(self):
        self.now = 10_000_000_000

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += int(seconds * 1e9)


def committed(clock=None, options=None):
    clock = clock or Clock()
    cache = RollingReference(options or RollingOptions(inpainting="soft"), clock_ns=clock)
    cache.commit(
        actions(),
        metadata(),
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
    return cache, clock


def test_rolling_keeps_four_future_targets_and_does_not_accumulate():
    cache, clock = committed()
    for index in range(1, 5):
        context = prepare_context(cache.context(index), metadata(), now_ns=clock())
        reference = context["reference"]
        np.testing.assert_array_equal(reference["absolute_actions"], actions()[26:30])
        assert reference["chunk_index"] == index - 1
        cache.commit(
            actions(),
            metadata(),
            {
                "status": "PASS",
                "policy_frames": 26,
                "completion": "commands_sent",
                "execution_target_count": 26,
                "frames": 26,
                "completed_monotonic_ns": clock(),
            },
            index,
        )


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"server_instance_id": "boot2"}, "server_restarted"),
        ({"normalization_sha256": "other"}, "model_identity_changed"),
        ({"checkpoint": "other"}, "model_identity_changed"),
    ],
)
def test_new_connections_survive_but_server_and_model_changes_clear(change, reason):
    cache, clock = committed()
    context = cache.context(1)
    assert prepare_context(context, dict(metadata()), now_ns=clock())["reference"] is not None
    updated = {**metadata(), **change}
    cleared = prepare_context(context, updated, now_ns=clock())
    assert cleared["reference"] is None
    assert cleared["invalidated_reason"] == reason


def test_expiry_progress_and_partial_execution():
    cache, clock = committed()
    wrong = cache.context(2)
    assert prepare_context(wrong, metadata(), now_ns=clock())["reference"] is None
    clock.advance(1.01)
    assert cache.context(1)["invalidated_reason"] == "reference_expired"
    with pytest.raises(RuntimeError, match="26 policy frames"):
        cache.commit(actions(), metadata(), {"status": "PASS", "policy_frames": 15}, 1)
    assert cache.reference is None
    with pytest.raises(ValueError, match="does not support"):
        prepare_context(cache.context(0), {}, now_ns=clock())
    cache.clear("home")
    assert cache.context(0)["reference"] is None


@pytest.mark.parametrize("delta", [True, False])
def test_reference_encoding_rebases_only_arms_and_normalizes_without_mutating(delta):
    state = np.linspace(-1, 1, 58, dtype=np.float32)
    absolute = actions()[:4]
    original = absolute.copy()

    def normalizer(data):
        return {"actions": (data["actions"] - 0.2) / 0.5}

    encoded = encode_reference(absolute, state, delta=delta, normalizer=normalizer)
    restored = encoded * 0.5 + 0.2
    if delta:
        restored[:, np.r_[0:7, 29:36]] += state[np.r_[0:7, 29:36]]
    np.testing.assert_allclose(restored, absolute, atol=1e-7)
    np.testing.assert_array_equal(absolute, original)


def test_identity_covariance_zero_propagation_and_bad_assets():
    assert not correction_matrix(np.eye(10), 4).any()
    with pytest.raises(ValueError, match="symmetric"):
        correction_matrix(np.array([[1, 2], [0, 1]]), 1)
    with pytest.raises(ValueError, match="semidefinite"):
        correction_matrix(np.array([[1, 2], [2, 1]]), 1)
    correlated = np.eye(10) + np.ones((10, 10)) * 0.2
    assert np.all(correction_matrix(correlated, 4) > 0)


def test_service_converts_references_and_never_stores_other_runs():
    cache, _clock = committed()
    context = cache.context(1)
    # Real server validates monotonic timestamps; inject current completion here.
    context["reference"] = {**context["reference"], "completed_monotonic_ns": time.monotonic_ns()}
    calls = []

    class Policy:
        received_ns = None

        def infer(self, observation, **kwargs):
            assert CONTEXT_KEY not in observation
            calls.append(kwargs)
            return {"actions": actions()}

    service = ExperimentalPolicy(Policy(), metadata(), normalizer=lambda x: x, delta=True)
    obs = {"observation/state": np.ones(58), CONTEXT_KEY: context}
    result = service.infer(obs)
    assert result["inference_experiment"]["reference_applied"]
    np.testing.assert_allclose(calls[-1]["initial_actions"][:, 0], actions()[26:30, 0] - 1)
    service.infer({"observation/state": np.ones(58)})
    assert calls[-1] == {}
    bad = copy.deepcopy(context)
    bad["options"]["inpainting"] = "correlation"
    with pytest.raises(ValueError, match="covariance asset"):
        service.infer({**obs, CONTEXT_KEY: bad})


def test_compression_preserves_endpoints_and_hand_motion_disables_whole_segment():
    options = RollingOptions(compression="cubic")
    candidate, report = compression_candidate(actions(), np.zeros(58), options)
    assert candidate.shape == (20, 58)
    assert report["applied"]
    np.testing.assert_array_equal(candidate[[0, -1]], actions()[[0, 25]])
    moving = actions()
    moving[12, 7] = 0.06
    candidate, report = compression_candidate(moving, np.zeros(58), options)
    assert candidate.shape == (26, 58)
    assert report["fallback_reason"] == "hand_variation"
    np.testing.assert_array_equal(candidate, moving[:26])


def test_strict_cubic_validation_falls_back_once_without_clipping_overshoot(monkeypatch):
    from deploy.fr3_sharpa_runtime import sdk_direct

    monkeypatch.setattr(sdk_direct, "hand_joint_limits", lambda: np.tile([-1.0, 1.0], (22, 1)))
    candidate, report = compression_candidate(actions(), np.zeros(58), RollingOptions(compression="cubic"))
    candidate[10, 7] = 1.01

    def ranges(values, **kwargs):
        if np.max(np.abs(values)) > 1:
            raise ValueError("synthetic native limits")

    frames, bridge, targets, audit, result = plan_execution(
        actions(), candidate, report, np.zeros(58), speed_scale=0.1, approach_speed_rad_s=0.03, range_validator=ranges
    )
    assert not result["applied"]
    assert len(targets) == 26
    assert len(frames) == bridge + 251
    assert result["fallback_reason"] == "validation_rejected"
    assert audit["corrected_values"] == 0

    def reject(values, **kwargs):
        raise ValueError("both plans invalid")

    with pytest.raises(ValueError, match="both plans invalid"):
        plan_execution(
            actions(), candidate, report, np.zeros(58), speed_scale=1, approach_speed_rad_s=0.03, range_validator=reject
        )


def test_three_rounds_commit_after_execution_preserve_predictions_and_trace(monkeypatch, tmp_path):
    from deploy.fr3_sharpa_runtime import sdk_direct

    monkeypatch.setattr(sdk_direct, "hand_joint_limits", lambda: np.tile([-1.0, 1.0], (22, 1)))
    clock = Clock()
    options = RollingOptions(inpainting="soft", compression="cubic")
    contexts, reports = [], []
    state = np.zeros(58, dtype=np.float32)
    traces = ExecutionTraceWriter(tmp_path)

    def observe(completed=None):
        return {"observation/state": state.copy(), "_captured_monotonic_ns": clock()}

    def infer(obs):
        context = obs[CONTEXT_KEY]
        contexts.append(context)
        result = actions() + state
        chunk = SharpaActionAdapter().adapt_chunk(
            result, start_timestamp_ns=clock(), session_id="test", source="openpi"
        )
        return metadata(), result, chunk

    class Transport:
        def request(self, request):
            assert request["execute_steps"] == 26
            assert len(request["actions"]) == 50
            frames, bridge, targets, _, compression = plan_execution(
                request["actions"],
                request["execution_targets"],
                request["compression"],
                state,
                speed_scale=0.1,
                approach_speed_rad_s=0.03,
                range_validator=lambda a, **kw: None,
            )
            clock.advance(len(frames) / 30 + 0.02)
            state[:] = targets[-1]
            report = {
                "status": "PASS",
                "published": True,
                "policy_frames": 26,
                "completion": "commands_sent",
                "execution_target_count": len(targets),
                "frames": len(frames),
                "bridge_frames": bridge,
                "completed_monotonic_ns": clock(),
                "compression": compression,
            }
            reports.append(report)
            return report

    report = prepare_and_run_chunks(
        infer,
        observe,
        Transport(),
        observe(),
        prepare_initial=False,
        num_chunks=3,
        execute_steps=26,
        execute_speed_scale=0.1,
        rolling_options=options,
        clock_ns=clock,
        trace_writer=traces,
    )
    assert report["num_chunks"] == 3
    assert contexts[0]["reference"] is None
    assert [c["reference"]["chunk_index"] for c in contexts[1:]] == [0, 1]
    assert all(r["execution_target_count"] == 20 and r["frames"] >= 191 for r in reports)
    records = [json.loads(Path(p).read_text()) for p in traces.paths]
    assert all(len(r["policy_output_unnormalized"]) == 50 for r in records)
    assert all(len(r["inference_experiment"]["execution_targets"]) == 20 for r in records)


def test_failed_execution_never_reuses_overlap():
    def infer(obs):
        return (
            metadata(),
            actions(),
            SharpaActionAdapter().adapt_chunk(actions(), start_timestamp_ns=1, session_id="test", source="openpi"),
        )

    class Fail:
        def request(self, request):
            return {"status": "FAIL"}

    with pytest.raises(RuntimeError, match="Chunk failed"):
        prepare_and_run_chunks(
            infer,
            lambda r: {},
            Fail(),
            {"observation/state": np.zeros(58)},
            prepare_initial=False,
            execute_steps=26,
            rolling_options=RollingOptions(),
        )


def test_covariance_asset_full_horizon_identity_and_identity_mismatch(tmp_path):
    from deploy.fr3_sharpa_runtime.correlation import ASSET_VERSION
    from deploy.fr3_sharpa_runtime.correlation import load_correction

    meta = {
        **metadata(),
        "action_order": ["left_arm_7", "left_hand_22", "right_arm_7", "right_hand_22"],
        "training_action_supervision": "delta_arm_joint_target_absolute_hand",
    }
    asset_meta = {
        "version": ASSET_VERSION,
        "model_identity": model_identity(meta),
        "shape": [50, 58],
        "flatten_order": "time-major",
        "space": "normalized_training_actions",
        "action_order": meta["action_order"],
        "training_action_supervision": meta["training_action_supervision"],
        "action_hz": 30,
        "sample_count": 10,
    }
    path = tmp_path / "synthetic-identity.npz"
    np.savez_compressed(path, covariance=np.eye(2900, dtype=np.float32), metadata=json.dumps(asset_meta))
    matrix, info = load_correction(path, meta)
    assert matrix.shape == (2668, 232)
    assert not matrix.any()
    assert info["noise"] == "independent"
    assert info["beta"] == 0.5
    with pytest.raises(ValueError, match="identity/space"):
        load_correction(path, {**meta, "normalization_sha256": "different"})


def test_incomplete_success_report_cannot_commit_reference():
    cache, clock = committed()
    with pytest.raises(RuntimeError, match="26 policy frames"):
        cache.commit(
            actions(),
            metadata(),
            {
                "status": "PASS",
                "policy_frames": 26,
                "completed_monotonic_ns": clock(),
                "completion": "commands_sent",
                "frames": 19,
                "execution_target_count": 20,
            },
            1,
        )
    assert cache.reference is None


def test_worker_rechecks_hand_variation_without_importing_scipy(monkeypatch):
    import sys

    from deploy.fr3_sharpa_runtime.compression import recheck_compression

    candidate, report = compression_candidate(actions(), np.zeros(58), RollingOptions(compression="cubic"))
    monkeypatch.setitem(sys.modules, "scipy", None)
    latest = np.zeros(58)
    latest[7] = 0.1
    selected, updated = recheck_compression(actions(), candidate, report, latest)
    assert not updated["applied"]
    assert updated["fallback_reason"] == "fresh_hand_variation"
    np.testing.assert_array_equal(selected, actions()[:26])
