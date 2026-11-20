import threading

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import pytest

from deploy.fr3_sharpa_runtime.fovea_policy import FoveaPolicy
from deploy.fr3_sharpa_runtime.fovea_runtime import FoveaRuntime
from openpi.shared.normalize import NormStats
from openpi.subtasks import fovea


class JointProbe(nnx.Module):
    def sample_joint(self, rng, observation):
        return {
            "actions": jnp.zeros((1, 50, 58)),
            "foveation_tokens": jnp.array(fovea.encode_box((0.4, 0.3, 0.7, 0.7), 640, 400))[None],
        }


def policy():
    stats = {
        k: NormStats(mean=np.zeros(58), std=np.ones(58), q01=-np.ones(58), q99=np.ones(58))
        for k in ["state", "actions"]
    }
    return FoveaPolicy(
        JointProbe(),
        stats,
        {
            "foveation": {"version": fovea.VERSION, "execute_steps": 15, "initial_box": list(fovea.INITIAL_BOX)},
            "warmed_up": True,
        },
    )


def observation():
    return {
        "observation/state": np.ones(58, np.float32) * 0.2,
        "observation/image": np.zeros((400, 640, 3), np.uint8),
        "observation/left_wrist_image": np.zeros((480, 640, 3), np.uint8),
        "observation/right_wrist_image": np.zeros((480, 640, 3), np.uint8),
        "prompt": "Right hand grasps and lifts the truss to the center",
    }


def context(runtime):
    return {
        "session_id": runtime.session,
        "query_index": runtime.query_index,
        "executed_steps": 0 if runtime.query_index == 0 else 15,
        "current_box": runtime.box.tolist(),
    }


def test_fixed_initial_predicted_next_crop_and_execution_ack():
    server = policy()
    client = FoveaRuntime("ws://127.0.0.1:8010")
    obs = observation()
    first = server.infer({**obs, "_fovea": context(client)})
    client.accept_result(server.metadata, first)
    np.testing.assert_array_equal(client.box, fovea.INITIAL_BOX)
    assert client.query_index == 0
    # Read-only retries don't advance the crop or the temporal query index.
    again = server.infer({**obs, "_fovea": context(client)})
    client.accept_result(server.metadata, again)
    with pytest.raises(ValueError, match="fifteen-step"):
        client.acknowledge_execution(7)
    client.acknowledge_execution(15)
    assert client.query_index == 1
    np.testing.assert_array_equal(client.box, first["foveation"]["next_box"])
    second = server.infer({**obs, "_fovea": context(client)})
    np.testing.assert_array_equal(second["foveation"]["used_box"], client.box)
    assert first["actions"].shape == (50, 58)
    np.testing.assert_allclose(first["actions"][0, :7], 0.2, atol=1e-6)
    np.testing.assert_allclose(first["actions"][0, 7:29], 0, atol=1e-6)
    np.testing.assert_allclose(first["actions"][0, 29:36], 0.2, atol=1e-6)
    with pytest.raises(ValueError, match="execution"):
        client.acknowledge_execution(15)


def test_reject_wrong_offset_and_stale_receipt():
    server = policy()
    client = FoveaRuntime("ws://127.0.0.1:8010")
    with pytest.raises(ValueError, match="exactly15"):
        FoveaRuntime("ws://localhost", execute_steps=8)
    ctx = {**context(client), "executed_steps": 15}
    with pytest.raises(ValueError, match="execution count"):
        server.infer({**observation(), "_fovea": ctx})
    result = server.infer({**observation(), "_fovea": context(client)})
    result["foveation"]["execute_steps"] = 8
    with pytest.raises(ValueError, match="receipt"):
        client.accept_result(server.metadata, result)
    result["foveation"]["execute_steps"] = 15
    result["foveation"]["session_id"] = "old"
    with pytest.raises(ValueError, match="receipt"):
        client.accept_result(server.metadata, result)


def test_websocket_two_queries_with_real_crop_transforms():
    import asyncio
    import socket

    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    server_policy = policy()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    loop = asyncio.new_event_loop()
    started = threading.Event()

    async def run():
        import websockets.asyncio.server as ws

        server = WebsocketPolicyServer(server_policy, host="127.0.0.1", port=port, metadata=server_policy.metadata)
        async with ws.serve(server._handler, "127.0.0.1", port, compression=None, max_size=None):  # noqa: SLF001
            started.set()
            await asyncio.Event().wait()

    def worker():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    future = asyncio.run_coroutine_threadsafe(run(), loop)
    try:
        assert started.wait(10)
        client = FoveaRuntime(f"ws://127.0.0.1:{port}")
        first = client.infer_chunk(observation())[1]
        client.acknowledge_execution(15)
        second = client.infer_chunk(observation())[1]
        assert first.shape == second.shape == (50, 58)
        assert client.query_index == 1
    finally:
        future.cancel()
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0.05), loop).result(timeout=2)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)


def test_fovea_handshake_rejects_wrong_checkpoint_even_with_matching_version():
    client = FoveaRuntime('ws://localhost:8010')
    client.expected_identity = 'selected-checkpoint'
    metadata = {**policy().metadata, 'fovea_identity': 'different-checkpoint'}
    with pytest.raises(ValueError, match='身份'):
        client.validate_metadata(metadata)
    metadata['fovea_identity'] = 'selected-checkpoint'
    assert client.validate_metadata(metadata) == metadata


def test_recording_returns_exact_model_crop_only_when_requested():
    from deploy.fr3_sharpa_runtime.four_view_recording import pixel_sha
    server = policy()
    client = FoveaRuntime('ws://localhost:8010')
    obs = observation()
    obs['observation/image'][:] = np.arange(640, dtype=np.uint8)[None, :, None]
    result = server.infer({**obs, '_fovea': {**context(client), 'record_roi': True, 'source_monotonic_ns': 123}})
    payload = result['recording_roi']
    np.testing.assert_array_equal(payload['pixels'], fovea.crop_rgb(obs['observation/image'], client.box))
    assert payload['metadata']['roi_pixel_sha256'] == pixel_sha(payload['pixels'])
    assert payload['metadata']['source_monotonic_ns'] == 123
    assert 'recording_roi' not in server.infer({**obs, '_fovea': context(client)})
