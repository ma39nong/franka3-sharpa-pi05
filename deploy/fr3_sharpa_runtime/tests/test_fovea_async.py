"""No hardware: observation clock, guided transforms and real streaming timeline."""
import threading
import time
from types import SimpleNamespace

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
from openpi_client import rtc
import pytest

from deploy.fr3_sharpa_runtime import fovea_async as asynchronous
from deploy.fr3_sharpa_runtime.fovea_async import FoveaAsyncRuntime
from deploy.fr3_sharpa_runtime.fovea_async import FoveaAsyncTask
from deploy.fr3_sharpa_runtime.fovea_async import source_time
from deploy.fr3_sharpa_runtime.fovea_policy import FoveaPolicy
from deploy.fr3_sharpa_runtime.rtc_stream import RTCTimeline
from deploy.fr3_sharpa_runtime.task_control import TaskControl
from deploy.fr3_sharpa_runtime.tests.test_fovea import observation
from deploy.fr3_sharpa_runtime.tests.test_fovea import policy
from deploy.fr3_sharpa_runtime.tests.test_tomato_task import FakeBackend
from deploy.fr3_sharpa_runtime.tests.test_tomato_task import command
from openpi.subtasks import fovea


def obs(now=None, age=0):
    now = time.monotonic() if now is None else now
    return {**observation(), '_captured_monotonic_ns': int(now * 1e9),
            '_alignment': {'source_age_ms': {'cam0': age}},
            '_task_record': {'run_id': 'test', 'stage_version': 0}}


def test_source_clock_uses_camera_stamp_not_encoding_completion():
    assert source_time(obs(12, age=80)) == pytest.approx(11.92)
    for age in (-1, 101, float('nan')):
        with pytest.raises(ValueError, match='camera age'):
            source_time(obs(12, age=age))
    with pytest.raises(ValueError):
        source_time({'_captured_monotonic_ns': 0})


def test_install_skip_includes_observation_transport_time():
    actions = np.repeat(np.arange(50, dtype=np.float32)[:, None], 58, axis=1)
    timeline = RTCTimeline(actions, 10, 1)
    context = timeline.snapshot(10.55, observed_at=10.5, delay_steps=5, schedule='exp', weight=5)
    np.testing.assert_allclose(context['previous_actions'][0], 15)
    installed = timeline.install(actions, rtc.make_receipt(context), 10.7)
    assert installed['skipped'] == 6  # 200ms total, not the 150ms model/IPC interval.
    assert timeline.sample(10.7)[0] == 6
    with pytest.raises(ValueError, match='recent and causal'):
        timeline.snapshot(11, observed_at=10.7, delay_steps=5, schedule='exp', weight=5)


def test_crop_commits_only_after_install_and_never_adds_install_delay(monkeypatch):
    runtime = FoveaAsyncRuntime('ws://localhost')
    monkeypatch.setattr(asynchronous, 'time', SimpleNamespace(monotonic=lambda: 10.2))
    runtime.prepare_observation(obs(10))
    runtime.pending = np.array([.3, .3, .7, .8])
    target = runtime.commit_installation({'sampled_monotonic_s': 10.2, 'phase': 0}, initial_start=True)
    assert target == pytest.approx(10.7)  # startup waited; execution starts at10.2.
    runtime.prepare_observation(obs(10.7))
    assert runtime.query_index == 1
    np.testing.assert_allclose(runtime.box, [.3, .3, .7, .8])
    runtime.pending = np.array([.4, .3, .8, .8])
    monkeypatch.setattr(asynchronous, 'time', SimpleNamespace(monotonic=lambda: 10.9))
    assert runtime.commit_installation({'skipped': 6}, initial_start=False) == pytest.approx(11.2)
    with pytest.raises(TimeoutError, match='target image'):
        runtime.prepare_observation(obs(11.4))
    new = obs(11.4)
    new['_task_record']['stage_version'] = 1
    runtime.prepare_observation(new)
    assert runtime.query_index == 0 and runtime.next_crop is None
    np.testing.assert_allclose(runtime.box, fovea.INITIAL_BOX)


def test_late_prediction_cannot_commit_box(monkeypatch):
    runtime = FoveaAsyncRuntime('ws://localhost')
    runtime.prepare_observation(obs(10))
    runtime.pending = np.array(fovea.INITIAL_BOX)
    monkeypatch.setattr(asynchronous, 'time', SimpleNamespace(monotonic=lambda: 10.49))
    with pytest.raises(TimeoutError, match='too late'):
        runtime.commit_installation({}, initial_start=False)
    assert runtime.next_crop is None


class GuidedProbe(nnx.Module):
    def sample_joint(self, rng, observation, *, rtc_actions=None, rtc_weights=None, rtc_max_guidance_weight=None):
        return {'actions': jnp.zeros((1, 50, 58)) if rtc_actions is None else rtc_actions,
                'foveation_tokens': jnp.asarray(fovea.encode_box((.3, .3, .7, .8), 640, 400))[None]}


def test_guided_prefix_roundtrip_reanchors_arms_and_preserves_absolute_hands():
    from openpi.shared.normalize import NormStats
    stats = {k: NormStats(mean=np.zeros(58), std=np.ones(58), q01=-np.ones(58), q99=np.ones(58))
             for k in ('state', 'actions')}
    server = FoveaPolicy(GuidedProbe(), stats, {}, enable_async=True)
    runtime = FoveaAsyncRuntime('ws://localhost')
    inputs = {**observation(), '_fovea': {'session_id': runtime.session, 'query_index': 0,
                'execution_mode': asynchronous.PROTOCOL, 'nominal_observation_steps': 15,
                'source_monotonic_s': 0., 'used_target_monotonic_s': None, 'current_box': list(fovea.INITIAL_BOX)}}
    actions = np.full((50, 58), .4, np.float32)
    timeline = RTCTimeline(actions, 0, 1)
    context = timeline.snapshot(.5, delay_steps=5, schedule='exp', weight=5)
    result = server.infer(inputs, rtc_context=context)
    rtc.validate_receipt(result['rtc'], context)
    np.testing.assert_allclose(result['actions'][:len(context['previous_actions'])], .4, atol=1e-5)
    np.testing.assert_allclose(actions, .4)  # delta transform must not mutate the stream.
    with pytest.raises(ValueError, match='observation-clock'):
        policy().infer(inputs, rtc_context=context)


class SimulatedRuntime(FoveaAsyncRuntime):
    def __init__(self):
        super().__init__('ws://localhost')
        self.calls = []
        self.block = None
        self.entered = threading.Event()

    def infer_chunk(self, observation, *, rtc_context=None, **kwargs):
        self.prepare_observation(observation)
        self.calls.append((self.prediction_source, self.query_index, self.session, rtc_context))
        self.entered.set()
        if self.block is not None:
            assert self.block.wait(2)
        else:
            time.sleep(.04)
        self.pending = np.array([.3, .3, .7, .8])
        self.last_receipt = rtc.make_receipt(rtc_context) if rtc_context else None
        return {'checkpoint': 'simulation', 'foveation_query': {'query_index': self.query_index}}, np.zeros((50, 58), np.float32), None


class Backend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.worker = self

    def start(self, actions, ticket, path, *, on_poll=None, admission=None):
        with admission():
            return super().start(actions, ticket, path, on_poll=on_poll)

    def request(self, payload, **kwargs):
        assert payload['operation'] == 'stream_snapshot'
        return {'context': self.plan.snapshot(time.monotonic(), observed_at=payload['observed_at'],
            delay_steps=payload['delay_steps'], schedule=payload['schedule'], weight=payload['weight'])}


@pytest.fixture
def engine(tmp_path):
    control = TaskControl(profile='tomato-in-hand119', initial_stage='pickup')
    runtime, backend = SimulatedRuntime(), Backend()
    args = SimpleNamespace(execute_steps=15, execute_speed_scale=1., num_chunks=0,
        rtc_initial_delay_steps=5, rtc_delay_buffer_size=10, rtc_prefix_attention_schedule='exp',
        rtc_max_guidance_weight=5.)
    def observe(**kwargs):
        if kwargs.get('on_poll'):
            kwargs['on_poll']()
        return obs()
    coordinator = FoveaAsyncTask(control, runtime, backend, observe, args, tmp_path)
    control.ready()
    yield coordinator, control, runtime, backend
    if runtime.block:
        runtime.block.set()
    coordinator.pool.shutdown(wait=True, cancel_futures=True)
    coordinator.observation_pool.shutdown(wait=True, cancel_futures=True)


def until(engine, condition, timeout=3):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline
        engine.tick()
        time.sleep(.002)


def test_stream_continues_during_inference_and_crop_queries_do_not_drift(engine):
    e, c, runtime, backend = engine
    command(c, 'start')
    until(e, lambda: e.chunk_count == 1)
    runtime.block = threading.Event()
    until(e, lambda: len(runtime.calls) == 2)
    before = backend.status()['phase']
    time.sleep(.08)
    e.tick()
    assert backend.status()['phase'] > before + 1
    assert e.chunk_count == 1 and e.future is not None
    runtime.block.set()
    runtime.block = None
    until(e, lambda: e.chunk_count == 3)
    sources = [row[0] for row in runtime.calls]
    assert sources[2] - sources[1] == pytest.approx(.5, abs=.065)
    assert [row[1] for row in runtime.calls[:3]] == [0, 1, 2]
    assert backend.plan.generation == 3
    assert runtime.calls[1][3] is not None


def test_stop_during_prediction_never_installs_or_commits(engine):
    e, c, runtime, backend = engine
    runtime.block = threading.Event()
    command(c, 'start')
    e.tick()
    assert runtime.entered.wait(1)
    command(c, 'stop')
    assert e.tick() is False
    runtime.block.set()
    assert e.future.result(timeout=1)
    assert e.tick() is False
    assert not e.streaming and runtime.next_crop is None
    assert not any(isinstance(call, tuple) and call[0] == 'install' for call in backend.calls)


def test_switch_bridge_discards_prebridge_box_and_reacquires(engine):
    e, c, runtime, backend = engine
    command(c, 'start')
    until(e, lambda: e.chunk_count == 1)
    command(c, 'next')
    until(e, lambda: c.snapshot()['active_stage'] == 'harvest_1')
    assert runtime.query_index == 0 and runtime.next_crop is None
    assert e.crop_target is not None
    until(e, lambda: e.chunk_count == 3)
    assert runtime.calls[-1][1] == 0
    assert runtime.calls[-1][3] is not None
    assert c.snapshot()['execution_state'] == 'RUNNING'


def test_real_joint_sampler_guides_actions_without_changing_box_branch(monkeypatch):
    import jax

    from openpi.models import pi0_fovea
    monkeypatch.setattr(pi0_fovea.base, 'preprocess_observation', lambda _, observation, **kw: observation)
    class LLM:
        def __call__(self, inputs, *, method=None, **kwargs):
            if method == 'embed':
                return jnp.zeros((*inputs.shape, 2))
            if method == 'decode_locations':
                return jnp.zeros((*inputs.shape[:-1], 1024))
            prefix, suffix = inputs
            return (prefix, None if suffix is None else suffix * .1), ()
    model = SimpleNamespace(image_keys=(), action_horizon=50, action_dim=58,
        PaliGemma=SimpleNamespace(llm=LLM()), action_out_proj=lambda value: value,
        embed_prefix=lambda obs: (jnp.zeros((1,2,2)), jnp.ones((1,2), bool), jnp.zeros((1,2), bool)),
        embed_suffix=lambda obs, x, t: (x, jnp.ones((1,50), bool), jnp.zeros((1,50), bool), None))
    observation = SimpleNamespace(state=jnp.zeros((1,58)))
    noise = jnp.ones((1,50,58))
    baseline = pi0_fovea.FoveaPi0.sample_joint(model, jax.random.key(0), observation, noise=noise)
    unguided = pi0_fovea.FoveaPi0.sample_joint(model, jax.random.key(0), observation, noise=noise,
        rtc_actions=jnp.zeros_like(noise), rtc_weights=jnp.zeros((1,50,1)))
    guided = pi0_fovea.FoveaPi0.sample_joint(model, jax.random.key(0), observation, noise=noise,
        rtc_actions=jnp.zeros_like(noise), rtc_weights=jnp.ones((1,50,1)))
    np.testing.assert_array_equal(baseline['actions'], unguided['actions'])
    assert np.isfinite(guided['actions']).all()
    assert np.linalg.norm(guided['actions']) < np.linalg.norm(baseline['actions'])
    np.testing.assert_array_equal(guided['foveation_tokens'], baseline['foveation_tokens'])


def test_async_websocket_receipts_and_predicted_crop_roundtrip():
    import asyncio
    import socket

    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.shared.normalize import NormStats
    stats = {k: NormStats(mean=np.zeros(58), std=np.ones(58), q01=-np.ones(58), q99=np.ones(58))
             for k in ('state', 'actions')}
    metadata = {'foveation': {'version': fovea.VERSION, 'execute_steps': 15,
                              'initial_box': list(fovea.INITIAL_BOX)}, 'warmed_up': True,
                'rtc_supported': True, 'rtc_warmed_up': True, 'rtc_protocol_version': rtc.PROTOCOL_VERSION,
                'rtc_action_dim': 58, 'rtc_algorithm': 'soft_prefix_guidance',
                'fovea_async_protocol': asynchronous.PROTOCOL}
    server_policy = FoveaPolicy(GuidedProbe(), stats, metadata, enable_async=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    loop, ready = asyncio.new_event_loop(), threading.Event()
    async def serve():
        import websockets.asyncio.server as ws
        server = WebsocketPolicyServer(server_policy, host='127.0.0.1', port=port, metadata=metadata)
        async with ws.serve(server._handler, '127.0.0.1', port, compression=None, max_size=None):
            ready.set()
            await asyncio.Event().wait()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    job = asyncio.run_coroutine_threadsafe(serve(), loop)
    try:
        assert ready.wait(2)
        client = FoveaAsyncRuntime(f'ws://127.0.0.1:{port}')
        first_meta, first, _ = client.infer_chunk(obs())
        timeline = RTCTimeline(first, time.monotonic(), 1)
        installed = timeline.status(time.monotonic())
        target = client.commit_installation(installed, initial_start=True)
        time.sleep(max(0, target - time.monotonic()))
        image = obs()
        context = timeline.snapshot(time.monotonic(), observed_at=source_time(image),
            delay_steps=5, schedule='exp', weight=5)
        result, second, _ = client.infer_chunk(image, rtc_context=context)
        assert first.shape == second.shape == (50, 58)
        assert result['foveation_query']['query_index'] == 1
        np.testing.assert_allclose(result['foveation_query']['used_box'], [.3, .3, .7, .8], atol=.002)
        assert client.last_receipt['token'] == context['token']
        timeline.install(second, client.last_receipt, time.monotonic())
        assert client.commit_installation({}, initial_start=False) > time.monotonic()
        with pytest.raises(ValueError):
            client.validate_metadata({**metadata, 'fovea_async_protocol': 'old'})
    finally:
        job.cancel()
        asyncio.run_coroutine_threadsafe(asyncio.sleep(.05), loop).result(timeout=2)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)


def test_superseded_background_prediction_does_not_commit_old_stage_box(engine):
    e, c, runtime, backend = engine
    command(c, 'start')
    until(e, lambda: e.chunk_count == 1)
    runtime.block = threading.Event()
    until(e, lambda: len(runtime.calls) == 2)
    old_session = runtime.session
    command(c, 'next')
    e.tick()  # invalidate old stream snapshot while inference is still running.
    runtime.block.set()
    runtime.block = None
    until(e, lambda: c.snapshot()['active_stage'] == 'harvest_1')
    assert runtime.session != old_session
    assert runtime.next_crop is None
    assert e.chunk_count == 2
    assert len([call for call in backend.calls if isinstance(call, tuple) and call[0] == 'install']) == 0


def test_overrun_prediction_hits_crop_deadline_while_old_stream_is_still_live(engine):
    e, c, runtime, backend = engine
    command(c, 'start')
    until(e, lambda: e.chunk_count == 1)
    runtime.block = threading.Event()
    until(e, lambda: len(runtime.calls) == 2)
    time.sleep(max(0, e.first_deadline - time.monotonic() + .01))
    assert backend.status()['phase'] < 50
    with pytest.raises(TimeoutError, match='deadline'):
        e.tick()
    assert e.chunk_count == 1


def test_first_worker_dispatch_rechecks_task_intent_before_start(engine):
    from deploy.fr3_sharpa_runtime.task_rtc import SDKTaskBackend
    e, c, runtime, backend = engine
    command(c, 'start')
    ticket = c.ticket()
    writes = []
    class Worker:
        def request(self, request, *, admission, **kwargs):
            command(c, 'stop')  # race after inference and before stdin.write.
            with admission():
                writes.append(request)
    e.backend = SDKTaskBackend(Worker(), SimpleNamespace(prediction_clip_rad=.05,
        execute_speed_scale=1., prepare_speed_rad_s=.03))
    with pytest.raises(InterruptedError, match='superseded'):
        e.start_prediction(np.zeros((50,58)), ticket, 'never-created.jsonl', lambda: None)
    assert writes == [] and e.backend.started is False


def test_async_context_does_not_claim_fifteen_completed_commands():
    client = FoveaAsyncRuntime('ws://localhost')
    client.prepare_observation(obs())
    context = client.fovea_context()
    assert 'executed_steps' not in context
    assert context['nominal_observation_steps'] == 15
    assert context['execution_mode'] == asynchronous.PROTOCOL
    assert context['source_monotonic_s'] == client.prediction_source


def test_camera_transport_lag_does_not_shorten_each_observation_period(engine):
    e, c, runtime, backend = engine
    e.observe = lambda **kwargs: obs(age=50)
    command(c, 'start')
    until(e, lambda: e.chunk_count == 3)
    assert e.camera_lag == pytest.approx(.05, abs=.001)
    sources = [row[0] for row in runtime.calls]
    assert sources[2] - sources[1] == pytest.approx(.5, abs=.025)
    assert runtime.query_index == 2
