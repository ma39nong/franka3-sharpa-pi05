"""Offline checks: timing equivalence, process isolation and bounded handoff."""
import copy
from pathlib import Path
import queue
from types import SimpleNamespace as S
import time

import numpy as np
import pytest

from experiments.weight_motion_eval import planner as slow
from .runtime.core import Admission, ConsumerGuard, Feedback
from experiments.weight_motion_eval.oneshot.test_continuous import Clock, Devices as SlowDevices
from .continuous import ContinuousConsumer, control_loop
from .deploy import parse_args, launch_commands
from .planner import build_plan
from .player import OneShot
from .prefetch import accept_prefetch
from .profile import ARM_SPEED_RAD_S, planning_config, session_limits


def inputs():
    limits = {"lower": np.full(54, -2.), "upper": np.full(54, 2.), "speed": np.full(54, 2.)}
    return limits, [str(i) for i in range(54)]


def make_player(start=None, *, continuation=False, request="first", final=False):
    start = np.zeros(54) if start is None else start
    config = planning_config("19999")
    config.update(hand_control="slider", execution_steps=20, continuation=continuation, final_chunk=final)
    raw = np.tile(start, (50, 1)) + np.linspace(0.001, 0.015, 50)[:, None]
    limits, names = inputs()
    return OneShot(build_plan(raw, start, config, limits, names),
                   Admission(9.8, 9.85, 9.9, request, "weights", 0), 10.)


class Devices(SlowDevices):
    def __init__(self, clock):
        super().__init__(clock)
        self.guard = ConsumerGuard(np.full(54, -2), np.full(54, 2), hand_control="slider",
                                   arm_speed_rad_s=ARM_SPEED_RAD_S)

    def feedback(self, now):
        return Feedback(self.q.copy(), np.zeros(54), (now,) * 4, (now,) * 4,
                        hand_control="slider", arm_speed_rad_s=ARM_SPEED_RAD_S)


@pytest.mark.parametrize("amplitude", [0.001, 0.02, 0.08])
@pytest.mark.parametrize("steps", [20, 50])
def test_same_arm_path_takes_exactly_quarter_slow_playback(amplitude, steps):
    limits, names = inputs()
    config = slow.read_config(Path(slow.__file__).with_name("config.yaml"))
    config.update(hand_control="slider", execution_steps=steps)
    raw = np.zeros((50, 54))
    raw[:, 0] = amplitude * np.sin(np.linspace(0, 8*np.pi, 50))
    first = slow.build_plan(raw, np.zeros(54), copy.deepcopy(config), copy.deepcopy(limits), names)
    medium = planning_config("19999")
    medium.update(hand_control="slider", execution_steps=steps)
    second = build_plan(raw, np.zeros(54), medium, limits, names)
    assert second.playback.duration == pytest.approx(first.playback.duration / 4)
    np.testing.assert_array_equal(first.knots, second.knots)
    for order in range(4):
        np.testing.assert_allclose(second.playback.sample(second.playback.duration * .37, order),
                                   first.playback.sample(first.playback.duration * .37, order) * 4**order, atol=1e-8)


def test_profile_does_not_patch_or_mutate_slow():
    validator, reader = slow.validate_config, slow.read_config
    original = reader(Path(slow.__file__).with_name("config.yaml"))
    limits, _ = inputs()
    before = limits['speed'].copy()
    for model in ('19999', '30000', '30000v2'):
        medium = planning_config(model)
        assert medium['minimum_time_scale'] == 0.625
        assert medium['arm_raw_initial_delta_rad'] == (.3 if model == '19999' else 1.5)
    assert slow.validate_config is validator and slow.read_config is reader
    assert reader(Path(slow.__file__).with_name("config.yaml")) == original
    bounded = session_limits(limits)
    assert bounded['speed'][0] == 2.
    np.testing.assert_array_equal(limits['speed'], before)
    assert original['minimum_time_scale'] == 2.5


def test_prefetch_overlaps_settle_but_cannot_activate_early():
    clock, incoming, events = Clock(), queue.Queue(), []
    devices = Devices(clock)
    first = make_player()
    second = make_player(first.plan.raw[-1], continuation=True, request='second', final=True)
    def notify(event):
        events.append(dict(event, at=clock()))
        if event['event'] == 'prefetch':
            assert first.state == 'settle_end'
            assert not any(e['event'] == 'round_complete' for e in events)
            second.admission = Admission(clock()-.01, clock(), clock(), 'second', 'weights', 0)
            second.prepared_at = clock()
            incoming.put(second)
    control_loop(devices, first, incoming, notify, lambda *a: None, lambda: False, 2,
                 clock=clock, sleep=clock.sleep)
    assert devices.prepares == devices.finishes == 1
    assert [f.sequence for f in devices.frames] == list(range(len(devices.frames)))
    assert len({f.run_id for f in devices.frames}) == 1
    prefetch = next(e['at'] for e in events if e['event']=='prefetch')
    complete = next(e['at'] for e in events if e['event']=='round_complete')
    started = next(e['at'] for e in events if e['event']=='round_started')
    assert prefetch < complete < started
    assert started - complete <= .011
    assert first.state == second.state == 'complete'
    for phase in (first.plan.playback, second.plan.approach, second.plan.playback):
        for order in (1, 2):
            np.testing.assert_allclose(phase.sample([0, phase.duration], order), 0, atol=1e-9)
    # Final round retains measured 0.5 second settling.
    transition = {item['to']: item['at'] for item in second.transitions}
    assert transition['complete'] - transition['settle_end'] >= .5


@pytest.mark.parametrize('fault', ['missing_plan', 'stale_plan', 'tick_overrun', 'missing_feedback'])
def test_medium_stops_on_stream_failures(fault):
    clock, incoming = Clock(), queue.Queue()
    devices, first = Devices(clock), make_player()
    if fault == 'stale_plan':
        incoming.put(make_player(first.plan.raw[-1], continuation=True, request='expired'))
    if fault in {'tick_overrun','missing_feedback'}:
        original = devices.submit
        def submit(frame, now):
            result = original(frame, now)
            if fault == 'tick_overrun':
                clock.now += .0151
                return result
            return None
        devices.submit = submit
    pattern = {'missing_plan':'Next fresh plan', 'stale_plan':'expired',
               'tick_overrun':'exceeded 100 Hz', 'missing_feedback':'lacks submission feedback'}[fault]
    with pytest.raises((RuntimeError, ValueError), match=pattern):
        control_loop(devices, first, incoming, lambda e: None, lambda *a: None,
                     lambda: False, 2, clock=clock, sleep=clock.sleep, idle_timeout=.1)
    assert devices.finishes == 0


def test_parent_prefetches_once_and_keeps_correct_round_across_completion(tmp_path):
    from openpi_client import msgpack_numpy
    requests = []
    devices = Devices(time.monotonic)
    devices.sock = S(close=lambda: None)
    devices.counter, devices.finish_policy = 0, 'disable'
    def recv(**kw):
        return msgpack_numpy.packb({'actions':np.full((50,54),len(requests)*.01)})
    ws = S(send=lambda packet: requests.append(packet), recv=recv)
    config = planning_config('19999')
    config['hand_control'] = 'slider'
    limits, names = inputs()
    consumer = ContinuousConsumer(devices,ws,{'checkpoint_manifest_sha256':'weights'},config,limits,names,
                                  tmp_path,S(event=lambda e:None),rounds=2)
    consumer.status, consumer.incoming = queue.Queue(),queue.Queue()
    process = S(start=lambda:None,is_alive=lambda:True)
    consumer.context = S(Process=lambda **kw:process)
    def consume():
        consumer.consume(None, {'observation/state':np.zeros(54)},
                         {'samples':{'state':{'stamp':time.time()-.001}}}, tmp_path)
    consume()
    consumer.status.put({'event':'prefetch','round':1,'target':[.01]*54})
    consume()
    assert consumer.rounds_completed == 0 and consumer.rounds_submitted == 2
    assert len(requests) == 2
    consumer.status.put({'event':'round_complete','round':1})
    consumer.status.put({'event':'between','round':1,'target':[.01]*54})
    consume()
    assert len(requests) == 2
    second = consumer.incoming.get_nowait()
    assert second.plan.config['final_chunk'] is True
    assert second.plan.start[0] == .01
    assert (tmp_path/'round-0002'/'plan'/'report.json').exists()
    consumer.status.put({'event':'round_started','round':2})
    consumer.status.put({'event':'round_complete','round':2})
    consumer.status.put({'event':'complete','rounds':2})
    consumer.poll()
    assert consumer.completed == 1


@pytest.mark.parametrize('model,port', [('19999','8001'),('30000','8002'),('30000v2','8003')])
def test_entry_and_transport_use_medium_settings_only(model,port,tmp_path):
    args = parse_args(['--model',model,'--execute','--supervised-trial','--continuous'])
    assert args.uri.endswith(port) and args.minimum_time_scale == 0.625
    commands = launch_commands(args,tmp_path,tmp_path)
    assert commands['gateway'][-2:] == ['--arm-speed-rad-s','2.0']
    assert 'deploy.fr3_wuji_medium.runtime.bridge' in commands['devices']


def test_prefetch_age_and_model_acquisition_bounds():
    player = make_player()
    accept_prefetch(player, 10.)
    with pytest.raises(ValueError,match='expired'):
        accept_prefetch(player,12.)
    limits,names = inputs()
    raw=np.zeros((50,54)); raw[:,0]=.5
    for model in ('19999','30000'):
        config=planning_config(model); config['hand_control']='slider'
        if model=='19999':
            with pytest.raises(ValueError,match='raw initial delta'):
                build_plan(raw,np.zeros(54),config,limits,names)
        else:
            build_plan(raw,np.zeros(54),config,limits,names)


def fake_bridge(sock, stop):
    """Only an in-memory robot; exercise real serialization and socket ownership."""
    import time

    from .runtime.ipc import encode
    from .runtime.ipc import receive
    from .runtime.ipc import unpack_frame
    from .runtime.ipc import wire_feedback

    device = Devices(time.monotonic)
    try:
        while not stop.is_set():
            request = receive(sock)
            op = request["operation"]
            now = time.monotonic()
            if op == "feedback":
                result = wire_feedback(device.feedback(now))
            elif op == "prepare":
                device.guard.arm(request["run_id"], request["plan_hash"], device.feedback(now), now)
                result = {"state": "armed"}
            elif op == "submit":
                frame_value = unpack_frame(request["frame"])
                fb = device.submit(frame_value, now)
                result = {"sequence": frame_value.sequence, "hands": {}, "feedback": wire_feedback(fb)}
            elif op == "next_plan":
                result = device.call(op, **{k: request[k] for k in ["run_id", "plan_hash", "start"]})
            elif op == "finish":
                result = {"state": "released"}
            elif op == "stop":
                result = {"state": "stopped", "physical_stop_confirmed": True}
            else:
                raise AssertionError(op)
            sock.sendall(encode({"id": request["id"], "ok": True, "result": result}))
    except (EOFError, OSError):
        pass
    finally:
        sock.close()


def test_spawned_controller_socket_transfer_and_two_rounds(tmp_path):
    import multiprocessing as mp
    import socket
    import threading
    import time

    from .continuous import StopFlag
    from .continuous import _worker

    clock = time.monotonic()
    first = make_player(np.zeros(54))
    second = make_player(first.plan.raw[-1], continuation=True, request="second")
    for player in (first, second):
        player.admission = Admission(clock - 0.1, clock - 0.05, clock, player.run_id, "weights", 0)
        player.prepared_at = clock
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    stop_server = threading.Event()
    server = threading.Thread(target=fake_bridge, args=(right, stop_server), daemon=True)
    server.start()
    context = mp.get_context("spawn")
    plans, status, stop = context.Queue(1), context.Queue(256), StopFlag(context)
    process = context.Process(target=_worker, args=(left, 0, "disable", first, plans, status, stop, str(tmp_path), 2))
    process.start()
    left.close()
    events = []
    try:
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            event = status.get(timeout=10)
            events.append(event)
            assert event["event"] != "failed", event
            if event["event"] == "prefetch" and event["round"] == 1:
                now = time.monotonic()
                second.admission = Admission(now-.01, now, now, "second", "weights", 0)
                second.prepared_at = now
                plans.put(second)
            if event["event"] == "complete":
                break
        assert events[-1]["event"] == "complete"
        process.join(5)
        assert process.exitcode == 0
    finally:
        stop.set()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(2)
        stop_server.set()
        server.join(1)




def test_continuation_removes_artificial_one_second_floor_with_bounds_intact():
    from .planner import caps
    limits, names = inputs()
    config = planning_config('19999')
    config.update(hand_control='slider', continuation=True)
    raw = np.zeros((50, 54))
    plan = build_plan(raw, np.zeros(54), config, limits, names)
    assert plan.approach.duration == pytest.approx(.025)
    raw[:, 0] = .1
    plan = build_plan(raw, np.zeros(54), config, limits, names)
    for order, derivative in enumerate(('velocity','acceleration','jerk'),1):
        peak = np.maximum(np.abs(plan.approach.lower[order]), np.abs(plan.approach.upper[order]))
        assert peak[0] / plan.approach.scale**order <= caps(config,'approach',derivative)[0] + 1e-9
