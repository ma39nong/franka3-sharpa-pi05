"""Timed speed changes select a prepared plan only at a chunk boundary."""

import queue
import time
from types import SimpleNamespace

import numpy as np
import pytest

from .continuous import ContinuousConsumer
from .continuous import PlanAlternatives
from .continuous import control_loop
from .deploy import parse_args
from .planner import build_plan
from .player import Admission
from .player import OneShot
from .profile import planning_config
from .profile import slow_motion_config
from .test_medium import Clock
from .test_medium import Devices
from .test_medium import inputs
from .test_medium import make_player


def alternatives(first):
    start = first.plan.raw[-1]
    actions = np.tile(start, (50, 1)) + np.linspace(0.001, 0.015, 50)[:, None]
    config = planning_config("19999")
    config.update(
        hand_control="slider", execution_steps=20, continuation=True, final_chunk=True, speed_profile="medium"
    )
    limits, names = inputs()
    medium = OneShot(
        build_plan(actions, start, config, limits, names), Admission(9.8, 9.85, 9.9, "medium", "weights", 0), 10.0
    )
    slow = OneShot(
        build_plan(actions, start, slow_motion_config(config), limits, names),
        Admission(9.8, 9.85, 9.9, "slow", "weights", 0),
        10.0,
    )
    return PlanAlternatives(medium, slow)


@pytest.mark.parametrize(("threshold", "expected"), [(1.1, "slow"), (25.0, "medium")])
def test_preplanned_choice_is_made_at_real_boundary(threshold, expected):
    clock, incoming, events = Clock(), queue.Queue(), []
    devices = Devices(clock)
    first = make_player()
    pair = alternatives(first)
    prefetch_elapsed = []

    def notify(event):
        events.append(dict(event, at=clock()))
        if event["event"] == "prefetch":
            prefetch_elapsed.append(clock() - first.started)
            for candidate in (pair.medium, pair.slow):
                candidate.admission = Admission(clock() - 0.01, clock(), clock(), candidate.run_id, "weights", 0)
                candidate.prepared_at = clock()
            incoming.put(pair)

    control_loop(
        devices,
        first,
        incoming,
        notify,
        lambda *_: None,
        lambda: False,
        2,
        clock=clock,
        sleep=clock.sleep,
        slow_after_seconds=threshold,
    )
    chosen = next(e for e in events if e["event"] == "round_started")
    assert chosen["profile"] == expected
    assert chosen["plan_hash"] == getattr(pair, expected).digest
    if expected == "slow":
        assert prefetch_elapsed[0] < threshold < chosen["at"] - first.started
        switch = [e for e in events if e["event"] == "speed_profile_changed"]
        assert len(switch) == 1
        assert switch[0]["round"] == 2
        assert switch[0]["elapsed_seconds"] >= threshold
        assert devices.frames[-1].plan_hash == pair.slow.digest
        assert devices.frames[-1].positions[0] == pytest.approx(pair.slow.plan.raw[-1, 0])
    else:
        assert not any(e["event"] == "speed_profile_changed" for e in events)
        assert devices.frames[-1].plan_hash == pair.medium.digest
    assert [frame.sequence for frame in devices.frames] == list(range(len(devices.frames)))


def test_slow_profile_matches_existing_slow_bounds_and_preserves_model_gate():
    from pathlib import Path

    from experiments.weight_motion_eval.planner import read_config

    medium = planning_config("30000")
    medium.update(hand_control="slider", execution_steps=20)
    slow = slow_motion_config(medium)
    original = read_config(Path(__file__).resolve().parents[2] / "experiments/weight_motion_eval/config.yaml")
    assert slow["minimum_time_scale"] == original["minimum_time_scale"] == 2.5
    assert slow["approach"] == original["approach"]
    assert slow["playback"] == original["playback"]
    assert slow["settle_seconds"] == original["settle_seconds"] == 0.5
    assert slow["slider_speed_rad_s"] == 1.0
    assert slow["arm_raw_initial_delta_rad"] == medium["arm_raw_initial_delta_rad"] == 1.5
    assert slow["max_smoothing_delta_rad"] == medium["max_smoothing_delta_rad"]


def test_switch_defaults_to_25_seconds_and_can_be_disabled():
    assert parse_args([]).slow_after_seconds == 25.0
    assert parse_args(["--slow-after-seconds", "0"]).slow_after_seconds == 0.0
    with pytest.raises(SystemExit):
        parse_args(["--slow-after-seconds", "nan"])
    with pytest.raises(SystemExit):
        parse_args(["--slow-after-seconds", "-1"])


def test_slider_limit_changes_only_at_validated_plan_boundary():
    from types import SimpleNamespace

    from .runtime.core import Frame
    from .runtime.devices import DeviceSession

    session = DeviceSession.__new__(DeviceSession)
    session.continuous = True
    session.state = "armed"
    session.fault = None
    session.last = Frame("run", "old", 4, 10.0, 10.02, 0.0, np.zeros(54), np.zeros(54), "settle_end")
    session.guard = SimpleNamespace(run_id="run", digest="old")
    session.plan_hashes = {"old"}
    session.hand_control = "slider"
    session.hands = {side: SimpleNamespace(slider_speed_rad_s=2.0) for side in ("left", "right")}
    session.check_endpoint = lambda: None
    with pytest.raises(ValueError, match="supported profile"):
        session.next_plan("run", "invalid", np.zeros(54), 0.5)
    assert all(hand.slider_speed_rad_s == 2.0 for hand in session.hands.values())
    assert session.next_plan("run", "slow", np.zeros(54), 1.0)["sequence"] == 5
    assert all(hand.slider_speed_rad_s == 1.0 for hand in session.hands.values())
    assert session.guard.digest == "slow"


def test_parent_preplans_both_profiles_from_one_fresh_inference(tmp_path):
    from openpi_client import msgpack_numpy

    now = time.monotonic()
    devices = Devices(time.monotonic)
    devices.sock = SimpleNamespace(close=lambda: None)
    devices.counter, devices.finish_policy = 0, "disable"
    requests = []
    ws = SimpleNamespace(
        send=lambda packet: requests.append(packet),
        recv=lambda **_: msgpack_numpy.packb({"actions": np.full((50, 54), 0.01 * len(requests))}),
    )
    config = planning_config("19999")
    config["hand_control"] = "slider"
    limits, names = inputs()
    consumer = ContinuousConsumer(
        devices,
        ws,
        {"checkpoint_manifest_sha256": "weights"},
        config,
        limits,
        names,
        tmp_path,
        SimpleNamespace(event=lambda _: None),
        rounds=2,
        slow_after_seconds=25.0,
    )
    consumer.status, consumer.incoming = queue.Queue(), queue.Queue()
    process = SimpleNamespace(start=lambda: None, is_alive=lambda: True)
    consumer.context = SimpleNamespace(Process=lambda **_: process)

    def consume():
        consumer.consume(
            None, {"observation/state": np.zeros(54)}, {"samples": {"state": {"stamp": time.time() - 0.001}}}, tmp_path
        )

    consume()
    consumer.status.put({"event": "prefetch", "round": 1, "target": [0.01] * 54})
    consume()
    candidates = consumer.incoming.get_nowait()
    assert isinstance(candidates, PlanAlternatives)
    assert len(requests) == 2
    np.testing.assert_array_equal(candidates.medium.plan.raw, candidates.slow.plan.raw)
    assert candidates.slow.plan.playback.duration > candidates.medium.plan.playback.duration
    assert candidates.slow.plan.config["slider_speed_rad_s"] == 1.0
    assert (tmp_path / "round-0002" / "plan" / "report.json").exists()
    assert (tmp_path / "round-0002" / "plan-slow" / "report.json").exists()
    assert candidates.medium.admission.sent_time >= now
    assert candidates.medium.admission == candidates.slow.admission


def test_spawned_controller_switches_without_resetting_wire_sequence(tmp_path):
    import multiprocessing as mp
    import socket
    import threading

    from .continuous import StopFlag
    from .continuous import _worker
    from .test_medium import fake_bridge

    prepared = time.monotonic()
    first = make_player(np.zeros(54))
    pair = alternatives(first)
    first.admission = Admission(prepared - 0.1, prepared - 0.05, prepared, "first", "weights", 0)
    first.prepared_at = prepared
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    stop_server = threading.Event()
    server = threading.Thread(target=fake_bridge, args=(right, stop_server), daemon=True)
    server.start()
    context = mp.get_context("spawn")
    plans, status, stop = context.Queue(1), context.Queue(256), StopFlag(context)
    process = context.Process(
        target=_worker, args=(left, 0, "disable", first, plans, status, stop, str(tmp_path), 2, 1.1)
    )
    process.start()
    left.close()
    events = []
    try:
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            event = status.get(timeout=max(1, deadline - time.monotonic()))
            events.append(event)
            assert event["event"] != "failed", event
            if event["event"] == "prefetch" and event["round"] == 1:
                now = time.monotonic()
                for candidate in (pair.medium, pair.slow):
                    candidate.admission = Admission(now - 0.01, now, now, candidate.run_id, "weights", 0)
                    candidate.prepared_at = now
                plans.put(pair)
            if event["event"] == "complete":
                break
        assert events[-1]["event"] == "complete"
        assert [e["profile"] for e in events if e["event"] == "round_started"] == ["slow"]
        assert len([e for e in events if e["event"] == "speed_profile_changed"]) == 1
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
