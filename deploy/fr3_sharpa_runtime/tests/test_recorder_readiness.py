"""Recorder boundaries and local receipt health, without ROS or hardware."""

from concurrent.futures import ThreadPoolExecutor
import json
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from deploy.fr3_sharpa_control.recorder_ros import SENSORS
from deploy.fr3_sharpa_control.recorder_ros import TOPICS
from deploy.fr3_sharpa_control.recorder_ros import BagCapture


class Clock:
    def __init__(self):
        self.now = 10_000_000_000

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += int(seconds * 1e9)


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    writers, serialized = [], []

    class Writer:
        def __init__(self):
            self.rows = []
            self.closed = False
            writers.append(self)

        def open(self, *args):
            pass

        def create_topic(self, topic):
            pass

        def write(self, *args):
            self.rows.append(args)

        def close(self):
            self.closed = True

    monkeypatch.setitem(
        sys.modules,
        "rosbag2_py",
        SimpleNamespace(
            SequentialWriter=Writer,
            StorageOptions=lambda **kw: kw,
            ConverterOptions=lambda *a: a,
            TopicMetadata=lambda **kw: kw,
        ),
    )
    clock = Clock()
    capture = BagCapture(tmp_path, stop_marker=tmp_path / "stop", clock=clock, sleep=clock.sleep)

    def serialize(message):
        serialized.append(message)
        return b"cdr:" + message

    def feed(topic, stamp=None, payload=b"image", ros=123456):
        capture.accept(topic, payload, ros, rx_mono_ns=clock() if stamp is None else stamp, serialize=serialize)

    return SimpleNamespace(bag=capture, clock=clock, writers=writers, serialized=serialized, feed=feed)


def healthy(r, exclude=()):
    for topic in SENSORS - set(exclude):
        r.feed(topic, r.clock() - 50_000_000)
        r.feed(topic)


def test_prepared_large_payloads_never_serialize_and_metadata_is_bounded(recorder):
    r = recorder
    image = b"x" * 1_000_000
    for _i in range(1000):
        r.clock.now += 1_000_000
        for topic in TOPICS:
            r.feed(topic, payload=image)
    assert not r.serialized
    assert not r.writers
    assert not any(r.bag.counts.values())
    assert r.bag.seen == set(TOPICS)
    assert all(len(samples) <= 512 for samples in r.bag.recent_rx_monotonic_ns.values())
    assert r.bag.readiness()["ready"]


def test_active_exactly_once_preserves_ros_receipt_and_old_entry_is_rejected(recorder):
    r = recorder
    healthy(r)
    old_entry = r.clock() - 1
    r.bag.request("start")
    r.feed("/cam0/color/image_raw", old_entry, payload=b"prepare")
    for topic in TOPICS:
        r.feed(topic, payload=b"task", ros=987654321)
    assert r.serialized == [b"task"] * len(TOPICS)
    assert r.writers[0].rows == [(topic, b"cdr:task", 987654321) for topic in TOPICS]
    r.bag.request("stop")
    for topic in TOPICS:
        r.feed(topic)
    assert len(r.serialized) == len(TOPICS)
    assert r.writers[0].closed
    assert json.loads((r.bag.directory / "recording.json").read_text())["status"] == "COMPLETE"
    with pytest.raises(RuntimeError):
        r.bag.request("start")


def test_prepared_callback_waiting_on_lock_cannot_cross_start(recorder):
    r = recorder
    healthy(r)
    entered = threading.Event()

    def delayed():
        stamp = r.clock()
        entered.set()
        r.feed("/cam0/color/image_raw", stamp, payload=b"old")

    with ThreadPoolExecutor() as pool:
        with r.bag.lock:
            future = pool.submit(delayed)
            assert entered.wait(1)
            r.clock.now += 1
            r.bag.request("start")
        future.result(timeout=1)
    assert not r.serialized
    assert not r.writers[0].rows


@pytest.mark.parametrize("mode", ["serialize", "write"])
def test_recording_errors_finish_and_block_later_serialization(recorder, mode):
    r = recorder
    healthy(r)
    r.bag.request("start")

    def fail(*args):
        raise ValueError("broken " + mode)

    if mode == "write":
        r.writers[0].write = fail
    r.bag.accept(
        "/cam0/color/image_raw", b"bad", 2, rx_mono_ns=r.clock(), serialize=fail if mode == "serialize" else bytes
    )
    assert r.bag.finished.is_set()
    assert not r.bag.active
    assert r.bag.report["status"] == "FAILED"
    for topic in TOPICS:
        r.feed(topic)
    assert not r.serialized
    with pytest.raises(RuntimeError, match="broken"):
        r.bag.request("start")
    r.bag.close()
    assert r.writers[0].closed
    assert json.loads((r.bag.directory / "recording.json").read_text())["status"] == "FAILED"


def test_marker_prevents_serialization_and_reactivation(recorder):
    r = recorder
    healthy(r)
    r.bag.request("start")
    r.bag.stop_marker.touch()
    r.feed("/cam0/color/image_raw")
    assert not r.serialized
    assert not r.writers[0].rows
    with pytest.raises(RuntimeError):
        r.bag.request("start")


@pytest.mark.parametrize("scenario", ["historical", "stale_cam1", "single_cam1", "missing_cam1"])
def test_timeout_has_diagnostics_and_never_opens_writer(recorder, scenario):
    r = recorder
    topic = "/cam1/color/image_raw"
    if scenario == "historical":
        for sensor in SENSORS:
            r.feed(sensor, r.clock() - 30_000_000_000)
    else:
        healthy(r, exclude={topic})
        if scenario == "stale_cam1":
            r.feed(topic, r.clock() - 300_000_000)
        if scenario == "single_cam1":
            r.feed(topic)
    initial = r.bag.request("status")
    assert initial["status"] == "PASS"
    assert not initial["active"]
    assert not initial["sensor_readiness"]["ready"]
    if scenario == "stale_cam1":

        def keep_six_live(seconds):
            r.clock.sleep(seconds)
            healthy(r, exclude={topic})

        r.bag.sleep = keep_six_live
    start = r.clock()
    with pytest.raises(TimeoutError) as error:
        r.bag.request("start")
    assert 5_000_000_000 <= r.clock() - start <= 5_020_000_000
    assert topic in str(error.value)
    assert "ages_ms" in str(error.value)
    if scenario == "stale_cam1":
        assert r.bag.readiness()["stale"] == [topic]
    assert "missing" in str(error.value)
    assert "stale" in str(error.value)
    assert "insufficient_progress" in str(error.value)
    assert not r.writers
    assert not r.serialized
    assert not any(r.bag.counts.values())
    # Fresh real callbacks restore eligibility after a failed readiness attempt.
    healthy(r)
    assert r.bag.request("start")["active"]


def test_two_distinct_receipts_required_and_200ms_boundary(recorder):
    r = recorder
    topic = "/cam1/color/image_raw"
    healthy(r, exclude={topic})
    r.feed(topic)
    r.feed(topic)  # Identical entry time cannot fake progress.
    assert topic in r.bag.readiness()["insufficient_progress"]
    r.clock.now += 1
    r.feed(topic)
    assert r.bag.readiness()["ready"]
    r.clock.now += 200_000_000
    assert topic not in r.bag.readiness()["stale"]
    r.clock.now += 1
    assert topic in r.bag.readiness()["stale"]


def test_background_callbacks_release_wait_lock_at_normal_camera_rates(recorder):
    r = recorder
    # Use the actual clock and events here to exercise cross-thread lock progress.
    r.bag.clock, r.bag.sleep = time.monotonic_ns, time.sleep
    done = threading.Event()

    def stream():
        deadline = time.monotonic() + 2
        next_rx = dict.fromkeys(SENSORS, 0)
        while not done.is_set() and time.monotonic() < deadline:
            now = time.monotonic_ns()
            for topic in SENSORS:
                if now >= next_rx[topic]:
                    r.feed(topic, now)
                    period = (
                        50_000_000
                        if topic.startswith("/cam0")
                        else 33_333_333
                        if topic.startswith("/cam")
                        else 100_000_000
                    )
                    next_rx[topic] = now + period
            time.sleep(0.001)

    with ThreadPoolExecutor() as pool:
        stream_future = pool.submit(stream)
        try:
            futures = [pool.submit(r.bag.request, "start") for _ in range(3)]
            assert all(f.result(timeout=2)["active"] for f in futures)
            assert len(r.writers) == 1
        finally:
            done.set()
        stream_future.result(timeout=2)


@pytest.mark.parametrize("stop", ["request", "marker"])
def test_stop_interrupts_waiting_start_without_opening_writer(recorder, stop):
    r = recorder
    waiting, release = threading.Event(), threading.Event()

    def sleep(seconds):
        waiting.set()
        assert release.wait(1)
        r.clock.sleep(seconds)

    r.bag.sleep = sleep
    with ThreadPoolExecutor() as pool:
        future = pool.submit(r.bag.request, "start")
        assert waiting.wait(1)
        if stop == "request":
            r.bag.request("stop")
        else:
            r.bag.stop_marker.touch()
        healthy(r)
        release.set()
        with pytest.raises(RuntimeError):
            future.result(timeout=1)
    assert not r.writers
    assert not r.serialized


def test_slow_writer_initialization_fails_closed_with_diagnostics(recorder, monkeypatch):
    r = recorder
    healthy(r)
    writer_class = sys.modules["rosbag2_py"].SequentialWriter
    monkeypatch.setattr(writer_class, "open", lambda *args: r.clock.sleep(0.3))
    with pytest.raises(RuntimeError, match="ages_ms"):
        r.bag.request("start")
    assert not r.bag.active
    assert r.bag.finished.is_set()
    assert r.bag.report["writer_initialization_ms"] == 300
    r.feed("/cam0/color/image_raw")
    assert not r.serialized
    assert not r.writers[0].rows
    r.bag.close()
    assert r.writers[0].closed
