import pytest

from deploy.fr3_sharpa_runtime.camera_clock import SourceClockSettling


def feed(clock, tick, *, bad_age=None, frozen=False):
    mono = tick * 100_000_000
    ros = 100_000_000_000 + mono
    for key in clock.streams:
        age = 30_000_000 if key != "cam0" else 120_000_000
        if key == "cam1" and bad_age is not None:
            age = bad_age
        stamp = ros - age
        if frozen and key == "cam1":
            stamp = clock.streams[key]["stamp"]
        clock.observe(key, stamp, now_ros=ros, now_mono=mono)
    return mono


def test_short_valid_window_before_late_clock_jump_cannot_admit_sdk():
    clock = SourceClockSettling(("cam0", "cam1", "cam2"), 0)
    assert not clock.ready(0)
    for tick in range(331):
        mono = feed(clock, tick, bad_age=11_600_000_000 if 220 <= tick <= 226 else None)
        if tick < 327:
            assert not clock.ready(mono)
    assert clock.ready(mono)
    assert clock.diagnostic(mono)["cameras"]["cam1"]["invalid_frames"] == 7


@pytest.mark.parametrize(("bad_age", "frozen"), [(201_000_000, False), (-1, False), (None, True)])
def test_stale_future_or_nonadvancing_stamp_resets_valid_clock_window(bad_age, frozen):
    clock = SourceClockSettling(("cam0", "cam1", "cam2"), 0)
    for tick in range(303):
        mono = feed(clock, tick, bad_age=bad_age if tick == 200 else None,
                    frozen=frozen and tick == 200)
        if tick == 300:
            assert not clock.ready(mono)
    assert clock.ready(mono)


def test_settled_stream_cannot_remain_ready_after_receiving_stops():
    clock = SourceClockSettling(("cam0", "cam1", "cam2"), 0)
    for tick in range(301):
        mono = feed(clock, tick)
    assert clock.ready(mono)
    assert not clock.ready(mono + 201_000_000)
    feed(clock, 304)
    assert not clock.ready(30_400_000_000)


def test_diagnostics_distinguish_continuous_old_timestamps_from_receive_stall():
    clock = SourceClockSettling(("cam1",), 0)
    for tick in range(5):
        clock.observe("cam1", 100_000_000_000 + tick * 33_000_000,
                      now_ros=109_000_000_000 + tick * 33_000_000, now_mono=tick * 33_000_000)
    details = clock.diagnostic(5 * 33_000_000)["cameras"]["cam1"]
    assert details["invalid_reasons"] == {"stale_source": 5}
    assert details["last_invalid"]["source_age_ms"] == 9000
    assert details["last_invalid"]["receive_gap_ms"] == 33
