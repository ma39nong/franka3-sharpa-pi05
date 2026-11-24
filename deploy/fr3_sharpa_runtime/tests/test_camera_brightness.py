from types import SimpleNamespace

import numpy as np

from deploy.fr3_sharpa_runtime.camera_brightness import HeadBrightness


def frame(value, padding=0):
    pixels = np.full((400, 640, 3), value, dtype=np.uint8)
    rows = np.pad(pixels.reshape(400, -1), ((0, 0), (0, padding)), constant_values=255)
    return SimpleNamespace(data=rows.tobytes(), height=400, width=640, step=1920 + padding)


def test_black_frames_with_valid_dimensions_fail_after_three_seconds():
    gate = HeadBrightness()
    gate.observe(frame(10, padding=16), 0)
    assert not gate.ready(0)
    assert not gate.failed(2_999_999_999)
    gate.observe(frame(10), 3_000_000_000)
    assert gate.failed(3_000_000_000)
    assert gate.diagnostic(3_000_000_000)["mean"] == 10


def test_recovery_requires_continuous_bright_frames():
    gate = HeadBrightness()
    gate.observe(frame(140), 0)
    assert gate.ready(3_000_000_000)
    gate.observe(frame(10), 4_000_000_000)
    assert not gate.ready(4_000_000_000)
    gate.observe(frame(140), 5_000_000_000)
    assert not gate.failed(8_000_000_000)
    assert not gate.ready(7_999_999_999)
    assert gate.ready(8_000_000_000)


def test_dark_background_with_visible_bright_content_passes():
    pixels = np.zeros((400, 640, 3), dtype=np.uint8)
    pixels[:, 320:] = 140
    message = SimpleNamespace(data=pixels.tobytes(), height=400, width=640, step=1920)
    gate = HeadBrightness()
    gate.observe(message, 0)
    assert gate.ready(3_000_000_000)
