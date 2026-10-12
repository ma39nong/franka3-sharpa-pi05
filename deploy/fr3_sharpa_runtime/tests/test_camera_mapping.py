"""Check camera identity through offline and live policy observation inputs."""

import base64
import io
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from deploy.fr3_sharpa_runtime import run


def camera_images():
    return {
        "cam0": np.full((2, 3, 3), [255, 0, 0], dtype=np.uint8),
        "cam1": np.full((2, 3, 3), [0, 255, 0], dtype=np.uint8),
        "cam2": np.full((2, 3, 3), [0, 0, 255], dtype=np.uint8),
    }


def assert_observation(observation, images, state):
    np.testing.assert_array_equal(observation["observation/image"], images["cam0"])
    np.testing.assert_array_equal(observation["observation/left_wrist_image"], images["cam2"])
    np.testing.assert_array_equal(observation["observation/right_wrist_image"], images["cam1"])
    np.testing.assert_array_equal(observation["observation/state"], state)
    assert observation["prompt"] == "test task"


def test_offline_camera_identity_and_state_order(tmp_path):
    images = camera_images()
    state = np.arange(58, dtype=np.float32)
    columns = {"observation.state": [state.tolist()]}
    for camera, pixels in images.items():
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        columns[f"observation.images.{camera}"] = [{"bytes": buffer.getvalue(), "path": None}]
    sample = tmp_path / "sample.parquet"
    pq.write_table(pa.table(columns), sample)

    assert_observation(run.load_sample(sample, "test task"), images, state)


def test_live_camera_identity_and_state_order(monkeypatch, tmp_path):
    images = camera_images()
    state = np.arange(58, dtype=np.float32)
    snapshot = {
        "state": state.tolist(),
        "images": {
            camera: {"shape": list(pixels.shape), "rgb_b64": base64.b64encode(pixels.tobytes()).decode()}
            for camera, pixels in images.items()
        },
        "captured_monotonic_ns": 123456,
    }
    (tmp_path / "setup.bash").touch()
    # Mock the ROS subprocess; this test never connects to hardware.
    monkeypatch.setattr(
        run.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(snapshot), stderr=""),
    )

    observation = run.live_observation(tmp_path, "test task")
    assert_observation(observation, images, state)
    assert observation["_captured_monotonic_ns"] == 123456
