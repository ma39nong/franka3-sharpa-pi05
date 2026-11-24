import sys

import yaml

from deploy.fr3_sharpa_runtime import camera_profile


def test_generated_head_profile_uses_rgb_only(tmp_path, monkeypatch):
    config = tmp_path / "gello/ros_ws/src/teleop_camera_bringup/config"
    config.mkdir(parents=True)
    original = {"time_domain": "global", "enable_depth": True, "enable_frame_sync": True,
                "enable_color_auto_exposure": True, "color_exposure": 30, "color_gain": 16,
                "color_width": 640, "color_height": 400, "color_fps": 20,
                "enable_sync_host_time": False}
    source = config / "camera_primary_params.yaml"
    source.write_text(yaml.safe_dump(original))
    output = tmp_path / "head.yaml"
    monkeypatch.setattr(sys, "argv", ["camera_profile", "--gello", str(tmp_path / "gello"),
                                    "--camera", "cam0", "--output", str(output)])
    camera_profile.main()
    actual = yaml.safe_load(output.read_text())
    assert actual == {**original, "enable_depth": False, "enable_frame_sync": False}
    assert yaml.safe_load(source.read_text()) == original
