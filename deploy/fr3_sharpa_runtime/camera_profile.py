"""Copy workstation parameters for MJPG right wrist or an RGB-only head."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gello", default=os.environ.get("SHARPA_GELLO_REPO"), required=not os.environ.get("SHARPA_GELLO_REPO"))
    parser.add_argument("--camera", choices=("cam0", "cam1"), default="cam1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    filename = "camera_primary_params.yaml" if args.camera == "cam0" else "camera_cam1_global_params.yaml"
    source = Path(args.gello) / "ros_ws/src/teleop_camera_bringup/config" / filename
    content = source.read_bytes()
    parameters = yaml.safe_load(content)
    if not isinstance(parameters, dict) or parameters.get("time_domain") != "global":
        raise ValueError("Expected the existing global source-timestamp camera profile")
    updates = {"enable_depth": False, "enable_frame_sync": False} if args.camera == "cam0" else {"color_format": "MJPG"}
    changed = {key: [parameters.get(key), value] for key, value in updates.items()}
    parameters.update(updates)
    if args.output is None:
        target = "camera_cam0_rgb_params.yaml" if args.camera == "cam0" else "camera_cam1_mjpg_params.yaml"
        args.output = Path(__file__).resolve().parents[2] / ".deployment" / target
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(parameters, sort_keys=False), encoding="utf-8")
    print(json.dumps({"profile": str(args.output.resolve()), "source": str(source.resolve()),
                      "source_sha256": hashlib.sha256(content).hexdigest(),
                      "changed": changed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
