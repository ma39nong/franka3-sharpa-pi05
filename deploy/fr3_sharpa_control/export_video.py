"""Export timestamp-aligned camera videos after the recording writer is closed."""
import argparse
from pathlib import Path

from .recording import write_json

CAMERAS = {"/cam0/color/image_raw": "head", "/cam2/color/image_raw": "left_wrist",
           "/cam1/color/image_raw": "right_wrist"}


def timed_samples(events, fps=30):
    """Hold last received image on one receipt-time axis, never zip frame numbers."""
    latest = {camera: None for camera in CAMERAS.values()}
    origin = None
    index = 0
    last_stamp = None
    for camera, stamp, frame in events:
        if origin is None:
            origin = stamp
        if last_stamp is not None and (stamp < last_stamp or stamp - last_stamp > 60_000_000_000):
            raise ValueError("Bag receipt clock moved backwards or jumped over 60 seconds")
        last_stamp = stamp
        while origin + round(index * 1e9 / fps) < stamp:
            yield index / fps, dict(latest)
            index += 1
        if camera is not None:
            latest[camera] = frame
    if origin is not None and origin + round(index * 1e9 / fps) <= last_stamp:
        yield index / fps, dict(latest)


def decode_image(message):
    import cv2
    import numpy as np
    encoding = message.encoding.lower()
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(encoding)
    if not channels or message.step < message.width * channels:
        raise ValueError(f"Unsupported RGB encoding/step: {message.encoding}")
    frame = np.frombuffer(message.data, np.uint8).reshape(message.height, message.step)
    frame = frame[:, :message.width * channels].reshape(message.height, message.width, channels)
    code = {"rgb8": cv2.COLOR_RGB2BGR, "rgba8": cv2.COLOR_RGBA2BGR,
            "bgra8": cv2.COLOR_BGRA2BGR, "mono8": cv2.COLOR_GRAY2BGR}.get(encoding)
    return cv2.cvtColor(frame, code) if code is not None else frame


def export(directory, *, fps=30, width=640, height=360):
    import cv2
    cv2.setNumThreads(1)
    import numpy as np
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image
    directory = Path(directory)
    output = directory / "video"
    output.mkdir(exist_ok=True)
    report_path = directory / "video_export.json"
    report = {"status": "EXPORTING", "fps": fps, "time_axis": "recorder_ros_receipt_time",
              "camera_mapping": CAMERAS, "frames": 0}
    write_json(report_path, report)
    writers = {}
    seen = set()
    try:
        for name in (*CAMERAS.values(), "three_cameras"):
            path = output / f"{name}.mp4"
            if path.exists():
                raise FileExistsError(f"已有视频，不覆盖：{path}")
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (width * (3 if name == "three_cameras" else 1), height))
            if not writer.isOpened():
                raise RuntimeError(f"Video encoder unavailable: {path}")
            writers[name] = writer
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=str(directory / "rosbag"), storage_id="sqlite3"),
                    rosbag2_py.ConverterOptions("", ""))
        def events():
            while reader.has_next():
                topic, data, stamp = reader.read_next()
                camera = CAMERAS.get(topic)
                if camera:
                    seen.add(camera)
                    yield camera, stamp, decode_image(deserialize_message(data, Image))
                else:
                    yield None, stamp, None
        for seconds, images in timed_samples(events(), fps):
            panels = []
            for name, frame in images.items():
                panel = np.zeros((height, width, 3), np.uint8)
                if frame is not None:
                    h, w = frame.shape[:2]
                    scale = min(width / w, height / h)
                    resized = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))))
                    rh, rw = resized.shape[:2]
                    panel[(height-rh)//2:(height-rh)//2+rh, (width-rw)//2:(width-rw)//2+rw] = resized
                label = f"{name}  {seconds:.2f}s" + ("  NO FRAME" if frame is None else "")
                cv2.putText(panel, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                writers[name].write(panel)
                panels.append(panel)
            writers["three_cameras"].write(np.concatenate(panels, axis=1))
            report["frames"] += 1
        if seen != set(CAMERAS.values()) or not report["frames"]:
            raise RuntimeError(f"缺少相机数据：{sorted(set(CAMERAS.values())-seen)}")
        report["status"] = "COMPLETE"
    except Exception as error:
        report.update(status="FAILED", error=str(error))
        raise
    finally:
        for writer in writers.values():
            writer.release()
        write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    export(args.directory)


if __name__ == "__main__":
    main()
