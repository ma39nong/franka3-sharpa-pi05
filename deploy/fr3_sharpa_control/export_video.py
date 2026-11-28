"""Export timestamp-aligned camera videos after the recording writer is closed."""
import argparse
import json
from pathlib import Path

from .recording import write_json, compact_run

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
    if compact_run(directory):
        return export_compact(directory,fps=fps,width=width,height=height)
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
        run_path = directory/'run.json'
        run = json.loads(run_path.read_text()) if run_path.exists() else {}
        four_required = bool((run.get('model',{}).get('tomato_model') or {}).get('online_roi_required'))
        if four_required or (directory/'policy_views').is_dir():
            from .export_four_view import export_four_views
            report['four_view'] = export_four_views(directory,fps=fps,width=width,height=height)
        report["status"] = "COMPLETE"
    except Exception as error:
        report.update(status="FAILED", error=str(error))
        raise
    finally:
        for writer in writers.values():
            writer.release()
        write_json(report_path, report)
    return report


def export_compact(directory, *, fps=30, width=640, height=360):
    """Exactly three camera MP4s and one triptych, for either storage backend."""
    import av
    import numpy as np
    from fractions import Fraction
    directory=Path(directory);output=directory/'video';output.mkdir(exist_ok=True)
    report={'status':'EXPORTING','fps':fps,'frames':0,'time_axis':'recorder_ros_receipt_time',
            'layout':'three-individual-and-one-triptych','four_view':False}
    write_json(directory/'video_export.json',report)
    names=('head','left_wrist','right_wrist');seen=set();writers={}
    try:
        for name in (*names,'three_cameras'):
            path=output/(name+'.mp4')
            if path.exists():raise FileExistsError(path)
            container=av.open(str(path),'w',format='mp4',options={'movflags':'+faststart'})
            stream=container.add_stream('libx264',rate=fps);stream.width=width*(3 if name=='three_cameras' else 1)
            stream.height=height;stream.pix_fmt='yuv420p';stream.codec_context.thread_count=1;stream.options={'preset':'fast','crf':'20','bf':'0'}
            writers[name]=(container,stream)
        def samples():
            if (directory/'mcap/data.mcap').is_file():
                from mcap.reader import make_reader
                from mcap_ros2.decoder import DecoderFactory
                decoders={name:av.CodecContext.create('h264','r') for name in names}
                with (directory/'mcap/data.mcap').open('rb') as handle:
                    reader=make_reader(handle,decoder_factories=[DecoderFactory()]);summary=reader.get_summary()
                    mapping={c.topic:names[(0,2,1)[i]] for c in summary.channels.values()
                             for i in range(3) if c.topic.startswith(f'/devices/cam{i}/') and c.topic.endswith('/image_h264')}
                    report['camera_mapping']=mapping
                    for _,ch,msg,data in reader.iter_decoded_messages(topics=list(mapping)):
                        name=mapping[ch.topic]
                        frames=decoders[name].decode(av.Packet(bytes(data.data)))
                        if len(frames)!=1:raise RuntimeError('Expected one H.264 frame per MCAP access unit')
                        seen.add(name);yield name,msg.log_time,frames[0].to_ndarray(format='rgb24')
            else:
                import rosbag2_py
                from rclpy.serialization import deserialize_message
                from sensor_msgs.msg import Image
                reader=rosbag2_py.SequentialReader();reader.open(rosbag2_py.StorageOptions(uri=str(directory/'rosbag'),storage_id='sqlite3'),rosbag2_py.ConverterOptions('',''))
                report['camera_mapping']=CAMERAS
                while reader.has_next():
                    topic,data,stamp=reader.read_next();name=CAMERAS.get(topic)
                    if name:
                        m=deserialize_message(data,Image)
                        if m.encoding not in ('rgb8','bgr8'):raise ValueError('Unsupported preview image encoding: '+m.encoding)
                        pixels=np.frombuffer(bytes(m.data),np.uint8).reshape(m.height,m.step)[:,:m.width*3].reshape(m.height,m.width,3)
                        if m.encoding=='bgr8':pixels=pixels[:,:,::-1].copy()
                        seen.add(name);yield name,stamp,pixels
        def write(name,pixels,index):
            container,stream=writers[name]
            frame=av.VideoFrame.from_ndarray(pixels,format='rgb24');frame.pts=index;frame.time_base=Fraction(1,fps)
            for packet in stream.encode(frame):container.mux(packet)
        for _,images in timed_samples(samples(),fps):
            panels=[]
            for name in names:
                panel=np.zeros((height,width,3),np.uint8);pixels=images[name]
                if pixels is not None:
                    h,w=pixels.shape[:2];scale=min(width/w,height/h)
                    rw,rh=max(1,round(w*scale)),max(1,round(h*scale))
                    resized=av.VideoFrame.from_ndarray(np.ascontiguousarray(pixels),format='rgb24').reformat(width=rw,height=rh).to_ndarray(format='rgb24')
                    panel[(height-rh)//2:(height-rh)//2+rh,(width-rw)//2:(width-rw)//2+rw]=resized
                write(name,panel,report['frames']);panels.append(panel)
            write('three_cameras',np.concatenate(panels,axis=1),report['frames']);report['frames']+=1
        if seen!=set(names) or not report['frames']:raise RuntimeError('Preview missing cameras: '+str(set(names)-seen))
        report['status']='COMPLETE'
    except Exception as error:
        report.update(status='FAILED',error=str(error));raise
    finally:
        close_error=None
        for container,stream in writers.values():
            try:
                for packet in stream.encode():container.mux(packet)
            except Exception as error:close_error=close_error or error
            finally:
                try:container.close()
                except Exception as error:close_error=close_error or error
        if close_error is not None:report.update(status='FAILED',error=str(close_error))
        write_json(directory/'video_export.json',report)
        if close_error is not None:raise close_error
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    export(args.directory)


if __name__ == "__main__":
    main()
