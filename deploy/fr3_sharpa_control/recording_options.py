"""Optional sensor channels shared by both recording backends."""
import os


def optional_channels():
    channels=[]
    if os.environ.get('SHARPA_RECORD_HEAD_DEPTH')=='1':
        channels.extend([
            dict(source_topic='/cam0/depth/image_raw',topic='/devices/cam0/cameras/depth/depth/image_raw/compressedDepth',
                 type='sensor_msgs/msg/CompressedImage',source_type='sensor_msgs/msg/Image',adapter='depth_rvl',
                 recording='enabled',unit='mm',invalid_policy='zero_invalid',width=640,height=400),
            dict(source_topic='/cam0/depth/camera_info',topic='/devices/cam0/cameras/depth/depth/camera_info',
                 type='sensor_msgs/msg/CameraInfo',recording='enabled')])
    if os.environ.get('SHARPA_RECORD_TACTILE')=='1':
        for side in ('left','right'):
            for finger in ('thumb','index','middle','ring','pinky'):
                for signal,kind in [('raw','sensor_msgs/msg/Image'),('deformation','sensor_msgs/msg/Image'),('wrench','geometry_msgs/msg/WrenchStamped')]:
                    channels.append(dict(source_topic=f'/sharpa/{side}/tactile/{finger}/{signal}',
                        topic=f'/devices/sharpa_{side}/tactile/{finger}/'+('wrench' if signal=='wrench' else 'image/'+signal),
                        type=kind,recording='enabled',value_origin='measured'))
    return channels


def enable_rosbag_channels():
    from . import recorder_ros as recorder
    for ch in optional_channels():
        recorder.TOPICS[ch['source_topic']]=ch.get('source_type',ch['type'])
    recorder.ALL_TOPICS={**recorder.TOPICS,**recorder.OPTIONAL_TOPICS}
    recorder.SENSORS=frozenset(recorder.TOPICS)-{'/teleop/arm_commands','/openpi/executed_joint_targets'}


def depth_payload(message):
    """Lossless ROS compressedDepth ConfigHeader + dimensions + Wilson RVL."""
    import struct
    import numpy as np
    if message.encoding!='16UC1':
        raise ValueError(f'Expected native millimetre uint16 depth, got {message.encoding}; no quantization allowed')
    dt='>u2' if message.is_bigendian else '<u2'
    rows=np.frombuffer(bytes(message.data),dtype=dt).reshape(message.height,message.step//2)
    pixels=np.ascontiguousarray(rows[:,:message.width], dtype=np.uint16).reshape(-1)
    import ctypes
    from pathlib import Path
    library=ctypes.CDLL(str(Path(__file__).resolve().parents[2]/'.deployment/mcap-rvl.so'))
    encode=library.sharpa_rvl
    encode.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p];encode.restype=ctypes.c_size_t
    output=np.empty(pixels.size*2+32,dtype='<u4')
    length=encode(pixels.ctypes.data,pixels.size,output.ctypes.data)
    return struct.pack('<iffII',0,0.,0.,message.width,message.height)+output.tobytes()[:length]



def main():
    import argparse
    from pathlib import Path
    import yaml
    parser=argparse.ArgumentParser();parser.add_argument('--depth-profile',required=True,type=Path);args=parser.parse_args()
    profile=yaml.safe_load(args.depth_profile.read_text());profile.update(enable_depth=True,depth_width=640,depth_height=400,depth_fps=20,depth_format='Y16')
    output=Path(__file__).resolve().parents[2]/'.deployment/camera_cam0_record_depth.yaml'
    output.write_text(yaml.safe_dump(profile,sort_keys=False));print(output)


if __name__=='__main__':main()
