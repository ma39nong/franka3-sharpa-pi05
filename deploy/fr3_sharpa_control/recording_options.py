"""Optional sensor channels shared by both recording backends."""
import os


def optional_channels():
    channels=[]
    cameras=[]
    if os.environ.get('SHARPA_RECORD_HEAD_DEPTH')=='1':cameras.append('cam0')
    # Legacy launch environments used HEAD_DEPTH for all three cameras.
    if os.environ.get('SHARPA_RECORD_WRIST_DEPTH',os.environ.get('SHARPA_RECORD_HEAD_DEPTH'))=='1':
        cameras.extend(('cam1','cam2'))
    for camera in cameras:
        channels.extend([
                dict(source_topic=f'/{camera}/depth/image_raw',
                     topic=f'/devices/{camera}/cameras/depth/depth/image_raw/compressedDepth',
                     type='sensor_msgs/msg/CompressedImage',source_type='sensor_msgs/msg/Image',adapter='depth_rvl',
                     recording='enabled',unit='mm',invalid_policy='zero_invalid'),
                dict(source_topic=f'/{camera}/depth/camera_info',
                     topic=f'/devices/{camera}/cameras/depth/depth/camera_info',
                     type='sensor_msgs/msg/CameraInfo',recording='enabled')])
    if os.environ.get('SHARPA_RECORD_TACTILE')=='1':
        from deploy.fr3_sharpa_runtime.tactile_recording import channels as tactile_channels
        channels.extend(tactile_channels())
    return channels


def enable_rosbag_channels():
    from . import recorder_ros as recorder
    for ch in optional_channels():
        target=recorder.OPTIONAL_TOPICS if ch.get('presence_policy')=='conditional' else recorder.TOPICS
        target[ch['source_topic']]=ch.get('source_type',ch['type'])
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



def depth_profile(source, camera):
    """Preserve RGB settings and select verified native depth profiles."""
    import yaml
    profile=yaml.safe_load(source.read_text())
    if camera=='cam0':
        profile.update(enable_depth=True,depth_width=640,depth_height=400,depth_fps=20,depth_format='Y16')
    elif camera in ('cam1','cam2'):
        # Both Gemini 305 serials verified live: native 640x480 Y16 at 30 Hz.
        profile.update(enable_depth=True,depth_width=640,depth_height=480,depth_fps=30,depth_format='Y16')
    else:
        raise ValueError(f'Unknown camera: {camera}')
    return profile


def main():
    import argparse
    from pathlib import Path
    import yaml
    parser=argparse.ArgumentParser()
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--depth-profile',type=Path)
    source.add_argument('--deployment-config',type=Path)
    parser.add_argument('--camera',choices=('cam0','cam1','cam2'),default='cam0')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.deployment_config:
        profile=yaml.safe_load(args.deployment_config.read_text())
        profile['camera_bringup']['cam1']['config_file']='camera_cam1_global_params.yaml'
        profile['camera_bringup']['cam2']['config_file']='camera_secondary_params.yaml'
        filename='cameras_record_depth.yaml'
    else:
        profile=depth_profile(args.depth_profile,args.camera)
        filename=f'camera_{args.camera}_record_depth.yaml'
    output=args.output or Path(__file__).resolve().parents[2]/'.deployment'/filename
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(yaml.safe_dump(profile,sort_keys=False));print(output)


if __name__=='__main__':main()
