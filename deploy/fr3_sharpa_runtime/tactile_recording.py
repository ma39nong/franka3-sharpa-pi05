"""Lossless tactile telemetry on the already-owned SDK handles; never sends commands."""
from array import array
import math
import queue
import threading
import time

import numpy as np

FINGERS=('thumb','index','middle','ring','pinky')
MANUAL='https://sharpa-robotics.github.io/sharpa-docs/#tactile-sensing'
FRAMES='https://github.com/sharpa-robotics/sharpa-urdf-usd-xml#model-description'
# Verified with both installed Wave hands, SDK 5.0.6, 2026-11-28.
IMAGE_SIGNALS={'RAW':('raw',(240,320)), 'DEFORM':('deformation',(240,240))}
# Preserve additional arrays if this SDK/device exposes them. Sparse contacts
# are conditional. Distributed variants require a verified descriptor before use.


def channel_index(side,finger):
    index=FINGERS.index(finger)
    return 9-index if side=='left' else index


def channels():
    result=[]
    for side in ('left','right'):
        for finger in FINGERS:
            base=f'/sharpa/{side}/tactile/{finger}'
            target=f'/devices/sharpa_{side}/tactile/{finger}'
            frame=f'{side}_{finger}_elastomer'
            common=dict(recording='enabled',value_origin='measured',frame_id=frame,
                timestamp_source='SDK frame.ts Unix seconds converted to integer nanoseconds',
                timestamp_limitation='SDK timestamp; physical exposure and device-host synchronization not independently certified',
                nominal_rate_hz=30,source_reference=MANUAL,coordinate_reference=FRAMES,
                axes='manufacturer right-handed elastomer frame; no axis permutation or sign change')
            for key,(name,shape) in IMAGE_SIGNALS.items():
                result.append(dict(common,source_topic=base+'/'+name,topic=target+'/image/'+name,
                    type='sensor_msgs/msg/Image',sdk_signal=key,encoding='mono8',native_dtype='uint8',
                    value_origin='measured' if key=='RAW' else 'derived',
                    native_shape=list(shape),height=shape[0],width=shape[1],dtype='uint8',unit='SDK grayscale code',compression='lossless MCAP Chunk Zstd',
                    upstream_codec='SDK JPEG transport decoded by SDK' if key=='RAW' else 'SDK native deformation image',
                    grid_axes=['row','column'],sdk_layout='leading batch singleton omitted; pixel order unchanged',
                    value_mapping='identity for RAW; nonlinear SDK deform_map_value(code) in mm for DEFORM'))
            result.append(dict(common,source_topic=base+'/wrench',topic=target+'/wrench',
                type='geometry_msgs/msg/WrenchStamped',sdk_signal='F6',value_origin='estimated',
                components=['Fx','Fy','Fz','Tx','Ty','Tz'],units=['N','N','N','N*m','N*m','N*m'],
                unit={'force':'N','torque':'N*m'},dtype='float64',
                native_dtype='float64 as exposed by installed Python SDK; underlying firmware dtype not exposed',
                torque_origin='thumb IP joint axis center' if finger=='thumb' else 'DIP joint axis center',
                conversion='identity; SDK engineering units, no raw-count conversion'))
            result.append(dict(common,source_topic=base+'/sample_info',topic=f'/recording/tactile/{side}/{finger}/sample_info',
                type='std_msgs/msg/UInt64MultiArray',adapter='tactile_sample_info',
                components=['sdk_frame_index','sdk_timestamp_ns','host_sdk_read_timestamp_ns'],
                units=['count','ns','ns'],native_dtype='uint64',value_origin='derived'))
            result.append(dict(common,source_topic=base+'/contact_point',topic=target+'/array/contact_point',
                type='sensor_msgs/msg/Image',sdk_signal='CONTACT_POINT',value_origin='estimated',recording='event_driven',presence_policy='conditional',
                native_dtype='preserved from SDK array; Image.encoding declares actual dtype',
                native_shape='SDK shape, stored in Image height/width/channels without resampling',
                components='SDK columns in original order; first two are deformation-map column,row',
                unit='pixel for column,row; any additional SDK components remain uninterpreted',
                semantics='raw SDK contact-point array, not a 3D point or a force measurement'))
    return result


def image_fields(value,shape=None):
    """Standard Image fields, preserving native numeric bytes, including NaN bits."""
    value=np.asarray(value)
    if value.dtype.kind not in 'uif' or value.dtype.itemsize not in (1,2,4,8):
        raise ValueError(f'Unsupported tactile dtype {value.dtype}; refusing lossy conversion')
    if shape is not None:
        if value.size!=math.prod(shape):raise ValueError('SDK tactile shape/byte count mismatch')
        value=value.reshape(shape)
    if value.ndim==1:value=value.reshape(1,-1)
    if value.ndim not in (2,3) or not value.size:raise ValueError('Expected nonempty tactile 2D/3D array')
    h,w=value.shape[:2];n=value.shape[2] if value.ndim==3 else 1
    dtype=value.dtype
    code={('u',1):'8U',('i',1):'8S',('u',2):'16U',('i',2):'16S',('i',4):'32S',('f',4):'32F',('f',8):'64F'}.get((dtype.kind,dtype.itemsize))
    if code is None:raise ValueError(f'No standard ROS Image encoding for {dtype}')
    contiguous=np.ascontiguousarray(value)
    return dict(height=h,width=w,encoding='mono8' if code=='8U' and n==1 else f'{code}C{n}',
        is_bigendian=dtype.byteorder=='>' or (dtype.byteorder=='=' and not np.little_endian),
        step=w*n*dtype.itemsize,data=contiguous.tobytes())


def native_shape(payload,key):
    shape=payload.get('shape',{}).get(key)
    if shape is None:return None
    if isinstance(shape,(list,tuple)):return tuple(int(x) for x in shape)
    return tuple(int(shape[i]) for i in range(shape.dim()))


def sdk_timestamp_ns(payload,received_ns):
    value=float(payload['ts'])
    if not math.isfinite(value) or value<=0:raise ValueError('Invalid SDK tactile timestamp')
    stamp=round(value*1_000_000_000)
    # Detect clock-domain mistakes instead of relabeling host time as capture time.
    if abs(stamp-received_ns)>2_000_000_000:
        raise ValueError('SDK tactile timestamp is outside host Unix clock domain (>2 s); verify clock basis')
    return stamp


def decode_payload(payload,side,finger,received_ns):
    if int(payload['channel'])!=channel_index(side,finger):raise ValueError('SDK tactile finger/channel mismatch')
    stamp=sdk_timestamp_ns(payload,received_ns)
    frame=int(payload['frame_id'])
    if frame<0:raise ValueError('Invalid SDK tactile frame index')
    content=payload['content']
    extras=[k for k,v in content.items() if k not in {'RAW','DEFORM','F6','CONTACT_POINT'} and v is not None and np.asarray(v).size]
    if extras:raise ValueError('New enabled SDK tactile outputs require recording descriptors: '+','.join(extras))
    messages={}
    for key,(name,shape) in IMAGE_SIGNALS.items():
        if content.get(key) is None:raise ValueError('Required SDK tactile output missing: '+key)
        value=np.asarray(content[key])
        if value.dtype!=np.uint8:raise ValueError(f'{key} changed native dtype to {value.dtype}; update descriptor, no uint8 quantization performed')
        declared=native_shape(payload,key)
        if declared is not None:
            while len(declared)>2 and declared[0]==1:declared=declared[1:]
            if declared!=shape:raise ValueError(f'{key} changed SDK native shape to {declared}; update descriptor, no resizing performed')
        messages[name]=image_fields(value,shape)
    wrench=np.asarray(content['F6'])
    if wrench.dtype!=np.float64 or wrench.size!=6 or not np.isfinite(wrench).all():raise ValueError('Invalid native SDK F6')
    messages['wrench']=wrench.reshape(-1).tolist()
    contact=content.get('CONTACT_POINT')
    if contact is not None and np.asarray(contact).size:
        shape=native_shape(payload,'CONTACT_POINT')
        if shape is not None:
            while len(shape)>3 and shape[0]==1:shape=shape[1:]
        messages['contact_point']=image_fields(contact,shape)
    return stamp,frame,messages


class TactileTelemetry:
    """Bounded FIFO, separate polling/serialization threads, no coalescing for recording."""
    def __init__(self,node,hands,*,poll_hz=60):
        from sensor_msgs.msg import Image
        from geometry_msgs.msg import WrenchStamped
        from std_msgs.msg import UInt64MultiArray,MultiArrayDimension
        self.types=Image,WrenchStamped,UInt64MultiArray,MultiArrayDimension
        self.node=node;self.hands=hands;self.interval=1/poll_hz;self.stop_event=threading.Event()
        self.queue=queue.Queue(64);self.error=None;self.previous={};self.sequences={}
        self.publishers={}
        for ch in channels():
            kind=Image if ch['type'].endswith('/Image') else WrenchStamped if ch['type'].endswith('/WrenchStamped') else UInt64MultiArray
            self.publishers[ch['source_topic']]=node.create_publisher(kind,ch['source_topic'],30)
        self.poll_thread=threading.Thread(target=self.poll,name='tactile-sdk-read',daemon=True)
        self.write_thread=threading.Thread(target=self.write,name='tactile-ros-publish',daemon=True)

    def start(self):
        self.write_thread.start();self.poll_thread.start()

    def fail(self,error):
        self.error=str(error);self.stop_event.set()
        self.node.get_logger().error('Tactile recording failed: '+str(error))
        # Required streams go stale and existing recorder health aborts the run.

    def poll(self):
        try:
            while not self.stop_event.is_set():
                started=time.monotonic()
                for side,hand in self.hands.items():
                    if not hand.tactile_available:raise RuntimeError(f'{side} SDK tactile unavailable')
                    for finger in FINGERS:
                        payload=hand._wave.fetch_tactile_frame(channel_index(side,finger),0.)
                        if payload is None:continue
                        received=time.time_ns();key=(side,finger);sequence=payload['frame_id']
                        if self.sequences.get(key)==sequence:continue
                        self.sequences[key]=sequence
                        owned=dict(payload,content={k:np.array(v,copy=True) if v is not None else None
                                                   for k,v in payload['content'].items()})
                        try:self.queue.put_nowait((side,finger,owned,received))
                        except queue.Full:raise RuntimeError('Tactile FIFO overflow; no frames silently discarded')
                self.stop_event.wait(max(0,self.interval-(time.monotonic()-started)))
        except Exception as error:self.fail(error)

    def write(self):
        try:
            while not self.stop_event.is_set():
                try:item=self.queue.get(timeout=.05)
                except queue.Empty:continue
                side,finger,payload,received=item
                stamp,index,values=decode_payload(payload,side,finger,received)
                key=(side,finger)
                if stamp<=self.previous.get(key,0):raise ValueError('Nonmonotonic SDK tactile timestamp')
                self.previous[key]=stamp
                Image,Wrench,Info,Dim=self.types
                for signal,value in values.items():
                    msg=Wrench() if signal=='wrench' else Image()
                    msg.header.stamp.sec=stamp//10**9;msg.header.stamp.nanosec=stamp%10**9
                    msg.header.frame_id=f'{side}_{finger}_elastomer'
                    if signal=='wrench':
                        for xyz,v in zip(('force.x','force.y','force.z','torque.x','torque.y','torque.z'),value):
                            group,axis=xyz.split('.');setattr(getattr(msg.wrench,group),axis,v)
                    else:
                        for field,data in value.items():setattr(msg,field,array('B',data) if field=='data' else data)
                    self.publishers[f'/sharpa/{side}/tactile/{finger}/{signal}'].publish(msg)
                info=Info();info.data=[index,stamp,received]
                dim=Dim();dim.label='sdk_frame_index,sdk_timestamp_ns,host_sdk_read_timestamp_ns';dim.size=3;dim.stride=3
                info.layout.dim=[dim]
                self.publishers[f'/sharpa/{side}/tactile/{finger}/sample_info'].publish(info)
        except Exception as error:self.fail(error)

    def close(self):
        self.stop_event.set()
        for thread in (self.poll_thread,self.write_thread):
            if thread.ident:thread.join(timeout=2)
        if any(t.is_alive() for t in (self.poll_thread,self.write_thread)):
            raise RuntimeError('Tactile workers failed to stop before SDK shutdown')
