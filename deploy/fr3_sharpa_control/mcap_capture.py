"""Direct MCAP capture. Profile is explicit so missing calibration never becomes invented data."""
import json
import os
from pathlib import Path
import queue
import signal
import socketserver
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import uuid
import yaml

from .recording import write_json
from .mcap_storage import McapEpisode, VIDEO_TYPE
from .mcap_video import VideoEncoder


def channel_role(channel):
    """Presence requirements are independent of storage or message frequency."""
    source = channel['source_topic']
    if channel.get('snapshot') or source == '/tf_static' or channel['type'] == 'sensor_msgs/msg/CameraInfo':
        return 'snapshot'
    if source in ('/dagger/frame', '/openpi/executed_joint_targets',
                  '/teleop/arm_commands', '/teleop/validated_arm_commands'):
        return 'action'
    if source == '/tf' or channel['type'] == 'std_msgs/msg/String' or source == '/teleop/arm_command_status':
        return 'event'
    return 'sensor'


def load_profile():
    path = Path(os.environ.get('SHARPA_MCAP_PROFILE', str(Path(__file__).resolve().parents[2]/'.deployment/mcap-recording.yaml')))
    if not path.is_file():
        raise ValueError(f'MCAP配置缺失：{path}；请参考 deploy/fr3_sharpa_control/mcap-profile.example.yaml')
    profile = yaml.safe_load(path.read_text())
    for key in ('model_root','calibration_root'):
        source = Path(profile[key]).expanduser()
        if not source.is_absolute(): source = path.parent/source
        source=source.resolve()
        if not source.is_dir() or not any(p.is_file() for p in source.rglob('*')):
            raise ValueError(f'MCAP真实资产缺失：{key}={source}')
        profile[key]=str(source)
    import xml.etree.ElementTree as ET
    for model in Path(profile['model_root']).rglob('*'):
        if model.suffix in ('.urdf', '.xml', '.sdf'):
            for element in ET.parse(model).iter():
                reference=element.attrib.get('filename') or (element.text.strip() if element.tag=='uri' and element.text else None)
                if reference and (Path(reference).is_absolute() or '://' in reference or not (model.parent/reference).is_file()):
                    raise ValueError(f'MCAP模型依赖未物化为有效相对路径：{model}: {reference}')
    if not profile.get('clock_mapping') or profile['clock_mapping'].get('valid_from_ns',0)<=0 or profile['clock_mapping'].get('sync_state') not in ('locked', 'sdk_mapped'):
        raise ValueError('MCAP requires measured clock_mapping configuration')
    from .recording_options import optional_channels
    profile['devices'].append(dict(device_id='optional_sensors', channels=optional_channels()))
    channels=[ch for d in profile['devices'] for ch in d['channels'] if ch.get('recording') in ('enabled','event_driven')]
    sources=[ch['source_topic'] for ch in channels]
    if len(sources)!=len(set(sources)): raise ValueError('Duplicate MCAP source topic')
    from .recorder_ros import TOPICS
    missing=set(TOPICS)-set(sources)
    if missing: raise ValueError(f'MCAP profile missing current channels: {sorted(missing)}')
    if not profile.get('session_id') or str(profile['session_id']).startswith('REQUIRED'):
        raise ValueError('MCAP requires a real session_id')
    for ch in channels:
        if ch['type']==VIDEO_TYPE:
            expected=18 if ch['role']=='head' else 16
            if ch['crf']!=expected or ch['width']%2 or ch['height']%2:
                raise ValueError('MCAP camera CRF/native dimensions invalid')
            if ch.get('color') != dict(range='limited',matrix='bt709',primaries='bt709',transfer='iec61966-2-1') or not ch.get('color_source') or ch['color_source'].startswith('REQUIRED'):
                raise ValueError('MCAP camera requires an explicit full-range sRGB input interpretation/source and declared output color')
    return profile, channels


def native(message):
    if hasattr(message,'get_fields_and_field_types'):
        return {k:native(getattr(message,k)) for k in message.get_fields_and_field_types()}
    if isinstance(message,(str,int,float,bool,bytes)): return message
    return [native(v) for v in message]


def stamp_ns(message):
    stamp=message.header.stamp
    return stamp.sec*1_000_000_000+stamp.nanosec


def recording_event(data, receipt_ns):
    """Adapt legacy DAgger events without confusing control and event phases."""
    event=json.loads(data)
    if not isinstance(event,dict):
        raise ValueError('MCAP recording event must be a JSON object')
    legacy=not event.get('event_type')
    if legacy:
        if not isinstance(event.get('event'),str) or not event['event'].strip():
            raise ValueError('MCAP recording event missing event_type/event')
        event['event_type']=event['event']
        # Legacy phase describes the controller, not an event lifecycle.
        if 'phase' in event:
            context=dict(event.get('context') or {})
            context['phase']=event['phase']
            event['context']=context
        event['phase']='instant'
    event.setdefault('phase','instant')
    event.setdefault('severity','info')
    event.setdefault('event_id',uuid.uuid4().hex)
    event.setdefault('stamp_ns',event.get('source_timestamp_ns',receipt_ns))
    if not isinstance(event['event_type'],str) or not event['event_type'].strip():
        raise ValueError('MCAP recording event_type must be a nonempty string')
    if event['phase'] not in ('start','end','instant'):
        raise ValueError('MCAP recording event has invalid lifecycle phase')
    if event['severity'] not in ('info','warning','error','fatal'):
        raise ValueError('MCAP recording event has invalid severity')
    return dict(data=json.dumps(event,ensure_ascii=False))


class Capture:
    def __init__(self, directory, profile, channels, *, serialize=None):
        from copy import deepcopy
        self.directory=Path(directory);self.profile=deepcopy(profile);self.channels=channels;self.serialize=serialize
        # Archive sparse channels as conditional, so zero events are not declared
        # to be missing enabled sensors by downstream conformance checks either.
        for device in self.profile['devices']:
            for channel in device['channels']:
                if channel.get('recording') in ('enabled','event_driven') and channel_role(channel) == 'event':
                    channel.update(recording='event_driven', presence_policy='conditional')
        self.finished=threading.Event();self.lock=threading.RLock();self.active=False
        self.error=None;self.writer=None;self.receipts={};self.static={};self.encoders={};self.video_workers={}
        self.queue=queue.Queue(maxsize=256);self.worker=None;self.started=None;self.close_lock=threading.Lock();self.last_context=None
        self.accepted_counts={ch['topic']:0 for ch in channels};self.queue_high_watermark=0
        self.report={'status':'PREPARED','recording_format':'mcap','started_unix_ns':None}
        self.persist()

    def persist(self):
        if self.writer is not None:
            self.report['payload_directory']=str(self.writer.target if self.writer.target.exists() else self.writer.directory)
        write_json(self.directory/'recording.json',dict(self.report,error=self.error,
            accepted_counts=dict(self.accepted_counts),queue_high_watermark=self.queue_high_watermark,
            counts={} if self.writer is None else {k:v['message_count'] for k,v in self.writer.stats.items()}))

    def fail(self,error):
        self.error=str(error);self.active=False;self.finished.set();self.persist()

    def accept(self,ch,message,rx):
        with self.lock:
            previous=self.receipts.get(ch['source_topic'])
            self.receipts[ch['source_topic']]=(time.monotonic(),1+(previous[1] if previous else 0))
            if channel_role(ch) == 'snapshot': self.static[ch['source_topic']]=(message,rx)
            if not self.active:return
            try:
                prepared = (self.video_workers[ch['topic']].submit(self.encode_video,ch,message)
                            if ch['type']==VIDEO_TYPE else None)
                self.queue.put_nowait((ch,message,rx,prepared))
                self.accepted_counts[ch['topic']]+=1
                self.queue_high_watermark=max(self.queue_high_watermark,self.queue.qsize())
            except queue.Full:self.fail('MCAP encoding queue overflow')

    def health(self):
        now=time.monotonic()
        missing, stale, insufficient = [], [], []
        with self.lock:
            for ch in self.channels:
                source = ch['source_topic']; role = channel_role(ch)
                if role == 'snapshot':
                    if source not in self.static: missing.append(source)
                elif role == 'sensor':
                    receipt = self.receipts.get(source)
                    if receipt is None: missing.append(source)
                    else:
                        if not 0 <= now-receipt[0] < .2: stale.append(source)
                        if receipt[1] < 2: insufficient.append(source)
        return dict(ready=not (missing or stale or insufficient), missing=missing,
                    stale=stale, insufficient_progress=insufficient)

    def readiness(self):
        return self.health()['ready']

    def prepare_resources(self):
        """Warm schemas, assets and encoders only; accept no data before the start gate."""
        if self.writer is not None:
            return
        quality=json.loads((self.directory/'data_quality.json').read_text())
        storage=quality.get('archive')
        target=(self.directory/'mcap').resolve()
        root=Path(storage['root']) if storage else self.directory
        run=json.loads((self.directory/'run.json').read_text())
        identity=dict(episode_id=storage['episode'] if storage else self.directory.name,
            episode_uid=uuid.uuid4().hex,session_id=self.profile['session_id'],recording_id=quality['run_id'],
            task_id=run.get('settings',{}).get('task_mode','deployment'),
            task_description=run.get('settings',{}).get('prompt') or 'Robot deployment',
            source_type='dagger',source_id=quality['run_id'],end_reason='normal stop')
        self.writer=McapEpisode(target,root=root,profile=self.profile,identity=identity,start_time_ns=0)
        for ch in self.channels:
            self.writer.register(ch['topic'],ch['type'])
            if ch['type']==VIDEO_TYPE:
                self.encoders[ch['topic']]=VideoEncoder(ch)
                self.video_workers[ch['topic']]=ThreadPoolExecutor(max_workers=1,thread_name_prefix='mcap-'+str(len(self.video_workers)))

    def request(self,operation):
        if self.error:raise RuntimeError(self.error)
        if operation=='status': return dict(status='PASS',active=self.active,sensor_readiness=self.health())
        if operation=='start':
            with self.lock:
                if self.finished.is_set():raise RuntimeError('Cannot reuse stopped MCAP recorder')
                if self.active:return dict(status='PASS',active=True)
                health = self.health()
                if not health['ready']:raise RuntimeError('MCAP channels/CameraInfo/TF snapshot not ready: '+json.dumps(health))
                if self.writer is None:self.prepare_resources()
                # Recheck after preparation, as rosbag does before activation.
                if not self.readiness():raise RuntimeError('MCAP sensors changed during preparation')
                self.started=time.time_ns();self.writer.start=self.started
                for ch in self.channels:
                    if channel_role(ch) == 'snapshot':
                        message,rx=self.static[ch['source_topic']]
                        self.queue.put_nowait((ch,message,self.started,None))
                        self.accepted_counts[ch['topic']]+=1
                self.active=True
                self.report.update(status='RECORDING',started_unix_ns=self.started)
                self.worker=threading.Thread(target=self.work,daemon=True);self.worker.start();self.persist()
                return dict(status='PASS',active=True)
        if operation=='stop':self.close();return dict(status='PASS',active=False)
        raise ValueError(operation)

    def encode_video(self, channel, message):
        """One FIFO encoder worker per camera; MCAP writes still follow receipt order."""
        stamp=stamp_ns(message)
        return dict(timestamp=dict(sec=stamp//10**9,nanosec=stamp%10**9),
                    frame_id=message.header.frame_id,format='h264',
                    data=self.encoders[channel['topic']].encode(message,stamp))

    def work(self):
        try:
            while True:
                item=self.queue.get()
                if item is None:break
                ch,msg,rx,prepared=item;topic=ch['topic'];kind=ch['type']
                if ch.get('adapter')=='depth_rvl':
                    from .recording_options import depth_payload
                    stamp=stamp_ns(msg)
                    value=dict(header=native(msg.header),format='16UC1; compressedDepth rvl',data=depth_payload(msg))
                elif ch.get('adapter')=='dagger_ownership':
                    frame=json.loads(msg.data);stamp=int(frame['source_timestamp_ns'])
                    value=dict(layout=dict(dim=[],data_offset=0),data=[{'hold':0,'policy':1,'human':2}[owner] for owner in frame['action_owner']])
                    context={key:frame.get(key) for key in ('mode','phase','generation','intervention_id','stage','prompt','model_version')}
                    if context!=self.last_context:
                        event=dict(event_id=uuid.uuid4().hex,event_type='control_context',phase='instant',severity='info',
                            stamp_ns=stamp,device_id='rig',topic=topic,error_code='',description='DAgger control context changed',context=context)
                        with (self.writer.directory/'events.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
                        self.last_context=context
                elif kind==VIDEO_TYPE:
                    stamp=stamp_ns(msg)
                    value=prepared.result()
                elif ch.get('adapter')=='arm_command':
                    value=dict(header=native(msg.header),name=list(msg.joint_names),position=list(msg.positions),velocity=[],effort=[])
                    stamp=stamp_ns(msg)
                elif kind=='std_msgs/msg/String':
                    value=(recording_event(msg.data,rx) if topic=='/recording/events' else
                           {'data':json.dumps(native(msg))} if ch.get('adapter')=='diagnostic_event' else
                           None if self.serialize is not None else native(msg));stamp=rx
                elif kind=='tf2_msgs/msg/TFMessage':
                    value=None if self.serialize is not None else native(msg);stamp=rx
                else:value=None;stamp=stamp_ns(msg)
                if self.serialize is not None and not ch.get('adapter') and kind!=VIDEO_TYPE and topic!='/recording/events':
                    self.writer.write_serialized(topic,self.serialize(msg),log_time=rx,publish_time=stamp)
                else:
                    if value is None:value=native(msg)
                    self.writer.write(topic,value,log_time=rx,publish_time=stamp)
        except Exception as error:self.fail(error)

    def close(self):
        with self.close_lock:
            return self._close()

    def _close(self):
        with self.lock:self.active=False;self.finished.set()
        if self.worker and self.worker.is_alive():
            self.queue.put(None,timeout=5);self.worker.join(timeout=30)
            if self.worker.is_alive():raise RuntimeError('MCAP writer failed to close')
        for worker in self.video_workers.values():worker.shutdown(wait=True)
        if self.writer and not self.writer.closed and self.started is None:
            for encoder in self.encoders.values():encoder.close()
            self.writer.cancel_prepared()
        if self.writer and not self.writer.closed:
            try:
                for encoder in self.encoders.values():encoder.close()
                missing = [ch['source_topic'] for ch in self.channels if channel_role(ch) != 'event'
                           and not self.writer.stats[ch['topic']]['message_count']]
                if not self.error and missing:
                    self.error='MCAP required channel contains no samples: '+', '.join(missing)
                mismatch={topic:(count,self.writer.stats[topic]['message_count'])
                          for topic,count in self.accepted_counts.items() if count!=self.writer.stats[topic]['message_count']}
                if not self.error and mismatch:self.error='MCAP accepted/written message count mismatch: '+str(mismatch)
                self.writer.close(self.error)
                self.report['status']='FAILED' if self.error else 'COMPLETE'
            except Exception as error:
                self.error=str(error);self.report['status']='FAILED'
        elif not self.started:self.report['status']='FAILED' if self.error else 'NOT_STARTED'
        self.report['ended_unix_ns']=time.time_ns();self.persist()


def main(args):
    profile,channels=load_profile()
    if args.check:
        from mcap_ros2._dynamic import serialize_dynamic
        from rosbags.typesys import Stores,get_typestore
        for ch in channels:
            if ch.get('adapter')=='depth_rvl':
                import ctypes
                ctypes.CDLL(str(Path(__file__).resolve().parents[2]/'.deployment/mcap-rvl.so'))
            if ch['type']==VIDEO_TYPE:
                encoder=VideoEncoder(ch);encoder.close()
            else:get_typestore(Stores.ROS2_HUMBLE).generate_msgdef(ch['type'],ros_version=2)
        print('MCAP dependencies/profile checked; no ROS node/hardware started');return
    import rclpy
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
    from rosidl_runtime_py.utilities import get_message
    rclpy.init();node=rclpy.create_node('sharpa_mcap_recorder')
    from rclpy.serialization import serialize_message
    capture=Capture(args.directory,profile,channels,serialize=serialize_message)
    try:capture.prepare_resources()
    except BaseException:
        capture.close();node.destroy_node();rclpy.shutdown();raise
    subscriptions=[]
    for ch in channels:
        qos=QoSProfile(depth=30,reliability=ReliabilityPolicy.BEST_EFFORT)
        if ch.get('snapshot'):qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        subscriptions.append(node.create_subscription(get_message(ch.get('source_type',ch['type'])),ch['source_topic'],
            lambda msg,ch=ch:capture.accept(ch,msg,node.get_clock().now().nanoseconds),qos))
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:reply=capture.request(json.loads(self.rfile.readline(65536))['operation'])
            except Exception as error:reply=dict(status='FAIL',error=str(error))
            self.wfile.write(json.dumps(reply).encode()+b'\n')
    server=socketserver.ThreadingUnixStreamServer(str(args.socket),Handler);server.daemon_threads=True
    args.socket.chmod(0o600);threading.Thread(target=server.serve_forever,daemon=True).start()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:capture.finished.set())
    try:
        while rclpy.ok() and not capture.finished.is_set():
            rclpy.spin_once(node,timeout_sec=.05)
            if (args.socket.parent/'stop-recording').exists():break
            if capture.active:
                health = capture.health()
                if not health['ready']:capture.fail('MCAP enabled channel disconnected/stale: '+json.dumps(health))
    finally:
        capture.close();server.shutdown();server.server_close();args.socket.unlink(missing_ok=True)
        node.destroy_node();rclpy.shutdown()
