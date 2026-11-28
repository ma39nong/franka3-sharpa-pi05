"""Real tactile-only ROS -> MCAP test; no joint commands, enable, tare or coefficient writes."""
import os,argparse,socket
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--hardware',action='store_true',help='Read tactile sensors only; never enables joints or sends robot targets')
args=parser.parse_args()
os.environ['ROS_DOMAIN_ID']='169';os.environ['ROS_LOCALHOST_ONLY']='1'
import hashlib,json,time,threading,struct
from collections import defaultdict,Counter
from pathlib import Path
from types import SimpleNamespace
import yaml,rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import serialize_message
from rosidl_runtime_py.utilities import get_message
from mcap.reader import make_reader
from litchi_hardware.hardware.sharpa.vendor import load_sharpa_sdk
from deploy.fr3_sharpa_runtime.tactile_recording import TactileTelemetry,channels
from deploy.fr3_sharpa_control.mcap_capture import Capture
from deploy.fr3_sharpa_control.quality import begin
from deploy.fr3_sharpa_control.recording import write_json
root=Path(__file__).resolve().parents[3];folder=root/'.deployment/validation/tactile-audit'/('live-'+str(time.time_ns()));folder.mkdir()
run=folder/'run';run.mkdir();begin(run,archive_parent=folder/'archive',recording_format='mcap');write_json(run/'run.json',{'settings':{'task_mode':'tactile_telemetry_validation_no_motion'}})
profile=yaml.safe_load((root/'.deployment/mcap-recording.yaml').read_text());chs=channels();profile['devices']=[dict(device_id='tactile_validation',channels=chs)]
if args.hardware:
 # Refuse before SDK discovery/connect if an existing owner occupies either port.
 for port in (50001,50021):
  with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as check:check.bind(('0.0.0.0',port))
 sdk=load_sharpa_sdk();manager=sdk.SharpaWaveManager.get_instance();deadline=time.monotonic()+8
 while time.monotonic()<deadline:
  if len(list(manager.get_all_devices()))>=2:break
  time.sleep(.1)
else:
 import numpy as np
 class FakeWave:
  def __init__(self):self.last={}
  def fetch_tactile_frame(self,ch,timeout):
   now=time.time();index=int(now*30)
   if self.last.get(ch)==index:return None
   self.last[ch]=index
   return dict(channel=ch,frame_id=index,ts=now,shape={'RAW':(1,240,320),'DEFORM':(1,240,240)},content={
    'RAW':np.arange(76800,dtype=np.uint8),'DEFORM':np.arange(57600,dtype=np.uint8),
    'F6':np.array([1,2,3,.1,.2,.3],dtype=np.float64),
    'CONTACT_POINT':np.array([[10.,20.]],dtype=np.float64) if index%5==0 else None})
  def stop(self):pass
  def destroy(self):pass
def fingerprint(msg):
 # DDS/CDR alignment padding may differ between independent serializations.
 # Compare exact image pixel bytes, numeric bits, header and layout instead.
 if hasattr(msg,'header'):
  prefix=json.dumps([msg.header.stamp.sec,msg.header.stamp.nanosec,msg.header.frame_id]).encode()
  if hasattr(msg,'encoding'):
   return prefix+json.dumps([msg.height,msg.width,msg.encoding,int(msg.is_bigendian),msg.step]).encode()+bytes(msg.data)
  w=msg.wrench
  return prefix+struct.pack('<6d',w.force.x,w.force.y,w.force.z,w.torque.x,w.torque.y,w.torque.z)
 return json.dumps([msg.layout.data_offset,[(d.label,d.size,d.stride) for d in msg.layout.dim]]).encode()+struct.pack('<3Q',*msg.data)
serialized_hashes=defaultdict(hashlib.sha256);message_topics={}
def serialize_for_capture(msg):
 data=serialize_message(msg);serialized_hashes[message_topics.pop(id(msg))].update(data);return data
rclpy.init();node=rclpy.create_node('tactile_validation');capture=Capture(run,profile,chs,serialize=serialize_for_capture);capture.prepare_resources()
expected=defaultdict(hashlib.sha256);received=Counter();subs=[]
def receive(ch,msg):
 with capture.lock:
  before=capture.accepted_counts[ch['topic']]
  if capture.active:message_topics[id(msg)]=ch['topic']
  capture.accept(ch,msg,time.time_ns())
  if capture.accepted_counts[ch['topic']]>before:
   expected[ch['topic']].update(fingerprint(msg));received[ch['topic']]+=1
for ch in chs:subs.append(node.create_subscription(get_message(ch['type']),ch['source_topic'],lambda m,ch=ch:receive(ch,m),qos_profile_sensor_data))
spinstop=threading.Event()
def spin():
 while not spinstop.is_set():rclpy.spin_once(node,timeout_sec=.01)
spinthread=threading.Thread(target=spin);spinthread.start();waves={};telemetry=None
try:
 if args.hardware:
  for side,serial in [('left','CD559038CD54'),('right','CC559039CC54')]:
   wave=manager.connect(serial,False);waves[side]=wave
   if not wave.start():raise RuntimeError(side+' tactile start failed')
 else:waves={'left':FakeWave(),'right':FakeWave()}
 telemetry=TactileTelemetry(node,{side:SimpleNamespace(_wave=w,tactile_available=True) for side,w in waves.items()});telemetry.start()
 deadline=time.monotonic()+8
 while time.monotonic()<deadline and not capture.readiness():
  if telemetry.error:raise RuntimeError(telemetry.error)
  time.sleep(.02)
 capture.request('start');time.sleep(3)
 capture.close()
 if telemetry.error or capture.error:raise RuntimeError(str(telemetry.error or capture.error))
 actual=defaultdict(hashlib.sha256);actual_cdr=defaultdict(hashlib.sha256);counts=Counter()
 from mcap_ros2.decoder import DecoderFactory
 factory=DecoderFactory()
 with (capture.writer.target/'data.mcap').open('rb') as f:
  for schema,ch,msg in make_reader(f,validate_crcs=True).iter_messages(log_time_order=False):
   decoded=factory.decoder_for(ch.message_encoding,schema)(msg.data)
   actual[ch.topic].update(fingerprint(decoded));actual_cdr[ch.topic].update(msg.data);counts[ch.topic]+=1
 assert counts==received and all(actual[t].digest()==expected[t].digest() for t in counts),'Native content/header data loss/alteration'
 assert all(actual_cdr[t].digest()==serialized_hashes[t].digest() for t in counts),'Serialized CDR changed during write'
 report=dict(status='PASS',mode='real_tactile_ROS_MCAP_no_motion_not_full_S1' if args.hardware else 'synthetic_payload_real_ROS_MCAP_not_hardware_S1',directory=str(folder),episode=str(capture.writer.target),
  required_streams=40,streams_with_samples=len(counts),messages=sum(counts.values()),cdr_bitwise_verified=True,crc_verified=True,
  error=telemetry.error,stats=list(capture.writer.stats.values()))
 write_json(folder/'result.json',report);write_json(root/'.deployment/validation/tactile-audit'/('latest-hardware-result.json' if args.hardware else 'latest-software-result.json'),report)
 print('RESULT',json.dumps({k:v for k,v in report.items() if k!='stats'}),flush=True)
finally:
 if telemetry:telemetry.close()
 capture.close();spinstop.set();spinthread.join(timeout=2);node.destroy_node();rclpy.shutdown()
 for wave in waves.values():wave.stop();wave.destroy()
