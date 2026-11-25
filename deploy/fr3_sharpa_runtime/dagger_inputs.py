"""Input-only left GELLO reader and named MANUS cache. No robot/hand SDK imports."""
import os
from pathlib import Path
import threading
import time

import yaml

from .dagger import vector
from .protocol import SHARPA_JOINT_NAMES

ARM_TOPIC = '/dagger/input/left_arm'
HAND_TOPIC = '/dagger/input/left_hand'


def configuration(repo):
    data = yaml.safe_load((Path(repo)/'config/modes/gello.yaml').read_text())
    left = data['left']
    return dict(sensitivity=left['joint_sensitivity'], max_delta=left['max_relative_delta'],
                margin=left.get('joint_limit_margin', [0.0]*7),
                timeout_s=min(.2, float(data['stale_timeout'])),
                arm_speed=min(.8, float(data['max_target_velocity'])))


class InputCache:
    def __init__(self, mapping):
        self.mapping = mapping
        self.lock = threading.Lock()
        self.values = {}
        self.errors = {}
        self.last_stamps = {}

    def receive(self, kind, message, ros_ns, mono=None):
        mono = time.monotonic() if mono is None else mono
        expected = tuple(f'left_fr3_joint{i}' for i in range(1,8)) if kind == 'arm' else SHARPA_JOINT_NAMES
        try:
            names = tuple(message.name)
            if len(names) != len(set(names)) or set(names) != set(expected):
                raise ValueError('Wrong DAgger input joint names: '+kind)
            stamp = int(message.header.stamp.sec)*1_000_000_000 + int(message.header.stamp.nanosec)
            if not 0 <= ros_ns-stamp <= self.mapping['timeout_s']*1e9:
                raise ValueError('Stale/future source timestamp')
            # Repeated source frames never refresh the receipt age.
            identity = (str(message.header.frame_id), stamp)
            with self.lock:
                previous = self.last_stamps.get(kind)
                if previous and identity[0] == previous[0] and stamp <= previous[1]:
                    raise ValueError('Repeated/backwards source timestamp')
                self.last_stamps[kind] = identity
                q = vector(message.position, len(expected))[[names.index(n) for n in expected]]
                self.values[kind] = (q, mono-(ros_ns-stamp)/1e9, identity[0])
                self.errors.pop(kind, None)
        except (ValueError, TypeError) as error:
            with self.lock:
                self.values.pop(kind, None)
                self.errors[kind] = str(error)

    def snapshot(self):
        with self.lock:
            if not all(k in self.values for k in ('arm','hand')):
                return None
            arm, hand = self.values['arm'], self.values['hand']
            return dict(self.mapping, arm=arm[0].copy(), hand=hand[0].copy(),
                        arm_time=arm[1], hand_time=hand[1], session=arm[2]+'|'+hand[2])


def teleop_python():
    """Locate an existing environment; never install/update it."""
    configured=os.environ.get('SHARPA_GELLO_INPUT_PYTHON')
    if configured:
        path=Path(configured).expanduser()
        if not path.is_file() or not os.access(path,os.X_OK):
            raise FileNotFoundError('Invalid SHARPA_GELLO_INPUT_PYTHON')
        return path
    candidates=[]
    if os.environ.get('CONDA_EXE'):
        candidates.append(Path(os.environ['CONDA_EXE']).resolve().parents[1])
    candidates += [Path.home()/name for name in ('miniforge3','miniconda3','anaconda3')]
    name=os.environ.get('TELEOP_CONDA_ENV','gello-upper-body-teleop')
    for base in candidates:
        path=base/'envs'/name/'bin/python'
        if path.is_file() and os.access(path,os.X_OK):
            return path
    raise FileNotFoundError('Set SHARPA_GELLO_INPUT_PYTHON to the existing GELLO teleop Python')


def reader_environment(python):
    # ROS overlays contain CPython-specific site packages. The serial child
    # must never import those packages into the existing teleop interpreter.
    env=dict(os.environ)
    env['PYTHONPATH']=str(Path(__file__).resolve().parents[2])
    env.pop('PYTHONHOME',None)
    env.pop('LD_LIBRARY_PATH',None)
    env['CONDA_PREFIX']=str(Path(python).parent.parent)
    env['PATH']=str(Path(python).parent)+os.pathsep+env.get('PATH','')
    return env


def main():
    import json
    import signal
    import subprocess
    import rclpy
    from sensor_msgs.msg import JointState
    python=teleop_python()
    # The child imports the established serial adapter in its original Python;
    # this ROS process only translates immutable, timestamped input packets.
    child=subprocess.Popen([str(python),'-m','deploy.fr3_sharpa_runtime.dagger_gello_reader'],
        stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,text=True,bufsize=1,env=reader_environment(python))
    rclpy.init()
    node=rclpy.create_node('dagger_left_gello_input')
    publisher=node.create_publisher(JointState,ARM_TOPIC,1)
    lock=threading.Lock()
    latest={}
    errors=[]
    def receive():
        try:
            for line in child.stdout:
                if len(line)>16384:
                    raise ValueError('Oversized GELLO input packet')
                packet=json.loads(line)
                vector(packet['positions'],7)
                with lock:
                    latest.clear();latest.update(packet)
        except Exception as error:
            errors.append(str(error))
    thread=threading.Thread(target=receive,daemon=True)
    thread.start()
    previous=None
    def tick():
        nonlocal previous
        if errors or child.poll() is not None:
            raise RuntimeError('GELLO input reader exited: '+str(errors))
        with lock:
            packet=dict(latest)
        if not packet:
            return
        identity=(packet['session'],packet['sequence'])
        age=time.monotonic()-packet['source_monotonic_s']
        if identity==previous or not 0<=age<=.2:
            return
        previous=identity
        ns=node.get_clock().now().nanoseconds-int(age*1e9)
        msg=JointState()
        msg.header.stamp.sec,msg.header.stamp.nanosec=divmod(ns,1_000_000_000)
        msg.header.frame_id=packet['session']
        msg.name=[f'left_fr3_joint{i}' for i in range(1,8)]
        msg.position=[float(x) for x in packet['positions']]
        publisher.publish(msg)
    node.create_timer(.01,tick)
    try:
        rclpy.spin(node)
    finally:
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.terminate()  # Only our input-only child, no motion owner.
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__=='__main__':
    main()
