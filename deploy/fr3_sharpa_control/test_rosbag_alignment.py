"""Rosbag contract and classification gate checks without devices or ROS nodes."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import threading
import time
import pytest
from . import quality
from .recording import COMPACT_HEADER,write_json,compact_finalize


@pytest.fixture
def recorder(monkeypatch):
    from . import recorder_ros
    module=importlib.reload(recorder_ros)
    for key in ('SHARPA_RECORD_HEAD_DEPTH','SHARPA_RECORD_WRIST_DEPTH','SHARPA_RECORD_TACTILE'):
        monkeypatch.setenv(key,'0')
    yield module
    importlib.reload(module)


def test_rosbag_commands_and_calibration_only_after_format_branch(recorder):
    from .rosbag_contract import configure,HAND_COMMANDS
    before=dict(recorder.TOPICS)
    assert '/openpi/executed_joint_targets' in before
    configure(recorder)
    assert '/teleop/arm_commands' not in recorder.ALL_TOPICS
    assert '/openpi/executed_joint_targets' not in recorder.ALL_TOPICS
    assert {('/teleop/validated_arm_commands'),*HAND_COMMANDS.values()}<=set(recorder.TOPICS)
    assert all(f'/cam{i}/color/camera_info' in recorder.ALL_TOPICS for i in range(3))
    assert {'/tf','/tf_static'}<=set(recorder.ALL_TOPICS)
    assert set(HAND_COMMANDS.values()).isdisjoint(recorder.SENSORS)
    assert '/teleop/validated_arm_commands' not in recorder.SENSORS


def test_mcap_dispatch_keeps_original_topic_contract(recorder,monkeypatch):
    from . import mcap_capture
    called=[];monkeypatch.setattr(mcap_capture,'main',lambda args:called.append(args))
    monkeypatch.setenv('SHARPA_RECORD_FORMAT','mcap');monkeypatch.setattr('sys.argv',['recorder'])
    recorder.main()
    assert len(called)==1
    assert '/teleop/arm_commands' in recorder.TOPICS
    assert '/openpi/executed_joint_targets' in recorder.TOPICS
    assert '/sharpa/left/command' not in recorder.ALL_TOPICS


def test_optional_channels_are_present_for_module_entrypoint(recorder,monkeypatch):
    from .rosbag_contract import configure
    monkeypatch.setenv('SHARPA_RECORD_HEAD_DEPTH','1');monkeypatch.setenv('SHARPA_RECORD_WRIST_DEPTH','1')
    configure(recorder)
    for i in range(3):
        assert f'/cam{i}/depth/image_raw' in recorder.ALL_TOPICS
        assert f'/cam{i}/depth/camera_info' in recorder.SENSORS


@pytest.mark.parametrize('interleaved',[False,True])
def test_sdk_hand_split_is_named_and_preserves_stamp(interleaved):
    pytest.importorskip('rclpy')
    from sensor_msgs.msg import JointState
    from .rosbag_contract import hand_commands
    from deploy.fr3_sharpa_runtime.protocol import FR3_POLICY_NAMES,SHARPA_JOINT_NAMES
    message=JointState();message.header.stamp.sec=17
    message.name=list(FR3_POLICY_NAMES)+[f'{side}_{joint}' for side in ('left','right') for joint in SHARPA_JOINT_NAMES]
    if interleaved:
        # Actual command_publisher order is left arm/hand, right arm/hand.
        message.name=([*FR3_POLICY_NAMES[:7],*[f'left_{joint}' for joint in SHARPA_JOINT_NAMES],
                       *FR3_POLICY_NAMES[7:],*[f'right_{joint}' for joint in SHARPA_JOINT_NAMES]])
    message.position=[float(i) for i in range(58)]
    messages=hand_commands(message)
    assert set(messages)=={'/sharpa/left/command','/sharpa/right/command'}
    assert messages['/sharpa/left/command'].name==list(SHARPA_JOINT_NAMES)
    assert list(messages['/sharpa/left/command'].position)==list(message.position[7:29] if interleaved else message.position[14:36])
    assert list(messages['/sharpa/right/command'].position)==list(message.position[36:58])
    assert all(m.header.stamp.sec==17 for m in messages.values())
    message.name[-1]='unexpected'
    with pytest.raises(ValueError):hand_commands(message)


def run(root,minute,stamp,recording_format,dagger=True):
    directory=root/'2026-11-28'/minute;directory.mkdir(parents=True)
    (directory/'run.log').write_text(COMPACT_HEADER+'\n')
    write_json(directory/'run.json',dict(settings=dict(dagger=dagger)))
    quality.begin(directory,archive_parent=root/'archive',recording_format=recording_format)
    bag=quality.payload(directory);bag.resolve().mkdir()
    (bag/'metadata.yaml').write_text('software fixture')
    write_json(directory/'recording.json',dict(status='COMPLETE',started_unix_ns=stamp,ended_unix_ns=(stamp+1 if stamp else None),counts={}))
    quality.finish(directory,outcome='FINISHED',reason='fixture',returncode=0,recorder_closed=True)
    return directory


@pytest.mark.parametrize('recording_format',['rosbag','mcap'])
def test_classification_required_for_both_formats_and_survives_restart(tmp_path,recording_format):
    directory=run(tmp_path,'10.00',100,recording_format)
    compact_finalize(directory)
    with pytest.raises(ValueError,match='质量分类'):quality.require_previous_classified(tmp_path)
    value=quality.read(directory)
    quality.rate(tmp_path,'2026-11-28/10.00',run_id=value['run_id'],grade='B',note='test',expected_revision=value['revision'])
    assert quality.previous_unclassified(tmp_path) is None
    # Gate has no Home condition, including when legacy metadata says Home failed.
    value=quality.read(directory);value['after_home']={'state':'failed'};quality.save_quality(directory,value)
    quality.require_previous_classified(tmp_path)


def test_gate_ignores_never_started_non_dagger_and_old_backlog(tmp_path):
    run(tmp_path,'10.00',10,'rosbag')
    never=run(tmp_path,'10.01',None,'mcap')
    assert quality.previous_unclassified(tmp_path)['directory']!=str(never)
    latest=run(tmp_path,'10.02',20,'mcap')
    value=quality.read(latest)
    quality.rate(tmp_path,'2026-11-28/10.02',run_id=value['run_id'],grade='A',expected_revision=0)
    run(tmp_path,'10.03',30,'rosbag',dagger=False)
    assert quality.previous_unclassified(tmp_path) is None


def test_controller_rejects_unclassified_before_starting_processes(tmp_path):
    from .controller import DeploymentController
    previous=run(tmp_path,'10.00',100,'rosbag')
    controller=object.__new__(DeploymentController)
    controller.agent=NS(set_mode=Mock());controller.lock=threading.RLock()
    controller.home_thread=controller.thread=controller.recorder=None
    controller.execution_directory=previous
    controller.preview=lambda values:dict(settings=dict(dagger=True,output=str(tmp_path)),approval='fixture')
    controller._require_model=Mock()
    with pytest.raises(ValueError,match='质量分类'):controller.start({})
    controller._require_model.assert_not_called()


def test_real_rosbag_contains_only_accepted_commands_and_calibrated_start(recorder,tmp_path):
    pytest.importorskip('rclpy')
    import numpy as np
    import rosbag2_py
    from rclpy.serialization import serialize_message,deserialize_message
    from sensor_msgs.msg import CameraInfo,Image,JointState
    from tf2_msgs.msg import TFMessage
    from geometry_msgs.msg import TransformStamped
    from .rosbag_contract import configure,hand_commands
    from deploy.fr3_sharpa_runtime.protocol import FR3_POLICY_NAMES,FR3_COMMAND_NAMES,SHARPA_JOINT_NAMES
    configure(recorder)
    directory=tmp_path/'run';directory.mkdir();now=time.monotonic_ns()
    specs=[];static=TFMessage()
    for index in range(3):
        edge=TransformStamped();edge.header.frame_id=f'cam{index}_link';edge.child_frame_id=f'cam{index}_color_optical_frame';edge.transform.rotation.w=1.
        static.transforms.append(edge)
    for camera,side in [('cam1','right'),('cam2','left')]:
        matrix=np.eye(4);matrix[0,3]=.1
        specs.append(dict(parent_frame=f'{side}_fr3_link8',child_frame=f'{camera}_color_optical_frame',matrix=matrix.tolist()))
    capture=recorder.BagCapture(directory,clock=lambda:now,static_extrinsics=specs)
    messages={}
    for topic in recorder.SENSORS:
        msg=Image() if topic.startswith('/cam') else JointState()
        msg.header.stamp.sec=17
        messages[topic]=msg
        for delta in (2,1):capture.accept(topic,msg,time.time_ns(),rx_mono_ns=now-delta,serialize=serialize_message)
    for index in range(3):
        info=CameraInfo();info.header.stamp.sec=16;info.width=640;info.height=480
        capture.accept(f'/cam{index}/color/camera_info',info,time.time_ns(),rx_mono_ns=now-1,serialize=serialize_message)
    capture.accept('/tf_static',static,time.time_ns(),rx_mono_ns=now-1,serialize=serialize_message)
    assert not any(capture.counts.values())
    capture.request('start')
    start=capture.report['started_unix_ns']
    for topic,msg in messages.items():capture.accept(topic,msg,start+1000,rx_mono_ns=now+1,serialize=serialize_message)
    arm=JointState();arm.name=list(FR3_COMMAND_NAMES);arm.position=[.1]*14
    capture.accept('/teleop/validated_arm_commands',arm,start+2000,rx_mono_ns=now+1,serialize=serialize_message)
    submitted=JointState();submitted.name=list(FR3_POLICY_NAMES)+[f'{side}_{n}' for side in ('left','right') for n in SHARPA_JOINT_NAMES];submitted.position=[.2]*58
    for topic,msg in hand_commands(submitted).items():capture.accept(topic,msg,start+3000,rx_mono_ns=now+1,serialize=serialize_message)
    capture.close()
    assert capture.report['status']=='COMPLETE'
    bag=directory/'rosbag';reader=rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag),storage_id='sqlite3'),rosbag2_py.ConverterOptions('',''))
    registered={topic.name for topic in reader.get_all_topics_and_types()}
    assert not {'/openpi/executed_joint_targets','/teleop/arm_commands'}&registered
    assert {'/sharpa/left/command','/sharpa/right/command','/teleop/validated_arm_commands'}<=registered
    rows=[]
    while reader.has_next():rows.append(reader.read_next())
    assert sum(capture.counts.values())==len(rows)
    snapshot=next(row for row in rows if row[0]=='/tf_static')
    assert snapshot[2]==start
    edges={edge.child_frame_id:edge for edge in deserialize_message(snapshot[1],TFMessage).transforms}
    assert edges['cam1_link'].header.frame_id=='right_fr3_link8'
    assert edges['cam2_link'].header.frame_id=='left_fr3_link8'
    assert json.loads((bag/'wrist_extrinsics.json').read_text())['static_extrinsics']==specs
