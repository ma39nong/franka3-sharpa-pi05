"""Rosbag-only saved command contract; never changes the MCAP profile or graph."""
from copy import deepcopy

VALIDATED = '/teleop/validated_arm_commands'
COMBINED_SOURCE = '/openpi/executed_joint_targets'
HAND_COMMANDS = {side:f'/sharpa/{side}/command' for side in ('left','right')}


def configure(recorder):
    """Called only after the MCAP dispatch branch has returned."""
    recorder.TOPICS = {topic:kind for topic,kind in recorder.TOPICS.items()
                       if topic not in ('/teleop/arm_commands',COMBINED_SOURCE)}
    from .recording_options import optional_channels
    for channel in optional_channels():
        target=recorder.OPTIONAL_TOPICS if channel.get('presence_policy')=='conditional' else recorder.TOPICS
        target[channel['source_topic']]=channel.get('source_type',channel['type'])
    recorder.TOPICS.update({VALIDATED:'sensor_msgs/msg/JointState',
                           **{topic:'sensor_msgs/msg/JointState' for topic in HAND_COMMANDS.values()}})
    recorder.OPTIONAL_TOPICS = {topic:kind for topic,kind in recorder.OPTIONAL_TOPICS.items()
                               if topic not in recorder.TOPICS}
    recorder.OPTIONAL_TOPICS.update({f'/cam{i}/color/camera_info':'sensor_msgs/msg/CameraInfo' for i in range(3)})
    recorder.OPTIONAL_TOPICS.update({'/tf':'tf2_msgs/msg/TFMessage','/tf_static':'tf2_msgs/msg/TFMessage'})
    recorder.ALL_TOPICS={**recorder.TOPICS,**recorder.OPTIONAL_TOPICS}
    # Preserve the existing receipt gate: commands and one-shot calibration are not sensors.
    recorder.SENSORS=frozenset(topic for topic in recorder.TOPICS
        if topic not in {VALIDATED,*HAND_COMMANDS.values()})


def hand_commands(message):
    """Extract SDK-submitted hand values; do not save combined/source arm targets."""
    from sensor_msgs.msg import JointState
    from deploy.fr3_sharpa_runtime.protocol import SHARPA_JOINT_NAMES, FR3_POLICY_NAMES
    names=list(message.name)
    expected={*FR3_POLICY_NAMES,*[f'{side}_{joint}' for side in ('left','right') for joint in SHARPA_JOINT_NAMES]}
    if len(names)!=58 or len(set(names))!=58 or len(message.position)!=58 or set(names)!=expected:
        raise ValueError('Expected complete named 58D SDK-dispatch record')
    for field in ('velocity','effort'):
        if len(getattr(message,field)) not in (0,58):raise ValueError('Invalid SDK-dispatch '+field)
    indices={name:i for i,name in enumerate(names)}
    result={}
    for side,topic in HAND_COMMANDS.items():
        msg=JointState();msg.header=deepcopy(message.header);msg.name=list(SHARPA_JOINT_NAMES)
        for field in ('position','velocity','effort'):
            values=getattr(message,field)
            setattr(msg,field,[float(values[indices[f'{side}_{joint}']]) for joint in SHARPA_JOINT_NAMES] if len(values) else [])
        result[topic]=msg
    return result
