"""Four standard JointState command streams, partitioned by verified joint names."""
from copy import deepcopy
import os
from deploy.fr3_sharpa_runtime.protocol import SHARPA_JOINT_NAMES, FR3_POLICY_TO_COMMAND

SOURCE_TOPIC = '/openpi/executed_joint_targets'
LEGACY_TOPIC = '/devices/rig/action/command'
GROUPS = {
    'left_arm': ('fr3_left', tuple(f'left_fr3_joint{i}' for i in range(1,8))),
    'left_hand': ('sharpa_left', tuple(f'left_{n}' for n in SHARPA_JOINT_NAMES)),
    'right_arm': ('fr3_right', tuple(f'right_fr3_joint{i}' for i in range(1,8))),
    'right_hand': ('sharpa_right', tuple(f'right_{n}' for n in SHARPA_JOINT_NAMES)),
}


def output_topic(group):
    return f'/devices/{GROUPS[group][0]}/action/command'


def expand_profile(profile):
    profile=deepcopy(profile)
    for device in profile['devices']:
        channels=[]
        for ch in device['channels']:
            if ch['source_topic']==SOURCE_TOPIC and ch.get('adapter')!='split_joint_command':
                for group,(_,names) in GROUPS.items():
                    channels.append(dict(ch,topic=output_topic(group),adapter='split_joint_command',
                        source_type='sensor_msgs/msg/JointState',command_group=group,joint_names=list(names),
                        unit='rad',action_basis='absolute',command_semantics='software_dispatched_target_not_hardware_ack'))
            elif ch['source_topic']=='/dagger/frame' and ch.get('adapter')=='dagger_ownership' and os.environ.get('SHARPA_COMPACT_RECORDS')=='1' and not any(c.get('adapter')=='dagger_dispatch' for c in device['channels']):
                channels.append(ch)
                for topic,kind,adapter in [('/recording/dispatch','std_msgs/msg/UInt64MultiArray','dagger_dispatch'),
                    ('/devices/rig/state/dispatch_joints','sensor_msgs/msg/JointState','dagger_snapshot'),
                    ('/recording/control_context','std_msgs/msg/String','dagger_context')]:
                    extra=dict(ch,topic=topic,type=kind,source_type='std_msgs/msg/String',adapter=adapter)
                    extra.pop('group_order',None);extra.pop('owner_codes',None)
                    if adapter=='dagger_snapshot':
                        extra.update(value_origin='measured',unit='rad',joint_names=[n for _,ns in GROUPS.values() for n in ns],
                            timestamp_semantics='dispatch_timestamp_of_latest_feedback_snapshot_not_native_sensor_stamp',
                            missing_fields=['velocity','effort'])
                    elif adapter=='dagger_dispatch':
                        extra.update(value_origin='derived',fields=['sequence','frame_index','monotonic_ns','gateway_sequence','gateway_valid'])
                    channels.append(extra)
            else:channels.append(ch)
        device['channels']=channels
        if 'action_space' in device:
            device['action_space']['command_topics']={g:output_topic(g) for g in GROUPS}
            device['action_space']['storage_layout']='four-joint-state-command-streams-v1'
    return profile


def split_message(message, header):
    names=list(message.name)
    expected={name for _,group in GROUPS.values() for name in group}
    if len(names)!=58 or len(set(names))!=58 or set(names)!=expected or len(message.position)!=58:
        raise ValueError('MCAP command split requires the exact 58 named robot joints')
    for field in ('velocity','effort'):
        if len(getattr(message,field)) not in (0,58):
            raise ValueError(f'MCAP JointState {field} must be empty or match all 58 names')
    indices={name:i for i,name in enumerate(names)}
    result={}
    for group,(_,joint_names) in GROUPS.items():
        result[group]=dict(header=deepcopy(header),name=list(joint_names))
        for field in ('position','velocity','effort'):
            source=getattr(message,field)
            result[group][field]=[float(source[indices[name]]) for name in joint_names] if len(source) else []
    return result


GATEWAY_TOPIC = '/teleop/validated_arm_commands'


def gateway_command_profile(profile):
    """Only arm action/command changes source; hand commands remain SDK targets."""
    profile=expand_profile(profile)
    for device in profile['devices']:
        for ch in device['channels']:
            group=ch.get('command_group')
            if ch.get('adapter') in ('split_joint_command','split_gateway_command') and group in ('left_arm','right_arm'):
                ch.update(source_topic=GATEWAY_TOPIC,source_type='sensor_msgs/msg/JointState',
                    adapter='split_gateway_command',command_semantics='gateway_validated_target_not_measured_state',controller_id='teleop_safety_gateway',
                    source_joint_names=[FR3_POLICY_TO_COMMAND[n] for n in GROUPS[group][1]],
                    joint_name_mapping={FR3_POLICY_TO_COMMAND[n]:n for n in GROUPS[group][1]})
            elif ch.get('adapter')=='split_joint_command' and group in ('left_hand','right_hand'):
                ch['command_semantics']='sdk_submitted_target_not_measured_state'
                ch['controller_id']='sharpa_sdk_direct'
        if 'action_space' in device:
            device['action_space']['command_semantics']='per_group_sources_declared_below'
            device['action_space']['arm_command_source']=GATEWAY_TOPIC
            device['action_space']['command_semantics_by_group']={
                'left_arm':'gateway_validated_target', 'right_arm':'gateway_validated_target',
                'left_hand':'sdk_submitted_target', 'right_hand':'sdk_submitted_target'}
            device['action_space']['storage_layout']='four-joint-state-command-streams-v2'
    from .recording_slim import slim_profile
    return slim_profile(profile)


def split_gateway_message(message, header):
    """Map the gateway's accepted sides, never synthesize a rejected side's target."""
    aliases={value:key for key,value in FR3_POLICY_TO_COMMAND.items()}
    names=[aliases.get(n,n) for n in message.name]
    expected={n for g in ('left_arm','right_arm') for n in GROUPS[g][1]}
    if not names or len(set(names))!=len(names) or not set(names)<=expected or len(message.position)!=len(names):
        raise ValueError('Invalid post-gateway JointState names/positions')
    for field in ('velocity','effort'):
        if len(getattr(message,field)) not in (0,len(names)):
            raise ValueError('Invalid post-gateway JointState array length')
    indices={name:i for i,name in enumerate(names)};result={}
    for group in ('left_arm','right_arm'):
        joint_names=GROUPS[group][1];present=set(joint_names)&set(names)
        if not present:continue
        if len(present)!=7:raise ValueError('Gateway side has an incomplete joint set')
        value=dict(header=deepcopy(header),name=list(joint_names))
        for field in ('position','velocity','effort'):
            source=getattr(message,field)
            value[field]=[float(source[indices[n]]) for n in joint_names] if len(source) else []
        result[group]=value
    return result
