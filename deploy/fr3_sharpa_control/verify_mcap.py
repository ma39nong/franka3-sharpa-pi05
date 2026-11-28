"""Synthetic software validation only; never starts ROS, SDKs or hardware."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import uuid
import yaml
from .mcap_storage import McapEpisode, VIDEO_TYPE, digest
from .mcap_video import VideoEncoder


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);parser.add_argument('--gateway-actions',action='store_true');args=parser.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    (root/'fixture-model').mkdir();(root/'fixture-model/robot.xml').write_text('<robot name="synthetic"><link name="base"/></robot>')
    (root/'fixture-calibration').mkdir();(root/'fixture-calibration/calibration.yaml').write_text('synthetic: true\n')
    topic='/devices/fixture/cameras/rgb/color/image_h264'
    ch=dict(topic=topic,type=VIDEO_TYPE,recording='enabled',role='head',crf=18,width=64,height=64,fps=30,
        color=dict(range='limited',matrix='bt709',primaries='bt709',transfer='iec61966-2-1'))
    profile=dict(model_root=str(root/'fixture-model'),calibration_root=str(root/'fixture-calibration'),
        devices=[dict(device_id='fixture',channels=[ch])],clock_mapping=dict(source_clock_id='fixture',method='ntp',valid_from_ns=1700000000000000000,
        raw_reference=1700000000000000000,unix_reference_ns=1700000000000000000,scale_ns_per_tick=1.,sync_state='locked',uncertainty_ns=1))
    if args.gateway_actions:
        from .mcap_actions import GROUPS, output_topic, gateway_command_profile, LEGACY_TOPIC, SOURCE_TOPIC
        profile['devices'][0]['channels'].append(dict(source_topic=SOURCE_TOPIC,topic=LEGACY_TOPIC,type='sensor_msgs/msg/JointState',recording='enabled'))
        # expand_profile expects input descriptors; RGB is an already prepared channel.
        ch['source_topic']='/synthetic/rgb'
        profile=gateway_command_profile(profile)
        for group in ('left_arm','right_arm'):
            profile['devices'][0]['channels'].append(dict(topic=output_topic(group).replace('/action/command','/state/joints'),type='sensor_msgs/msg/JointState',recording='enabled',value_origin='measured'))
    identity=dict(episode_id='episode0',episode_uid=uuid.uuid4().hex,session_id='synthetic',recording_id=uuid.uuid4().hex,
        task_id='fixture',task_description='Synthetic validation',source_type='synthetic',source_id='synthetic',end_reason='normal stop')
    if args.gateway_actions:
        identity.update(action_storage_layout='four-joint-state-command-streams-v2',action_semantics={'arms':'gateway_validated_target','hands':'sdk_submitted_target'})
    origin=1700000000000000000
    episode=McapEpisode(root/'pending/episode0',root=root,profile=profile,identity=identity,start_time_ns=origin)
    episode.register(topic,VIDEO_TYPE);encoder=VideoEncoder(ch)
    if args.gateway_actions:
        for channel in profile['devices'][0]['channels'][1:]:episode.register(channel['topic'],channel['type'])
    for i in range(65):
        stamp=origin+i*1_000_000_000//30
        msg=SimpleNamespace(width=64,height=64,encoding='rgb8',step=192,data=bytes([i%255])*64*64*3)
        payload=encoder.encode(msg,stamp)
        episode.write(topic,dict(timestamp=dict(sec=stamp//10**9,nanosec=stamp%10**9),frame_id='fixture_optical',format='h264',data=payload),
            log_time=stamp+1000,publish_time=stamp)
        if args.gateway_actions:
            from .mcap_actions import split_gateway_message, split_message
            from deploy.fr3_sharpa_runtime.protocol import FR3_POLICY_TO_COMMAND
            head=dict(stamp=dict(sec=(stamp+100)//10**9,nanosec=(stamp+100)%10**9),frame_id='synthetic')
            names=[n for _,ns in GROUPS.values() for n in ns]
            pre=SimpleNamespace(name=names,position=[.1]*58,velocity=[],effort=[])
            hands=split_message(pre,head)
            post=SimpleNamespace(name=[FR3_POLICY_TO_COMMAND[n] for g in ('left_arm','right_arm') for n in GROUPS[g][1]],position=[.001]*14,velocity=[],effort=[])
            pieces={**{g:hands[g] for g in ('left_hand','right_hand')},**split_gateway_message(post,head)}
            for group in GROUPS:episode.write(output_topic(group),pieces[group],log_time=stamp+1100,publish_time=stamp+100)
            for group in ('left_arm','right_arm'):
                state=dict(header=head,name=list(GROUPS[group][1]),position=[.05]*7,velocity=[],effort=[])
                episode.write(output_topic(group).replace('/action/command','/state/joints'),state,log_time=stamp+1200,publish_time=stamp+100)
    encoder.close();episode.close()
    info=json.loads((episode.target/'episode.json').read_text());shared=episode.assets['devices_ref'].parent
    template=Path.home()/'.local/share/sharpa-conformance/recorder-conformance/templates/certification_target.yaml'
    target=yaml.safe_load(template.read_text());target['target']['recorder']=info['recorder']
    target['target']['config_files']=[dict(path='devices.yaml',sha256=digest(shared/'devices.yaml'))]
    target['event_types']=dict(unavailable=['unavailable'],forced_keyframe=['forced_idr'],clock=['clock'],interrupted=['interrupted'],recovery=['recovery'])
    (root/'target.yaml').write_text(yaml.safe_dump(target))
    (root/'runs.yaml').write_text(yaml.safe_dump(dict(runs=[dict(run_id='synthetic',scenario='S1',episode_dir=str(episode.target),shared_dir=str(shared))])))
    print(root)


if __name__=='__main__':main()
