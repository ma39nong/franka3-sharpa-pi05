"""Migrate recorded MCAP commands to four streams; preserve original episodes by rename."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import uuid

import yaml
from mcap.reader import make_reader
from mcap.writer import Writer, CompressionType, IndexType
from mcap_ros2.decoder import DecoderFactory
from mcap_ros2._dynamic import serialize_dynamic
from .mcap_actions import GROUPS, LEGACY_TOPIC, expand_profile, output_topic, split_message
from .mcap_storage import digest, seal, sync_tree
from .recording import write_json


def replace_bytes(path, data):
    temp=path.with_name('.'+path.name+'.new')
    with temp.open('wb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def hash_message(h, message, data):
    h.update(struct.pack('<QQI',message.log_time,message.publish_time,message.sequence));h.update(struct.pack('<Q',len(data)));h.update(data)


def updated_shared(episode, root, cache):
    info=json.loads((episode/'episode.json').read_text());refs=info['assets']
    devices=(episode/refs['devices_ref']['path']).resolve();old_shared=devices.parent
    if str(old_shared) in cache:return cache[str(old_shared)]
    for key in ('model_ref','calibration_ref','devices_ref','config_ref'):
        path=(episode/refs[key]['path']).resolve()
        if not path.exists() or digest(path)!=refs[key]['sha256']:
            raise ValueError(f'Missing/modified original shared asset: {key}: {path}')
    temp=root/'_shared'/('.split-'+uuid.uuid4().hex);shutil.copytree(old_shared,temp)
    device_data=yaml.safe_load((temp/'devices.yaml').read_text())
    device_data=expand_profile(device_data)
    replace_bytes(temp/'devices.yaml',yaml.safe_dump(device_data,allow_unicode=True,sort_keys=False).encode())
    recorder=temp/'configs/recorder.yaml'
    if recorder.exists():replace_bytes(recorder,yaml.safe_dump(expand_profile(yaml.safe_load(recorder.read_text())),allow_unicode=True,sort_keys=False).encode())
    checksum=digest(temp);dest=root/'_shared'/checksum
    if dest.exists():
        if digest(dest)!=checksum:raise ValueError('Shared destination hash collision')
        shutil.rmtree(temp)
    else:
        sync_tree(temp);temp.rename(dest)
    result={k:{'path':os.path.relpath(path,episode),'sha256':digest(path)} for k,path in {
        'devices_ref':dest/'devices.yaml','calibration_ref':dest/'calibration','config_ref':dest/'configs'}.items()}
    cache[str(old_shared)]=result
    return result


def convert(episode,root,backup_root,*,commit,cache):
    source_manifest_sha=digest(episode/'manifest.json')
    before=json.loads((episode/'manifest.json').read_text())
    # Refuse to transform damaged or concurrently changed source payloads.
    for entry in before['files']:
        path=episode/entry['path']
        if not path.is_file() or path.stat().st_size!=entry['size'] or digest(path)!=entry['sha256']:
            raise ValueError(f'Original manifest mismatch: {path}')
    source=episode/'data.mcap'
    with source.open('rb') as f:
        reader=make_reader(f,validate_crcs=True);summary=reader.get_summary()
        if not any(c.topic==LEGACY_TOPIC for c in summary.channels.values()):
            return dict(episode=str(episode),status='already_split_or_no_legacy_command')
    refs=updated_shared(episode,root,cache)
    stage=episode.with_name('.'+episode.name+'.split-'+uuid.uuid4().hex)
    shutil.copytree(episode,stage,copy_function=os.link,symlinks=True)
    temp=stage/'.data.mcap.new';expected=defaultdict(hashlib.sha256);counts=Counter();types={};command_count=0
    try:
        with source.open('rb') as src,temp.open('wb') as dst:
            reader=make_reader(src,validate_crcs=True);summary=reader.get_summary()
            writer=Writer(dst,compression=CompressionType.ZSTD,index_types=IndexType.ALL,enable_crcs=True)
            writer.start(profile='ros2',library='sharpa-command-split/1')
            schemas={i:writer.register_schema(name=s.name,encoding=s.encoding,data=s.data) for i,s in summary.schemas.items()}
            channel_ids={};encoders={};factory=DecoderFactory()
            for cid,ch in summary.channels.items():
                schema=summary.schemas[ch.schema_id]
                if ch.topic==LEGACY_TOPIC:
                    if schema.name!='sensor_msgs/msg/JointState':raise ValueError('Unexpected legacy command schema')
                    encoders[cid]=serialize_dynamic(schema.name,schema.data.decode())[schema.name]
                    for group in GROUPS:
                        topic=output_topic(group)
                        if any(c.topic==topic for c in summary.channels.values()):raise ValueError('Both split and combined action channels exist')
                        channel_ids[(cid,group)]=writer.register_channel(topic=topic,message_encoding='cdr',schema_id=schemas[ch.schema_id],metadata=ch.metadata)
                        types[topic]=schema.name
                else:
                    channel_ids[(cid,None)]=writer.register_channel(topic=ch.topic,message_encoding=ch.message_encoding,schema_id=schemas[ch.schema_id],metadata=ch.metadata)
                    types[ch.topic]=schema.name
            for schema,ch,msg in reader.iter_messages(log_time_order=False):
                if ch.topic==LEGACY_TOPIC:
                    value=factory.decoder_for(ch.message_encoding,schema)(msg.data)
                    header=dict(stamp=dict(sec=value.header.stamp.sec,nanosec=value.header.stamp.nanosec),frame_id=value.header.frame_id)
                    if header['stamp']['sec']*10**9+header['stamp']['nanosec']!=msg.publish_time:raise ValueError('Legacy command header/publish_time mismatch')
                    pieces=split_message(value,header);command_count+=1
                    records=[(group,output_topic(group),encoders[ch.id](part)) for group,part in pieces.items()]
                else:records=[(None,ch.topic,msg.data)]
                for group,topic,data in records:
                    writer.add_message(channel_id=channel_ids[(ch.id,group)],log_time=msg.log_time,publish_time=msg.publish_time,sequence=msg.sequence,data=data)
                    counts[topic]+=1;hash_message(expected[topic],msg,data)
            for a in reader.iter_attachments():writer.add_attachment(create_time=a.create_time,log_time=a.log_time,name=a.name,media_type=a.media_type,data=a.data)
            for m in reader.iter_metadata():writer.add_metadata(name=m.name,data=m.metadata)
            writer.finish();dst.flush();os.fsync(dst.fileno())
        # Independent readback: CRCs, counters and exact binary payload/envelope hashes.
        actual=defaultdict(hashlib.sha256);observed=Counter()
        with temp.open('rb') as f:
            reader=make_reader(f,validate_crcs=True)
            for schema,ch,msg in reader.iter_messages(log_time_order=False):
                if ch.topic==LEGACY_TOPIC:raise ValueError('Combined action remains')
                observed[ch.topic]+=1;hash_message(actual[ch.topic],msg,msg.data)
        if counts!=observed or any(expected[t].digest()!=actual[t].digest() for t in counts):raise ValueError('MCAP roundtrip payload verification failed')
        if any(counts[output_topic(g)]!=command_count for g in GROUPS):raise ValueError('Split command sample loss')
        os.replace(temp,stage/'data.mcap')
        stat=json.loads((stage/'channel_stats.json').read_text());key='streams' if 'streams' in stat else 'channels';rows=[]
        for row in stat[key]:
            if row['topic']==LEGACY_TOPIC:
                for group in GROUPS:rows.append(dict(row,topic=output_topic(group),command_group=group,message_count=command_count))
            else:rows.append(row)
        stat[key]=rows;write_json(stage/'channel_stats.json',stat)
        info=json.loads((stage/'episode.json').read_text());info['assets'].update(refs)
        info['action_storage_layout']='four-joint-state-command-streams-v1'
        info['format_migration']=dict(name='sharpa-command-split',version=1,tool_sha256=digest(Path(__file__)),
            migrated_at_utc=datetime.now(timezone.utc).isoformat(),original_mcap_sha256=digest(source),
            original_backup=str(backup_root/episode.relative_to(root)),preserved='all non-command CDR payloads and original message timestamps/sequences')
        write_json(stage/'episode.json',info)
        metadata=yaml.safe_load((stage/'metadata.yaml').read_text());bag=metadata['rosbag2_bagfile_information'];topics=[]
        for entry in bag['topics_with_message_count']:
            if entry['topic_metadata']['name']==LEGACY_TOPIC:
                for group in GROUPS:
                    new=deepcopy(entry);new['topic_metadata']['name']=output_topic(group);new['message_count']=command_count;topics.append(new)
            else:topics.append(entry)
        bag['topics_with_message_count']=topics;bag['message_count']=sum(counts.values())
        for entry in bag.get('files',[]):entry['message_count']=sum(counts.values())
        replace_bytes(stage/'metadata.yaml',yaml.safe_dump(metadata,sort_keys=False).encode())
        seal(stage,before['status']);sync_tree(stage)
        result=dict(episode=str(episode),status='verified_staging',staging=str(stage),command_samples=command_count,
            video_frames=sum(n for t,n in counts.items() if types.get(t)=='foxglove_msgs/msg/CompressedVideo'),
            original_messages=summary.statistics.message_count,new_messages=sum(counts.values()),backup=str(backup_root/episode.relative_to(root)))
        if commit:
            if digest(episode/'manifest.json')!=source_manifest_sha:
                raise RuntimeError('Source episode changed during conversion; original left untouched')
            backup=backup_root/episode.relative_to(root);backup.parent.mkdir(parents=True,exist_ok=True)
            episode.rename(backup)
            try:stage.rename(episode)
            except BaseException:backup.rename(episode);raise
            result['status']='converted';result.pop('staging')
        return result
    except BaseException:
        # Never replace the original on error; leave staging for diagnosis.
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--episode',action='append');parser.add_argument('--commit',action='store_true')
    parser.add_argument('--backup-root',type=Path)
    args=parser.parse_args();root=args.root.resolve();backup=(args.backup_root or root/'.mcap-format-backups'/datetime.now().strftime('%Y%m%d-%H%M%S')).resolve()
    paths=[root/p for p in args.episode] if args.episode else sorted(p for grade in 'ABCD' for p in (root/grade).glob('episode*') if p.is_dir() and not p.is_symlink())
    report=[];cache={};reportfile=backup.parent/(backup.name+'-report.json')
    reportfile.parent.mkdir(parents=True,exist_ok=True)
    for episode in paths:
        if not (episode/'data.mcap').is_file():
            result=dict(episode=str(episode),status='skipped_non_mcap')
        else:
            if not episode.resolve().is_relative_to(root):raise ValueError('Episode outside root')
            result=convert(episode,root,backup,commit=args.commit,cache=cache)
        report.append(result);write_json(reportfile,dict(backup_root=str(backup),episodes=report))
        print(json.dumps(result,ensure_ascii=False),flush=True)
    print('REPORT',reportfile,flush=True)


if __name__=='__main__':main()
