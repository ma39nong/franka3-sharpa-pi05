"""Add operator-confirmed static wrist calibration to closed MCAP episodes.

Original MCAP and metadata are retained outside episodes. All non-static-TF
messages retain their exact CDR bytes, sequence, log_time and publish_time.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time

import yaml
from mcap.reader import make_reader
from mcap.writer import Writer, CompressionType
from mcap_ros2.decoder import DecoderFactory
from mcap_ros2._dynamic import serialize_dynamic
from rosbags.typesys import get_typestore, Stores
from .calibrated_tf import merge_static
from .mcap_storage import digest, seal, sync_tree
from .recording import write_json


def plain_transform(t):
    return dict(header=dict(stamp=dict(sec=t.header.stamp.sec,nanosec=t.header.stamp.nanosec),frame_id=t.header.frame_id),
        child_frame_id=t.child_frame_id,transform=dict(translation={k:getattr(t.transform.translation,k) for k in 'xyz'},
        rotation={k:getattr(t.transform.rotation,k) for k in ('x','y','z','w')}))


def fingerprint(channel,message,h):
    h.update(channel.topic.encode()+b'\0');h.update(struct.pack('<QQI',message.log_time,message.publish_time,message.sequence));h.update(message.data)


def rewrite(source, destination, specs):
    edges={};tf_old_count=0
    with source.open('rb') as stream:
        reader=make_reader(stream,validate_crcs=True,decoder_factories=[DecoderFactory()]);summary=reader.get_summary()
        for _,_,_,decoded in reader.iter_decoded_messages(topics=['/tf_static'],log_time_order=False):
            tf_old_count+=1
            for t in decoded.transforms:
                value=plain_transform(t)
                if t.child_frame_id in edges:
                    a=deepcopy(edges[t.child_frame_id]);b=deepcopy(value)
                    a['header'].pop('stamp');b['header'].pop('stamp')
                    if a!=b:raise ValueError('Changing/ambiguous static frame in old episode: '+t.child_frame_id)
                edges[t.child_frame_id]=value
    before=hashlib.sha256();counts={};last_log=0;first=None
    with source.open('rb') as src,destination.open('wb') as out:
        reader=make_reader(src,validate_crcs=True)
        writer=Writer(out,compression=CompressionType.ZSTD,enable_crcs=True)
        writer.start(profile=reader.get_header().profile,library='sharpa-wrist-static-backfill/1')
        schemas={i:writer.register_schema(name=s.name,encoding=s.encoding,data=s.data) for i,s in summary.schemas.items()}
        channels={i:writer.register_channel(topic=c.topic,message_encoding=c.message_encoding,schema_id=schemas.get(c.schema_id,0),metadata=c.metadata) for i,c in summary.channels.items()}
        existing=next((i for i,c in summary.channels.items() if c.topic=='/tf_static'),None)
        definition=get_typestore(Stores.ROS2_HUMBLE).generate_msgdef('tf2_msgs/msg/TFMessage',ros_version=2)[0]
        encode=serialize_dynamic('tf2_msgs/msg/TFMessage',definition)['tf2_msgs/msg/TFMessage']
        if existing is None:
            schema=writer.register_schema(name='tf2_msgs/msg/TFMessage',encoding='ros2msg',data=definition.encode())
            tf_channel=writer.register_channel(topic='/tf_static',message_encoding='cdr',schema_id=schema)
        else:tf_channel=channels[existing]
        for _,channel,msg in reader.iter_messages(log_time_order=False):
            if msg.log_time<last_log:raise ValueError('Old MCAP has nonmonotonic receipt order; manual inspection required')
            last_log=msg.log_time
            if first is None:
                first=msg.log_time
                transformed=merge_static({'transforms':list(edges.values())},specs,first)
                writer.add_message(channel_id=tf_channel,log_time=first,publish_time=first,sequence=0,data=encode(transformed))
                counts['/tf_static']=1
            if channel.topic=='/tf_static':continue
            fingerprint(channel,msg,before)
            writer.add_message(channel_id=channels[msg.channel_id],log_time=msg.log_time,publish_time=msg.publish_time,sequence=msg.sequence,data=msg.data)
            counts[channel.topic]=counts.get(channel.topic,0)+1
        for attachment in reader.iter_attachments():
            writer.add_attachment(create_time=attachment.create_time,log_time=attachment.log_time,name=attachment.name,media_type=attachment.media_type,data=attachment.data)
        for metadata in reader.iter_metadata():writer.add_metadata(metadata.name,metadata.metadata)
        if first is None:raise ValueError('Empty MCAP cannot establish a retrospective sample window')
        writer.finish();out.flush();os.fsync(out.fileno())
    after=hashlib.sha256();read_counts={}
    with destination.open('rb') as stream:
        for _,channel,msg in make_reader(stream,validate_crcs=True).iter_messages(log_time_order=False):
            read_counts[channel.topic]=read_counts.get(channel.topic,0)+1
            if channel.topic!='/tf_static':fingerprint(channel,msg,after)
    if before.digest()!=after.digest() or counts!=read_counts:raise ValueError('Non-TF data or count changed during rewrite')
    return dict(counts=counts,tf_old_count=tf_old_count,tf_new_count=1,first_log_ns=first,
                non_static_tf_sha256=after.hexdigest(),non_static_tf_payloads_unchanged=True)


def augment_assets(episode,specs,calibration_source,archive_root):
    info=json.loads((episode/'episode.json').read_text());assets=deepcopy(info['assets'])
    old_shared=(episode/assets['devices_ref']['path']).resolve().parent
    signature=hashlib.sha256((str(old_shared)+digest(calibration_source)+json.dumps(specs,sort_keys=True)).encode()).hexdigest()
    new_shared=archive_root/'_shared'/signature
    if not new_shared.exists():
        temp=new_shared.with_name('.'+signature+'.wrist-tmp')
        if temp.exists():raise FileExistsError(temp)
        shutil.copytree(old_shared,temp,symlinks=False)
        for name in ('wrist_left.json','wrist_right.json','wrist_acceptance.json'):
            dest=temp/'calibration/handeye'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(calibration_source/'handeye'/name,dest)
        write_json(temp/'calibration/wrist_extrinsics.json',{'static_extrinsics':specs})
        sync_tree(temp);temp.rename(new_shared)
    for key,part in [('devices_ref','devices.yaml'),('calibration_ref','calibration'),('config_ref','configs')]:
        path=new_shared/part;assets[key]=dict(path=os.path.relpath(path,episode),sha256=digest(path))
    return assets


def backfill(episode,specs,calibration_source,archive_root,backup_root):
    info=json.loads((episode/'episode.json').read_text());tag=hashlib.sha256((json.dumps(specs,sort_keys=True)+digest(calibration_source/'handeye/wrist_acceptance.json')).encode()).hexdigest()
    if info.get('wrist_extrinsics_backfill',{}).get('calibration_sha256')==tag:return {'episode':str(episode),'status':'ALREADY_APPLIED'}
    status=json.loads((episode/'manifest.json').read_text())['status']
    names=['episode.json','metadata.yaml','channel_stats.json','manifest.json','checksums.sha256']
    original={name:(episode/name).read_bytes() for name in names}
    source=episode/'data.mcap';original_stat=source.stat()
    backup=backup_root/episode.parent.name/episode.name;backup.mkdir(parents=True,exist_ok=False)
    os.link(source,backup/'data.mcap')
    for name,data in original.items():(backup/name).write_bytes(data)
    source_sha=digest(source)
    write_json(backup/'transaction.json',dict(status='PREPARING',episode=str(episode),original_data_sha256=source_sha))
    temporary=backup/'data.calibrated.mcap'
    result=rewrite(source,temporary,specs)
    assets=augment_assets(episode,specs,calibration_source,archive_root)
    if source.stat().st_mtime_ns!=original_stat.st_mtime_ns or source.stat().st_ino!=original_stat.st_ino or any((episode/name).read_bytes()!=data for name,data in original.items()):
        raise RuntimeError('Episode changed concurrently; not replacing original')
    metadata=yaml.safe_load(original['metadata.yaml']);bag=metadata['rosbag2_bagfile_information'];rows=bag['topics_with_message_count']
    tf=next((r for r in rows if r['topic_metadata']['name']=='/tf_static'),None)
    if tf is None:
        tf={'topic_metadata':dict(name='/tf_static',type='tf2_msgs/msg/TFMessage',serialization_format='cdr',offered_qos_profiles='')};rows.append(tf)
    tf['message_count']=1;bag['message_count']=sum(result['counts'].values())
    for row in bag.get('files',[]):
        if row.get('path')=='data.mcap':row['message_count']=bag['message_count']
    stats=json.loads(original['channel_stats.json']);stream_key='streams' if 'streams' in stats else 'channels';stat_rows=stats[stream_key]
    stat_rows[:]=[r for r in stat_rows if r['topic']!='/tf_static']
    stat_rows.append(dict(topic='/tf_static',message_count=1,first_time_ns=result['first_log_ns'],last_time_ns=result['first_log_ns'],actual_rate_hz=None,dropped_frames=0,encode_errors=0,decode_errors=0,null_reasons={'actual_rate_hz':'One static calibration snapshot'}))
    info['assets']=assets
    if isinstance(info.get('supplemental_calibration'),dict):
        info['supplemental_calibration']['wrist_extrinsics']='retrospectively supplemented from operator-supplied calibration; mounting unchanged'
    info['wrist_static_tf_status']='supplemented'
    info['dynamic_tf_recorded']=bool(result['counts'].get('/tf',0))
    info['wrist_extrinsics_backfill']=dict(calibration_sha256=tag,applied_at_unix_ns=time.time_ns(),
        original_data_sha256=source_sha,non_static_tf_sha256=result['non_static_tf_sha256'],
        basis='Operator confirms both wrist installations unchanged since original recordings',
        dynamic_tf_reconstructed=False,backup=str(backup))
    try:
        os.replace(temporary,source)
        write_json(episode/'episode.json',info)
        (episode/'metadata.yaml').write_text(yaml.safe_dump(metadata,sort_keys=False))
        write_json(episode/'channel_stats.json',stats);seal(episode,status);sync_tree(episode)
    except BaseException:
        restore=backup/'restore.mcap';os.link(backup/'data.mcap',restore);os.replace(restore,source)
        for name,data in original.items():(episode/name).write_bytes(data)
        raise
    write_json(backup/'transaction.json',dict(status='COMPLETE',episode=str(episode),original_data_sha256=source_sha,new_data_sha256=digest(source),**result))
    return dict(episode=str(episode),status='COMPLETE',**result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--profile',type=Path,required=True);p.add_argument('--episode',action='append');args=p.parse_args()
    profile=yaml.safe_load(args.profile.read_text());specs=profile['static_extrinsics']
    backup=args.root/'.calibration-backups'/str(time.time_ns());backup.mkdir(parents=True)
    episodes=[args.root/x for x in args.episode] if args.episode else sorted(ep.parent for grade in ('A','B','C','D') for ep in (args.root/grade).glob('episode*/data.mcap'))
    results=[]
    for ep in episodes:
        result=backfill(ep,specs,Path(profile['calibration_root']),args.root,backup);results.append(result);print(json.dumps({k:v for k,v in result.items() if k!='counts'}),flush=True)
        write_json(backup/'results.json',results)


if __name__=='__main__':main()
