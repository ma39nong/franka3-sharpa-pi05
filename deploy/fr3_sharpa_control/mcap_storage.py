"""ROS-independent standard-message MCAP writer and content-addressed assets."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

import yaml

from .recording import write_json

VERSION = '1.2.0'
MAX_BYTES = 10_000_000_000 - 16_000_000
VIDEO_TYPE = 'foxglove_msgs/msg/CompressedVideo'
VIDEO_SCHEMA = '''builtin_interfaces/Time timestamp
string frame_id
string format
uint8[] data
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
'''


def digest(path):
    path = Path(path)
    if path.is_file():
        with path.open('rb') as stream:
            h = hashlib.file_digest(stream, 'sha256')
        return h.hexdigest()
    rows = [(p.relative_to(path).as_posix(), digest(p)) for p in sorted(path.rglob('*')) if p.is_file()]
    # Deterministic directory digest: sorted relative paths and file hashes.
    h = hashlib.sha256()
    for name, checksum in rows:
        h.update(name.encode()); h.update(b'\0'); h.update(checksum.encode()); h.update(b'\n')
    return h.hexdigest()


def seal(directory, status):
    """Refresh manifests after a metadata edit; callers serialize edits via quality lock."""
    directory = Path(directory)
    files = [dict(path=p.relative_to(directory).as_posix(), size=p.stat().st_size, sha256=digest(p))
             for p in sorted(directory.rglob('*')) if p.is_file()
             and p.name not in ('manifest.json', 'checksums.sha256')]
    write_json(directory/'manifest.json', dict(schema_version=2, status=status, files=files))
    entries = files + [dict(path='manifest.json', sha256=digest(directory/'manifest.json'))]
    tmp = directory/'.checksums.tmp'
    tmp.write_text(''.join(f'{row["sha256"]}  {row["path"]}\n' for row in entries))
    os.replace(tmp, directory/'checksums.sha256')


def sync_tree(directory):
    for path in Path(directory).rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _copy_asset(source, destination):
    """Reject external symlinks; referenced meshes must already be materialized."""
    source = Path(source).resolve()
    if not source.exists() or not any(p.is_file() for p in source.rglob('*')):
        raise ValueError(f'MCAP asset directory is missing/empty: {source}')
    for p in source.rglob('*'):
        if p.is_symlink():
            raise ValueError(f'MCAP assets must be materialized, no symlinks: {p}')
    if not destination.exists():
        temporary = destination.with_name('.'+destination.name+'-'+uuid.uuid4().hex)
        temporary.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, temporary)
        try:
            temporary.rename(destination)
        except OSError:
            if not destination.is_dir():
                raise
            shutil.rmtree(temporary)
    if digest(source) != digest(destination):
        raise ValueError(f'Existing shared asset changed: {destination}')


def materialize(root, profile):
    """Copy real model/calibration once, keeping per-episode references relocatable."""
    root = Path(root)
    model_source = Path(profile['model_root'])
    model_hash = digest(model_source)
    model = root/'assets/models'/model_hash
    _copy_asset(model_source, model)
    signature = hashlib.sha256(json.dumps(dict(profile=profile, model_sha256=model_hash,
        calibration_sha256=digest(profile['calibration_root'])), sort_keys=True).encode()).hexdigest()
    shared = root/'_shared'/signature
    if not shared.exists():
        temp = shared.with_name('.'+signature+'-'+uuid.uuid4().hex)
        temp.mkdir(parents=True)
        _copy_asset(profile['calibration_root'], temp/'calibration')
        (temp/'configs').mkdir()
        (temp/'devices.yaml').write_text(yaml.safe_dump({'devices': profile['devices']}, allow_unicode=True))
        (temp/'configs/recorder.yaml').write_text(yaml.safe_dump(profile, allow_unicode=True))
        (temp/'configs/clock_mapping.yaml').write_text(yaml.safe_dump(profile['clock_mapping']))
        try:
            temp.rename(shared)
        except OSError:
            if not shared.is_dir():
                raise
            shutil.rmtree(temp)
    return {'model_ref': model, 'calibration_ref': shared/'calibration',
            'devices_ref': shared/'devices.yaml', 'config_ref': shared/'configs'}


class McapEpisode:
    def __init__(self, target, *, root, profile, identity, start_time_ns):
        from mcap.writer import Writer, CompressionType, IndexType
        self.target = Path(target)
        if self.target.exists():
            raise FileExistsError(self.target)
        self.assets = materialize(root, profile)
        self.directory = self.target.with_name('.'+self.target.name+'-'+uuid.uuid4().hex+'.partial')
        self.directory.mkdir(parents=True)
        self.path = self.directory/'data.mcap'
        self.stream = self.path.open('wb')
        self.writer = Writer(self.stream, compression=CompressionType.ZSTD, index_types=IndexType.ALL,
                             enable_crcs=True, use_statistics=True, chunk_size=1_048_576)
        self.writer.start(profile='ros2', library=f'sharpa-mcap/{VERSION}')
        self.channels = {}; self.schemas = {}; self.stats = {}; self.last_log = 0
        self.identity = dict(identity)
        self.start = start_time_ns
        self.last_publish = {}
        self.end = start_time_ns
        self.closed = False
        self.failed = None
        seal(self.directory, 'recording')

    def register(self, topic, kind):
        from mcap_ros2._dynamic import serialize_dynamic
        from rosbags.typesys import get_typestore, Stores
        if kind not in self.schemas:
            definition = (VIDEO_SCHEMA if kind == VIDEO_TYPE else
                          get_typestore(Stores.ROS2_HUMBLE).generate_msgdef(kind, ros_version=2)[0])
            schema_id = self.writer.register_schema(name=kind, encoding='ros2msg', data=definition.encode())
            from .mcap_cdr import binary_encoder
            encode = binary_encoder(kind) or serialize_dynamic(kind, definition)[kind]
            self.schemas[kind] = schema_id, encode
        schema_id, encode = self.schemas[kind]
        channel_id = self.writer.register_channel(topic=topic, message_encoding='cdr', schema_id=schema_id)
        self.channels[topic] = channel_id, encode, kind
        self.stats[topic] = dict(topic=topic, message_count=0, first_time_ns=None, last_time_ns=None,
                                 dropped_frames=0, encode_errors=0, decode_errors=None,
                                 null_reasons={'decode_errors': 'Full decode is performed by external conformance validation'})

    def write(self, topic, message, *, log_time, publish_time):
        _, encode, _ = self.channels[topic]
        return self.write_serialized(topic, encode(message), log_time=log_time, publish_time=publish_time)

    def write_serialized(self, topic, data, *, log_time, publish_time):
        """Write original ROS CDR without decoding/re-encoding binary arrays."""
        if self.closed:
            raise RuntimeError('MCAP writer is closed')
        if log_time <= 0 or publish_time <= 0 or log_time < self.last_log:
            raise ValueError(f'MCAP invalid/nonmonotonic receipt time: {topic}')
        if topic not in ('/tf','/tf_static') and publish_time <= self.last_publish.get(topic, 0):
            raise ValueError(f'MCAP nonmonotonic source time: {topic}')
        channel_id, encode, _ = self.channels[topic]
        self.writer.add_message(channel_id=channel_id, log_time=int(log_time), publish_time=int(publish_time),
                                data=data, sequence=self.stats[topic]['message_count'])
        self.last_log = log_time; self.last_publish[topic] = publish_time
        self.end = max(self.end, publish_time)
        row = self.stats[topic]; row['message_count'] += 1
        if row['first_time_ns'] is None: row['first_time_ns'] = publish_time
        row['last_time_ns'] = publish_time
        if self.stream.tell() > MAX_BYTES:
            raise RuntimeError('MCAP approaches 10 GB; recorder does not support splitting')

    def cancel_prepared(self):
        """Discard only this writer's empty preparation, never a recorded episode."""
        if any(row['message_count'] for row in self.stats.values()):
            raise RuntimeError('Cannot discard a recording containing messages')
        if not self.closed:
            self.stream.close(); self.closed = True
            shutil.rmtree(self.directory)

    def close(self, error=None):
        if self.closed:
            return
        self.closed = True
        self.failed = str(error) if error else None
        try:
            self.writer.finish()
            self.stream.flush(); os.fsync(self.stream.fileno())
        finally:
            self.stream.close()
        from mcap.reader import make_reader
        # Re-read CRCs and counts before declaring a short episode complete.
        with self.path.open('rb') as stream:
            reader = make_reader(stream, validate_crcs=True)
            counts = {}
            for _, channel, _ in reader.iter_messages(log_time_order=False):
                counts[channel.topic] = counts.get(channel.topic, 0)+1
        if counts != {topic: row['message_count'] for topic,row in self.stats.items() if row['message_count']}:
            raise RuntimeError('MCAP readback count mismatch')
        for row in self.stats.values():
            duration = (row['last_time_ns']-row['first_time_ns']) if row['message_count'] else 0
            row['actual_rate_hz'] = ((row['message_count']-1)*1e9/duration if duration > 0 else None)
            if row['actual_rate_hz'] is None:
                row['null_reasons']['actual_rate_hz'] = 'Fewer than two distinct samples'
        write_json(self.directory/'channel_stats.json', {'streams': list(self.stats.values())})
        info = dict(schema_version=2, **self.identity, start_time_ns=self.start, end_time_ns=self.end,
                    task_result='unknown', recorder={'name':'sharpa-mcap','version':VERSION,
                    'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip()},
                    assets={key: {'path': os.path.relpath(path, self.target), 'sha256': digest(path)}
                            for key,path in self.assets.items()})
        if error: info['recording_error'] = str(error)
        write_json(self.directory/'episode.json', info)
        total = sum(row['message_count'] for row in self.stats.values())
        topics = [{'topic_metadata':{'name':topic,'type':kind,'serialization_format':'cdr','offered_qos_profiles':''},
                   'message_count': self.stats[topic]['message_count']} for topic,(_,_,kind) in self.channels.items()]
        metadata = {'rosbag2_bagfile_information': dict(version=5, storage_identifier='mcap',
            relative_file_paths=['data.mcap'], duration={'nanoseconds':max(0,self.end-self.start)},
            starting_time={'nanoseconds':self.start}, message_count=total, topics_with_message_count=topics,
            compression_format='', compression_mode='', files=[dict(path='data.mcap',
            starting_time={'nanoseconds':self.start}, duration={'nanoseconds':max(0,self.end-self.start)}, message_count=total)])}
        (self.directory/'metadata.yaml').write_text(yaml.safe_dump(metadata))
        self.identity['end_reason'] = info.get('end_reason','operator stop')
        seal(self.directory, 'failed' if error else 'complete')
        sync_tree(self.directory)
        self.directory.rename(self.target)
        fd=os.open(self.target.parent, os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
