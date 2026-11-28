"""Recording gate shared by control workers; never owns robot shutdown."""
from contextlib import suppress
import json
import os
from pathlib import Path
import socket
import threading


def rpc(path, operation, *, timeout=10):
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(timeout)
        client.connect(str(path))
        client.sendall(json.dumps({"operation": operation}).encode() + b"\n")
        data = b""
        while b"\n" not in data:
            block = client.recv(4096)
            if not block:
                raise RuntimeError("录制器提前关闭连接")
            data += block
            if len(data) > 65536:
                raise ValueError("录制器应答超长")
    result = json.loads(data.split(b"\n", 1)[0])
    if result.get("status") != "PASS":
        raise RuntimeError(result.get("error", "录制请求失败"))
    return result


def mark_stop(path):
    path = Path(path)
    if path.parent.is_dir():
        (path.parent / "stop-recording").touch(exist_ok=True)


class RecordingGate:
    def __init__(self, path=None):
        self.path = path if path is not None else os.environ.get("SHARPA_RECORD_SOCKET")
        self.started = False

    @property
    def enabled(self):
        return bool(self.path)

    def start(self):
        if self.enabled and not self.started:
            rpc(self.path, "start")
            self.started = True

    def stop(self, *, wait=True):
        if not self.enabled:
            return
        if Path(self.path).exists():
            with suppress(OSError):
                mark_stop(self.path)
        if not wait:
            threading.Thread(target=self.stop, daemon=True).start()
            return
        try:
            rpc(self.path, "stop", timeout=2)
        except (OSError, RuntimeError, ValueError) as error:
            # The recorder may already have been stopped by the UI. Persist
            # the warning but never prevent the existing hand disable path.
            from deploy.fr3_sharpa_runtime.shutdown import stderr
            stderr(f"[recording] stop: {error}")


def play_task_frames(frames, bridge_frames, send, feedback, gate, *, resume, play=None):
    if play is None:
        from deploy.fr3_sharpa_runtime.sdk_direct import play_command_frames
        play = play_command_frames
    if gate.enabled and not gate.started:
        if bridge_frames:
            play(frames[:bridge_frames], send, feedback)
        gate.start()  # Recording ready BEFORE the first task frame, not Home/bridge.
        resume()      # Recorder startup must not trip the inter-phase watchdog.
        play(frames[bridge_frames:], send, feedback)
    else:
        play(frames, send, feedback)


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def command_publisher(node, gate):
    if not gate.enabled:
        return lambda values, stamp: None
    from sensor_msgs.msg import JointState
    from deploy.fr3_sharpa_runtime.protocol import SHARPA_JOINT_NAMES
    publisher = node.create_publisher(JointState, "/openpi/executed_joint_targets", 30)
    names = ([f"left_fr3_joint{i}" for i in range(1, 8)] + [f"left_{n}" for n in SHARPA_JOINT_NAMES]
             + [f"right_fr3_joint{i}" for i in range(1, 8)] + [f"right_{n}" for n in SHARPA_JOINT_NAMES])
    def publish(values, stamp):
        message = JointState()
        message.header.stamp.sec = stamp // 1_000_000_000
        message.header.stamp.nanosec = stamp % 1_000_000_000
        message.name, message.position = names, [float(value) for value in values]
        publisher.publish(message)
    return publish


COMPACT_HEADER = '[record-layout compact-v1]'


def compact_run(directory):
    """New runs opt in explicitly; never infer a migration for historical runs."""
    try:
        with (Path(directory) / 'run.log').open() as stream:
            return stream.readline().strip() == COMPACT_HEADER
    except OSError:
        return False


def compact_finalize(directory):
    """Consolidate transient lifecycle reports only after all readers/writers close."""
    directory = Path(directory)
    if not compact_run(directory):
        return
    from . import quality
    with quality.locked(directory):
        try:
            value = quality.read(directory)
        except (OSError, ValueError):
            value = None
        if value and not value.get('review_ready'):
            return
        export = directory/'video_export.json'
        if export.exists() and json.loads(export.read_text()).get('status') in ('QUEUED','EXPORTING','RUNNING'):
            return
        # Final reports are folded into episode metadata and the one diagnostic
        # log. Raw arrays and policy images are never copied into the log.
        reports = {}
        for name in ('run.json','model_metadata.json','dagger_manifest.json','recording.json','video_export.json'):
            path = directory/name
            if path.is_file():
                reports[name] = json.loads(path.read_text())
        trace = directory/'logs'
        diagnostics = {}
        if trace.is_dir():
            for path in trace.glob('*.json'):
                diagnostics[path.name] = json.loads(path.read_text())
        if not reports and not diagnostics:
            return
        stored_in_episode=False
        if value:
            bag = quality.payload(directory, value)
            if bag.is_dir():
                episode_path = bag/'episode.json'
                episode = json.loads(episode_path.read_text()) if episode_path.exists() else {'schema_version':2}
                episode.update(record_layout='compact-v1',quality=value,
                    run_summary={**episode.get('run_summary',{}),**reports},
                    diagnostics={**episode.get('diagnostics',{}),**diagnostics})
                write_json(episode_path, episode)
                stored_in_episode=True
                if (bag/'manifest.json').is_file():
                    from .mcap_storage import seal
                    seal(bag, json.loads((bag/'manifest.json').read_text())['status'])
        with (directory/'run.log').open('a') as stream:
            summary=(dict(component='final_summary',episode=str(bag.resolve()),
                          execution=value.get('execution'),video_export=reports.get('video_export.json',{}).get('status'))
                     if stored_in_episode else dict(component='final_summary',quality=value,reports=reports,diagnostics=diagnostics))
            stream.write(json.dumps(summary,ensure_ascii=False)+'\n')
        # Only files generated by the new compact lifecycle are transient.
        for name in (*reports,'data_quality.json','ui_status.json','operator-stop.json'):
            (directory/name).unlink(missing_ok=True)
        if trace.is_dir():
            for path in trace.glob('*.json'):path.unlink()
            try:trace.rmdir()
            except OSError:pass


def run_report(directory, name):
    """Read a lifecycle report before or after compact consolidation."""
    directory=Path(directory)
    try:
        return json.loads((directory/name).read_text())
    except FileNotFoundError:
        if not compact_run(directory):raise
    for alias in ('mcap','rosbag'):
        episode=directory/alias/'episode.json'
        if episode.is_file():
            return json.loads(episode.read_text()).get('run_summary',{}).get(name,{})
    result={}
    with (directory/'run.log').open() as stream:
        for line in stream:
            try:
                item=json.loads(line)
                if item.get('component')=='final_summary':result=item.get('reports',{}).get(name,result)
            except (ValueError,AttributeError):pass
    return result
