"""Bounded auditing and authenticated local DAgger requests; no hardware access."""
import json
from pathlib import Path
import queue
import threading
import time

from .task_control import TaskServer, task_rpc

FRAME_TOPIC = '/dagger/frame'
EVENT_TOPIC = '/dagger/events'


class AuditWriter:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.queue = queue.Queue(maxsize=2048)
        self.error = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.work, daemon=True, name='dagger-audit')
        self.thread.start()

    def put(self, item):
        self.check()
        try:
            self.queue.put_nowait(item)
        except queue.Full as error:
            self.error = 'audit backlog full'
            raise RuntimeError('DAgger audit backlog full; stop rather than lose labels') from error

    def check(self):
        if self.error:
            raise RuntimeError('DAgger audit failed: ' + self.error)

    def work(self):
        try:
            with (self.directory/'frames.jsonl').open('x') as frames, (self.directory/'events.jsonl').open('x') as events:
                while not self.stopped.is_set() or not self.queue.empty():
                    try:
                        item = self.queue.get(timeout=.1)
                    except queue.Empty:
                        frames.flush(); events.flush()
                        continue
                    stream = events if 'event' in item else frames
                    stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False)+'\n')
        except Exception as error:
            self.error = str(error)

    def close(self):
        self.stopped.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('DAgger audit did not drain; recording incomplete')
        self.check()


class ControlChannel:
    """RPC threads enqueue intents; only the dispatch loop mutates authority."""
    def __init__(self, path, authority):
        self.authority = authority
        self.lock, self.run_id = authority.lock, authority.run_id
        self.requests = queue.Queue(maxsize=32)
        self.server = TaskServer(path, self)

    @property
    def last_contact(self):
        return self.authority.last_contact

    @last_contact.setter
    def last_contact(self, value):
        self.authority.last_contact = value

    def snapshot(self):
        return self.authority.snapshot()

    def command(self, command, **fields):
        done = threading.Event()
        reply = {}
        request = dict(command=command, fields=fields, done=done, reply=reply,
                       expires=time.monotonic()+.4)
        self.requests.put_nowait(request)
        if not done.wait(.6):
            raise TimeoutError('DAgger executor did not confirm the request')
        if 'error' in reply:
            raise ValueError(reply['error'])
        return reply['result']

    def apply(self, measured, human):
        pending = []
        for _ in range(8):
            try:
                request = self.requests.get_nowait()
            except queue.Empty:
                break
            try:
                if time.monotonic() >= request['expires']:
                    raise ValueError('Expired DAgger request; authority unchanged')
                self.authority.command(request['command'], measured=measured, human=human, **request['fields'])
                pending.append(request)
            except Exception as error:
                request['reply']['error'] = str(error)
                request['done'].set()
        return pending

    def confirm(self, pending):
        for request in pending:
            request['reply']['result'] = self.snapshot()
            request['done'].set()

    def close(self):
        self.server.close()


def rpc(path, run_id, command='status', **fields):
    return task_rpc(path, run_id, command, timeout=.8, **fields)
