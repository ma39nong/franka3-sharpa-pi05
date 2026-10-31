"""Cross-checkout camera lease held throughout startup, execution and cleanup."""
# ruff: noqa: RUF001 -- Chinese operator diagnostics use Chinese punctuation.

import fcntl
import os
from pathlib import Path
import tempfile

LEASE_ENV = "SHARPA_CAMERA_LEASE_FD"


class CameraLease:
    def __init__(self, path=None):
        self.path = Path(path or Path(tempfile.gettempdir()) / f"sharpa-deployment-camera-{os.getuid()}.lock")
        self.fd = None
        self.borrowed = False

    def acquire(self):
        inherited = os.environ.get(LEASE_ENV)
        if inherited:
            fd = int(inherited)
            stat = self.path.stat()
            opened = os.fstat(fd)
            if fd < 3 or (stat.st_dev, stat.st_ino) != (opened.st_dev, opened.st_ino):
                raise RuntimeError("部署相机锁继承无效；拒绝接管")
            self.fd, self.borrowed = fd, True
            return self
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(fd)
            raise RuntimeError("已有部署正在启动、执行或清理；禁止重复启动或接管相机") from error
        self.fd = fd
        return self

    def close(self):
        if self.fd is not None and not self.borrowed:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
        self.fd = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_):
        self.close()
