"""Record directly to a file; terminal mirroring cannot break runtime output."""

import argparse
from contextlib import suppress
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading


def record_command(command, path, *, env=None, cwd=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    child = None
    interrupted = None
    finished = threading.Event()

    def forward(number, _frame):
        nonlocal interrupted
        if interrupted is not None:
            return
        interrupted = number
        if child is not None and child.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(child.pid, number)

    def mirror():
        # Only this reader writes to the terminal. Runtime stderr always points
        # to the regular file, even if this terminal disappears.
        with path.open("rb") as reader:
            while True:
                data = reader.read(65536)
                if data:
                    try:
                        sys.stdout.buffer.write(data)
                        sys.stdout.buffer.flush()
                    except (OSError, ValueError):
                        # Python flushes stdout again at interpreter exit.
                        # Redirect that descriptor too, preserving child status.
                        try:
                            with open(os.devnull, "wb") as sink:
                                os.dup2(sink.fileno(), sys.stdout.fileno())
                        except (OSError, ValueError):
                            pass
                        return
                elif finished.is_set():
                    return
                else:
                    finished.wait(0.05)

    previous = {number: signal.signal(number, forward) for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        # Reject accidental reuse of a previous run's console log.
        with path.open("xb", buffering=0) as output:
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True, env=env, cwd=cwd)
            if interrupted is not None:
                os.killpg(child.pid, interrupted)
            reader = threading.Thread(target=mirror, daemon=True)
            reader.start()
            try:
                status = child.wait()
            finally:
                finished.set()
                reader.join(timeout=2)
        return 128 - status if status < 0 else status
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    raise SystemExit(record_command(command, args.log))


if __name__ == "__main__":
    main()
