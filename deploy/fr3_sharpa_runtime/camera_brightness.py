"""Detect sustained underexposure in the head-camera RGB stream."""

import numpy as np


class HeadBrightness:
    stable_ns = 3_000_000_000

    def __init__(self):
        self.bright_since = None
        self.dark_since = None
        self.mean = None
        self.p95 = None

    def observe(self, message, now_ns):
        rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
        pixels = rows[::8, :message.width * 3].reshape(-1, message.width, 3)[:, ::8]
        # Mean channel intensity treats RGB and BGR identically.
        intensity = pixels.mean(axis=2)
        self.mean = float(intensity.mean())
        self.p95 = float(np.percentile(intensity, 95))
        dark = self.mean < 20 and self.p95 < 48
        if dark:
            self.bright_since = None
            if self.dark_since is None:
                self.dark_since = now_ns
        else:
            self.dark_since = None
            if self.bright_since is None:
                self.bright_since = now_ns

    def ready(self, now_ns):
        return self.bright_since is not None and now_ns - self.bright_since >= self.stable_ns

    def failed(self, now_ns):
        return self.dark_since is not None and now_ns - self.dark_since >= self.stable_ns

    def diagnostic(self, now_ns):
        return {"mean": self.mean, "p95": self.p95,
                "dark_seconds": (now_ns - self.dark_since) / 1e9 if self.dark_since is not None else 0,
                "required_bright_seconds": self.stable_ns / 1e9,
                "mean_threshold": 20, "p95_threshold": 48}
