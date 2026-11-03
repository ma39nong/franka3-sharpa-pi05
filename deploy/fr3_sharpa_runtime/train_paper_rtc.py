"""Normalize short references on the new Paper axis, then fix trained prefixes."""

from .paper_rtc import PaperRTCPrefixTransform
from .train_rtc import TrainRTCPolicy


class TrainPaperRTCPolicy(TrainRTCPolicy):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self._input_transform = PaperRTCPrefixTransform(self._input_transform)
