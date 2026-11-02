"""Paper noise and trained hard prefixes, matching training commit e93a3aa."""

from dataclasses import dataclass

from flax import nnx

from .paper_model import PaperPi0
from .paper_model import PaperPi0Config
from .train_rtc_model import TrainRTCPi0


@dataclass(frozen=True)
class TrainPaperRTCPi0Config(PaperPi0Config):
    rtc_max_delay: int = 0

    def __post_init__(self):
        # Apply the same shape/delay rules as the ordinary trained sampler.
        super().__post_init__()
        if not self.pi05 or type(self.rtc_max_delay) is not int or not 0 < self.rtc_max_delay < self.action_horizon:
            raise ValueError("Train Paper RTC requires pi05 and 0 < rtc_max_delay < horizon")

    def create(self, rng):
        return TrainPaperRTCPi0(self, rngs=nnx.Rngs(rng))


class TrainPaperRTCPi0(PaperPi0, TrainRTCPi0):
    """PaperPi0 supplies correlated noise before TrainRTCPi0 fixes the prefix."""
