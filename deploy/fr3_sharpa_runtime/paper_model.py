"""Inference adapter matching tomato-openpi PaperPi0 and its checkpoint state.

Base Pi0 retains RTC/Soft algorithms. Only the training distribution and the
normalization bridge are selected here; robot commands remain absolute 58D.
"""
from dataclasses import dataclass, replace

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.pi0 import Pi0
from openpi.models.pi0_config import Pi0Config
from openpi.shared import delta_paper as paper
from openpi.training.config import LeRobotFr3SharpaDataConfig
from openpi import transforms


@dataclass(frozen=True)
class PaperPi0Config(Pi0Config):
    paper_asset_dir: str = ""
    paper_mode: str = "timestamp_zscore_arms"
    paper_beta: float = 0.5
    paper_std_floor: float = 1e-4
    paper_asset_sha256: str = ""

    def create(self, rng):
        return PaperPi0(self, rngs=nnx.Rngs(rng))


class PaperPi0(Pi0):
    def __init__(self, config, *, rngs):
        _, arrays = paper.load_bundle(config.paper_asset_dir, mode=config.paper_mode,
            beta=config.paper_beta, horizon=config.action_horizon, std_floor=config.paper_std_floor,
            expected_sha256=config.paper_asset_sha256)
        super().__init__(config, rngs=rngs)
        self.paper_beta = config.paper_beta
        if self.paper_beta:
            self.paper_noise_cholesky = nnx.Variable(jnp.asarray(arrays["noise_cholesky"]))

    def noise(self, rng, shape):
        value = jax.random.normal(rng, shape)
        if not self.paper_beta:
            return value
        flat = value.reshape(*shape[:-2], self.action_horizon * self.action_dim)
        return (flat @ self.paper_noise_cholesky.value.astype(value.dtype).T).reshape(shape)

    def sample_actions(self, rng, observation, *, noise=None, **kwargs):
        if noise is None:
            noise = self.noise(rng, (observation.state.shape[0], self.action_horizon, self.action_dim))
        return super().sample_actions(rng, observation, noise=noise, **kwargs)


def normalization_bridge(model, stats):
    metadata, arrays = paper.load_bundle(model.paper_asset_dir, mode=model.paper_mode,
        beta=model.paper_beta, horizon=model.action_horizon, std_floor=model.paper_std_floor,
        expected_sha256=model.paper_asset_sha256)
    return paper.ArmNormalization(model.paper_mode, arrays["arm_mean"], arrays["arm_std"],
        stats["actions"].q01, stats["actions"].q99, model.paper_std_floor, metadata["bundle_sha256"])


@dataclass(frozen=True)
class PaperSharpaData(LeRobotFr3SharpaDataConfig):
    def create(self, assets_dirs, model_config):
        base = super().create(assets_dirs, model_config)
        bridge = normalization_bridge(model_config, base.norm_stats)
        return replace(base, model_transforms=transforms.Group(
            inputs=(bridge, *base.model_transforms.inputs),
            outputs=(*base.model_transforms.outputs, replace(bridge, inverse=True))))


def reference_normalizer(stats, model):
    """Short references occupy NEW chunk indices 0..N, never old indices26..29."""
    normalize = transforms.Normalize(stats, use_quantiles=True)
    if not isinstance(model, PaperPi0Config):
        return normalize
    bridge = normalization_bridge(model, stats)

    def apply(data):
        values = np.asarray(data["actions"], dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 58 or not 1 <= len(values) <= 50:
            raise ValueError("Paper reference must be1..50x58")
        padded = np.zeros((50, 58), dtype=np.float32)
        padded[:len(values)] = values
        return {**data, "actions": bridge(normalize({"actions": padded}))["actions"][:len(values)]}
    return apply
