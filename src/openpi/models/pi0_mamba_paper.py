"""Mamba with the existing T0/T1/T2/T3/Tg data/prior contracts.

Multiple inheritance shares the original Paper configuration identity, allowing
unmodified PaperSharpaData/statistics/provenance to serve both architectures.
RTC uses the same clean-prefix training semantics as the Transformer.
"""

from dataclasses import dataclass
import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.pi0_mamba import MambaPi0
from openpi.models.pi0_mamba_config import MambaPi0Config
from openpi.shared import delta_paper as paper


@dataclass(frozen=True)
class PaperMambaPi0Config(MambaPi0Config):
    paper_asset_dir: str = ""
    paper_asset_sha256: str | None = None
    paper_mode: str = "global_quantile"
    paper_beta: float = 0.0
    paper_std_floor: float = 1e-4
    stage: str = "paper_delta_fm"
    action_representation: str = "delta_arms_absolute_hands"
    normalization: str = "paper"

    def validate_algorithm(self):
        if (
            self.architecture != "pi05_mamba3_jax"
            or self.stage != "paper_delta_fm"
            or self.action_representation != "delta_arms_absolute_hands"
            or self.normalization != "paper"
        ):
            raise ValueError("Paper Mamba requires arm Delta and matching Paper normalization")
        if self.paper_mode not in paper.MODES or not 0 <= self.paper_beta < 1:
            raise ValueError("Invalid Paper mode/beta")
        if not math.isfinite(self.paper_std_floor) or self.paper_std_floor <= 0:
            raise ValueError("Paper standard deviation floor must be positive and finite")

    def load(self, params, *, remove_extra_params=False):
        if self.paper_beta:
            _, arrays = paper.load_bundle(
                self.paper_asset_dir,
                mode=self.paper_mode,
                beta=self.paper_beta,
                horizon=self.action_horizon,
                std_floor=self.paper_std_floor,
                expected_sha256=self.paper_asset_sha256,
            )
            if "paper_noise_cholesky" not in params or not np.array_equal(
                np.asarray(params["paper_noise_cholesky"]), arrays["noise_cholesky"]
            ):
                raise ValueError("Checkpoint noise factor differs from authenticated Paper assets")
        return super().load(params, remove_extra_params=False)

    def create(self, rng):
        return PaperMambaPi0(self, rngs=nnx.Rngs(rng))


class PaperMambaPi0(MambaPi0):
    def __init__(self, config, *, rngs):
        if not config.paper_asset_sha256:
            raise ValueError("Compute matching Paper statistics and reload configuration before model creation")
        _, arrays = paper.load_bundle(
            config.paper_asset_dir,
            mode=config.paper_mode,
            beta=config.paper_beta,
            horizon=config.action_horizon,
            std_floor=config.paper_std_floor,
            expected_sha256=config.paper_asset_sha256,
        )
        super().__init__(config, rngs)
        self.paper_beta = config.paper_beta
        if self.paper_beta:
            self.paper_noise_cholesky = nnx.Variable(jnp.asarray(arrays["noise_cholesky"]))

    # Exact same time-major z @ L.T as the existing Transformer Paper model.
    def noise(self, rng, shape):
        value = jax.random.normal(rng, shape)
        if not self.paper_beta:
            return value
        flat = value.reshape(*shape[:-2], self.action_horizon * self.action_dim)
        return (flat @ self.paper_noise_cholesky.value.T).reshape(shape)
