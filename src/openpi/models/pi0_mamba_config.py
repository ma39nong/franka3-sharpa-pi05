"""Opt-in Phase A architecture; Pi05 identity preserves tokenizer/state semantics."""

import dataclasses

from flax import nnx

from openpi.models.mamba3.config import Mamba3Config
from openpi.models.pi0_config import Pi0Config


@dataclasses.dataclass(frozen=True)
class MambaPi0Config(Pi0Config):
    # Inherited action_expert_variant is serialization compatibility only; it is
    # never constructed and never determines Mamba width or pretrained loading.
    pi05: bool = True
    action_dim: int = 58
    action_horizon: int = 50
    max_token_len: int = 256
    rtc_max_delay: int = 0
    architecture: str = "pi05_mamba3_jax"
    stage: str = "phase_a_plain_fm"
    action_representation: str = "absolute"
    normalization: str = "ordinary_per_dimension"
    diagnostic_only: bool = False
    mamba: Mamba3Config = dataclasses.field(default_factory=Mamba3Config)
    mamba_depth: int = 4
    mamba_mlp_dim: int = 2048
    film_mlp_dim: int = 2048
    context_heads: int = 8
    context_vision_tokens: int = 44
    context_language_tokens: int = 16
    context_global_tokens: int = 4

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05 or not self.discrete_state_input:
            raise ValueError("Mamba Phase A requires Pi05 discrete state text")
        if type(self.rtc_max_delay) is not int or not 0 <= self.rtc_max_delay < self.action_horizon:
            raise ValueError("rtc_max_delay must be an integer in [0, action_horizon)")
        self.validate_algorithm()
        if "lora" in self.paligemma_variant or "lora" in self.action_expert_variant:
            raise ValueError("Mamba Phase A supports FullFT only, no LoRA")
        if self.dtype != self.mamba.dtype:
            raise ValueError("VLM and Mamba compute dtype must match")
        if not self.diagnostic_only and (
            self.action_dim != 58 or self.action_horizon != 50 or self.paligemma_variant != "gemma_2b"
        ):
            raise ValueError("Production Phase A requires complete gemma_2b VLM, H50/D58")
        for name in (
            "mamba_depth",
            "mamba_mlp_dim",
            "film_mlp_dim",
            "context_heads",
            "context_vision_tokens",
            "context_language_tokens",
            "context_global_tokens",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.mamba.hidden_size % self.context_heads:
            raise ValueError("Context heads must divide Mamba width")

    def validate_algorithm(self):
        if self.action_representation != "absolute" or self.normalization != "ordinary_per_dimension":
            raise ValueError("Mamba Phase A rejects Delta and Paper normalization")
        if self.architecture != "pi05_mamba3_jax" or self.stage != "phase_a_plain_fm":
            raise ValueError("Unsupported Mamba architecture/stage")

    def load(self, params, *, remove_extra_params=False):
        import jax
        import jax.numpy as jnp

        from openpi.shared import array_typing as at

        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=True)

        if not bool(jax.device_get(jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in jax.tree.leaves(params)])))):
            raise ValueError("Mamba checkpoint contains non-finite parameters")
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def create(self, rng):
        from openpi.models.pi0_mamba import MambaPi0

        return MambaPi0(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter(self):
        return nnx.Nothing

    def load_pytorch(self, train_config, weight_path):
        raise ValueError("Mamba Phase A is JAX-only; use the architecture-specific checkpoint loader")
