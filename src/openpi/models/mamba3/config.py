"""Minimal mixer hyperparameters (no language-model configuration).

Defaults verified against MambaAct 97414cf/mamba/mamba_ssm/modules/mamba3.py.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Mamba3Config:
    hidden_size: int = 512
    state_size: int = 64
    head_dim: int = 64
    expand: int = 1
    ngroups: int = 1
    chunk_size: int = 64
    rope_fraction: float = 0.5
    A_floor: float = 1e-4
    bcnorm_eps: float = 1e-5
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    dtype: str = "bfloat16"

    def __post_init__(self):
        for name in ("hidden_size", "state_size", "head_dim", "expand", "ngroups", "chunk_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.intermediate_size % self.head_dim or self.num_heads % self.ngroups:
            raise ValueError("Mamba width/head/groups must divide exactly")
        if self.state_size % 2 or self.num_rope_angles <= 0 or self.rope_fraction not in (0.5, 1.0):
            raise ValueError("Mamba rotary state must have valid even pairs")
        if not 0 < self.time_step_floor <= self.time_step_min <= self.time_step_max:
            raise ValueError("Invalid SSM dt initialization")
        if self.A_floor <= 0 or self.bcnorm_eps <= 0 or self.dtype not in ("float32", "bfloat16"):
            raise ValueError("Invalid Mamba precision/numerics")

    @property
    def intermediate_size(self):
        return self.hidden_size * self.expand

    @property
    def num_heads(self):
        return self.intermediate_size // self.head_dim

    @property
    def num_rope_angles(self):
        return int(self.state_size * self.rope_fraction) // 2

    @property
    def d_in_proj(self):
        return (
            2 * self.intermediate_size + 2 * self.state_size * self.ngroups + 3 * self.num_heads + self.num_rope_angles
        )
