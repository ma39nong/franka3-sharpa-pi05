"""One mapping of model identities to checkpoints, ports and contracts."""

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class ModelSpec:
    name: str
    port: int
    checkpoint: str
    source_hz: int
    contract_module: str

    @property
    def uri(self) -> str:
        return f"ws://127.0.0.1:{self.port}"

    def contract(self, checkpoint):
        return import_module(self.contract_module).checkpoint_contract(checkpoint)


MODELS = {
    "19999": ModelSpec("19999", 8001, "checkpoints/19999", 30, "deploy.fr3_wuji_models.model_19999.serve"),
    "30000": ModelSpec("30000", 8002, "checkpoints/30000", 30, "deploy.fr3_wuji_models.model_30000.serve"),
    "30000v2": ModelSpec("30000v2", 8003, "checkpoints/30000v2", 30, "deploy.fr3_wuji_models.model_30000v2.contract"),
    "25000": ModelSpec("25000", 8004, "checkpoints/25000", 15, "deploy.fr3_wuji_models.model_25000.contract"),
    "25000-single": ModelSpec(
        "25000-single", 8005, "checkpoints/25000-single", 15, "deploy.fr3_wuji_models.model_25000_single.contract"
    ),
    "20hz": ModelSpec(
        "20hz",
        8006,
        "checkpoints/pi05_fr3_wuji_20hz/tomato_lora_0918_20hz/19999",
        20,
        "deploy.fr3_wuji_models.model_20hz.serve",
    ),
}

# Descriptive operator names; checkpoint/service identities stay stable.
MODEL_ALIASES = {
    "54": "19999",
    "64-lora-30hz": "30000",
    "64-full-30hz": "30000v2",
    "64-full-15hz-ab": "25000",
    "64-full-15hz-a": "25000-single",
    "64": "30000",  # Compatibility with the initial dimension-only name.
}


def profile(name: str) -> ModelSpec:
    try:
        return MODELS[MODEL_ALIASES.get(name, name)]
    except KeyError as error:
        raise ValueError(f"Unknown model profile: {name}") from error
