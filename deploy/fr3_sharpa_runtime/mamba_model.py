"""Small deployment factories; training dataloaders and initialization are excluded."""

from dataclasses import dataclass
from dataclasses import replace

import numpy as np

from openpi.policies.policy import Policy
from openpi.training.config import AssetsConfig
from openpi.training.config import LeRobotFr3SharpaDataConfig
from openpi.training.config import TrainConfig

from .paper_model import PaperSharpaData
from .paper_rtc import PaperRTCPolicy
from .train_paper_rtc import TrainPaperRTCPolicy
from .train_rtc import TrainRTCPolicy


@dataclass(frozen=True)
class MambaSharpaData(LeRobotFr3SharpaDataConfig):
    def create(self, assets_dirs, model_config):
        return replace(super().create(assets_dirs, model_config), use_quantile_norm=False)


def build_config(metadata, config):
    from .mamba_contract import reconstruct_model

    model = reconstruct_model(metadata, config.NORM_STATS)
    data_cls = PaperSharpaData if metadata["paper_algorithm"] else MambaSharpaData
    data = data_cls(
        repo_id=config.NORM_STATS.parent.name,
        assets=AssetsConfig(assets_dir=str(config.NORM_STATS.parent.parent), asset_id=config.NORM_STATS.parent.name),
        extra_delta_transform=bool(metadata["paper_algorithm"]),
        action_hz=30.0,
    )
    return TrainConfig(name=config.CONFIG_NAME, model=model, data=data, freeze_filter=model.get_freeze_filter())


class MambaPolicy(Policy):
    soft_rtc = False

    def infer(self, obs, *, noise=None, rtc_context=None):
        if noise is not None:
            noise = np.asarray(noise)
            if noise.shape not in ((50, 58), (1, 50, 58)) or not np.isfinite(noise).all():
                raise ValueError("Mamba noise must be finite (50,58) or (1,50,58)")
        if rtc_context is not None and not (self.soft_rtc or isinstance(self, TrainRTCPolicy)):
            raise ValueError("Mamba serial policy rejects RTC; select an explicit Soft/Hard RTC policy")
        result = super().infer(obs, noise=noise, rtc_context=rtc_context)
        if rtc_context is not None and isinstance(self, TrainRTCPolicy):
            from openpi_client import rtc

            context = rtc.validate_context(rtc_context)
            # Match upstream offline policy's exact physical prefix after the
            # numerically lossy normalization round trip. The sampler itself
            # already fixes the encoded prefix at every Euler step.
            result["actions"] = np.array(result["actions"], copy=True)
            delay = context.inference_delay_steps
            result["actions"][:delay] = context.previous_actions[:delay]
        from .protocol import as_finite_action_chunk

        result["actions"] = as_finite_action_chunk(result["actions"])
        return result


class MambaSoftRTCPolicy(MambaPolicy):
    soft_rtc = True
    paper_soft_rtc = False

    def __init__(self, model, **kwargs):
        expected = "paper" if self.paper_soft_rtc else "ordinary_per_dimension"
        if model.rtc_max_delay or model.config.normalization != expected:
            raise ValueError("Mamba Soft RTC requires untrained RTC weights and the matching Plain/Paper policy")
        raw = set(kwargs.get("sample_kwargs") or {})
        if raw.intersection(
            {"prefix_actions", "prefix_length", "rtc_actions", "rtc_weights", "rtc_max_guidance_weight"}
        ):
            raise ValueError("Mamba RTC inputs must come from the validated request context")
        super().__init__(model, **kwargs)

    def _rtc_sample_kwargs(self, context, target):
        from openpi.models.mamba3.soft_rtc import validate_soft_inputs

        values = super()._rtc_sample_kwargs(context, target)
        validate_soft_inputs(
            values["rtc_actions"],
            values["rtc_weights"],
            values["rtc_max_guidance_weight"],
            (1, self._model.action_horizon, self._model.action_dim),
        )
        return values


class MambaPaperSoftRTCPolicy(MambaSoftRTCPolicy, PaperRTCPolicy):
    paper_soft_rtc = True


class MambaTrainRTCPolicy(MambaPolicy, TrainRTCPolicy):
    pass


class MambaTrainPaperRTCPolicy(MambaPolicy, TrainPaperRTCPolicy):
    pass
