"""Synchronized Soft RTC warmup using the existing wire/receipt protocol."""

import hashlib
import time

import numpy as np
from openpi_client import rtc

from .protocol import as_finite_action_chunk
from .rtc_stream import PAPER_RTC_ADAPTER


def warmup_soft_rtc(policy, observation, reference, *, paper=False):
    reference = as_finite_action_chunk(reference)
    timings, errors = [], []
    lengths = (50, 50, 35, 1) if paper else (50, 50)
    for number, length in enumerate(lengths):
        horizon = min(25, length)
        context = rtc.validate_context(
            {
                "protocol_version": rtc.PROTOCOL_VERSION,
                "token": f"mamba-soft-warmup-{number}",
                "source_number": 1,
                "source_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
                "previous_actions": reference[:length],
                "inference_delay_steps": min(5, length),
                "prefix_attention_horizon": horizon,
                "prefix_attention_schedule": "exp",
                "max_guidance_weight": 5.0,
            }
        )
        started = time.perf_counter()
        result = policy.infer(observation, rtc_context=context)
        # Host conversion waits for JAX execution, including VJP, to complete.
        actions = as_finite_action_chunk(result["actions"])
        timings.append((time.perf_counter() - started) * 1000)
        rtc.validate_receipt(result.get("rtc"), context)
        error = np.abs(actions[:horizon] - reference[:horizon])
        errors.append(
            {
                "reference_length": length,
                "overlap_mae_rad": float(error.mean()),
                "overlap_p95_rad": float(np.percentile(error, 95)),
                "overlap_max_rad": float(error.max()),
                "first_frame_linf_rad": float(np.max(np.abs(actions[0] - reference[0]))),
                "first_velocity_linf_rad_s": float(
                    np.max(np.abs(np.diff(actions[:2], axis=0) - np.diff(reference[:2], axis=0))) * 30
                ),
            }
        )
    metadata = {
        "rtc_supported": True,
        "rtc_protocol_version": rtc.PROTOCOL_VERSION,
        "rtc_action_dim": 58,
        "rtc_algorithm": "soft_prefix_guidance",
        "rtc_warmed_up": True,
        "rtc_smoke_action_shape": [50, 58],
        "rtc_warm_infer_ms": timings[1],
        "rtc_soft_warmup": {
            "wall_ms": timings,
            "overlap_errors": errors,
            "timing_scope": "host policy with synchronization; first call includes compilation",
            "performance_admission": "NOT_MEASURED",
        },
    }
    if paper:
        metadata.update(
            rtc_input_adapter=PAPER_RTC_ADAPTER, rtc_short_prefix_warmed_up=True, rtc_smoke_prefix_lengths=[50, 35, 1]
        )
    return metadata
