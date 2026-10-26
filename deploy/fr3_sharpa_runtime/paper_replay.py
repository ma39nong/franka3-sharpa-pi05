"""Synthetic CPU ablation. No weights, cameras, ROS, sockets or SDK connections."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import jax.numpy as jnp
import numpy as np

from openpi.models.inpainting import correct_inpainting

from .compression import compression_candidate
from .compression import padded_targets
from .rolling import RollingOptions
from .sdk_direct import chunk_command_frames

GROUPS = {"left_arm": slice(0, 7), "left_hand": slice(7, 29), "right_arm": slice(29, 36), "right_hand": slice(36, 58)}


def toy_sample(noise, target, reference=None, matrix=None):
    """Deliberately imperfect toy vector field, not a learned policy or robot model."""
    x = jnp.asarray(noise)[None]
    noise = x
    target = jnp.asarray(target)[None]
    counts = jnp.zeros(2, dtype=jnp.int32)
    for step in range(10):
        t = 1 - step / 10
        x = x - 0.1 * (0.7 * (x - target) / max(t, 0.2))
        if reference is not None:
            x, audit = correct_inpainting(
                x, jnp.asarray(reference)[None], noise, jnp.float32(t - 0.1), correction_matrix=matrix
            )
            counts += audit
    return np.asarray(x[0]), np.asarray(counts)


def run_ablation(*, seed=42, rounds=5, speed_scale=1.0):
    modes = [
        ("default_50_15", 15, "off", "off"),
        ("rolling_off", 26, "off", "off"),
        ("soft", 26, "soft", "off"),
        ("correlation", 26, "correlation", "off"),
        ("soft_cubic", 26, "soft", "cubic"),
        ("correlation_cubic", 26, "correlation", "cubic"),
    ]
    rng = np.random.default_rng(seed)
    noises = rng.normal(0, 0.005, (rounds, 50, 58)).astype(np.float32)
    # Analytic cross-block of synthetic Sigma = I + 0.1*11^T after beta=.5 shrinkage.
    # It is never presented as a training-data covariance asset.
    matrix = jnp.full((46 * 58, 4 * 58), 0.05 / (1 + 0.05 * 232), dtype=jnp.float32)
    rows = []
    for name, consumed, inpainting, compression in modes:
        previous = None
        state = np.zeros(58, np.float32)
        times, commands, jumps, errors = [], [], [], []
        elapsed, total_bridge, total_frames, compressed_rounds = 0.0, 0, 0, 0
        skipped = applied = 0
        started = time.perf_counter()
        for index in range(rounds):
            # Same controlled observations/targets across all rolling variants.
            phase = index * consumed + np.arange(50)
            target = np.broadcast_to((0.02 * np.sin(phase / 40))[:, None], (50, 58)).astype(np.float32).copy()
            reference = previous[26:30] if previous is not None and inpainting != "off" else None
            predicted, counts = toy_sample(
                noises[index], target, reference, matrix if inpainting == "correlation" else None
            )
            applied += int(counts[0])
            skipped += int(counts[1])
            if previous is not None:
                jumps.append(np.abs(predicted[0] - previous[consumed - 1]))
            errors.append((predicted - target) ** 2)
            if consumed == 26:
                selected, detail = compression_candidate(
                    predicted, state, RollingOptions(inpainting=inpainting, compression=compression)
                )
                compressed_rounds += int(detail["applied"])
            else:
                selected = predicted[:consumed]
            # Synthetic planning bounds only; no native hardware range validation is claimed.
            if np.max(np.abs(selected)) > 1:
                raise ValueError("Synthetic trace outside +/-1 rad toy bounds")
            frames, bridge = chunk_command_frames(
                padded_targets(selected),
                state,
                speed_scale=speed_scale,
                execute_steps=len(selected),
                approach_speed_rad_s=0.03,
            )
            # Model/communication pause is an explicit 60ms hold in this toy timeline.
            if commands:
                elapsed += 0.06
                commands.append(state.copy())
                times.append(elapsed)
                elapsed += 1 / 30
            for frame in frames:
                commands.append(frame)
                times.append(elapsed)
                elapsed += 1 / 30
            state = frames[-1].copy()
            previous = predicted
            total_bridge += bridge
            total_frames += len(frames)
        values, stamps = np.asarray(commands), np.asarray(times)
        velocity = np.diff(values, axis=0) / np.diff(stamps)[:, None]
        velocity_times = (stamps[1:] + stamps[:-1]) / 2
        acceleration = np.diff(velocity, axis=0) / np.diff(velocity_times)[:, None]
        jumps = np.asarray(jumps) if jumps else np.zeros((1, 58))
        squared = np.concatenate(errors)
        rows.append(
            {
                "mode": name,
                "rounds": rounds,
                "consumed_model_steps_per_round": consumed,
                "synthetic_noise_sha256": hashlib.sha256(noises.tobytes()).hexdigest(),
                "groups": {
                    group: {
                        "max_boundary_jump_rad": float(jumps[:, part].max()),
                        "toy_target_rmse_rad": float(np.sqrt(squared[:, part].mean())),
                        "max_command_speed_rad_s": float(np.abs(velocity[:, part]).max()),
                        "max_command_acceleration_rad_s2": float(np.abs(acceleration[:, part]).max()),
                    }
                    for group, part in GROUPS.items()
                },
                "bridge_frames": total_bridge,
                "command_frames": total_frames,
                "synthetic_elapsed_s": elapsed,
                "compressed_rounds": compressed_rounds,
                "inpainting_applied_steps": applied,
                "skipped_propagations": skipped,
                "cpu_computation_ms": (time.perf_counter() - started) * 1000,
            }
        )
    return {
        "test_level": "synthetic_cpu_only",
        "seed": seed,
        "speed_scale": speed_scale,
        "checkpoint": None,
        "norm_stats": None,
        "limitations": "Toy vector field and synthetic covariance; no GPU model, dynamics, contact or task success validation.",
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error("--rounds must be at least 2 for boundary metrics")
    report = run_ablation(seed=args.seed, rounds=args.rounds, speed_scale=args.speed_scale)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Synthetic CPU ablation saved to {args.report}; no learned model or hardware was used")


if __name__ == "__main__":
    main()
