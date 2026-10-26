"""Slow initialization planner. Targets are explicit policy positions in rad."""
from dataclasses import dataclass
import numpy as np
import time
import sys

from .feedback import LiveFeedback
from .preflight import execute_preflight
from .protocol import (
    DT_NS, FPS, LEFT_FR3_SLICE, LEFT_SHARPA_SLICE, RIGHT_FR3_SLICE,
    RIGHT_SHARPA_SLICE, SHARPA_JOINT_NAMES, as_finite_action_chunk,
)


ERROR_GROUPS = {
    "left_fr3": LEFT_FR3_SLICE,
    "left_sharpa": LEFT_SHARPA_SLICE,
    "right_fr3": RIGHT_FR3_SLICE,
    "right_sharpa": RIGHT_SHARPA_SLICE,
}

HOME_TOLERANCE_RAD = 0.02


def target_chunk(target):
    target = np.asarray(target, dtype=np.float32)
    if target.shape != (58,) or not np.isfinite(target).all():
        raise ValueError("Prepare requires a finite 58D absolute target in rad")
    return as_finite_action_chunk(np.repeat(target[None, :], 50, axis=0))


def check_validation_excursion(target, current, *, mode, excursion_rad, source='openpi'):
    """Reuse an existing guarded-mode bound, not a replacement authorization."""
    from .protocol import LEFT_SHARPA_SLICE, RIGHT_SHARPA_SLICE
    # Match validation_safety.py: OpenPI task prepare is exempt from the
    # acceptance baseline bound; acquisition/continuity/slew remain enforced.
    if mode == 'supervised_hardware_validation' and source != 'openpi':
        for side, part in [('left', LEFT_SHARPA_SLICE), ('right', RIGHT_SHARPA_SLICE)]:
            delta = float(np.max(np.abs(target_chunk(target)[0, part]-target_chunk(current)[0, part])))
            if delta > excursion_rad:
                raise ValueError(f"Existing supervised validation total excursion {excursion_rad} rad blocks {side} prepare displacement {delta:.6f} rad; slow steps do not bypass authorization")


@dataclass
class SlowApproach:
    target: np.ndarray
    previous: np.ndarray
    speed_rad_s: float = 0.03
    tolerance_rad: float = 0.01
    stability_delta_rad: float = 0.005
    stable_duration_s: float = 0.5
    timeout_s: float = 60.0

    def __post_init__(self):
        self.target = target_chunk(self.target)[0].copy()
        self.previous = target_chunk(self.previous)[0].copy()
        values = (self.speed_rad_s, self.tolerance_rad, self.stability_delta_rad,
                  self.stable_duration_s, self.timeout_s)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("Prepare motion parameters must be finite and positive")
        self.started_ns = None
        self.stable_since_ns = None
        self.last_feedback = self.previous.copy()
        travel_time = float(np.max(np.abs(self.target - self.previous))) / self.speed_rad_s
        self.effective_timeout_s = (
            max(self.timeout_s, 120.0, travel_time * 2.0 + self.stable_duration_s + 5.0)
            if self.timeout_s == 60.0 else self.timeout_s
        )

    def step(self, feedback: LiveFeedback, *, now_ros_ns: int, now_monotonic_ns: int,
             enforce_execute_delta: bool = True):
        feedback.validate_freshness(now_ros_ns=now_ros_ns, now_monotonic_ns=now_monotonic_ns)
        if self.started_ns is None:
            self.started_ns = now_monotonic_ns
        if now_monotonic_ns-self.started_ns >= self.effective_timeout_s*1e9:
            errors = {
                name: float(np.max(np.abs(self.last_feedback[part] - self.target[part])))
                for name, part in ERROR_GROUPS.items()
            }
            remaining = max(errors.values())
            left_hand_errors = self.last_feedback[LEFT_SHARPA_SLICE] - self.target[LEFT_SHARPA_SLICE]
            worst_left = int(np.argmax(np.abs(left_hand_errors)))
            left_detail = {
                "joint": SHARPA_JOINT_NAMES[worst_left],
                "index": worst_left,
                "target_rad": float(self.target[LEFT_SHARPA_SLICE][worst_left]),
                "feedback_rad": float(self.last_feedback[LEFT_SHARPA_SLICE][worst_left]),
                "error_rad": float(left_hand_errors[worst_left]),
            }
            raise TimeoutError(
                "Prepare did not reach and settle at policy first target; "
                f"chunks blocked (remaining_error_rad={remaining:.6f}, "
                f"timeout_s={self.effective_timeout_s:.1f}, "
                f"error_by_group={errors}, worst_left_sharpa={left_detail})"
            )
        # A straight joint-space trajectory; never alter the policy target.
        # Progress only one nominal 30Hz interval even if scheduling is delayed.
        error = self.target-self.previous
        maximum = float(np.max(np.abs(error)))
        alpha = min(1.0, self.speed_rad_s*(DT_NS/1e9)/maximum) if maximum else 1.0
        command = self.previous+alpha*error
        # Prepare is deliberately allowed to bridge a large current-to-policy
        # displacement.  It still requires fresh feedback; execute's first
        # target delta gate is applied only after preparation completes.
        if enforce_execute_delta:
            execute_preflight(target_chunk(command), feedback, now_ros_ns=now_ros_ns,
                              now_monotonic_ns=now_monotonic_ns)
        measured_error = float(np.max(np.abs(feedback.state-self.target)))
        stable = float(np.max(np.abs(feedback.state-self.last_feedback))) <= self.stability_delta_rad
        at_command_target = np.max(np.abs(command-self.target)) <= 1e-6
        if at_command_target and measured_error <= self.tolerance_rad and stable:
            if self.stable_since_ns is None:
                self.stable_since_ns = now_monotonic_ns
        else:
            self.stable_since_ns = None
        settled = (self.stable_since_ns is not None
                   and now_monotonic_ns-self.stable_since_ns >= self.stable_duration_s*1e9)
        self.previous = command.copy()
        self.last_feedback = feedback.state.copy()
        return command, settled, measured_error


def prepare_and_run_chunks(infer, acquire_after, transport, initial_observation,
                           *, num_chunks=1, speed_rad_s=0.03,
                           execute_speed_scale=1.0, trace_writer=None, execute_steps=50,
                           initial_target=None, home_metadata=None, prediction_clip_rad=0.05,
                           prepare_initial=True, rolling_options=None, clock_ns=time.monotonic_ns):
    """Optionally prepare, then infer/execute; zero chunks means until Ctrl-C.

    Retain full predictions for audit, execute only the requested prefix, then
    reobserve. An explicit initial target prepares Home without a model call.
    With prepare_initial=False, infer directly from the initial live observation.
    """
    if type(num_chunks) is not int or num_chunks < 0:
        raise ValueError("num_chunks must be a nonnegative integer (0 = continuous)")
    if type(execute_steps) is not int or not 1 <= execute_steps <= 50:
        raise ValueError("execute_steps must be an integer in [1, 50]")
    if not np.isfinite(prediction_clip_rad) or prediction_clip_rad < 0:
        raise ValueError("prediction_clip_rad must be finite and nonnegative")
    if not prepare_initial and initial_target is not None:
        raise ValueError("An initial Home target requires prepare_initial=True")
    rolling = None
    if rolling_options is not None:
        from .rolling import RollingReference
        if execute_steps != 26:
            raise ValueError("paper-rolling requires execute_steps=26")
        rolling = RollingReference(rolling_options, clock_ns=clock_ns)
    initialization = "current" if not prepare_initial else "home" if initial_target is not None else "policy"
    results = []
    count = 0
    prepared = None
    trace_path = None
    stage = "initial_inference"
    metadata = {}

    def event(name, **fields):
        nonlocal stage
        stage = name
        print(f"[chunk-runtime] {name} chunk={count + 1}", file=sys.stderr, flush=True)
        if trace_writer is not None:
            trace_writer.event(name, chunk_index=count, **fields)

    try:
        event("initialization", mode=initialization)
        if prepare_initial:
            if initial_target is None:
                event("initial_inference")
                metadata, initial_actions, _ = infer(initial_observation)
                target = as_finite_action_chunk(initial_actions)[0]
            else:
                target = target_chunk(initial_target)[0]
            tolerance = HOME_TOLERANCE_RAD if initial_target is not None else 0.049
            event("prepare", target=target.tolist(), speed_rad_s=speed_rad_s,
                  tolerance_rad=tolerance, home=home_metadata)
            prepared = transport.request({"operation": "prepare", "target": target.tolist(),
                                          "speed_rad_s": speed_rad_s,
                                          "tolerance_rad": tolerance})
            if home_metadata is not None:
                prepared["home"] = home_metadata
            if prepared.get("status") != "PASS":
                raise RuntimeError("Prepare did not pass; chunks blocked")
            event("prepare_complete", report=prepared)
        completed = prepared
        next_observation = None if prepare_initial else initial_observation
        while num_chunks == 0 or count < num_chunks:
            trace_path = None
            event("observation")
            observation = acquire_after(completed) if next_observation is None else next_observation
            next_observation = None
            event("inference")
            if rolling is not None:
                from .rolling import CONTEXT_KEY
                observation = {**observation, CONTEXT_KEY: rolling.context(count)}
            metadata, actions, adapted_chunk = infer(observation)
            actions = as_finite_action_chunk(actions)
            if trace_writer is not None:
                trace_path = trace_writer.begin_chunk(
                    chunk_index=count, metadata=metadata,
                    observation_before=observation, actions=actions,
                    adapted_chunk=adapted_chunk, execute_steps=execute_steps)
            event("execute", predicted_steps=50, execute_steps=execute_steps, prediction_clip_rad=prediction_clip_rad)
            command_started = time.monotonic()
            request = {"operation": "chunk", "actions": actions.tolist(),
                       "execute_steps": execute_steps,
                       "prediction_clip_rad": prediction_clip_rad,
                       "execute_speed_scale": execute_speed_scale,
                       "approach_speed_rad_s": speed_rad_s}
            if rolling is not None:
                from .compression import compression_candidate
                candidate, compression = compression_candidate(
                    actions, observation["observation/state"], rolling_options)
                request.update(execution_targets=candidate.tolist(), compression=compression,
                               record_command_frames=True)
                diagnostic = metadata.get("inference_experiment", {})
                reference_stamp = diagnostic.get("reference_completed_monotonic_ns")
                if reference_stamp is not None and not 0 <= clock_ns() - reference_stamp <= rolling_options.reference_max_age_s * 1e9:
                    raise RuntimeError("Rolling reference expired before execution")
                if trace_writer is not None:
                    trace_writer.record_experiment(trace_path, context=observation["_inference_context"],
                        diagnostics=diagnostic, execution_targets=candidate, compression=compression)
            completed = transport.request(request)
            if completed.get("status") != "PASS":
                raise RuntimeError("Chunk failed; further inference stopped")
            if rolling is not None:
                rolling.commit(actions, metadata, completed, count)
            command_duration = time.monotonic() - command_started
            if trace_writer is not None:
                # Mark transmission complete even if the following sensor
                # acquisition fails. That distinction matters in a postmortem.
                trace_writer.finish_chunk(trace_path, observation_after=None,
                    command_duration_s=command_duration,
                    end_to_end_duration_s=time.monotonic()-command_started,
                    execution_report=completed)
            count += 1
            # In continuous mode records are on disk, not an unbounded RAM list.
            if num_chunks:
                results.append(completed)
            event("chunk_complete", report=completed)
            if trace_writer is not None:
                event("post_chunk_observation")
                next_observation = acquire_after(completed)
                trace_writer.finish_chunk(trace_path, observation_after=next_observation,
                    command_duration_s=command_duration,
                    end_to_end_duration_s=time.monotonic()-command_started,
                    execution_report=completed)
        report = {"mode": "execute", "status": "COMPLETE", "initialization": initialization,
                  "prepare": prepared, "chunks": results,
                  "checkpoint": metadata["checkpoint"], "norm_stats": metadata["norm_stats"],
                  "action_shape": [50, 58], "execute_steps": execute_steps,
                  "action_hz": FPS, "num_chunks": count,
                  "published": bool((prepared or {}).get("published", False) or any(r.get("published", False) for r in results))}
        if trace_writer is not None:
            trace_writer.finish_run(report)
        return report
    except BaseException as error:
        if rolling is not None:
            rolling.clear("interrupted_or_failed")
        if trace_writer is not None:
            status = "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED"
            if trace_path is not None:
                trace_writer.fail_chunk(trace_path, error=str(error), stage=stage)
            trace_writer.finish_run({"status": status, "stage": stage, "num_chunks": count,
                                     "initialization": initialization,
                                     "prepare": prepared, "error": str(error) or type(error).__name__})
        raise
