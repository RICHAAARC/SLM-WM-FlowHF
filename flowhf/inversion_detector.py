from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from flowhf.digest import stable_digest, tensor_content_sha256
from flowhf.model_runtime import PromptConditioning, guided_flow_velocity


@dataclass(frozen=True)
class FlowMatchSchedule:
    timesteps: tuple[Any, ...]
    sigmas: tuple[float, ...]
    record: dict[str, Any]


@dataclass(frozen=True)
class InversionResult:
    recovered_latent: Any
    replayed_terminal_latent: Any
    record: dict[str, Any]


def flowmatch_euler_step_actual_dtype(
    sample: Any,
    model_output: Any,
    *,
    sigma: float,
    sigma_next: float,
) -> Any:
    """Match Diffusers 0.38 FlowMatch Euler's deterministic dtype order."""

    import torch

    if not isinstance(sample, torch.Tensor) or not isinstance(
        model_output,
        torch.Tensor,
    ):
        raise TypeError("FlowMatch sample and model output must be tensors")
    delta = sigma_next - sigma
    if not math.isfinite(delta) or not delta < 0.0:
        raise RuntimeError("FlowMatch sigma order must strictly descend")
    updated_float32 = sample.to(torch.float32) + delta * model_output
    return updated_float32.to(model_output.dtype)


def capture_flowmatch_schedule(
    pipeline: Any,
    *,
    inference_steps: int,
) -> FlowMatchSchedule:
    scheduler = pipeline.scheduler
    scheduler.set_timesteps(
        inference_steps,
        device=pipeline._execution_device,
    )
    timesteps = tuple(item.detach().clone() for item in scheduler.timesteps)
    sigmas = tuple(float(item) for item in scheduler.sigmas)
    if len(timesteps) != inference_steps or len(sigmas) != inference_steps + 1:
        raise RuntimeError("FlowMatch schedule has unexpected lengths")
    config = dict(scheduler.config)
    if config.get("stochastic_sampling", False):
        raise RuntimeError("stochastic FlowMatch sampling cannot be inverted here")
    payload = {
        "scheduler_class": (
            f"{type(scheduler).__module__}.{type(scheduler).__qualname__}"
        ),
        "timesteps": [float(item.detach().cpu().item()) for item in timesteps],
        "sigmas": list(sigmas),
        "shift": config.get("shift"),
        "use_dynamic_shifting": config.get("use_dynamic_shifting"),
        "base_shift": config.get("base_shift"),
        "max_shift": config.get("max_shift"),
        "time_shift_type": config.get("time_shift_type"),
        "stochastic_sampling": config.get("stochastic_sampling", False),
    }
    return FlowMatchSchedule(
        timesteps=timesteps,
        sigmas=sigmas,
        record={**payload, "schedule_digest": stable_digest(payload)},
    )


def invert_terminal_to_callback_state(
    pipeline: Any,
    terminal_latent: Any,
    *,
    schedule: FlowMatchSchedule,
    conditioning: PromptConditioning,
    guidance_scale: float,
    callback_step_index: int,
    fixed_point_iterations: int,
) -> InversionResult:
    """Invert explicit FlowMatch Euler intervals by fixed-point solving.

    Generation interval ``i`` is
    ``x[i+1] = x[i] + (sigma[i+1]-sigma[i]) * v(x[i], t[i], c)``.
    The inverse solves this exact discrete equation for ``x[i]`` while walking
    from sigma=0 toward the callback output state.
    """

    import torch

    target_sigma_index = callback_step_index + 1
    if target_sigma_index >= len(schedule.sigmas) - 1:
        raise ValueError("callback must leave at least one generation interval")
    recovered = terminal_latent.detach().clone()
    interval_records: list[dict[str, Any]] = []
    with torch.no_grad():
        for interval_index in range(
            len(schedule.timesteps) - 1,
            target_sigma_index - 1,
            -1,
        ):
            terminal_for_interval = recovered.detach().clone()
            estimate = terminal_for_interval.detach().clone()
            timestep = schedule.timesteps[interval_index]
            sigma = schedule.sigmas[interval_index]
            sigma_next = schedule.sigmas[interval_index + 1]
            delta = sigma_next - sigma
            if not math.isfinite(delta) or not delta < 0.0:
                raise RuntimeError("FlowMatch sigma order must strictly descend")
            for _ in range(fixed_point_iterations):
                velocity = guided_flow_velocity(
                    pipeline,
                    estimate,
                    timestep,
                    conditioning,
                    guidance_scale,
                )
                estimate = (
                    terminal_for_interval.to(torch.float32)
                    - delta * velocity
                ).to(terminal_for_interval.dtype)
            final_velocity = guided_flow_velocity(
                pipeline,
                estimate,
                timestep,
                conditioning,
                guidance_scale,
            )
            replayed = flowmatch_euler_step_actual_dtype(
                estimate,
                final_velocity,
                sigma=sigma,
                sigma_next=sigma_next,
            )
            residual = replayed.float() - terminal_for_interval.float()
            residual_rms = float(residual.square().mean().sqrt().item())
            interval_records.append(
                {
                    "interval_index": interval_index,
                    "timestep": float(timestep.detach().cpu().item()),
                    "sigma": sigma,
                    "sigma_next": sigma_next,
                    "fixed_point_iterations": fixed_point_iterations,
                    "replay_residual_rms": residual_rms,
                }
            )
            recovered = estimate
        replayed_terminal = recovered
        for interval_index in range(
            target_sigma_index,
            len(schedule.timesteps),
        ):
            timestep = schedule.timesteps[interval_index]
            sigma = schedule.sigmas[interval_index]
            sigma_next = schedule.sigmas[interval_index + 1]
            velocity = guided_flow_velocity(
                pipeline,
                replayed_terminal,
                timestep,
                conditioning,
                guidance_scale,
            )
            replayed_terminal = flowmatch_euler_step_actual_dtype(
                replayed_terminal,
                velocity,
                sigma=sigma,
                sigma_next=sigma_next,
            )
    payload = {
        "operator": "flowmatch_euler_discrete_fixed_point_inverse",
        "actual_dtype_policy": (
            "sample_float32_euler_then_cast_to_model_output_dtype"
        ),
        "equation": (
            "x_i=x_i_plus_1-(sigma_i_plus_1-sigma_i)*"
            "v_theta(x_i,t_i,conditioning)"
        ),
        "callback_step_index": callback_step_index,
        "target_sigma_index": target_sigma_index,
        "target_sigma": schedule.sigmas[target_sigma_index],
        "fixed_point_iterations": fixed_point_iterations,
        "schedule_digest": schedule.record["schedule_digest"],
        "terminal_latent_content_sha256": tensor_content_sha256(
            terminal_latent
        ),
        "recovered_latent_content_sha256": tensor_content_sha256(recovered),
        "replayed_terminal_latent_content_sha256": tensor_content_sha256(
            replayed_terminal
        ),
        "interval_records": interval_records,
        "wrong_key_feedback_allowed": False,
    }
    return InversionResult(
        recovered_latent=recovered,
        replayed_terminal_latent=replayed_terminal,
        record={**payload, "inversion_digest": stable_digest(payload)},
    )
