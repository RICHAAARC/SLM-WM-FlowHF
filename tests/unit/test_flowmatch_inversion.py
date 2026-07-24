from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from flowhf.inversion_detector import (
    FlowMatchSchedule,
    flowmatch_euler_step_actual_dtype,
    invert_terminal_to_callback_state,
)
from flowhf.model_runtime import PromptConditioning


class ConstantVelocityTransformer:
    def __init__(self) -> None:
        self._parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

    def parameters(self):
        return iter((self._parameter,))

    def __call__(self, *, hidden_states, **_kwargs):
        return (torch.ones_like(hidden_states),)


def test_flowmatch_inverse_solves_the_actual_discrete_interval() -> None:
    pipeline = SimpleNamespace(transformer=ConstantVelocityTransformer())
    schedule = FlowMatchSchedule(
        timesteps=(torch.tensor(2.0), torch.tensor(1.0)),
        sigmas=(1.0, 0.5, 0.0),
        record={"schedule_digest": "schedule"},
    )
    conditioning = PromptConditioning(
        positive_prompt_embeds=torch.zeros((1, 1, 1)),
        positive_pooled_prompt_embeds=torch.zeros((1, 1)),
        negative_prompt_embeds=torch.zeros((1, 1, 1)),
        negative_pooled_prompt_embeds=torch.zeros((1, 1)),
    )
    result = invert_terminal_to_callback_state(
        pipeline,
        torch.zeros((1, 1, 2, 2)),
        schedule=schedule,
        conditioning=conditioning,
        guidance_scale=1.0,
        callback_step_index=0,
        fixed_point_iterations=2,
    )
    assert result.recovered_latent.tolist() == [[[[0.5, 0.5], [0.5, 0.5]]]]
    assert torch.equal(
        result.replayed_terminal_latent,
        torch.zeros((1, 1, 2, 2)),
    )
    assert result.record["wrong_key_feedback_allowed"] is False
    assert result.record["interval_records"][0]["replay_residual_rms"] == 0.0


def test_fp16_euler_step_matches_diffusers_038_scheduler() -> None:
    diffusers = pytest.importorskip("diffusers")
    assert diffusers.__version__ == "0.38.0"
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(2)
    sample = torch.tensor(
        [[[[1.0000, -0.7500], [0.1250, 2.0000]]]],
        dtype=torch.float16,
    )
    model_output = torch.tensor(
        [[[[0.3333, -0.1666], [0.0625, -0.5000]]]],
        dtype=torch.float16,
    )
    expected = scheduler.step(
        model_output,
        scheduler.timesteps[0],
        sample,
        return_dict=False,
    )[0]
    actual = flowmatch_euler_step_actual_dtype(
        sample,
        model_output,
        sigma=float(scheduler.sigmas[0]),
        sigma_next=float(scheduler.sigmas[1]),
    )
    assert actual.dtype == torch.float16
    assert torch.equal(actual, expected)


def test_fp16_inverse_then_forward_reports_quantized_error() -> None:
    pipeline = SimpleNamespace(transformer=ConstantVelocityTransformer())
    schedule = FlowMatchSchedule(
        timesteps=(torch.tensor(2.0), torch.tensor(1.0)),
        sigmas=(1.0, 0.5, 0.0),
        record={"schedule_digest": "fp16-schedule"},
    )
    conditioning = PromptConditioning(
        positive_prompt_embeds=torch.zeros((1, 1, 1)),
        positive_pooled_prompt_embeds=torch.zeros((1, 1)),
        negative_prompt_embeds=torch.zeros((1, 1, 1)),
        negative_pooled_prompt_embeds=torch.zeros((1, 1)),
    )
    sample = torch.tensor([[[[0.7, -0.2]]]], dtype=torch.float16)
    velocity = torch.ones_like(sample)
    terminal = flowmatch_euler_step_actual_dtype(
        sample,
        velocity,
        sigma=0.5,
        sigma_next=0.0,
    )
    result = invert_terminal_to_callback_state(
        pipeline,
        terminal,
        schedule=schedule,
        conditioning=conditioning,
        guidance_scale=1.0,
        callback_step_index=0,
        fixed_point_iterations=2,
    )
    error = float(
        (
            result.replayed_terminal_latent.float() - terminal.float()
        ).abs().max().item()
    )
    assert error <= torch.finfo(torch.float16).eps
    assert result.record["actual_dtype_policy"] == (
        "sample_float32_euler_then_cast_to_model_output_dtype"
    )
