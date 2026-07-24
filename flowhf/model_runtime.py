from __future__ import annotations

from dataclasses import dataclass
import platform
import sys
from typing import Any

from flowhf.digest import stable_digest
from flowhf.run_spec import FlowHFRunSpec, SCHEDULER_CLASS


@dataclass(frozen=True)
class PromptConditioning:
    positive_prompt_embeds: Any
    positive_pooled_prompt_embeds: Any
    negative_prompt_embeds: Any
    negative_pooled_prompt_embeds: Any


@dataclass(frozen=True)
class ModelRuntime:
    pipeline: Any
    torch_module: Any
    record: dict[str, Any]


def _qualified_name(value: Any) -> str:
    resolved = type(value)
    return f"{resolved.__module__}.{resolved.__qualname__}"


def fixed_model_identity_digest(spec: FlowHFRunSpec) -> str:
    return stable_digest(
        {
            "model_id": spec.model_id,
            "model_revision": spec.model_revision,
        }
    )


def load_model_runtime(
    spec: FlowHFRunSpec,
    *,
    hf_token: str | None,
) -> ModelRuntime:
    import diffusers
    import torch
    import transformers
    from diffusers import StableDiffusion3Pipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    properties = torch.cuda.get_device_properties(0)
    total_mib = int(properties.total_memory // (1024 * 1024))
    if "A100" not in properties.name or total_mib < 75 * 1024:
        raise RuntimeError("the first smoke requires an A100 with at least 75 GiB")
    dtype = getattr(torch, spec.latent_dtype)
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        spec.model_id,
        revision=spec.model_revision,
        torch_dtype=dtype,
        token=hf_token,
    )
    pipeline = pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=False)
    if _qualified_name(pipeline.scheduler) != SCHEDULER_CLASS:
        raise RuntimeError("SD3.5 did not load FlowMatchEulerDiscreteScheduler")
    scheduler_config = dict(pipeline.scheduler.config)
    if scheduler_config.get("stochastic_sampling", False):
        raise RuntimeError("numerical inversion requires deterministic sampling")
    environment = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "diffusers_version": str(diffusers.__version__),
        "transformers_version": str(transformers.__version__),
        "python_implementation": sys.implementation.name,
        "gpu_name": properties.name,
        "gpu_total_memory_mib": total_mib,
        "latent_dtype": spec.latent_dtype,
    }
    record = {
        "model_id": spec.model_id,
        "model_revision": spec.model_revision,
        "model_identity_digest": fixed_model_identity_digest(spec),
        "pipeline_class": _qualified_name(pipeline),
        "scheduler_class": _qualified_name(pipeline.scheduler),
        "environment": environment,
        "environment_digest": stable_digest(environment),
    }
    return ModelRuntime(
        pipeline=pipeline,
        torch_module=torch,
        record={**record, "model_runtime_digest": stable_digest(record)},
    )


def build_prompt_conditioning(
    pipeline: Any,
    spec: FlowHFRunSpec,
) -> PromptConditioning:
    positive, _, positive_pooled, _ = pipeline.encode_prompt(
        prompt=spec.prompt,
        prompt_2=spec.prompt,
        prompt_3=spec.prompt,
        device=pipeline._execution_device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    negative, _, negative_pooled, _ = pipeline.encode_prompt(
        prompt=spec.negative_prompt,
        prompt_2=spec.negative_prompt,
        prompt_3=spec.negative_prompt,
        device=pipeline._execution_device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    return PromptConditioning(
        positive_prompt_embeds=positive,
        positive_pooled_prompt_embeds=positive_pooled,
        negative_prompt_embeds=negative,
        negative_pooled_prompt_embeds=negative_pooled,
    )


def guided_flow_velocity(
    pipeline: Any,
    latent: Any,
    timestep: Any,
    conditioning: PromptConditioning,
    guidance_scale: float,
) -> Any:
    import torch

    if guidance_scale > 1.0:
        latent_input = torch.cat((latent, latent), dim=0)
        prompt_embeds = torch.cat(
            (
                conditioning.negative_prompt_embeds,
                conditioning.positive_prompt_embeds,
            ),
            dim=0,
        )
        pooled = torch.cat(
            (
                conditioning.negative_pooled_prompt_embeds,
                conditioning.positive_pooled_prompt_embeds,
            ),
            dim=0,
        )
    else:
        latent_input = latent
        prompt_embeds = conditioning.positive_prompt_embeds
        pooled = conditioning.positive_pooled_prompt_embeds
    timestep_batch = timestep.reshape(1).expand(latent_input.shape[0])
    transformer_dtype = next(pipeline.transformer.parameters()).dtype
    prediction = pipeline.transformer(
        hidden_states=latent_input.to(dtype=transformer_dtype),
        timestep=timestep_batch,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled,
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]
    if guidance_scale > 1.0:
        unconditional, conditional = prediction.chunk(2)
        prediction = unconditional + guidance_scale * (
            conditional - unconditional
        )
    return prediction.to(dtype=latent.dtype)


def make_base_latent(pipeline: Any, spec: FlowHFRunSpec) -> Any:
    import torch

    shape = (
        1,
        int(pipeline.transformer.config.in_channels),
        spec.height // int(pipeline.vae_scale_factor),
        spec.width // int(pipeline.vae_scale_factor),
    )
    generator = torch.Generator(device="cpu").manual_seed(spec.seed)
    latent = torch.randn(
        shape,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    return latent.to(
        device=pipeline._execution_device,
        dtype=next(pipeline.transformer.parameters()).dtype,
    )


def decode_latent_to_pil(pipeline: Any, latent: Any) -> Any:
    vae_dtype = next(pipeline.vae.parameters()).dtype
    scaled = latent.to(dtype=vae_dtype) / pipeline.vae.config.scaling_factor
    scaled = scaled + pipeline.vae.config.shift_factor
    decoded = pipeline.vae.decode(scaled, return_dict=False)[0]
    images = pipeline.image_processor.postprocess(
        decoded.detach(),
        output_type="pil",
    )
    if type(images) is not list or len(images) != 1:
        raise RuntimeError("VAE decode must return exactly one PIL image")
    return images[0]


def encode_pil_to_latent(pipeline: Any, image: Any) -> Any:
    pixels = pipeline.image_processor.preprocess(image).to(
        device=pipeline._execution_device,
        dtype=next(pipeline.vae.parameters()).dtype,
    )
    encoded = pipeline.vae.encode(pixels).latent_dist.mode()
    return (
        encoded - float(pipeline.vae.config.shift_factor)
    ) * float(pipeline.vae.config.scaling_factor)
