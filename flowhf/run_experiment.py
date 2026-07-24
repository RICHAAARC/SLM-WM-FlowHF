from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import traceback
from typing import Any, Callable

from flowhf.digest import (
    file_sha256,
    stable_digest,
    stable_json_dumps,
    tensor_content_sha256,
)
from flowhf.evaluate_keys import evaluate_key_plan
from flowhf.hf_injector import apply_fixed_energy_hf, build_hf_template
from flowhf.image_quality import paired_quality
from flowhf.inversion_detector import (
    FlowMatchSchedule,
    InversionResult,
    capture_flowmatch_schedule,
    invert_terminal_to_callback_state,
)
from flowhf.key_plan import (
    KeyPlan,
    build_key_plan,
    build_registered_key_candidate,
)
from flowhf.model_runtime import (
    ModelRuntime,
    PromptConditioning,
    build_prompt_conditioning,
    decode_latent_to_pil,
    encode_pil_to_latent,
    load_model_runtime,
    make_base_latent,
)
from flowhf.run_spec import FlowHFRunSpec


@dataclass(frozen=True)
class ChainOutput:
    role: str
    callback_latent: Any
    terminal_latent: Any
    image: Any
    injection_record: dict[str, Any] | None


@dataclass(frozen=True)
class FrozenChainObservations:
    chain: ChainOutput
    direct_latent: Any
    inverted: InversionResult
    replay_image: Any


@dataclass(frozen=True)
class ExperimentDependencies:
    load_runtime: Callable[..., ModelRuntime] = load_model_runtime


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(stable_json_dumps(value) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _safe_failure_text(error: BaseException, secrets: tuple[str, ...]) -> str:
    value = f"{type(error).__name__}: {error}"
    for secret in secrets:
        if secret:
            value = value.replace(secret, "[REDACTED]")
    return value[:4000]


def _assert_repository_identity(spec: FlowHFRunSpec) -> None:
    repository = Path(__file__).resolve().parents[1]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != spec.repository_commit:
        raise RuntimeError("repository HEAD differs from the run request")
    tracked = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if tracked:
        raise RuntimeError("repository tracked worktree is not clean")


def _run_chain(
    runtime: ModelRuntime,
    spec: FlowHFRunSpec,
    conditioning: PromptConditioning,
    base_latent: Any,
    *,
    role: str,
    registered_key_material: str,
) -> ChainOutput:
    captured: dict[str, Any] = {}

    def callback(
        _pipeline: Any,
        step_index: int,
        _timestep: Any,
        callback_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        latent = callback_kwargs["latents"]
        if step_index == spec.injection_step_index:
            if captured:
                raise RuntimeError("callback index 18 was observed more than once")
            if role == "watermarked":
                template = build_hf_template(
                    latent,
                    registered_key_material,
                    runtime.record["model_identity_digest"],
                )
                written = apply_fixed_energy_hf(
                    latent,
                    template,
                    relative_l2=spec.hf_relative_l2,
                )
                latent = written.latent
                callback_kwargs["latents"] = latent
                captured["injection_record"] = written.record
            captured["latent"] = latent.detach().clone()
        return callback_kwargs

    output = runtime.pipeline(
        prompt=None,
        negative_prompt=None,
        prompt_embeds=conditioning.positive_prompt_embeds,
        pooled_prompt_embeds=conditioning.positive_pooled_prompt_embeds,
        negative_prompt_embeds=conditioning.negative_prompt_embeds,
        negative_pooled_prompt_embeds=(
            conditioning.negative_pooled_prompt_embeds
        ),
        height=spec.height,
        width=spec.width,
        num_inference_steps=spec.inference_steps,
        guidance_scale=spec.guidance_scale,
        latents=base_latent.detach().clone(),
        output_type="latent",
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    terminal = output.images
    if "latent" not in captured:
        raise RuntimeError("pipeline did not expose callback index 18")
    image = decode_latent_to_pil(runtime.pipeline, terminal)
    return ChainOutput(
        role=role,
        callback_latent=captured["latent"],
        terminal_latent=terminal.detach().clone(),
        image=image,
        injection_record=captured.get("injection_record"),
    )


def _tensor_rms(left: Any, right: Any) -> float:
    difference = left.detach().float() - right.detach().float()
    return float(difference.square().mean().sqrt().item())


def _save_tensor(torch_module: Any, path: Path, tensor: Any) -> dict[str, Any]:
    torch_module.save(tensor.detach().cpu(), path)
    return {
        "path": path.relative_to(path.parents[1]).as_posix(),
        "file_sha256": file_sha256(path),
        "tensor_content_sha256": tensor_content_sha256(tensor),
        "shape": [int(value) for value in tensor.shape],
        "dtype": str(tensor.dtype),
    }


def _freeze_chain_observations(
    runtime: ModelRuntime,
    spec: FlowHFRunSpec,
    schedule: FlowMatchSchedule,
    conditioning: PromptConditioning,
    chain: ChainOutput,
) -> FrozenChainObservations:
    direct_latent = encode_pil_to_latent(runtime.pipeline, chain.image)
    inverted = invert_terminal_to_callback_state(
        runtime.pipeline,
        direct_latent,
        schedule=schedule,
        conditioning=conditioning,
        guidance_scale=spec.guidance_scale,
        callback_step_index=spec.injection_step_index,
        fixed_point_iterations=spec.inversion_fixed_point_iterations,
    )
    replay_image = decode_latent_to_pil(
        runtime.pipeline,
        inverted.replayed_terminal_latent,
    )
    return FrozenChainObservations(
        chain=chain,
        direct_latent=direct_latent.detach().clone(),
        inverted=inverted,
        replay_image=replay_image,
    )


def _score_frozen_chain(
    runtime: ModelRuntime,
    key_plan: KeyPlan,
    frozen: FrozenChainObservations,
    tensor_directory: Path,
) -> dict[str, Any]:
    chain = frozen.chain
    direct_latent = frozen.direct_latent
    inverted = frozen.inverted
    model_digest = runtime.record["model_identity_digest"]
    direct_scores = evaluate_key_plan(
        direct_latent,
        key_plan=key_plan,
        model_identity_digest=model_digest,
        observation_domain="final_image_vae_reencoded",
    )
    inverted_scores = evaluate_key_plan(
        inverted.recovered_latent,
        key_plan=key_plan,
        model_identity_digest=model_digest,
        observation_domain="flowmatch_inverted_index18",
    )
    oracle_scores = evaluate_key_plan(
        chain.callback_latent,
        key_plan=key_plan,
        model_identity_digest=model_digest,
        observation_domain="generation_callback_index18",
    )
    tensor_records = {
        "callback_index18": _save_tensor(
            runtime.torch_module,
            tensor_directory / f"{chain.role}_callback_index18.pt",
            chain.callback_latent,
        ),
        "terminal_generation": _save_tensor(
            runtime.torch_module,
            tensor_directory / f"{chain.role}_terminal_generation.pt",
            chain.terminal_latent,
        ),
        "final_image_reencoded": _save_tensor(
            runtime.torch_module,
            tensor_directory / f"{chain.role}_final_reencoded.pt",
            direct_latent,
        ),
        "flowmatch_inverted_index18": _save_tensor(
            runtime.torch_module,
            tensor_directory / f"{chain.role}_inverted_index18.pt",
            inverted.recovered_latent,
        ),
        "flowmatch_replayed_terminal": _save_tensor(
            runtime.torch_module,
            tensor_directory / f"{chain.role}_replayed_terminal.pt",
            inverted.replayed_terminal_latent,
        ),
    }
    return {
        "role": chain.role,
        "injection_record": chain.injection_record,
        "tensor_records": tensor_records,
        "oracle_index18_scores": oracle_scores,
        "direct_final_image_scores": direct_scores,
        "inverted_index18_scores": inverted_scores,
        "inversion_record": inverted.record,
        "round_trip": {
            "oracle_vs_inverted_index18_rms": _tensor_rms(
                chain.callback_latent,
                inverted.recovered_latent,
            ),
            "direct_reencode_vs_replayed_terminal_rms": _tensor_rms(
                direct_latent,
                inverted.replayed_terminal_latent,
            ),
            "final_image_vs_replayed_image": paired_quality(
                chain.image,
                frozen.replay_image,
            ),
        },
    }


def _schedule_injection_record(
    schedule: FlowMatchSchedule,
    injection_step_index: int,
) -> dict[str, Any]:
    return {
        "callback_step_index": injection_step_index,
        "callback_timestep": schedule.record["timesteps"][
            injection_step_index
        ],
        "callback_output_sigma": schedule.sigmas[injection_step_index + 1],
        "remaining_generation_intervals": (
            len(schedule.timesteps) - injection_step_index - 1
        ),
    }


def _build_inventory(run_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(run_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_root).as_posix()
        if relative in {"inventory.json", "run_state.json"}:
            continue
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return records


def run_experiment(
    spec: FlowHFRunSpec,
    *,
    watermark_key: str,
    hf_token: str | None = None,
    dependencies: ExperimentDependencies = ExperimentDependencies(),
) -> dict[str, Any]:
    """Run the fixed one-prompt experiment.

    The raw watermark key is accepted only as an in-memory argument. Neither it
    nor the optional model-download token is included in persisted records.
    """

    if not watermark_key:
        raise ValueError("watermark key is required")
    run_root = Path(spec.run_root)
    if run_root.exists():
        raise FileExistsError("run root already exists")
    run_root.mkdir(parents=True)
    state_path = run_root / "run_state.json"
    _atomic_json(
        state_path,
        {
            "run_id": spec.run_id,
            "status": "running",
            "run_spec": spec.public_record(),
            "diagnostic_only": True,
            "supports_paper_claim": False,
        },
    )
    secrets = (watermark_key, hf_token or "")
    runtime_record: dict[str, Any] | None = None
    schedule_record: dict[str, Any] | None = None
    completed_chain_roles: list[str] = []
    try:
        _assert_repository_identity(spec)
        runtime = dependencies.load_runtime(spec, hf_token=hf_token)
        runtime_record = runtime.record
        registered_key = build_registered_key_candidate(watermark_key)
        schedule = capture_flowmatch_schedule(
            runtime.pipeline,
            inference_steps=spec.inference_steps,
        )
        schedule_record = schedule.record
        conditioning = build_prompt_conditioning(runtime.pipeline, spec)
        base_latent = make_base_latent(runtime.pipeline, spec)
        clean = _run_chain(
            runtime,
            spec,
            conditioning,
            base_latent,
            role="clean",
            registered_key_material=registered_key.material,
        )
        completed_chain_roles.append("clean")
        watermarked = _run_chain(
            runtime,
            spec,
            conditioning,
            base_latent,
            role="watermarked",
            registered_key_material=registered_key.material,
        )
        completed_chain_roles.append("watermarked")
        if clean.injection_record is not None:
            raise RuntimeError("clean chain unexpectedly wrote an HF carrier")
        if watermarked.injection_record is None:
            raise RuntimeError("watermarked chain did not write the HF carrier")
        image_directory = run_root / "images"
        tensor_directory = run_root / "tensors"
        image_directory.mkdir()
        tensor_directory.mkdir()
        base_latent_record = _save_tensor(
            runtime.torch_module,
            tensor_directory / "base_latent.pt",
            base_latent,
        )
        clean_image_path = image_directory / "clean.png"
        watermarked_image_path = image_directory / "watermarked.png"
        clean.image.save(clean_image_path, format="PNG")
        watermarked.image.save(watermarked_image_path, format="PNG")
        clean_frozen = _freeze_chain_observations(
            runtime,
            spec,
            schedule,
            conditioning,
            clean,
        )
        watermarked_frozen = _freeze_chain_observations(
            runtime,
            spec,
            schedule,
            conditioning,
            watermarked,
        )
        key_plan = build_key_plan(watermark_key)
        if (
            key_plan.registered.material_digest_random
            != registered_key.material_digest_random
        ):
            raise RuntimeError("registered key identity changed before scoring")
        clean_record = _score_frozen_chain(
            runtime,
            key_plan,
            clean_frozen,
            tensor_directory,
        )
        watermarked_record = _score_frozen_chain(
            runtime,
            key_plan,
            watermarked_frozen,
            tensor_directory,
        )
        result_payload = {
            "run_spec": spec.public_record(),
            "diagnostic_only": True,
            "supports_paper_claim": False,
            "candidate_promotion_allowed": False,
            "qualification_evidence": False,
            "chain_count": 2,
            "chain_order": ["clean", "watermarked"],
            "model_runtime": runtime.record,
            "scheduler": schedule.record,
            "conditioning": {
                "positive_prompt_embeds_content_sha256": (
                    tensor_content_sha256(
                        conditioning.positive_prompt_embeds
                    )
                ),
                "positive_pooled_prompt_embeds_content_sha256": (
                    tensor_content_sha256(
                        conditioning.positive_pooled_prompt_embeds
                    )
                ),
                "negative_prompt_embeds_content_sha256": (
                    tensor_content_sha256(
                        conditioning.negative_prompt_embeds
                    )
                ),
                "negative_pooled_prompt_embeds_content_sha256": (
                    tensor_content_sha256(
                        conditioning.negative_pooled_prompt_embeds
                    )
                ),
            },
            "injection": {
                **_schedule_injection_record(
                    schedule,
                    spec.injection_step_index,
                ),
                "relative_l2": spec.hf_relative_l2,
                "routing": "uniform_hf_tail",
            },
            "base_latent_content_sha256": tensor_content_sha256(base_latent),
            "base_latent_tensor": base_latent_record,
            "key_plan": key_plan.public_record(),
            "wrong_key_feedback_allowed": False,
            "clean": clean_record,
            "watermarked": watermarked_record,
            "image_quality": paired_quality(clean.image, watermarked.image),
            "image_files": {
                "clean": {
                    "path": "images/clean.png",
                    "sha256": file_sha256(clean_image_path),
                },
                "watermarked": {
                    "path": "images/watermarked.png",
                    "sha256": file_sha256(watermarked_image_path),
                },
            },
        }
        result_payload["result_digest"] = stable_digest(result_payload)
        _atomic_json(run_root / "result.json", result_payload)
        inventory = {
            "run_id": spec.run_id,
            "files": _build_inventory(run_root),
        }
        inventory["inventory_digest"] = stable_digest(inventory)
        _atomic_json(run_root / "inventory.json", inventory)
        completed = {
            "run_id": spec.run_id,
            "status": "success",
            "run_spec": spec.public_record(),
            "result_digest": result_payload["result_digest"],
            "inventory_digest": inventory["inventory_digest"],
            "diagnostic_only": True,
            "supports_paper_claim": False,
        }
        _atomic_json(state_path, completed)
        return completed
    except BaseException as error:
        failure = {
            "run_id": spec.run_id,
            "status": "failure",
            "run_spec": spec.public_record(),
            "failure": _safe_failure_text(error, secrets),
            "model_runtime": runtime_record,
            "scheduler": schedule_record,
            "completed_chain_roles": completed_chain_roles,
            "traceback_digest": stable_digest(
                {
                    "traceback": "\n".join(
                        line
                        for line in traceback.format_exception(error)
                        if not any(secret and secret in line for secret in secrets)
                    )
                }
            ),
            "diagnostic_only": True,
            "supports_paper_claim": False,
        }
        _atomic_json(state_path, failure)
        raise
