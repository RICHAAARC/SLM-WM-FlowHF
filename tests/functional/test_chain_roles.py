from __future__ import annotations

from types import SimpleNamespace
import inspect

from PIL import Image
import torch

from flowhf.key_plan import build_key_plan
from flowhf.model_runtime import ModelRuntime, PromptConditioning
from flowhf.run_experiment import (
    _freeze_chain_observations,
    _run_chain,
    run_experiment,
)
from flowhf.run_spec import FlowHFRunSpec


class FakePipeline:
    def __call__(self, **kwargs):
        latent = kwargs["latents"]
        callback = kwargs["callback_on_step_end"]
        for step in range(20):
            callback_values = callback(
                self,
                step,
                torch.tensor(float(20 - step)),
                {"latents": latent},
            )
            latent = callback_values["latents"] + 0.01
        return SimpleNamespace(images=latent)


def test_clean_and_watermarked_have_exactly_one_role_specific_write(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "flowhf.run_experiment.decode_latent_to_pil",
        lambda _pipeline, _latent: Image.new("RGB", (16, 16)),
    )
    spec = FlowHFRunSpec(
        run_id="unit",
        repository_commit="a" * 40,
        run_root=str(tmp_path / "run"),
        prompt_id="p1",
        prompt="test prompt",
        negative_prompt="",
        seed=1,
    )
    runtime = ModelRuntime(
        pipeline=FakePipeline(),
        torch_module=torch,
        record={"model_identity_digest": "model"},
    )
    conditioning = PromptConditioning(None, None, None, None)
    base = torch.ones((1, 4, 8, 8), dtype=torch.float16)
    plan = build_key_plan("functional-key")
    clean = _run_chain(
        runtime,
        spec,
        conditioning,
        base,
        role="clean",
        registered_key_material=plan.registered.material,
    )
    watermarked = _run_chain(
        runtime,
        spec,
        conditioning,
        base,
        role="watermarked",
        registered_key_material=plan.registered.material,
    )
    assert clean.injection_record is None
    assert watermarked.injection_record is not None
    assert watermarked.injection_record["write_policy"] == (
        "single_actual_dtype_addition"
    )
    assert not torch.equal(clean.callback_latent, watermarked.callback_latent)


def test_wrong_key_roster_is_built_only_after_both_chains_are_frozen() -> None:
    run_source = inspect.getsource(run_experiment)
    assert run_source.index("clean_frozen = _freeze_chain_observations") < (
        run_source.index("key_plan = build_key_plan")
    )
    assert run_source.index("watermarked_frozen = _freeze_chain_observations") < (
        run_source.index("key_plan = build_key_plan")
    )
    assert "key_plan" not in inspect.signature(_run_chain).parameters
    assert "key_plan" not in inspect.signature(
        _freeze_chain_observations
    ).parameters
