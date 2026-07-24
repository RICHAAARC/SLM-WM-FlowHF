from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from flowhf.digest import tensor_content_sha256
from flowhf.evaluate_keys import build_rank_record, evaluate_key_plan
from flowhf.hf_injector import apply_fixed_energy_hf, build_hf_template
from flowhf.key_plan import build_key_plan
from flowhf.model_runtime import fixed_model_identity_digest
from flowhf.run_spec import FlowHFRunSpec


FIXTURE = Path(__file__).parents[1] / "fixtures" / "hf_template_golden.json"


def test_hf_template_matches_v1_golden_vector() -> None:
    golden = json.loads(FIXTURE.read_text(encoding="utf-8"))
    reference = torch.zeros(tuple(golden["shape"]), dtype=torch.float32)
    template = build_hf_template(
        reference,
        golden["key_material_for_test_only"],
        golden["model_identity_digest"],
    )
    assert template.selected_element_count == golden["selected_element_count"]
    assert (
        tensor_content_sha256(template.tensor)
        == golden["template_content_sha256"]
    )
    assert template.tensor.reshape(-1)[:8].tolist() == pytest.approx(
        golden["first_eight_values"],
        abs=0.0,
    )


def test_environment_changes_do_not_enter_template_model_identity(
    tmp_path,
) -> None:
    golden = json.loads(FIXTURE.read_text(encoding="utf-8"))
    spec = FlowHFRunSpec(
        run_id="identity",
        repository_commit="a" * 40,
        run_root=str(tmp_path / "run"),
        prompt_id="p1",
        prompt="prompt",
        negative_prompt="",
        seed=1,
    )
    identity = fixed_model_identity_digest(spec)
    assert identity == golden["model_identity_digest"]
    reference = torch.zeros(tuple(golden["shape"]), dtype=torch.float32)
    environment_a = {"torch": "cpu", "gpu": "none"}
    environment_b = {"torch": "2.11.0+cu128", "gpu": "A100"}
    assert environment_a != environment_b
    template_a = build_hf_template(
        reference,
        golden["key_material_for_test_only"],
        identity,
    )
    template_b = build_hf_template(
        reference,
        golden["key_material_for_test_only"],
        fixed_model_identity_digest(spec),
    )
    assert tensor_content_sha256(template_a.tensor) == tensor_content_sha256(
        template_b.tensor
    )
    assert template_a.template_digest == template_b.template_digest


def test_fixed_energy_write_and_33_key_rank_are_separate() -> None:
    latent = torch.linspace(-2.0, 2.0, 4 * 8 * 8).reshape(1, 4, 8, 8)
    key_plan = build_key_plan("unit-test-registered-key")
    template = build_hf_template(latent, key_plan.registered.material, "model")
    written = apply_fixed_energy_hf(latent, template)
    assert written.record["target_relative_l2"] == 0.012
    assert written.record["realized_relative_l2"] == pytest.approx(
        0.012,
        rel=1e-5,
    )
    scores = evaluate_key_plan(
        written.latent,
        key_plan=key_plan,
        model_identity_digest="model",
        observation_domain="unit",
    )
    assert len(scores["key_score_records"]) == 33
    assert [item["wrong_key_index"] for item in scores["key_score_records"][1:]] == list(
        range(32)
    )
    public = json.dumps(key_plan.public_record(), sort_keys=True)
    assert "unit-test-registered-key" not in public
    assert "unit-test-registered-key" not in repr(key_plan)


def test_rank_record_keeps_exact_wrong_key_denominator() -> None:
    with pytest.raises(ValueError, match="exactly 32"):
        build_rank_record(1.0, (0.0,))
