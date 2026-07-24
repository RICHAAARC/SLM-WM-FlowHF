from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from flowhf.digest import stable_digest, tensor_content_sha256
from flowhf.key_plan import KEYED_PRG_VERSION, build_keyed_gaussian_tensor


HF_RELATIVE_L2 = 0.012
HF_TAIL_FRACTION = 0.20
HF_LOW_PASS_KERNEL = 5
HF_LOW_PASS_PADDING = 2


@dataclass(frozen=True)
class HFTemplate:
    tensor: Any
    selected_element_count: int
    model_identity_digest: str
    scoring_key_identity_digest_random: str
    template_digest: str


@dataclass(frozen=True)
class HFWrite:
    latent: Any
    update: Any
    record: dict[str, Any]


def build_hf_template(
    reference_latent: Any,
    key_material: str,
    model_identity_digest: str,
) -> HFTemplate:
    import torch
    import torch.nn.functional as functional

    if not isinstance(reference_latent, torch.Tensor):
        raise TypeError("reference latent must be a Tensor")
    if reference_latent.ndim != 4 or int(reference_latent.shape[0]) != 1:
        raise ValueError("reference latent must have shape [1,C,H,W]")
    shape = tuple(int(value) for value in reference_latent.shape)
    raw = build_keyed_gaussian_tensor(
        shape,
        key_material,
        {
            "operator": "latent_carrier_template",
            "branch_name": "hf_tail_robust",
            "model_identity_digest": model_identity_digest,
        },
    )
    low_pass = functional.avg_pool2d(
        raw,
        kernel_size=HF_LOW_PASS_KERNEL,
        stride=1,
        padding=HF_LOW_PASS_PADDING,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    )
    high_pass = raw - low_pass
    flat = high_pass.reshape(-1)
    values = flat.tolist()
    selected_count = max(1, math.ceil(len(values) * HF_TAIL_FRACTION))
    selected_indices = sorted(
        range(len(values)),
        key=lambda index: (-abs(values[index]), index),
    )[:selected_count]
    sparse = torch.zeros_like(flat)
    selected = torch.tensor(selected_indices, dtype=torch.long, device="cpu")
    sparse[selected] = flat[selected]
    sparse = sparse.reshape(shape)
    norm = torch.linalg.vector_norm(sparse.reshape(-1))
    if not bool(torch.isfinite(norm)) or float(norm.item()) == 0.0:
        raise RuntimeError("HF template has no finite nonzero energy")
    normalized = sparse / norm
    key_digest = stable_digest({"key_material": key_material})
    content_digest = tensor_content_sha256(normalized)
    template_digest = stable_digest(
        {
            "carrier_template": "high_frequency_tail",
            "latent_shape": list(shape),
            "scoring_key_identity_digest": key_digest,
            "model_identity_digest": model_identity_digest,
            "prg_version": KEYED_PRG_VERSION,
            "prg_domain": "hf_tail_robust",
            "high_pass": "avg_pool2d_5x5_zero_padding_input_minus_low_pass",
            "tail_fraction": HF_TAIL_FRACTION,
            "tail_order": (
                "absolute_value_descending_then_flat_index_ascending"
            ),
            "selected_element_count": selected_count,
            "template_content_sha256": content_digest,
        }
    )
    return HFTemplate(
        tensor=normalized.to(
            device=reference_latent.device,
            dtype=torch.float32,
        ),
        selected_element_count=selected_count,
        model_identity_digest=model_identity_digest,
        scoring_key_identity_digest_random=key_digest,
        template_digest=template_digest,
    )


def apply_fixed_energy_hf(
    latent: Any,
    template: HFTemplate,
    *,
    relative_l2: float = HF_RELATIVE_L2,
) -> HFWrite:
    import torch

    if relative_l2 != HF_RELATIVE_L2:
        raise ValueError("the first experiment fixes HF relative L2 at 0.012")
    latent_float = latent.detach().to(dtype=torch.float32)
    latent_norm = torch.linalg.vector_norm(latent_float.reshape(-1))
    if not bool(torch.isfinite(latent_norm)) or float(latent_norm.item()) == 0.0:
        raise ValueError("latent must have finite nonzero L2 energy")
    update = template.tensor * latent_norm * relative_l2
    written = (latent_float + update).to(dtype=latent.dtype)
    actual_update = written.detach().to(dtype=torch.float32) - latent_float
    actual_norm = torch.linalg.vector_norm(actual_update.reshape(-1))
    realized = float((actual_norm / latent_norm).item())
    if not math.isfinite(realized) or realized <= 0.0:
        raise RuntimeError("HF update vanished after actual-dtype materialization")
    record = {
        "write_policy": "single_actual_dtype_addition",
        "injection_routing": "uniform_hf_tail",
        "target_relative_l2": relative_l2,
        "realized_relative_l2": realized,
        "template_digest": template.template_digest,
        "input_latent_content_sha256": tensor_content_sha256(latent),
        "update_content_sha256": tensor_content_sha256(actual_update),
        "written_latent_content_sha256": tensor_content_sha256(written),
    }
    return HFWrite(latent=written, update=actual_update, record=record)

