from __future__ import annotations

from typing import Any

from flowhf.evaluate_keys import evaluate_key_plan
from flowhf.key_plan import KeyPlan
from flowhf.model_runtime import encode_pil_to_latent


def evaluate_final_image(
    pipeline: Any,
    image: Any,
    *,
    key_plan: KeyPlan,
    model_identity_digest: str,
) -> tuple[Any, dict[str, Any]]:
    latent = encode_pil_to_latent(pipeline, image)
    result = evaluate_key_plan(
        latent,
        key_plan=key_plan,
        model_identity_digest=model_identity_digest,
        observation_domain="final_image_vae_reencoded",
    )
    return latent, result

