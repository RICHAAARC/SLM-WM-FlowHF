from __future__ import annotations

from statistics import fmean
import math
from typing import Any

from flowhf.digest import stable_digest, tensor_content_sha256
from flowhf.hf_injector import build_hf_template
from flowhf.key_plan import KeyPlan


def normalized_correlation(observed: Any, template: Any) -> float:
    import torch

    left = observed.detach().to(dtype=torch.float32).reshape(-1)
    right = template.detach().to(dtype=torch.float32).reshape(-1)
    if left.numel() != right.numel():
        raise ValueError("score inputs must have the same element count")
    left = left - left.mean()
    right = right - right.mean()
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if (
        not bool(torch.isfinite(left_norm))
        or not bool(torch.isfinite(right_norm))
        or float(left_norm.item()) == 0.0
        or float(right_norm.item()) == 0.0
    ):
        raise ValueError("score inputs must have finite centered energy")
    return float(torch.dot(left / left_norm, right / right_norm).item())


def build_rank_record(
    registered_score: float,
    wrong_scores: tuple[float, ...],
) -> dict[str, Any]:
    if len(wrong_scores) != 32:
        raise ValueError("rank record requires exactly 32 wrong-key scores")
    if not math.isfinite(registered_score) or not all(
        math.isfinite(value) for value in wrong_scores
    ):
        raise ValueError("all scores must be finite")
    mean = fmean(wrong_scores)
    variance = fmean((value - mean) ** 2 for value in wrong_scores)
    payload = {
        "registered_score": registered_score,
        "wrong_score_count": len(wrong_scores),
        "wrong_score_mean": mean,
        "wrong_score_population_variance": variance,
        "registered_rank": 1
        + sum(value >= registered_score for value in wrong_scores),
        "registered_empirical_percentile": (
            sum(value < registered_score for value in wrong_scores)
            / len(wrong_scores)
        ),
        "registered_minus_max_wrong_margin": (
            registered_score - max(wrong_scores)
        ),
        "registered_minus_mean_wrong_margin": registered_score - mean,
    }
    return {**payload, "rank_record_digest": stable_digest(payload)}


def evaluate_key_plan(
    observed_latent: Any,
    *,
    key_plan: KeyPlan,
    model_identity_digest: str,
    observation_domain: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for candidate in key_plan.ordered:
        template = build_hf_template(
            observed_latent,
            candidate.material,
            model_identity_digest,
        )
        score = normalized_correlation(observed_latent, template.tensor)
        records.append(
            {
                "key_role": candidate.role,
                "wrong_key_index": candidate.index,
                "key_material_digest_random": (
                    candidate.material_digest_random
                ),
                "hf_score": score,
                "template_digest": template.template_digest,
            }
        )
    rank = build_rank_record(
        records[0]["hf_score"],
        tuple(record["hf_score"] for record in records[1:]),
    )
    payload = {
        "observation_domain": observation_domain,
        "score_method_role": "hf_tail_only_content",
        "score_operator": "centered_float32_normalized_correlation",
        "observed_latent_content_sha256": tensor_content_sha256(
            observed_latent
        ),
        "key_plan_digest_random": key_plan.plan_digest_random,
        "key_score_records": records,
        "rank_record": rank,
        "wrong_key_feedback_allowed": False,
    }
    return {**payload, "evaluation_digest": stable_digest(payload)}
