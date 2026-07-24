from __future__ import annotations

import math
from typing import Any


def image_to_rgb_tensor(image: Any) -> Any:
    import numpy as np
    import torch

    array = np.asarray(image.convert("RGB"), dtype="float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def measured_ssim(reference_image: Any, candidate_image: Any) -> float:
    import torch
    import torch.nn.functional as functional

    if reference_image.size != candidate_image.size:
        raise ValueError("paired images must have the same size")
    window_size = min(11, *reference_image.size)
    if window_size % 2 == 0:
        window_size -= 1
    axis = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    kernel_1d = torch.exp(-(axis.square()) / (2.0 * 1.5**2))
    kernel_1d /= kernel_1d.sum()
    kernel = (kernel_1d[:, None] @ kernel_1d[None, :]).expand(
        3, 1, window_size, window_size
    )
    reference = image_to_rgb_tensor(reference_image)
    candidate = image_to_rgb_tensor(candidate_image)
    mean_ref = functional.conv2d(reference, kernel, groups=3)
    mean_candidate = functional.conv2d(candidate, kernel, groups=3)
    var_ref = functional.conv2d(reference.square(), kernel, groups=3) - mean_ref.square()
    var_candidate = (
        functional.conv2d(candidate.square(), kernel, groups=3)
        - mean_candidate.square()
    )
    covariance = (
        functional.conv2d(reference * candidate, kernel, groups=3)
        - mean_ref * mean_candidate
    )
    c1 = 0.01**2
    c2 = 0.03**2
    score = (
        (2 * mean_ref * mean_candidate + c1)
        * (2 * covariance + c2)
        / (
            (mean_ref.square() + mean_candidate.square() + c1)
            * (var_ref + var_candidate + c2)
        )
    )
    return float(score.mean().item())


def paired_quality(reference_image: Any, candidate_image: Any) -> dict[str, Any]:
    reference = image_to_rgb_tensor(reference_image)
    candidate = image_to_rgb_tensor(candidate_image)
    difference = reference - candidate
    mse = float(difference.square().mean().item())
    return {
        "mse": mse,
        "mean_abs_error": float(difference.abs().mean().item()),
        "psnr": "inf" if mse == 0.0 else 20.0 * math.log10(1.0 / math.sqrt(mse)),
        "ssim": measured_ssim(reference_image, candidate_image),
    }

