from __future__ import annotations

import os

import pytest

from flowhf.run_experiment import run_experiment
from flowhf.run_spec import load_run_spec


@pytest.mark.gpu
def test_real_one_prompt_a100_smoke() -> None:
    request = os.environ.get("FLOWHF_RUN_REQUEST")
    key = os.environ.get("FLOWHF_WATERMARK_KEY")
    if not request or not key:
        pytest.skip("explicit real-run request and watermark key are required")
    run_experiment(
        load_run_spec(request),
        watermark_key=key,
        hf_token=os.environ.get("HF_TOKEN"),
    )
