from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowhf.run_experiment import run_experiment
from flowhf.run_spec import load_run_spec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    arguments = parser.parse_args()
    watermark_key = os.environ.pop("FLOWHF_WATERMARK_KEY", "")
    hf_token = os.environ.pop("HF_TOKEN", None)
    if not watermark_key:
        parser.error("FLOWHF_WATERMARK_KEY is required")
    run_experiment(
        load_run_spec(arguments.request),
        watermark_key=watermark_key,
        hf_token=hf_token,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
