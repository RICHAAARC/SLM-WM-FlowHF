from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from flowhf.digest import stable_digest
from flowhf.hf_injector import HF_RELATIVE_L2


MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
MODEL_REVISION = "b940f670f0eda2d07fbb75229e779da1ad11eb80"
SCHEDULER_CLASS = (
    "diffusers.schedulers.scheduling_flow_match_euler_discrete."
    "FlowMatchEulerDiscreteScheduler"
)


@dataclass(frozen=True)
class FlowHFRunSpec:
    run_id: str
    repository_commit: str
    run_root: str
    prompt_id: str
    prompt: str
    negative_prompt: str
    seed: int
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    width: int = 512
    height: int = 512
    inference_steps: int = 20
    guidance_scale: float = 4.5
    latent_dtype: str = "float16"
    injection_step_index: int = 18
    hf_relative_l2: float = HF_RELATIVE_L2
    inversion_fixed_point_iterations: int = 8

    def __post_init__(self) -> None:
        if not self.run_id or "/" in self.run_id or "\\" in self.run_id:
            raise ValueError("run_id must be one safe path component")
        if (
            len(self.repository_commit) != 40
            or any(
                character not in "0123456789abcdef"
                for character in self.repository_commit
            )
        ):
            raise ValueError("repository_commit must be a full lowercase SHA")
        if not self.prompt_id or not self.prompt:
            raise ValueError("prompt_id and prompt are required")
        if self.model_id != MODEL_ID or self.model_revision != MODEL_REVISION:
            raise ValueError("the first experiment fixes the SD3.5 model")
        if (self.width, self.height, self.inference_steps) != (512, 512, 20):
            raise ValueError("the first experiment fixes 512x512 and 20 steps")
        if self.guidance_scale != 4.5 or self.latent_dtype != "float16":
            raise ValueError("guidance and dtype are fixed")
        if self.injection_step_index != 18:
            raise ValueError("the first experiment injects at callback index 18")
        if self.hf_relative_l2 != HF_RELATIVE_L2:
            raise ValueError("the first experiment fixes HF relative L2")
        if not 1 <= self.inversion_fixed_point_iterations <= 32:
            raise ValueError("inversion iterations must be between 1 and 32")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        root = Path(self.run_root)
        if not root.is_absolute():
            raise ValueError("run_root must be absolute")

    def public_record(self) -> dict[str, Any]:
        payload = asdict(self)
        return {**payload, "run_spec_digest": stable_digest(payload)}


def load_run_spec(path: str | Path) -> FlowHFRunSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise TypeError("run request must be one JSON object")
    return FlowHFRunSpec(**payload)

