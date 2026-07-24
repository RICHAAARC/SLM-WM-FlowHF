from __future__ import annotations

import json

import pytest

from flowhf.run_experiment import run_experiment
from flowhf.run_spec import FlowHFRunSpec


def test_preflight_failure_is_packable_and_does_not_persist_key(
    monkeypatch,
    tmp_path,
) -> None:
    spec = FlowHFRunSpec(
        run_id="preflight-failure",
        repository_commit="a" * 40,
        run_root=str(tmp_path / "run"),
        prompt_id="p1",
        prompt="prompt",
        negative_prompt="",
        seed=3,
    )
    monkeypatch.setattr(
        "flowhf.run_experiment._assert_repository_identity",
        lambda _spec: (_ for _ in ()).throw(RuntimeError("identity failed")),
    )
    with pytest.raises(RuntimeError, match="identity failed"):
        run_experiment(spec, watermark_key="failure-test-raw-key")
    state_path = tmp_path / "run" / "run_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failure"
    assert b"failure-test-raw-key" not in state_path.read_bytes()
