from __future__ import annotations

import hashlib
import json
from pathlib import Path

from flowhf import colab_entry


class FakeDrive:
    def __init__(self) -> None:
        self.events: list[str] = []

    def mount(self) -> None:
        self.events.append("mount")

    def unmount(self) -> None:
        self.events.append("unmount")

    def copy_file(self, _source: Path, _destination: Path) -> None:
        raise AssertionError("input preparation must not copy output")


def test_notebook_is_thin_output_free_and_final_cell_is_independent() -> None:
    notebook_path = (
        Path(__file__).parents[2]
        / "notebooks"
        / "content_survival_flow_hf_colab.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert len(code_cells) == 3
    assert all(cell["outputs"] == [] for cell in code_cells)
    combined = "\n".join("".join(cell["source"]) for cell in code_cells)
    assert "run_experiment" not in combined
    assert "watermark_raw_key.txt" not in combined
    bootstrap_source = "".join(code_cells[0]["source"])
    assert bootstrap_source.index("expected_commit =") < (
        bootstrap_source.index("'pip', 'install'")
    )
    assert bootstrap_source.index("'pip', 'install'") < (
        bootstrap_source.index("from flowhf import colab_entry")
    )
    assert bootstrap_source.index("from flowhf import colab_entry") < (
        bootstrap_source.index("load_verified_drive_input")
    )
    final_source = "".join(code_cells[-1]["source"])
    assert "prepared" not in final_source
    assert "package_and_deliver_from_disk" in final_source


def test_notebook_registry_matches_bytes() -> None:
    root = Path(__file__).parents[2]
    registry = json.loads(
        (root / "notebooks" / "registry.json").read_text(encoding="utf-8")
    )
    for relative, expected in registry.items():
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        assert actual == expected


def test_drive_input_persists_request_but_not_raw_key(
    monkeypatch,
    tmp_path,
) -> None:
    drive_input = tmp_path / "drive"
    drive_input.mkdir()
    request_path = drive_input / "run_request.json"
    key_path = drive_input / "watermark_raw_key.txt"
    local_path = tmp_path / "local" / "run_request.json"
    request = {
        "run_id": "smoke",
        "repository_commit": "a" * 40,
        "run_root": str(tmp_path / "smoke"),
        "prompt_id": "p1",
        "prompt": "one prompt",
        "negative_prompt": "",
        "seed": 7,
    }
    request_path.write_text(json.dumps(request), encoding="utf-8")
    key_path.write_text("drive-only-test-key\n", encoding="utf-8")
    monkeypatch.setattr(colab_entry, "DRIVE_REQUEST_PATH", request_path)
    monkeypatch.setattr(colab_entry, "DRIVE_KEY_PATH", key_path)
    monkeypatch.setattr(colab_entry, "LOCAL_INPUT_ROOT", local_path.parent)
    monkeypatch.setattr(colab_entry, "LOCAL_REQUEST_PATH", local_path)
    monkeypatch.setattr(colab_entry, "LOCAL_RUNS_ROOT", tmp_path)
    local_path.parent.mkdir()
    local_path.write_bytes(request_path.read_bytes())
    drive = FakeDrive()
    prepared = colab_entry.load_verified_drive_input(
        adapter=drive,
        secret_getter=lambda _name: (_ for _ in ()).throw(KeyError()),
    )
    assert prepared.watermark_key == "drive-only-test-key"
    assert drive.events == ["mount", "unmount"]
    assert b"drive-only-test-key" not in local_path.read_bytes()
