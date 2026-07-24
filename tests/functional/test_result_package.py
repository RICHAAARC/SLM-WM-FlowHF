from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from flowhf.result_package import (
    deliver_package_to_drive,
    package_local_result,
)


class FakeDrive:
    def __init__(self) -> None:
        self.events: list[str] = []

    def mount(self) -> None:
        self.events.append("mount")

    def unmount(self) -> None:
        self.events.append("unmount")

    def copy_file(self, source: Path, destination: Path) -> None:
        self.events.append(f"copy:{destination.name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def _run_root(tmp_path: Path, status: str = "success") -> Path:
    root = tmp_path / "run"
    root.mkdir()
    (root / "run_state.json").write_text(
        json.dumps({"status": status}),
        encoding="utf-8",
    )
    (root / "evidence.json").write_text('{"diagnostic_only":true}\n')
    return root


@pytest.mark.parametrize("status", ["success", "failure"])
def test_success_and_failure_can_be_packaged_and_delivered(
    tmp_path,
    status,
) -> None:
    root = _run_root(tmp_path, status)
    package = package_local_result(
        root,
        tmp_path / "delivery",
        watermark_key="package-test-key",
    )
    drive = FakeDrive()
    result = deliver_package_to_drive(
        package,
        adapter=drive,
        drive_results_directory=tmp_path / "drive",
    )
    assert result["archive_sha256"] == package.archive_sha256
    assert drive.events == [
        "mount",
        f"copy:{package.archive_path.name}",
        f"copy:{package.checksum_path.name}",
        "unmount",
    ]


def test_secret_leak_rejects_before_drive_mount(tmp_path) -> None:
    root = _run_root(tmp_path)
    (root / "evidence.json").write_text("package-test-key", encoding="utf-8")
    with pytest.raises(RuntimeError, match="secret scan"):
        package_local_result(
            root,
            tmp_path / "delivery",
            watermark_key="package-test-key",
        )

