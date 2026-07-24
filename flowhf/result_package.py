from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tarfile
import tempfile
from typing import Any, Protocol

from flowhf.digest import file_sha256


class DriveAdapter(Protocol):
    def mount(self) -> None: ...

    def unmount(self) -> None: ...

    def copy_file(self, source: Path, destination: Path) -> None: ...


@dataclass(frozen=True)
class PackageResult:
    archive_path: Path
    checksum_path: Path
    archive_sha256: str
    scanned_file_count: int


def _scan_for_secrets(
    run_root: Path,
    *,
    secrets: tuple[str, ...],
) -> int:
    required = tuple(secret.encode("utf-8") for secret in secrets if secret)
    if not required:
        raise ValueError("at least the watermark key must be scanned")
    if any(secret in run_root.name.encode("utf-8") for secret in required):
        raise RuntimeError("secret scan rejected the run directory name")
    scanned = 0
    for path in sorted(run_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_root).as_posix().encode("utf-8")
        if any(secret in relative for secret in required):
            raise RuntimeError("secret scan rejected a local path")
        data = path.read_bytes()
        scanned += 1
        if any(secret in data for secret in required):
            raise RuntimeError("secret scan rejected the local run")
    return scanned


def _archive_members(run_root: Path) -> list[Path]:
    return [
        path
        for path in sorted(run_root.rglob("*"))
        if path.is_file()
    ]


def _verify_archive(
    archive_path: Path,
    run_root: Path,
) -> None:
    expected = {
        path.relative_to(run_root).as_posix(): (
            path.stat().st_size,
            file_sha256(path),
        )
        for path in _archive_members(run_root)
    }
    with tempfile.TemporaryDirectory(prefix="flowhf-package-verify-") as temp:
        extraction = Path(temp)
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive.getmembers():
                resolved = (extraction / member.name).resolve()
                if extraction.resolve() not in resolved.parents:
                    raise RuntimeError("archive member escaped verification root")
            archive.extractall(extraction, filter="data")
        extracted_root = extraction / run_root.name
        actual = {
            path.relative_to(extracted_root).as_posix(): (
                path.stat().st_size,
                file_sha256(path),
            )
            for path in sorted(extracted_root.rglob("*"))
            if path.is_file()
        }
    if actual != expected:
        raise RuntimeError("archive extraction did not reproduce the run")


def package_local_result(
    run_root: str | Path,
    delivery_directory: str | Path,
    *,
    watermark_key: str,
    hf_token: str | None = None,
) -> PackageResult:
    source = Path(run_root)
    if not source.is_dir():
        raise FileNotFoundError("run root does not exist")
    state = json.loads((source / "run_state.json").read_text(encoding="utf-8"))
    if state.get("status") not in {"success", "failure"}:
        raise RuntimeError("only terminal success or failure can be packaged")
    scanned = _scan_for_secrets(
        source,
        secrets=(watermark_key, hf_token or ""),
    )
    delivery = Path(delivery_directory)
    delivery.mkdir(parents=True, exist_ok=True)
    archive_path = delivery / f"{source.name}.tar.gz"
    checksum_path = delivery / f"{source.name}.tar.gz.sha256"
    if archive_path.exists() or checksum_path.exists():
        raise FileExistsError("delivery files already exist")
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in _archive_members(source):
            relative = path.relative_to(source)
            archive.add(
                path,
                arcname=(Path(source.name) / relative).as_posix(),
                recursive=False,
            )
    digest = file_sha256(archive_path)
    checksum_path.write_text(
        f"{digest}  {archive_path.name}\n",
        encoding="ascii",
    )
    _verify_archive(archive_path, source)
    return PackageResult(
        archive_path=archive_path,
        checksum_path=checksum_path,
        archive_sha256=digest,
        scanned_file_count=scanned,
    )


def deliver_package_to_drive(
    package: PackageResult,
    *,
    adapter: DriveAdapter,
    drive_results_directory: str | Path,
) -> dict[str, Any]:
    destination = Path(drive_results_directory)
    adapter.mount()
    try:
        archive_destination = destination / package.archive_path.name
        checksum_destination = destination / package.checksum_path.name
        adapter.copy_file(package.archive_path, archive_destination)
        adapter.copy_file(package.checksum_path, checksum_destination)
    finally:
        adapter.unmount()
    return {
        "archive_name": package.archive_path.name,
        "checksum_name": package.checksum_path.name,
        "archive_sha256": package.archive_sha256,
        "scanned_file_count": package.scanned_file_count,
    }
