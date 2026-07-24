from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable

from flowhf.result_package import (
    DriveAdapter,
    deliver_package_to_drive,
    package_local_result,
)
from flowhf.run_experiment import run_experiment
from flowhf.run_spec import FlowHFRunSpec, load_run_spec


PUBLIC_REPOSITORY_URL = "https://github.com/RICHAAARC/SLM-WM-FlowHF.git"
COLAB_REPOSITORY_ROOT = Path("/content/SLM-WM-FlowHF")
LOCAL_INPUT_ROOT = Path("/content/flowhf/input")
LOCAL_REQUEST_PATH = LOCAL_INPUT_ROOT / "run_request.json"
LOCAL_RUNS_ROOT = Path("/content/flowhf/runs")
LOCAL_DELIVERY_ROOT = Path("/content/flowhf/delivery")
DRIVE_INPUT_ROOT = Path("/content/drive/MyDrive/SLM/flow-hf/inputs")
DRIVE_RESULTS_ROOT = Path("/content/drive/MyDrive/SLM/flow-hf/results")
DRIVE_REQUEST_PATH = DRIVE_INPUT_ROOT / "run_request.json"
DRIVE_KEY_PATH = DRIVE_INPUT_ROOT / "watermark_raw_key.txt"
HF_TOKEN_SECRET_NAME = "HF_TOKEN"


@dataclass(frozen=True)
class PreparedInput:
    spec: FlowHFRunSpec
    watermark_key: str = field(repr=False)
    hf_token: str | None = field(default=None, repr=False)


class GoogleColabDriveAdapter:
    def __init__(self) -> None:
        from google.colab import drive

        self._drive = drive

    def mount(self) -> None:
        self._drive.mount("/content/drive", force_remount=False)

    def unmount(self) -> None:
        self._drive.flush_and_unmount()

    def copy_file(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def _optional_hf_token(secret_getter: Callable[[str], str]) -> str | None:
    try:
        token = secret_getter(HF_TOKEN_SECRET_NAME)
    except (KeyError, PermissionError):
        return None
    return token or None


def load_verified_drive_input(
    *,
    adapter: DriveAdapter,
    secret_getter: Callable[[str], str],
) -> PreparedInput:
    """Load the raw key only after verified project code is installed."""

    local_request_bytes = LOCAL_REQUEST_PATH.read_bytes()
    spec = load_run_spec(LOCAL_REQUEST_PATH)
    adapter.mount()
    try:
        drive_request_bytes = DRIVE_REQUEST_PATH.read_bytes()
        watermark_key = DRIVE_KEY_PATH.read_text(encoding="utf-8").strip()
    finally:
        adapter.unmount()
    if drive_request_bytes != local_request_bytes:
        raise RuntimeError("Drive request changed after commit verification")
    if not watermark_key:
        raise RuntimeError("Drive watermark key file is empty")
    expected_run_root = LOCAL_RUNS_ROOT / spec.run_id
    if Path(spec.run_root) != expected_run_root:
        raise RuntimeError("Colab run_root must use the fixed local runs root")
    return PreparedInput(
        spec=spec,
        watermark_key=watermark_key,
        hf_token=_optional_hf_token(secret_getter),
    )


def bootstrap_public_checkout(
    spec: FlowHFRunSpec,
    *,
    repository_root: Path = COLAB_REPOSITORY_ROOT,
    command_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, str]:
    remote_url = command_runner(
        ["git", "remote", "get-url", "origin"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if remote_url != PUBLIC_REPOSITORY_URL:
        raise RuntimeError("checkout origin is not the fixed public repository")
    remote_line = command_runner(
        ["git", "ls-remote", "origin", "refs/heads/main"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    remote_sha = remote_line.split()[0] if remote_line else ""
    if remote_sha != spec.repository_commit:
        raise RuntimeError("published main differs from the run request")
    head = command_runner(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != remote_sha:
        raise RuntimeError("fresh clone HEAD differs from published main")
    command_runner(
        ["git", "switch", "--detach", remote_sha],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "repository_url": PUBLIC_REPOSITORY_URL,
        "resolved_commit": remote_sha,
        "checkout_mode": "detached",
    }


def run_prepared_input(prepared: PreparedInput) -> dict[str, Any]:
    return run_experiment(
        prepared.spec,
        watermark_key=prepared.watermark_key,
        hf_token=prepared.hf_token,
    )


def package_and_deliver_from_disk(
    *,
    adapter: DriveAdapter,
    secret_getter: Callable[[str], str],
) -> dict[str, Any]:
    """Independent final-cell entry; relies only on the fixed local request."""

    spec = load_run_spec(LOCAL_REQUEST_PATH)
    adapter.mount()
    try:
        watermark_key = DRIVE_KEY_PATH.read_text(encoding="utf-8").strip()
    finally:
        adapter.unmount()
    if not watermark_key:
        raise RuntimeError("Drive watermark key file is empty")
    hf_token = _optional_hf_token(secret_getter)
    package = package_local_result(
        spec.run_root,
        LOCAL_DELIVERY_ROOT,
        watermark_key=watermark_key,
        hf_token=hf_token,
    )
    return deliver_package_to_drive(
        package,
        adapter=adapter,
        drive_results_directory=DRIVE_RESULTS_ROOT,
    )


def colab_secret_getter(name: str) -> str:
    from google.colab import userdata

    return userdata.get(name)
