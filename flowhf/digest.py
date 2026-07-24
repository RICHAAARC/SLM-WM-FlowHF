from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


TENSOR_CONTENT_DIGEST_VERSION = "slm_wm_tensor_content"


def stable_json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_digest(value: Any) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def tensor_content_sha256(tensor: Any) -> str:
    import torch

    values = tensor.detach().cpu().contiguous()
    raw = values.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(TENSOR_CONTENT_DIGEST_VERSION.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            [int(value) for value in values.shape],
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(raw)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

