from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flowhf.colab_entry import (
    GoogleColabDriveAdapter,
    colab_secret_getter,
    package_and_deliver_from_disk,
)


def main() -> int:
    package_and_deliver_from_disk(
        adapter=GoogleColabDriveAdapter(),
        secret_getter=colab_secret_getter,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
