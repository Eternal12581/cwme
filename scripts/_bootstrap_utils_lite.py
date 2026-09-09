"""Install lightweight utils stub for offline sanity scripts."""
from __future__ import annotations

import sys


def bootstrap_utils_lite() -> None:
    import utils_lite

    sys.modules["utils"] = utils_lite
