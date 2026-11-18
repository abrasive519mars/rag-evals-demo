"""Cache files for generations and judge verdicts.

A long run is checkpointed by copying the cache directory while workers are still writing to it, so an
entry has to appear whole or not at all: a half-written file restored on the next run would be read as a
finished answer.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path


def write_atomic(path: Path, text: str) -> None:
    """Write next to the target under a per-thread name, then rename over it in one step."""
    tmp = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
