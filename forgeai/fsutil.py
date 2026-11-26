"""Crash-safe file writes that also survive Windows sharing violations (another thread/process or an antivirus scanner
holding the target open makes os.replace fail with 'Access is denied' for a few milliseconds)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def atomic_write(path: str | Path, obj, retries: int = 12) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = obj if isinstance(obj, str) else json.dumps(obj, indent=2, ensure_ascii=False)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{id(text) & 0xFFFF:x}.tmp")
    tmp.write_text(text, encoding="utf-8")
    last: Exception | None = None
    for i in range(retries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:  # Windows: target momentarily locked
            last = e
            time.sleep(0.02 * (i + 1))
    try:  # last resort: non-atomic write, never lose the data
        path.write_text(text, encoding="utf-8")
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    if last and not path.exists():
        raise last
