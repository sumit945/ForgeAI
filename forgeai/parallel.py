"""Ordered parallel map for model calls.

Every step of the pipeline still runs and every check still applies; independent model calls (one per passage, topic or eval case) are
simply issued two at a time instead of one after another. Ollama serves 2 requests at once on a 4 GB GPU (measured: 6 calls 27 s
sequential, 13.5 s with 2 workers, no further gain with 3-4); raise FORGEAI_WORKERS together with OLLAMA_NUM_PARALLEL on bigger hardware,
or set FORGEAI_WORKERS=1 to run strictly sequentially.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def workers() -> int:
    try:
        return max(1, int(os.environ.get("FORGEAI_WORKERS", "2")))
    except ValueError:
        return 2


def pmap(fn: Callable[[T], R], items: Iterable[T], n: int | None = None) -> list[R]:
    """fn over items with up to n threads. Results keep the input order; the first exception is re-raised (budget stops included)."""
    items = list(items)
    n = min(n or workers(), len(items))
    if n <= 1:
        return [fn(i) for i in items]
    with ThreadPoolExecutor(n) as ex:
        return list(ex.map(fn, items))
