"""Cache of generated eval sets.

An eval set is a pure function of: the document contents, the requirement fields the generator reads, the model and its sampling settings,
the generator's prompts and code. When all of these are identical a rebuild can reuse the set that was already generated and machine-verified,
which skips the slowest pipeline stage. Anything that differs, even by one byte, gives a different key, so a wrong reuse cannot happen
(the worst case is a miss and a normal generation). Weak sets (fewer than 20 cases or any warning) are never cached.

`--fresh-evalset` (or fresh_evalset=True) ignores the cache and writes a new entry.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .docs import Document

CACHE_VERSION = "1"
REQ_FIELDS = ("purpose", "must_answer_topics", "must_refuse_topics", "known_gaps", "language_code", "language")
_PKG = Path(__file__).resolve().parent


def cache_dir() -> Path:
    return Path(os.environ.get("FORGEAI_CACHE", Path.home() / ".cache" / "forgeai")) / "evalsets"


def _code_digest() -> str:
    h = hashlib.sha1()
    files = [_PKG / n for n in ("evalgen.py", "textutil.py", "i18n.py")] + sorted((_PKG / "prompts").glob("evalgen_*.txt"))
    for f in files:
        try:
            h.update(f.read_bytes())
        except OSError:
            h.update(f.name.encode())
    return h.hexdigest()[:16]


def key(docs: list[Document], req: dict, llm, quotas: dict, seed: int = 11) -> str:
    payload = {
        "v": CACHE_VERSION,
        "docs": sorted((d.name, hashlib.sha1(d.text.encode("utf-8", "ignore")).hexdigest()) for d in docs),
        "req": {k: req.get(k) for k in REQ_FIELDS},
        "model": llm.model, "temperature": llm.temperature, "llm_seed": llm.seed, "num_ctx": llm.num_ctx,
        "quotas": quotas, "seed": seed, "code": _code_digest(),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()[:24]


def load(k: str) -> dict | None:
    """The cached set, or None (missing, unreadable or no longer valid: treated as a miss, never an error)."""
    p = cache_dir() / f"{k}.json"
    try:
        from .schemas import validate_evalset

        ev = json.loads(p.read_text(encoding="utf-8"))
        validate_evalset(ev)
        return ev if ev.get("total", 0) >= 20 and not ev.get("warnings") else None
    except Exception:  # noqa: BLE001
        return None


def save(k: str, ev: dict) -> bool:
    if ev.get("warnings") or ev.get("total", 0) < 20:
        return False  # a weak set must not stick
    try:
        d = cache_dir()
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f"{k}.json.tmp"
        tmp.write_text(json.dumps({**ev, "cache": {"key": k, "created": time.time(), "reused": False}}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(d / f"{k}.json")
        return True
    except OSError:
        return False  # caching must never break a build


def stamp_reused(ev: dict) -> dict:
    c = dict(ev.get("cache") or {})
    c["reused"] = True
    return {**ev, "cache": c}
