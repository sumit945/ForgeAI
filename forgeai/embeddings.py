"""Embedders for hybrid retrieval.

  HashingEmbedder - deterministic, offline, zero-download: hashed word + character-trigram features. Gives typo/morphology
                    tolerance ("hols" ~ "holidays") on top of BM25. Always available.
  OllamaEmbedder  - neural embeddings from a local Ollama model (e.g. nomic-embed-text, bge-m3). Used when the model is installed.

`get_embedder("hashing")` or `get_embedder("ollama:nomic-embed-text")`; an unavailable neural model degrades to hashing
(the caller is told which one it actually got).
"""
from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path

import numpy as np
import requests

from .errors import ModelUnavailable
from .retrieval import tokenize


def _norm(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class HashingEmbedder:
    name = "hashing"
    floor = 0.12  # minimum similarity threshold used by the hybrid retriever

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _features(self, text: str):
        for w in tokenize(text):
            yield w, 1.0
            p = f"#{w}#"
            for i in range(len(p) - 2):
                yield p[i : i + 3], 0.5

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            for f, wgt in self._features(t):
                h = zlib.crc32(f.encode("utf-8"))
                out[r, h % self.dim] += wgt * (1.0 if (h >> 20) & 1 else -1.0)
        return _norm(out)


_VEC_CACHE: dict[str, np.ndarray] = {}  # process-wide: rebuilt agents (every eval iteration) never re-embed the same chunk
_VEC_LOCK = __import__("threading").Lock()


class OllamaEmbedder:
    floor = 0.35

    def __init__(self, model: str, host: str = "http://localhost:11434", cache_path: str | Path | None = None, timeout: int = 120):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout
        self.name = f"ollama:{model}"
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, list[float]] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self._cache = {}

    def _key(self, t: str) -> str:
        return hashlib.sha1(f"{self.model}|{t}".encode()).hexdigest()

    def available(self) -> bool:
        try:
            names = [m["name"] for m in requests.get(f"{self.host}/api/tags", timeout=5).json().get("models", [])]
            return self.model in names or (":" not in self.model and any(n.split(":")[0] == self.model for n in names))
        except Exception:  # noqa: BLE001
            return False

    def embed(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if self._key(t) not in _VEC_CACHE and self._key(t) not in self._cache]
        for i in range(0, len(missing), 16):
            batch = missing[i : i + 16]
            try:
                r = requests.post(f"{self.host}/api/embed", json={"model": self.model, "input": [b[:6000] for b in batch]}, timeout=self.timeout)
                r.raise_for_status()
                vecs = r.json()["embeddings"]
            except Exception as e:  # noqa: BLE001
                raise ModelUnavailable(f"Embedding model '{self.model}' failed ({type(e).__name__}).", hint=f"Run: ollama pull {self.model}") from e
            with _VEC_LOCK:
                for t, v in zip(batch, vecs):
                    _VEC_CACHE[self._key(t)] = np.asarray(v, dtype=np.float32)
        if missing and self.cache_path:
            self._cache = {k: v.tolist() for k, v in _VEC_CACHE.items()}
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
            except OSError:
                pass
        return _norm(np.array([_VEC_CACHE[self._key(t)] if self._key(t) in _VEC_CACHE else np.asarray(self._cache[self._key(t)], dtype=np.float32) for t in texts], dtype=np.float32))


def get_embedder(spec: str | None, host: str = "http://localhost:11434", cache_path: str | Path | None = None):
    """Returns (embedder, note). Never raises: a missing neural model degrades to the offline hashing embedder."""
    if spec and spec.startswith("ollama:"):
        e = OllamaEmbedder(spec.split(":", 1)[1], host, cache_path)
        if e.available():
            return e, ""
        return HashingEmbedder(), f"embedding model '{spec}' is not installed; using the offline hashing embedder"
    return HashingEmbedder(), ""
