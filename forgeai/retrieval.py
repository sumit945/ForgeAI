"""Dependency-free BM25 retriever with light stemming, glossary query expansion and neighbour expansion."""
from __future__ import annotations

import math
import re
from collections import Counter

from .docs import Chunk
from .lang import STOP_INDIC, TOKEN_RE, normalize_text, stem_indic

STOP = set(
    """a an the and or but if of to in on at by for with from as is are was were be been being it its this that these those
    do does did can could should would may might will shall i you we they he she them his her our your their my me us
    what which who whom whose when where why how any all some no not than then so such there here about into over under
    per also please tell give need want know get have has had""".split()
)


def stem(w: str) -> str:
    for suf in ("ations", "ation", "ingly", "ings", "ing", "edly", "ies", "ied", "ed", "es", "s", "ly"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            if suf in ("ies", "ied"):
                return w[: -len(suf)] + "y"
            return w[: -len(suf)]
    return w


def tokenize(text: str, keep_stop: bool = False) -> list[str]:
    """Unicode-aware tokens: English words are stemmed as before; Indian-language words are normalised (native digits ->
    ASCII, danda/nukta/joiners removed), stop-listed and, for the Dravidian scripts, truncation-stemmed."""
    out: list[str] = []
    for tok in TOKEN_RE.findall(normalize_text(text)):
        tok = tok.replace(",", "")
        if tok.isascii():
            if keep_stop or tok not in STOP:
                out.append(stem(tok))
        elif keep_stop or tok not in STOP_INDIC:
            out.append(stem_indic(tok))
    return out


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        k1: float = 1.4,
        b: float = 0.75,
        glossary: dict[str, list[str]] | None = None,
    ):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.glossary = {stem(k.lower()): v for k, v in (glossary or {}).items()}
        self.tf = [Counter(tokenize(c.text)) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 1.0
        df: Counter = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(chunks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.by_doc: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            self.by_doc.setdefault(c.doc, []).append(i)

    def expand(self, query: str) -> list[str]:
        toks = tokenize(query)
        extra: list[str] = []
        if self.glossary:
            for t in list(toks):
                for syn in self.glossary.get(t, []):
                    extra.extend(tokenize(syn))
        return toks + extra

    def scores(self, query: str) -> list[float]:
        q = self.expand(query)
        out = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf.get(w, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            out.append(s)
        return out

    def search(self, query: str, top_k: int = 4, min_score: float = 0.0, neighbors: int = 0) -> list[tuple[Chunk, float]]:
        sc = self.scores(query)
        order = sorted(range(len(sc)), key=lambda i: -sc[i])
        picked = [i for i in order[:top_k] if sc[i] > 0 and sc[i] >= min_score]
        if neighbors and picked:
            seen = set(picked)
            for i in list(picked[:2]):  # only expand around the strongest hits
                c = self.chunks[i]
                pos = self.by_doc[c.doc].index(i)
                for d in range(1, neighbors + 1):
                    for j in (pos - d, pos + d):
                        if 0 <= j < len(self.by_doc[c.doc]):
                            k = self.by_doc[c.doc][j]
                            if k not in seen:
                                picked.append(k)
                                seen.add(k)
        return [(self.chunks[i], sc[i]) for i in picked]

    def rank_of(self, query: str, predicate) -> int | None:
        """1-based rank of the first chunk (by BM25) satisfying predicate(chunk.text); None if absent."""
        sc = self.scores(query)
        order = sorted(range(len(sc)), key=lambda i: -sc[i])
        for r, i in enumerate(order, 1):
            if sc[i] <= 0:
                break
            if predicate(self.chunks[i].text):
                return r
        return None

    def self_score(self, text: str) -> float:
        """Score of a text against itself - used to normalise 'how well is this covered' signals."""
        q = tokenize(text)
        return sum(self.idf.get(w, 0.5) for w in set(q)) or 1.0


class HybridRetriever(Retriever):
    """BM25 + dense similarity fused with Reciprocal Rank Fusion.

    Dense candidates only count above a similarity floor calibrated on the corpus with unrelated probe questions, so an
    off-topic question still retrieves nothing and the agent can refuse without a model call.
    """

    def __init__(self, chunks, embedder, glossary=None, rrf_k: int = 60, dense_min_sim: float | None = None, **kw):
        super().__init__(chunks, glossary=glossary, **kw)
        import numpy as np

        self.np, self.embedder, self.rrf_k = np, embedder, rrf_k
        self.E = embedder.embed([c.text for c in chunks]) if chunks else np.zeros((0, 1))
        self.tau = dense_min_sim if dense_min_sim is not None else self._calibrate()

    PROBES = [
        "What is the capital of France?", "Write a short poem about the ocean.", "How do I bake sourdough bread?",
        "Who won the football world cup in 2018?", "Explain how photosynthesis works.", "Write python code to sort a list.",
        "What is the best way to learn the guitar?", "Translate good morning into Spanish.", "Tell me a joke about cats.",
        "How tall is Mount Everest?", "What should I cook for dinner tonight?", "Recommend a good science fiction novel.",
    ]

    def _calibrate(self) -> float:
        """Similarity floor = the highest similarity that clearly UNRELATED probe questions reach against this corpus (+ margin)."""
        floor = getattr(self.embedder, "floor", 0.2)
        if not len(self.chunks):
            return floor
        P = self.embedder.embed(self.PROBES)
        return max(floor, float((P @ self.E.T).max()) + 0.02)

    def dense_scores(self, query: str):
        q = self.embedder.embed([" ".join(self.expand(query)) or query])[0]
        return self.E @ q

    def scores(self, query: str) -> list[float]:
        b = super().scores(query)
        d = self.dense_scores(query)
        fused = [0.0] * len(b)
        for rank, i in enumerate(sorted((i for i in range(len(b)) if b[i] > 0), key=lambda i: -b[i]), 1):
            fused[i] += 1.0 / (self.rrf_k + rank)
        for rank, i in enumerate(sorted((i for i in range(len(d)) if d[i] >= self.tau), key=lambda i: -d[i]), 1):
            fused[i] += 1.0 / (self.rrf_k + rank)
        return fused
