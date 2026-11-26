"""Small shared text helpers used by guard, eval generator, judge and optimizer."""
from __future__ import annotations

import re

from .retrieval import tokenize


def norm(s: str) -> str:
    s = s.lower().replace("’", "'")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # 1,200 -> 1200
    s = re.sub(r"[^a-z0-9.%$ ]+", " ", s)
    s = re.sub(r"(?<![0-9])\.|\.(?![0-9])", " ", s)  # sentence dots, keep decimals
    return re.sub(r"\s+", " ", s).strip()


def contains(text: str, snippet: str) -> bool:
    return bool(snippet.strip()) and norm(snippet) in norm(text)


def quote_in(quote: str, text: str, thresh: float = 0.9) -> bool:
    """Is `quote` (nearly) verbatim in `text`? Used to catch LLMs that invent supporting quotes."""
    q = tokenize(quote, keep_stop=True)
    if not q:
        return False
    if contains(text, quote):  # exact (normalised) containment is enough even for short quotes like "48 hours"
        return True
    if len(q) < 3:  # a fuzzy match on 1-2 tokens would be meaningless
        return False
    tt = set(tokenize(text, keep_stop=True))
    return sum(1 for w in q if w in tt) / len(q) >= thresh


def clean_escalation(e: str, default: str = "the relevant team") -> str:
    """The requirements model sometimes returns an instruction ("Point users to HR for ...") instead of a contact.
    Reduce it to the noun phrase ("HR"), or fall back to a safe default, so the refusal message reads naturally."""
    e = (e or "").strip().rstrip(".")
    e = re.sub(r"^(please\s+)?(point|direct|refer|redirect|tell|advise|ask|send|route|escalate)\s+(?:\w+\s+){0,2}?(to|toward|towards)\s+", "", e, flags=re.I)
    e = re.sub(r"^(please\s+)?contact\s+", "", e, flags=re.I)
    e = re.split(r"\s+(for|when|if|whenever|in case|about|regarding|that|who)\s+", e, maxsplit=1, flags=re.I)[0]
    e = e.strip(" ,;:.")
    return e if e and len(e.split()) <= 8 else default


def numbers(s: str) -> set[str]:
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    return {n.rstrip(".") for n in re.findall(r"\d+(?:\.\d+)?", s)}


def jaccard(a: str, b: str) -> float:
    A, B = set(tokenize(a)), set(tokenize(b))
    return len(A & B) / len(A | B) if A | B else 0.0
