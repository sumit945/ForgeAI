"""Stage 2 - Groundability guard.

For every requested topic (and the overall purpose) we ask: do the documents actually contain this?
Independent signals are combined so a small model can neither talk itself into false support nor veto covered topics:
  1. lexical coverage of the topic's content words in the top retrieved excerpts (deterministic)
  2. a deterministic "evidence sentence": one sentence containing most topic words that also states something concrete
  3. an LLM verdict that must come with a verbatim quote - the quote is verified against the excerpts.
Unsupported topics are removed from `must_answer_topics`, recorded as `known_gaps`, and become refusals.
"""
from __future__ import annotations

import re

from .docs import Document, chunk_documents
from .errors import BudgetExceeded, ModelUnavailable
from .llm import LLM
from .prompts import render
from .schemas import GroundCheckOut
from .retrieval import Retriever, tokenize
from .textutil import quote_in
from .tools import evidence_hits

MODAL = re.compile(
    r"\b(must|shall|may|entitled|receive[sd]?|required?|within|at least|no more than|up to|per|pay|paid|approved?|accepted)\b", re.I
)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def evidence_sentence(topic: str, texts: list[str]) -> tuple[str | None, float]:
    """A single sentence that contains most of the topic words AND states something concrete (a number or a rule word)."""
    tt = {t for t in tokenize(topic) if len(t) > 2}
    if not tt:
        return None, 0.0
    best, best_s = None, 0.0
    for t in texts:
        for sent in SENT_SPLIT.split(t):
            if len(sent.split()) < 4 or not (re.search(r"\d", sent) or MODAL.search(sent)):
                continue
            sc = len(tt & set(tokenize(sent))) / len(tt)
            if sc > best_s:
                best, best_s = sent.strip(), sc
    return (best, best_s) if best_s >= 0.5 else (None, best_s)


def check_topic(topic: str, retr: Retriever, llm: LLM | None) -> dict:
    wide = 8 if len({c.doc for c in retr.chunks}) > 10 else 3  # many documents share a pattern: look at more of them
    hits = retr.search(topic, top_k=wide)
    excerpts = "\n---\n".join(f"[{c.doc}] {c.text}" for c, _ in hits[:3])
    union = set(tokenize(" ".join(c.text for c, _ in hits)))
    topic_toks = [t for t in set(tokenize(topic)) if len(t) > 2]
    coverage = (sum(1 for t in topic_toks if t in union) / len(topic_toks)) if topic_toks else 0.0
    sent, _ = evidence_sentence(topic, [c.text for c, _ in hits])
    res = {
        "topic": topic, "coverage": round(coverage, 2), "top_docs": sorted({c.doc for c, _ in hits}),
        "evidence_sentence": sent, "llm_verdict": None, "quote": "", "quote_verified": False, "notes": [],
    }
    if not hits:
        res.update(verdict="unsupported", notes=["no document text overlaps this topic"])
        return res
    verdict = None
    if llm is not None:
        sysm, user = render("groundability_check", topic=topic, excerpts=excerpts)
        try:
            d = llm.json(sysm, user, task="groundability_check", schema=GroundCheckOut, max_tokens=250)
            verdict = str(d.get("verdict", "")).lower().strip()
            res["llm_verdict"] = verdict
            res["quote"] = str(d.get("quote", ""))
            res["quote_verified"] = quote_in(res["quote"], "\n".join(c.text for c, _ in hits))
        except (BudgetExceeded, ModelUnavailable):  # never swallow a budget stop or an outage
            raise
        except Exception as e:  # noqa: BLE001
            res["notes"].append(f"LLM check failed ({type(e).__name__}); using deterministic signals only")
    if verdict == "yes" and not res["quote_verified"]:
        res["notes"].append("LLM said 'yes' but its quote is not in the documents -> its verdict is not trusted")
        verdict = "partial"
    if coverage < 0.3:
        final = "unsupported"
    elif sent:  # concrete sentence found deterministically
        final = "supported"
        if verdict == "no":
            res["notes"].append(f'LLM said "no" but the documents state: "{sent[:110]}" -> supported by deterministic evidence')
    elif verdict == "yes":
        final = "supported" if coverage >= 0.5 else "partial"
    elif verdict is None:
        final = "partial" if coverage >= 0.5 else "unsupported"
    elif verdict == "no":
        final = "unsupported"
    else:
        final = "partial"
    res["verdict"] = final
    return res


def _tool_covered(topic: str) -> bool:
    return any(evidence_hits(t, topic) for t in ("open_meteo_weather", "current_date"))


def groundability_guard(req: dict, docs: list[Document], llm: LLM | None) -> tuple[dict, dict]:
    """Returns (report, narrowed_requirements)."""
    retr = Retriever(chunk_documents(docs, 120, 30))
    topics = []
    # A topic only counts if the BRIEF asked for it. Small models like to "helpfully" add topics lifted from the document
    # inventory; those would be trivially supported and could mask an unsupportable brief.
    brief_toks = {t for t in tokenize(req.get("brief", "")) if len(t) > 3}

    def _requested(topic: str) -> bool:
        tt = {t for t in tokenize(topic) if len(t) > 3}
        return not brief_toks or not tt or bool(tt & brief_toks)

    flags = {t: _requested(t) for t in req["must_answer_topics"]}
    if not any(flags.values()):  # nothing overlaps the brief wording at all: fall back to trusting the list
        flags = {t: True for t in flags}
    for t in req["must_answer_topics"]:
        if not flags[t]:
            topics.append({"topic": t, "verdict": "not_requested", "coverage": None,
                           "notes": ["not asked for in the brief (added by the requirements model); ignored and removed from scope"]})
            continue
        if _tool_covered(t):
            topics.append({"topic": t, "verdict": "supported_by_tool", "coverage": None, "notes": ["answerable via a live tool, not documents"]})
        else:
            topics.append(check_topic(t, retr, llm))
    purpose = check_topic(req["purpose"], retr, llm) if not _tool_covered(req["purpose"]) else {"verdict": "supported_by_tool"}

    ignored = [t for t in topics if t["verdict"] == "not_requested"]
    topics_req = [t for t in topics if t["verdict"] != "not_requested"]
    ok = [t for t in topics_req if t["verdict"] in ("supported", "supported_by_tool")]
    partial = [t for t in topics_req if t["verdict"] == "partial"]
    bad = [t for t in topics_req if t["verdict"] == "unsupported"]
    n = max(1, len(topics_req))
    if (not ok and not partial) or len(bad) / n >= 0.6:
        status = "blocked"
    elif bad or partial:
        status = "warn"
    else:
        status = "ok"

    gaps = [t["topic"] for t in bad]
    narrowed = dict(req)
    narrowed["must_answer_topics"] = [t["topic"] for t in topics if t["verdict"] not in ("unsupported", "not_requested")]
    narrowed["known_gaps"] = gaps
    narrowed["must_refuse_topics"] = list(req["must_refuse_topics"]) + [
        f"{g} (the documents do not cover this - say so instead of guessing)" for g in gaps
    ]
    report = {
        "status": status,
        "purpose_check": purpose,
        "topics": topics,
        "summary": {"supported": len(ok), "partial": len(partial), "unsupported": len(bad), "ignored_not_in_brief": len(ignored)},
        "known_gaps": gaps,
        "message": {
            "ok": "All requested topics are supported by the documents.",
            "warn": "Some requested topics are weakly or not supported. Unsupported topics were removed from scope and will be refused, not guessed.",
            "blocked": "The documents cannot support this brief. No agent was generated because it would have to hallucinate.",
        }[status],
    }
    return report, narrowed


def groundability_markdown(rep: dict) -> str:
    rows = "\n".join(
        f"| {t['topic']} | {t['verdict']} | {t.get('coverage')} | {'; '.join(t.get('notes', [])) or ''} |" for t in rep["topics"]
    )
    return f"""# Groundability report - **{rep['status'].upper()}**

{rep['message']}

| Topic | Verdict | Lexical coverage | Notes |
|---|---|---|---|
{rows}

Known gaps: {', '.join(rep['known_gaps']) or 'none'}
"""
