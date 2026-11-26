"""Generic agent runtime + chat runner. Loads ANY definition produced by the factory (forgeai.agent/1)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .docs import Document, chunk_documents, load_folder
from .generator import render_system_prompt
from .llm import LLM, extract_json
from .embeddings import get_embedder
from .errors import ModelUnavailable
from .retrieval import HybridRetriever, Retriever
from .tools import run_tool


_SPLIT = re.compile(r"(?<=\?)\s+|\s+(?:and also|also,?|as well as|additionally,?)\s+", re.I)


def split_question(q: str) -> list[str]:
    """Split a compound question into its parts (each at least 4 words); returns [q] when it is a single question."""
    parts = [p.strip() for p in _SPLIT.split(q) if len(p.split()) >= 4]
    return parts if len(parts) >= 2 else [q]


MAX_DENSE_CHUNKS = 2500


class Agent:
    def __init__(self, defn: dict, llm: LLM, docs: list[Document] | None = None, base_dir: str | Path | None = None, embedder=None):
        self.defn, self.llm = defn, llm
        if docs is None:
            docs, _ = load_folder(Path(base_dir or ".") / defn["knowledge"]["documents_dir"])
        self.docs = docs
        self.doc_names = {d.name for d in docs}
        kn = defn["knowledge"]
        self.chunks = chunk_documents(docs, kn["chunking"]["size_words"], kn["chunking"]["overlap_words"])
        self.retrieval_note = ""
        self.retr = self._build_retriever(kn["retrieval"], embedder)
        self.system = render_system_prompt(defn)
        self.tools = defn.get("tools", [])

    def _build_retriever(self, r: dict, embedder=None):
        gl = r.get("glossary") or {}
        if r.get("method") != "hybrid":
            return Retriever(self.chunks, glossary=gl)
        if len(self.chunks) > MAX_DENSE_CHUNKS:  # embedding thousands of chunks takes many minutes: degrade, say so, keep working
            self.retrieval_note = f"corpus too large for dense retrieval ({len(self.chunks)} chunks > {MAX_DENSE_CHUNKS}); using BM25 only"
            return Retriever(self.chunks, glossary=gl)
        try:
            if embedder is None:
                embedder, note = get_embedder(r.get("embedder"), getattr(self.llm, "host", "http://localhost:11434"))
                self.retrieval_note = note
            return HybridRetriever(self.chunks, embedder, glossary=gl)
        except ModelUnavailable as e:  # embedding server died: degrade to BM25 rather than fail every question
            self.retrieval_note = f"hybrid retrieval unavailable ({e}); using BM25 only"
            return Retriever(self.chunks, glossary=gl)

    @classmethod
    def load(cls, definition_path: str | Path, llm: LLM, docs_dir: str | Path | None = None) -> "Agent":
        p = Path(definition_path)
        defn = json.loads(p.read_text(encoding="utf-8"))
        if docs_dir:
            docs, _ = load_folder(docs_dir)
            return cls(defn, llm, docs=docs)
        return cls(defn, llm, base_dir=p.parent)

    # -------------------------------------------------------------------------------
    def retrieve(self, question: str, history: list[dict] | None = None):
        r = self.defn["knowledge"]["retrieval"]
        q = question
        if history and len(question.split()) < 6:  # short follow-up: borrow the previous user turn
            prev = [h["q"] for h in history if h.get("q")]
            if prev:
                q = prev[-1] + " " + question
        parts = split_question(q)
        if len(parts) == 1:
            return self.retr.search(q, r["top_k"], r.get("min_score", 0.0), r.get("neighbors", 0))
        # compound question ("X? Also, Y?"): retrieve for each part and interleave, so every part gets its own passages
        per = [self.retr.search(p, r["top_k"], r.get("min_score", 0.0), r.get("neighbors", 0)) for p in parts]
        merged, seen = [], set()
        for rank in range(max(len(x) for x in per)):
            for lst in per:
                if rank < len(lst) and lst[rank][0].id not in seen:
                    seen.add(lst[rank][0].id)
                    merged.append(lst[rank])
        return merged[: min(8, max(r["top_k"], r["top_k"] * len(parts) // 2 + 1))]

    def answer(self, question: str, history: list[dict] | None = None) -> dict:
        sp = self.defn["system_prompt"]
        hits = self.retrieve(question, history)
        retrieved = [{"id": c.id, "doc": c.doc, "score": round(s, 2), "text": c.text} for c, s in hits]
        base = {"question": question, "retrieved": retrieved, "tool_calls": [], "violations": []}
        if not hits and not self.tools:
            return {**base, "answer": sp["refusal_message"], "refused": True, "sources": [], "evidence": [], "deterministic_refusal": True}

        ctx = "\n\n".join(f"[{c.doc}]\n{c.text}" for c, _ in hits) or "(no relevant excerpts were found)"
        msgs = [{"role": "system", "content": self.system}]
        for h in (history or [])[-3:]:
            msgs += [{"role": "user", "content": h["q"]}, {"role": "assistant", "content": h["a"]}]
        msgs.append({"role": "user", "content": f"CONTEXT:\n{ctx}\n\nQUESTION: {question}\n\nReply with the JSON object only."})

        temp = self.defn["model"].get("temperature", 0.1)
        parsed, raw = None, ""
        for _ in range(3):
            raw = self.llm.chat(msgs, task="agent_answer", json_mode=True, temperature=temp, max_tokens=500)
            try:
                parsed = extract_json(raw)
            except Exception:  # noqa: BLE001
                parsed = None
                break
            if isinstance(parsed, dict) and parsed.get("tool") in self.tools and len(base["tool_calls"]) < 2:
                res = run_tool(parsed["tool"], parsed.get("args", {}), {"documents": self.docs})
                base["tool_calls"].append({"tool": parsed["tool"], "args": parsed.get("args", {}), "result": res})
                msgs += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"TOOL RESULT ({parsed['tool']}): {json.dumps(res, ensure_ascii=False)[:1500]}\nNow reply with the final JSON object (answer/sources/evidence/refused)."},
                ]
                continue
            break
        return {**base, **self._normalise(parsed, raw, base["tool_calls"], retrieved)}

    def _normalise(self, parsed, raw: str, tool_calls: list, retrieved: list | None = None) -> dict:
        v: list[str] = []
        if not isinstance(parsed, dict) or "answer" not in parsed:
            return {"answer": (raw or "").strip()[:800], "refused": False, "sources": [], "evidence": [], "violations": ["unparseable_output"]}
        ans = parsed.get("answer")
        if isinstance(ans, (list, dict)):
            ans = json.dumps(ans, ensure_ascii=False)
        ans = str(ans or "").strip()
        ref = parsed.get("refused", False)
        refused = ref if isinstance(ref, bool) else str(ref).strip().lower() in ("true", "yes", "1")
        srcs = parsed.get("sources", [])
        srcs = [srcs] if isinstance(srcs, str) else srcs if isinstance(srcs, list) else []
        clean = []
        for s in srcs:
            s = str(s).split("#")[0].strip()
            if s in self.doc_names:
                clean.append(s)
            else:
                v.append(f"invented_source:{s}")
        ev = parsed.get("evidence", [])
        ev = [ev] if isinstance(ev, str) else ev if isinstance(ev, list) else []
        ev = [str(e) for e in ev if str(e).strip()]
        if not ans:
            v.append("empty_answer")
        if not refused and self.defn["output_contract"].get("citation_required") and not clean and not tool_calls:
            v.append("missing_citation")  # the MODEL broke the contract; recorded and counted, then repaired by code below
            clean = self._attribute(ans, retrieved or [])
            if clean:
                v.append("sources_auto_attributed")
        return {"answer": ans, "refused": refused, "sources": sorted(set(clean)), "evidence": ev, "violations": v}


def _attr_tokens(text: str) -> set[str]:
    from .retrieval import tokenize

    return {t for t in tokenize(text) if len(t) > 2}


def _attribute(self, answer: str, retrieved: list) -> list[str]:
    """Code-side citation repair: cite the retrieved document(s) whose text best covers the answer's content words.
    Small models often leave `sources` empty; the contract is enforced by the runtime, and every repair is flagged."""
    at = _attr_tokens(answer)
    if not at or not retrieved:
        return []
    scored: dict[str, float] = {}
    for h in retrieved:
        cov = len(at & _attr_tokens(h["text"])) / len(at)
        scored[h["doc"]] = max(scored.get(h["doc"], 0.0), cov)
    best = max(scored.values())
    return sorted(d for d, c in scored.items() if c >= max(0.35, 0.8 * best))


Agent._attribute = _attribute  # type: ignore[attr-defined]


def chat(definition_path: str, model: str, docs_dir: str | None = None, verbose: bool = False, host: str = "http://localhost:11434") -> None:
    llm = LLM(model=model, host=host)
    agent = Agent.load(definition_path, llm, docs_dir)
    print(f"\n=== {agent.defn['name']} ===  (tools: {', '.join(agent.tools) or 'none'};  type 'exit' to quit)\n")
    hist: list[dict] = []
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if q.lower() in ("exit", "quit", ""):
            if q:
                return
            continue
        r = agent.answer(q, hist)
        tag = " [declined]" if r["refused"] else ""
        print(f"\nagent>{tag} {r['answer']}")
        if r["sources"]:
            print(f"        sources: {', '.join(r['sources'])}")
        for tc in r["tool_calls"]:
            print(f"        tool: {tc['tool']}({json.dumps(tc['args'])}) -> {json.dumps(tc['result'])[:160]}")
        if r["violations"] and verbose:
            print(f"        contract violations: {r['violations']}")
        if verbose:
            for h in r["retrieved"]:
                print(f"        retrieved {h['id']} score={h['score']}")
        print()
        hist.append({"q": q, "a": r["answer"]})
