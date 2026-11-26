"""Stage 6 - Critic/optimizer loop (max 3 iterations).

run evals -> classify EVERY failure (retrieval_miss | prompt_defect | hallucination | wrong_refusal)
          -> apply a TARGETED patch (one config value / one prompt rule / one glossary entry - never a regeneration)
          -> re-run. A patch set is accepted only if the dev pass-rate strictly improves, otherwise it is reverted.
The hold-out split is scored but never used for decisions, so the report can show how much the loop overfitted dev.
"""
from __future__ import annotations

import copy
import hashlib
import json
import threading
from collections import Counter, defaultdict
from pathlib import Path

from .docs import Document
from .generator import definition_hash
from .judge import judge_case, scorecard
from .errors import BudgetExceeded, ModelOutputError, ModelUnavailable, SchemaError
from .llm import LLM
from .parallel import pmap
from .prompts import render
from .schemas import GlossaryOut, RuleOut, validate_definition
from .retrieval import stem, tokenize
from .runner import Agent
from .textutil import contains, jaccard

CATEGORIES = ["retrieval_miss", "prompt_defect", "hallucination", "wrong_refusal"]
KIND_DESC = {
    "hallucination": "the assistant answered with details that are not in the context, or answered a question it should have declined",
    "wrong_refusal": "the assistant refused although the context contained the answer",
    "prompt_defect": "the assistant had the right context but omitted required facts, ignored scope, or skipped citations",
}
FALLBACK_RULES = {
    "hallucination": "Before answering, verify that every number, name and date in your answer appears in the CONTEXT; if the specific detail asked for is missing, refuse instead of approximating.",
    "wrong_refusal": "If the CONTEXT contains information relevant to the question, answer from it; refuse only when the context clearly lacks the answer.",
    "prompt_defect": "Include every specific number, date, condition and exception from the relevant excerpts, and always cite the source file names.",
}
MAX_RULES = 14


# ---------------------------------------------------------------------------------- evaluation
class EvalCache:
    """Append-only JSONL cache of (definition, case) -> agent answer + verdict, so an interrupted run resumes mid-evaluation."""

    def __init__(self, path: Path | None):
        self.path, self.data = path, {}
        self._lock = threading.Lock()
        if path and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec
                except (BudgetExceeded, ModelUnavailable):  # never swallow a budget stop or an outage
                    raise
                except Exception:  # noqa: BLE001
                    continue  # a torn last line from a crash is simply ignored

    @staticmethod
    def key(defn_hash: str, case: dict, judge_model: str) -> str:
        return hashlib.sha1(f"{defn_hash}|{case['id']}|{case['question']}|{judge_model}".encode()).hexdigest()

    def put(self, key: str, raw: dict, judged: dict) -> None:
        rec = {"key": key, "raw": raw, "judged": judged}
        self.data[key] = rec
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _error_answer(question: str, msg: str) -> dict:
    return {"question": question, "answer": "", "refused": False, "sources": [], "evidence": [], "retrieved": [], "tool_calls": [],
            "violations": [f"agent_error: {msg[:160]}"]}


def evaluate(defn: dict, docs: list[Document], cases: list[dict], agent_llm: LLM, judge_llm: LLM | None, cache: EvalCache | None = None) -> tuple[Agent, list[dict], dict, dict]:
    agent = Agent(defn, agent_llm, docs=docs)
    dh = definition_hash(defn)
    jm = judge_llm.model if judge_llm else "none"
    def one(c: dict) -> tuple[dict, dict]:
        key = EvalCache.key(dh, c, jm)
        if cache and key in cache.data:
            return cache.data[key]["raw"], cache.data[key]["judged"]
        try:
            res = agent.answer(c["question"])
        except ModelOutputError as e:  # one bad generation must not abort the whole evaluation
            res = _error_answer(c["question"], str(e))
        try:
            j = judge_case(c, res, judge_llm)
        except ModelOutputError:
            j = judge_case(c, res, None)  # judge model failed: fall back to deterministic checks only
            j.setdefault("notes", []).append("LLM judge unavailable for this case; deterministic checks only")
        j["sources"], j["evidence"] = res.get("sources", []), res.get("evidence", [])
        j["retrieved"] = [h["id"] for h in res.get("retrieved", [])]
        j["tool_calls"] = [{"tool": t["tool"], "args": t["args"]} for t in res.get("tool_calls", [])]
        if cache:
            cache.put(key, res, j)
        return res, j

    pairs = pmap(one, cases)  # independent cases, answered and judged in parallel; order is kept
    results = [j for _, j in pairs]
    raw = {c["id"]: res for c, (res, _) in zip(cases, pairs)}
    return agent, results, scorecard(results), raw


# ---------------------------------------------------------------------------------- classification
def classify(case: dict, j: dict) -> dict:
    notes = " ".join(j.get("notes", []))
    if any(str(v).startswith("agent_error") for v in j.get("violations", [])):
        return {"id": case["id"], "category": "system_error", "why": "the model call failed after retries; not a quality defect", "question": case["question"], "answer": "", "notes": j.get("violations", [])}
    if case["should_refuse"]:
        if "leaked" in notes or "ignore" in case["question"].lower():
            cat, why = "prompt_defect", "instruction-override / system-prompt leak attempt was not resisted"
        else:
            cat, why = "hallucination", "answered a question the documents cannot support instead of refusing"
    elif j["refused"]:
        if j["retrieval_hit"] < 0.99:
            cat, why = "retrieval_miss", "refused because the needed passage was not in the retrieved context"
        else:
            cat, why = "wrong_refusal", "the needed passage WAS in context but the agent refused"
    elif j["retrieval_hit"] < 0.99:
        cat, why = "retrieval_miss", "answer lacks the expected facts and the needed passage was not retrieved"
    elif not j["grounded"] and (j["grounding"]["unsupported_numbers"] or j["grounding"]["lexical_support"] < 0.5):
        cat, why = "hallucination", "answer contains details not present in the retrieved context"
    else:
        cat, why = "prompt_defect", "passage was retrieved and answer is grounded, but required facts/citation are missing"
    return {"id": case["id"], "category": cat, "why": why, "question": case["question"], "answer": j["answer"][:300], "notes": j.get("notes", [])}


# ---------------------------------------------------------------------------------- fixes
def _failure_text(items: list[tuple[dict, dict]], k: int = 3) -> str:
    out = []
    for c, f in items[:k]:
        exp = "; ".join(c["expected_facts"]) if c["expected_facts"] else "(should have refused)"
        out.append(f"Q: {c['question']}\nExpected: {exp}\nAgent said: {f['answer'][:220]}\nProblem: {f['why']}")
    return "\n\n".join(out)


def _rule_fix(cat: str, items: list[tuple[dict, dict]], defn: dict, llm: LLM) -> dict | None:
    rules = defn["system_prompt"]["rules"]
    if len(rules) >= MAX_RULES:
        return None
    sysm, user = render(
        "fix_prompt_rule", kind_desc=KIND_DESC[cat], rules="\n".join(f"- {r['text']}" for r in rules), failures=_failure_text(items)
    )
    rule, origin = "", "llm"
    try:
        rule = str(llm.json(sysm, user, task="fix_prompt_rule", schema=RuleOut, max_tokens=150).get("rule", "")).strip()
    except (BudgetExceeded, ModelUnavailable):  # never swallow a budget stop or an outage
        raise
    except Exception:  # noqa: BLE001
        pass
    nw = len(rule.split())
    if not (6 <= nw <= 45) or any(jaccard(rule, r["text"]) > 0.6 for r in rules):
        rule, origin = FALLBACK_RULES[cat], "fallback-template"
        if any(jaccard(rule, r["text"]) > 0.6 for r in rules):
            return None
    rid = f"fix-{cat}-{len(rules)}"
    rules.append({"id": rid, "text": rule, "origin": origin})
    return {"type": "add_prompt_rule", "target": f"system_prompt.rules[{rid}]", "before": None, "after": rule, "origin": origin}


def _retrieval_fixes(items: list[tuple[dict, dict]], defn: dict, agent: Agent, docs: list[Document], llm: LLM, cases: dict) -> list[dict]:
    kn = defn["knowledge"]
    topk = kn["retrieval"]["top_k"]
    low_rank, no_overlap, split = [], [], []
    for c, f in items:
        facts = c["expected_facts"] or []
        missing = [x for x in facts if not any(contains(h["text"], x) for h in agent.answer_cache.get(c["id"], []))] or facts
        for fact in missing[:1]:
            rank = agent.retr.rank_of(c["question"], lambda t, fact=fact: contains(t, fact))
            in_doc = any(contains(d.text, fact) for d in docs)
            if rank is not None and rank > topk:
                low_rank.append((c, rank))
            elif rank is None and in_doc and not any(contains(ch.text, fact) for ch in agent.chunks):
                split.append(c)
            else:
                no_overlap.append((c, fact))
    changes: list[dict] = []
    if low_rank and kn["retrieval"]["top_k"] < 8:
        new = min(8, max(topk + 2, min(max(r for _, r in low_rank), 8)))
        kn["retrieval"]["top_k"] = new
        changes.append({"type": "set_param", "target": "knowledge.retrieval.top_k", "before": topk, "after": new,
                        "why": f"evidence ranked {sorted(r for _, r in low_rank)} but only top {topk} are passed to the model", "cases": [c['id'] for c, _ in low_rank]})
    if split:
        ch = kn["chunking"]
        before = dict(ch)
        ch["size_words"] = min(260, ch["size_words"] + 40)
        ch["overlap_words"] = min(70, ch["overlap_words"] + 15)
        changes.append({"type": "set_param", "target": "knowledge.chunking", "before": before, "after": dict(ch),
                        "why": "expected fact exists in the document but is cut by a chunk boundary", "cases": [c["id"] for c in split]})
    if no_overlap:
        gl = kn["retrieval"].setdefault("glossary", {})
        added, used = {}, []
        for c, fact in no_overlap[:5]:
            passage = next((ch.text for ch in agent.chunks if contains(ch.text, fact)), None) or next(
                (d.text[:900] for d in docs if contains(d.text, fact)), None)
            if not passage:
                continue
            sysm, user = render("fix_glossary", question=c["question"], passage=passage[:1200])
            try:
                pairs = llm.json(sysm, user, task="fix_glossary", schema=GlossaryOut, max_tokens=200).get("pairs", [])
            except (BudgetExceeded, ModelUnavailable):  # never swallow a budget stop or an outage
                raise
            except Exception:  # noqa: BLE001
                pairs = []
            qtoks = set(tokenize(c["question"]))
            ptoks = set(tokenize(passage))
            for p in pairs if isinstance(pairs, list) else []:
                qw, dw = str(p.get("question_word", "")).lower().strip(), str(p.get("document_word", "")).lower().strip()
                if qw and dw and stem(qw) in qtoks and stem(dw) in ptoks and dw not in gl.get(qw, []):  # both words verified
                    gl.setdefault(qw, []).append(dw)
                    added.setdefault(qw, []).append(dw)
                    used.append(c["id"])
        if added:
            changes.append({"type": "add_glossary", "target": "knowledge.retrieval.glossary", "before": None, "after": added,
                            "why": "question wording shares no search terms with the document; verified synonym pairs added to query expansion", "cases": sorted(set(used))})
        elif not changes:
            kn["retrieval"]["neighbors"] = max(kn["retrieval"].get("neighbors", 0), 1)
            changes.append({"type": "set_param", "target": "knowledge.retrieval.neighbors", "before": 0, "after": 1,
                            "why": "no verified synonym pair found; widening context with adjacent chunks instead", "cases": [c["id"] for c, _ in no_overlap]})
    return changes


def apply_fixes(defn: dict, failures: list[dict], agent: Agent, docs: list[Document], cases: dict, llm: LLM, skip: set[str], only: str | None = None) -> tuple[dict, list[dict]]:
    new = copy.deepcopy(defn)
    new["version"] = int(defn.get("version", 0)) + 1
    by_cat: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for f in failures:
        by_cat[f["category"]].append((cases[f["id"]], f))
    changes: list[dict] = []
    for cat in sorted(by_cat, key=lambda k: -len(by_cat[k])):
        if cat == "system_error" or cat in skip or (only and cat != only):
            continue
        items = by_cat[cat]
        if cat == "retrieval_miss":
            for ch in _retrieval_fixes(items, new, agent, docs, llm, cases):
                changes.append({**ch, "category": cat, "cases": ch.get("cases") or [c["id"] for c, _ in items]})
        else:
            ch = _rule_fix(cat, items, new, llm)
            if ch:
                changes.append({**ch, "category": cat, "why": f"{len(items)} case(s): {KIND_DESC[cat]}", "cases": [c["id"] for c, _ in items]})
                if cat == "hallucination" and new["model"].get("temperature", 0.1) > 0:
                    changes.append({"type": "set_param", "target": "model.temperature", "before": new["model"]["temperature"], "after": 0.0,
                                    "why": "reduce sampling randomness to curb invented details", "category": cat, "cases": [c["id"] for c, _ in items]})
                    new["model"]["temperature"] = 0.0
    try:
        validate_definition(new)
    except SchemaError:  # a patch must never yield an invalid agent; drop all changes of this attempt
        return defn, []
    return new, changes


# ---------------------------------------------------------------------------------- loop
def optimise(
    defn0: dict, docs: list[Document], cases: list[dict], agent_llm: LLM, judge_llm: LLM | None, out_dir: Path,
    max_iters: int = 3, target: float = 0.9, log=print,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    case_by_id = {c["id"]: c for c in cases}
    history: list[dict] = []
    stop_reason = f"reached max_iterations={max_iters}"

    cache = EvalCache(out_dir / "eval_cache.jsonl")

    def run(defn, label):
        agent, results, sc, raw = evaluate(defn, docs, cases, agent_llm, judge_llm, cache)
        agent.answer_cache = {k: v.get("retrieved", []) for k, v in raw.items()}
        (out_dir / f"{label}_results.json").write_text(json.dumps({"definition_hash": definition_hash(defn), "scorecard": sc, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
        return agent, results, sc

    log(f"[optimizer] baseline evaluation ({len(cases)} cases)")
    cur_defn = defn0
    cur_agent, cur_res, cur_sc = run(cur_defn, "iter_0")
    baseline = {"iteration": 0, "definition_hash": definition_hash(cur_defn), "scorecard": cur_sc}
    skip: set[str] = set()
    try:
        for it in range(1, max_iters + 1):
            dev_fail = [r for r in cur_res if r["split"] == "dev" and not r["passed"]]
            dev_rate = cur_sc["dev"]["pass_rate"] or 0
            if not dev_fail:
                stop_reason = "no failures left on the dev split"
                break
            if dev_rate >= target:
                stop_reason = f"dev pass-rate {dev_rate:.0%} >= target {target:.0%}"
                break
            classified = [classify(case_by_id[r["id"]], r) for r in dev_fail]
            counts_before = dict(Counter(c["category"] for c in classified))
            log(f"[optimizer] iteration {it}: dev pass {dev_rate:.0%}; failures {counts_before}")
            new_defn, changes = apply_fixes(cur_defn, classified, cur_agent, docs, case_by_id, agent_llm, skip)
            if not changes:
                stop_reason = "no applicable targeted fix remains"
                break
            new_agent, new_res, new_sc = run(new_defn, f"iter_{it}")
            delta = (new_sc["dev"]["pass_rate"] or 0) - dev_rate
            before_pass = {r["id"] for r in cur_res if r["passed"]}
            after_pass = {r["id"] for r in new_res if r["passed"]}
            targeted = sorted({cid for ch in changes for cid in ch.get("cases", [])})
            accepted = delta > 1e-9
            entry = {
                "iteration": it, "accepted": accepted,
                "score_before": {"dev": cur_sc["dev"]["pass_rate"], "holdout": cur_sc["holdout"]["pass_rate"], "overall": cur_sc["overall"]["pass_rate"]},
                "score_after": {"dev": new_sc["dev"]["pass_rate"], "holdout": new_sc["holdout"]["pass_rate"], "overall": new_sc["overall"]["pass_rate"]},
                "failures_classified": classified, "failure_counts_before": counts_before,
                "changes": changes, "failures_targeted": targeted,
                "fixed": sorted(after_pass - before_pass), "regressed": sorted(before_pass - after_pass),
                "definition_hash_before": definition_hash(cur_defn), "definition_hash_after": definition_hash(new_defn),
            }
            if accepted:
                log(f"[optimizer]   accepted: dev {dev_rate:.0%} -> {new_sc['dev']['pass_rate']:.0%}; fixed {entry['fixed']}")
                cur_defn, cur_agent, cur_res, cur_sc = new_defn, new_agent, new_res, new_sc
                entry["decision"] = "accepted: dev pass-rate improved"
            else:
                tried = {ch["category"] for ch in changes}
                entry["decision"] = f"REVERTED: dev pass-rate {dev_rate:.0%} -> {new_sc['dev']['pass_rate']:.0%} (no strict improvement)"
                log(f"[optimizer]   {entry['decision']}")
                entry["ablation_trials"] = []
                if len(tried) > 1:  # the combined patch regressed: retry each failure category on its own
                    for cat in sorted(tried, key=lambda k: -counts_before.get(k, 0)):
                        sub_defn, sub_changes = apply_fixes(cur_defn, classified, cur_agent, docs, case_by_id, agent_llm, skip, only=cat)
                        if not sub_changes:
                            continue
                        sub_agent, sub_res, sub_sc = run(sub_defn, f"iter_{it}_only_{cat}")
                        sub_delta = (sub_sc["dev"]["pass_rate"] or 0) - dev_rate
                        entry["ablation_trials"].append({"category": cat, "changes": sub_changes, "dev_after": sub_sc["dev"]["pass_rate"], "holdout_after": sub_sc["holdout"]["pass_rate"], "accepted": sub_delta > 1e-9})
                        log(f"[optimizer]   ablation '{cat}' alone: dev {dev_rate:.0%} -> {sub_sc['dev']['pass_rate']:.0%}")
                        if sub_delta > 1e-9:
                            sub_pass = {r["id"] for r in sub_res if r["passed"]}
                            entry.update({
                                "accepted": True, "changes": sub_changes, "failures_targeted": sorted({c for ch in sub_changes for c in ch.get("cases", [])}),
                                "score_after": {"dev": sub_sc["dev"]["pass_rate"], "holdout": sub_sc["holdout"]["pass_rate"], "overall": sub_sc["overall"]["pass_rate"]},
                                "fixed": sorted(sub_pass - before_pass), "regressed": sorted(before_pass - sub_pass),
                                "definition_hash_after": definition_hash(sub_defn),
                                "decision": f"accepted after ablation: the combined patch regressed, but the '{cat}' fix alone improved dev {dev_rate:.0%} -> {sub_sc['dev']['pass_rate']:.0%}",
                            })
                            cur_defn, cur_agent, cur_res, cur_sc = sub_defn, sub_agent, sub_res, sub_sc
                            accepted = True
                            break
                if not accepted:
                    skip |= tried
            history.append(entry)
            (out_dir.parent / "iteration_log.json").write_text(json.dumps({"baseline": baseline, "iterations": history}, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            stop_reason = f"reached max_iterations={max_iters}"
    except BudgetExceeded as e:
        stop_reason = f"budget stop: {e}"
    final_failures = [classify(case_by_id[r["id"]], r) for r in cur_res if not r["passed"]]
    return {
        "definition": cur_defn, "baseline": baseline, "history": history, "stop_reason": stop_reason,
        "final_results": cur_res, "final_scorecard": cur_sc, "final_failures": final_failures,
    }
