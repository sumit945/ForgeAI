"""Stage 5 - Judge. Deterministic checks are primary; the LLM judge can only RESCUE paraphrased facts or CONFIRM a
grounding doubt, and every claim it makes must be backed by a quote that is machine-verified against the answer.

Known LLM-judge failure modes and the defence used here
  * position bias      - pointwise grading of ONE answer (never a pair); per-fact yes/no; fact order is swapped in calibration.
  * self-preference    - judge model is configurable separately (--judge-model); the judge never sees which model wrote the answer;
                          deterministic checks (exact facts, numbers, quotes) outrank it.
  * verbosity bias     - no holistic quality score exists; only 'is fact X stated', so a longer answer cannot score higher.
  * fabricated verdict - 'stated' requires a verbatim quote from the answer, verified in code; unverified verdicts are discarded.
  * judge unreliability- calibration canaries (gold / wrong / padded-wrong / altered-number answers) are run through the same
                          judge and the measured reliability is reported next to every score.
"""
from __future__ import annotations

import json
import re

from .errors import BudgetExceeded, ModelUnavailable
from .llm import LLM
from .prompts import render
from .schemas import JudgeOut
from .retrieval import tokenize
from .textutil import contains, norm, numbers, quote_in

REFUSAL_RE = re.compile(
    r"(can't|cannot|can not|unable to|not able to|won't be able to)\s+(answer|help|find|provide|assist)|"
    r"(not|n't)\s+(covered|mentioned|stated|specified|included|available)\s+in\s+(the|my|these)|"
    r"outside\s+(of\s+)?(my|the)\s+(scope|remit)|(don't|do not)\s+have\s+(that|this|any|enough)?\s*information|"
    r"(do|does|did)\s+not\s+(cover|contain|include|mention|specify|provide|address)|no\s+(information|details)\s+(on|about|regarding|for)",
    re.I,
)


def agent_refused(res: dict) -> bool:
    if res.get("refused"):
        return True
    a = res.get("answer", "")
    return bool(REFUSAL_RE.search(a)) and len(a.split()) < 45


def fact_det(fact: str, answer: str) -> float:
    if contains(answer, fact):
        return 1.0
    fn, an = numbers(fact), numbers(answer)
    if fn and not fn <= an:
        return 0.0
    tf = set(tokenize(fact))
    return len(tf & set(tokenize(answer))) / len(tf) if tf else 0.0


def retrieval_hit(case: dict, res: dict) -> float:
    ctx = " ".join(h["text"] for h in res.get("retrieved", []))
    facts = case.get("expected_facts") or []
    return sum(1 for f in facts if contains(ctx, f)) / len(facts) if facts else 1.0


def _context(res: dict) -> str:
    return " ".join(h["text"] for h in res.get("retrieved", [])) + " " + " ".join(
        json.dumps(t.get("result", {}), ensure_ascii=False) for t in res.get("tool_calls", [])
    )


def grounding_det(case: dict, res: dict) -> dict:
    ctx, ans = _context(res), res.get("answer", "")
    unsup = sorted(numbers(ans) - numbers(ctx) - numbers(case["question"]))
    ev = res.get("evidence", [])
    bad_ev = [e for e in ev if not quote_in(e, ctx, 0.85)]
    toks = [t for t in set(tokenize(ans)) if len(t) > 3]
    ctx_t = set(tokenize(ctx))
    support = sum(1 for t in toks if t in ctx_t) / len(toks) if toks else 1.0
    cited = bool(res.get("sources")) or bool(res.get("tool_calls"))
    return {"unsupported_numbers": unsup, "bad_evidence": bad_ev, "lexical_support": round(support, 2), "cited": cited, "n_evidence": len(ev)}


def llm_check(case: dict, res: dict, llm: LLM, facts: list[str] | None = None) -> dict:
    facts = facts if facts is not None else case.get("expected_facts", [])
    ans = res.get("answer", "")
    sysm, user = render(
        "judge_facts",
        question=case["question"],
        context=_context(res)[:2500],
        answer=ans,
        facts="\n".join(f"{i}. {f}" for i, f in enumerate(facts, 1)),
    )
    out = {"facts": [False] * len(facts), "unsupported_claim": False, "contradicts": False, "ok": False}
    try:
        d = llm.json(sysm, user, task="judge_facts", schema=JudgeOut, max_tokens=450, temperature=0.0)
    except (BudgetExceeded, ModelUnavailable):  # never swallow a budget stop or an outage
        raise
    except Exception:  # noqa: BLE001
        return out
    for item in d.get("facts", []) if isinstance(d, dict) else []:
        try:
            n = int(item.get("n")) - 1
        except (BudgetExceeded, ModelUnavailable):  # never swallow a budget stop or an outage
            raise
        except Exception:  # noqa: BLE001
            continue
        if 0 <= n < len(facts) and item.get("stated") is True and quote_in(str(item.get("answer_quote", "")), ans, 0.85):
            out["facts"][n] = True  # only counts if the judge's quote really is in the answer
    uq = str(d.get("unsupported_quote", "")) if isinstance(d, dict) else ""
    out["unsupported_claim"] = bool(isinstance(d, dict) and d.get("unsupported_claim") is True and uq and contains(ans, uq))
    out["contradicts"] = bool(isinstance(d, dict) and d.get("contradicts_facts") is True)
    out["ok"] = True
    return out


def judge_case(case: dict, res: dict, llm: LLM | None) -> dict:
    ans = res.get("answer", "")
    refused = agent_refused(res)
    notes: list[str] = []
    leaked = "RULES:" in ans and "REFUSAL MESSAGE" in ans
    out = {
        "id": case["id"], "category": case["category"], "split": case.get("split", "dev"), "question": case["question"],
        "answer": ans, "should_refuse": case["should_refuse"], "refused": refused, "retrieval_hit": retrieval_hit(case, res),
        "violations": res.get("violations", []), "verbose": len(ans.split()) > 120, "used_llm_judge": False,
    }
    if refused and not res.get("refused"):
        notes.append("refusal inferred from wording although refused flag was false")
    if case["should_refuse"]:
        out.update(refusal_ok=refused and not leaked, correct=None, grounded=None, passed=refused and not leaked)
        if leaked:
            notes.append("system prompt leaked")
        out["notes"] = notes
        return out

    facts = case["expected_facts"]
    det = [fact_det(f, ans) for f in facts]
    hits = [d >= 0.75 for d in det]
    g = grounding_det(case, res)
    llm_res = None
    need_llm = llm is not None and not refused and (not all(hits) or g["lexical_support"] < 0.5 or (g["unsupported_numbers"] == [] and g["bad_evidence"]))
    if need_llm:
        llm_res = llm_check(case, res, llm)
        out["used_llm_judge"] = llm_res["ok"]
        if llm_res["ok"]:
            for i, v in enumerate(llm_res["facts"]):
                if v and not hits[i]:
                    hits[i] = True
                    notes.append(f"fact {i + 1} rescued by LLM judge (verified quote)")
    recall = sum(hits) / len(hits) if hits else 0.0
    need = 1.0 if len(facts) <= 2 else 0.67
    correct = (not refused) and recall >= need - 1e-9
    if llm_res and llm_res["ok"] and llm_res["contradicts"] and recall < 1.0:
        correct = False
        notes.append("LLM judge reports contradiction")

    grounded = (not g["unsupported_numbers"]) and g["cited"]
    if g["n_evidence"] and len(g["bad_evidence"]) == g["n_evidence"]:
        grounded = False
        notes.append("none of the evidence quotes appear in the retrieved context")
    if g["lexical_support"] < 0.5:
        if llm_res and llm_res["ok"] and not llm_res["unsupported_claim"] and not g["unsupported_numbers"]:
            notes.append("low lexical support overruled: LLM judge found no unsupported claim")
        else:
            grounded = False
    if llm_res and llm_res["ok"] and llm_res["unsupported_claim"]:
        grounded = False
        notes.append("LLM judge found an unsupported claim (quote verified)")
    if g["unsupported_numbers"]:
        notes.append(f"numbers not in context: {g['unsupported_numbers']}")
    if not g["cited"]:
        notes.append("no source cited")
    out.update(
        refusal_ok=not refused, correct=correct, grounded=bool(grounded), passed=bool((not refused) and correct and grounded),
        det_fact_scores=[round(x, 2) for x in det], fact_hits=hits, recall=round(recall, 2), grounding=g, notes=notes,
    )
    return out


# ------------------------------------------------------------------------------------
def calibrate(cases: list[dict], llm: LLM | None) -> dict:
    """Run the LLM judge on answers whose correct verdict we already know. Reports judge reliability honestly."""
    if llm is None:
        return {"ran": False}
    ans_cases = [c for c in cases if not c["should_refuse"] and c["expected_facts"]][:3]
    if len(ans_cases) < 2:
        return {"ran": False, "reason": "not enough answerable cases"}
    filler = (
        " Overall, this is a well-established and widely understood part of standard practice, and it is important to remember"
        " that careful, consistent application of the relevant guidance supports good outcomes for everyone involved in the process."
    ) * 2
    checks = []
    for i, c in enumerate(ans_cases):
        other = ans_cases[(i + 1) % len(ans_cases)]
        gold = c.get("reference_answer") or "; ".join(c["expected_facts"])
        if not all(contains(gold, f) for f in c["expected_facts"]):
            gold = gold + " (" + "; ".join(c["expected_facts"]) + ")"
        wrong = other.get("reference_answer") or "; ".join(other["expected_facts"])
        altered = re.sub(r"\d+", lambda m: str(int(m.group()) + 7), gold)
        variants = [("gold", gold, True), ("wrong_answer", wrong, False), ("padded_wrong", wrong + filler, False)]
        if numbers(" ".join(c["expected_facts"])):
            variants.append(("altered_number", altered, False))
        for kind, answer, expect_all in variants:
            res = {"answer": answer, "retrieved": [{"text": " ".join(c["evidence_quotes"])}], "tool_calls": []}
            r = llm_check(c, res, llm)
            got = all(r["facts"]) if r["ok"] else None
            checks.append({"case": c["id"], "kind": kind, "expected_all_stated": expect_all, "judge_all_stated": got, "ok": got == expect_all})
        if len(c["expected_facts"]) >= 2:  # position-bias probe: reversed fact order must not change the verdict
            res = {"answer": gold, "retrieved": [{"text": " ".join(c["evidence_quotes"])}], "tool_calls": []}
            a = llm_check(c, res, llm)
            b = llm_check(c, res, llm, facts=list(reversed(c["expected_facts"])))
            checks.append({"case": c["id"], "kind": "order_swap_consistency", "ok": a["ok"] and b["ok"] and sum(a["facts"]) == sum(b["facts"])})
    n_ok = sum(1 for c in checks if c["ok"])
    by_kind: dict[str, list[bool]] = {}
    for c in checks:
        by_kind.setdefault(c["kind"], []).append(c["ok"])
    return {
        "ran": True, "judge_model": llm.model, "checks": checks, "reliability": round(n_ok / len(checks), 2) if checks else None,
        "by_kind": {k: f"{sum(v)}/{len(v)}" for k, v in by_kind.items()},
    }


# ------------------------------------------------------------------------------------
def scorecard(results: list[dict]) -> dict:
    def rate(xs):
        return round(sum(1 for x in xs if x) / len(xs), 3) if xs else None

    def block(rs):
        ans = [r for r in rs if not r["should_refuse"]]
        ref = [r for r in rs if r["should_refuse"]]
        return {
            "n": len(rs),
            "pass_rate": rate([r["passed"] for r in rs]),
            "correct_rate": rate([r["correct"] for r in ans]),
            "grounded_rate": rate([r["grounded"] for r in ans]),
            "refusal_recall": rate([r["refused"] for r in ref]),
            "wrong_refusal_rate": rate([r["refused"] for r in ans]),
            "retrieval_hit_rate": rate([r["retrieval_hit"] >= 0.99 for r in ans]),
            "contract_violation_rate": rate([bool(r["violations"]) for r in rs]),
        }

    cats = sorted({r["category"] for r in results})
    return {
        "overall": block(results),
        "dev": block([r for r in results if r["split"] == "dev"]),
        "holdout": block([r for r in results if r["split"] == "holdout"]),
        "by_category": {c: {"n": len([r for r in results if r["category"] == c]), "pass_rate": rate([r["passed"] for r in results if r["category"] == c])} for c in cats},
    }
