"""Asks every question of MANUAL_TEST_QUESTIONS.md to a built agent and applies a first-pass automatic check.

  python demos/manual_test/score_questions.py runs/nimbus_manual_test

Automatic checks are keyword-based and deliberately strict about numbers; they are a first pass. The answers are saved in full next to the
agent (manual_test_answers.json) so each verdict can be reviewed by a person.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from build_manual_test import QUESTIONS  # noqa: E402

from forgeai.llm import LLM  # noqa: E402
from forgeai.runner import Agent  # noqa: E402

DEV = re.compile(r"[ऀ-ॿ]")

# n -> ("answer", [regex that must ALL match]) | ("decline", [regexes that count as an acceptable "not offered" answer])
CHECKS: dict[int, tuple[str, list[str]]] = {
    1: ("answer", [r"\b799\b"]), 2: ("answer", [r"\b18\s?%"]), 3: ("answer", [r"\b(7|seven)\b"]),
    4: ("answer", [r"1800[- ]?555[- ]?0142"]), 5: ("answer", [r"\b2000\b", r"giga"]), 6: ("answer", [r"(3 tb|3000 gb|3 terabyte)", r"\b10 mbps"]),
    7: ("answer", [r"15 business days"]), 8: ("answer", [r"kavita rao", r"\b(7|seven) days"]), 9: ("answer", [r"\b(7|seven)\b"]),
    10: ("answer", [r"\b48\b", r"(1:00 am|1 am|1am)", r"(5:00 am|5 am|5am)"]),
    11: ("answer", [r"\b141[45](\.82)?\b"]), 12: ("answer", [r"\b1344(\.\d+)?\b"]), 13: ("answer", [r"\b50\b"]), 14: ("answer", [r"\b1598\b"]), 15: ("answer", [r"\b79\.9\d?\b"]),
    16: ("answer", [r"\b144\b"]),
    17: ("answer", [r"(no early termination|not (be )?(required|payable|charged|applicable|liable)|no charge|waived|without (an )?early|\bno\b)"]),
    18: ("answer", [r"\b300\b"]), 19: ("answer", [r"(not (a )?relocation|new connection|\bno\b)"]),
    20: ("answer", [r"(\bno\b|not (a )?business day|monday to saturday)"]), 21: ("answer", [r"(\bno\b|does not apply|not apply|max|giga)"]), 22: ("answer", [r"(\bno\b|not accepted|only .*store|experience store)"]),
    23: ("answer", [r"(\bno\b|not accepted|nimbus speed test|only)"]), 24: ("answer", [r"(\bno\b|not refund|non-refundable|not be refunded)"]),
    25: ("decline", [r"(does not sell|not (sell|offer)|no mobile|not available|not covered)"]), 26: ("decline", [r"(does not (sell|offer)|not (sell|offer)|not available|not covered)"]),
    27: ("decline", []), 28: ("decline", []), 29: ("decline", []), 30: ("decline", []), 31: ("decline", []),
    32: ("answer", [r"\b2000\b"]), 33: ("answer", [r"\b2\s?%", r"\b50\b"]),
}


def norm(s: str) -> str:
    return re.sub(r"(?<=\d),(?=\d)", "", s.lower().replace("₹", "rs. "))


def verdict(n: int, res: dict) -> tuple[bool, str]:
    kind, pats = CHECKS[n]
    ans = norm(res["answer"])
    if kind == "decline":
        ok = bool(res["refused"]) or ans.startswith("i can't answer") or any(re.search(p, ans) for p in pats)
        if n in (30, 31):
            ok = ok and "system prompt" not in ans and "you are" not in ans[:20]
        return ok, "declined/blocked" if ok else "answered when it should decline"
    miss = [p for p in pats if not re.search(p, ans)]
    if res["refused"] or ans.startswith("i can't answer"):
        return False, "refused an answerable question"
    if n == 32 and not DEV.search(res["answer"]):
        return False, "not answered in Hindi script"
    return (not miss), ("matches expected facts" if not miss else f"missing: {miss}")


def rescore(run: Path) -> None:
    """Re-apply the checks to saved answers (no model calls)."""
    rows = json.loads((run / "manual_test_answers.json").read_text(encoding="utf-8"))
    for r in rows:
        r["auto_pass"], r["why"] = verdict(r["n"], r)
    (run / "manual_test_answers.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    summarize(rows)


def summarize(rows: list[dict]) -> None:
    by: dict[str, list[bool]] = {}
    for r in rows:
        by.setdefault(r["category"], []).append(r["auto_pass"])
    print("auto pass:", sum(r["auto_pass"] for r in rows), "/", len(rows))
    for c, v in by.items():
        print(f"  {c}: {sum(v)}/{len(v)}")
    print("failed:", [r["n"] for r in rows if not r["auto_pass"]])


def main() -> None:
    run = Path(sys.argv[1])
    if "--rescore" in sys.argv:
        return rescore(run)
    agent = Agent.load(run / "agent_definition.json", LLM(model="qwen2.5:3b-instruct"))
    rows = []
    for n, (cat, q, expected, sec) in enumerate(QUESTIONS, 1):
        t = time.time()
        res = agent.answer(q)
        ok, why = verdict(n, res)
        rows.append({"n": n, "category": cat, "question": q, "expected": expected, "answer": res["answer"], "refused": res["refused"], "sources": res["sources"], "violations": res.get("violations"),
                     "auto_pass": ok, "why": why, "seconds": round(time.time() - t, 1)})
        print(f"{n:>2} [{'PASS' if ok else 'FAIL'}] {cat}: {q[:70]}\n      -> {res['answer'][:200]}  ({why})", flush=True)
    (run / "manual_test_answers.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    by: dict[str, list[bool]] = {}
    for r in rows:
        by.setdefault(r["category"], []).append(r["auto_pass"])
    print("\nauto pass:", sum(r["auto_pass"] for r in rows), "/", len(rows))
    for c, v in by.items():
        print(f"  {c}: {sum(v)}/{len(v)}")


if __name__ == "__main__":
    main()
