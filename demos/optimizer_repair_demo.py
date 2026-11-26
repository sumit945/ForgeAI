"""Optimizer demonstration from a DELIBERATELY DEGRADED agent.

The generated agents of the two demos already pass their dev split at baseline, so the critic/optimizer loop has nothing to do.
To prove that it works, this script takes the ship-spares agent, sabotages it (top_k=1, tiny chunks, the refusal / scope / exactness /
combine rules removed, temperature 0.7), keeps the SAME eval set and documents, and lets the optimizer repair it.

  python demos/optimizer_repair_demo.py [source_run_folder] [out_folder]

This is a labelled fault-injection test of the loop, not a claim about the factory's first-pass quality.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forgeai.docs import load_folder  # noqa: E402
from forgeai.fsutil import atomic_write  # noqa: E402
from forgeai.generator import validate_definition  # noqa: E402
from forgeai.llm import LLM  # noqa: E402
from forgeai.optimizer import optimise  # noqa: E402
from forgeai.report import pct, scorecard_markdown  # noqa: E402

src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "ship_spares_optimizer_demo"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "runs" / "optimizer_repair_demo"
out.mkdir(parents=True, exist_ok=True)

docs, _ = load_folder(src / "docs")
good = json.loads((src / "05_agent_definition.v0.json").read_text(encoding="utf-8"))
cases = json.loads((src / "06_eval_set.json").read_text(encoding="utf-8"))["cases"]

bad = copy.deepcopy(good)
bad["knowledge"]["retrieval"].update(top_k=1, neighbors=0, glossary={})
bad["knowledge"]["chunking"] = {"size_words": 40, "overlap_words": 0}
bad["system_prompt"]["rules"] = [r for r in bad["system_prompt"]["rules"] if r["id"] not in ("not-found", "scope", "specifics", "combine")]
bad["model"]["temperature"] = 0.7
validate_definition(bad)
atomic_write(out / "degraded_definition.json", bad)

llm = LLM(trace_path=out / "llm_trace.jsonl")
llm.preflight()
res = optimise(bad, docs, cases, llm, llm, out / "iterations", max_iters=3, target=1.0, log=print)
atomic_write(out / "agent_definition.json", res["definition"])
atomic_write(out / "iteration_log.json", {"baseline": res["baseline"], "iterations": res["history"], "stop_reason": res["stop_reason"]})
atomic_write(out / "scorecard.md", scorecard_markdown(res["baseline"], res["history"], res["final_scorecard"]))

lines = ["# Optimizer repair demo (fault injection)", "",
         "Start: the ship-spares agent with top_k=1, 40-word chunks, no refusal/scope/exactness/combine rules, temperature 0.7.",
         f"Same {len(cases)} eval cases and documents as `{src.name}`. Optimizer stop reason: **{res['stop_reason']}**.", "",
         scorecard_markdown(res["baseline"], res["history"], res["final_scorecard"]), "", "## Change log"]
for h in res["history"]:
    lines.append(f"\n### Iteration {h['iteration']} - {h['decision']}")
    lines.append(f"Dev {pct(h['score_before']['dev'])} -> {pct(h['score_after']['dev'])}; hold-out {pct(h['score_before']['holdout'])} -> {pct(h['score_after']['holdout'])}. "
                 f"Failures before: {h['failure_counts_before']}. Now passing: {h['fixed']}. Newly failing: {h['regressed']}.")
    for t in h.get("ablation_trials", []):
        lines.append(f"- ablation `{t['category']}` alone: dev -> {pct(t['dev_after'])} ({'kept' if t['accepted'] else 'rejected'})")
    for c in h["changes"]:
        lines.append(f"- **{c['type']}** `{c['target']}` ({c.get('category', '')}): {c.get('before')} -> {c.get('after')}  \n  why: {c.get('why', '')}  cases: {c.get('cases')}")
if res["final_failures"]:
    lines += ["", "## Still failing after the repair"] + [f"- **{f['id']}** [{f['category']}] {f['question'][:110]} - {f['why']}" for f in res["final_failures"]]
atomic_write(out / "SUMMARY.md", "\n".join(lines) + "\n")
print(f"done -> {out / 'SUMMARY.md'}")
