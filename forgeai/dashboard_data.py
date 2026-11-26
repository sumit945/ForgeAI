"""Read-only data layer for the dashboard: turns run folders into the numbers the UI shows. Nothing here is mock data -
every value comes from the artifacts a run wrote (run_status.json, scorecard.json, 06_eval_set.json, ...). Missing files
degrade to None, never to an exception, so a half-finished or older run folder still renders."""
from __future__ import annotations

import json
from pathlib import Path


def _read(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _pct(x) -> int | None:
    return None if x is None else round(float(x) * 100)


def effective_scorecard(d: Path) -> dict:
    """scorecard.json when present; otherwise rebuilt from iteration_log.json + iterations/iter_N_results.json
    (older or partial runs, e.g. the optimizer fault-injection demo, have no scorecard.json)."""
    sc = _read(d / "scorecard.json", {}) or {}
    if sc.get("final"):
        return sc
    it = _read(d / "iteration_log.json", {}) or {}
    if not it.get("baseline"):
        return sc
    accepted = [h["iteration"] for h in it.get("iterations", []) if h.get("accepted")]
    k = max(accepted) if accepted else 0
    res = _read(d / "iterations" / f"iter_{k}_results.json", {}) or {}
    final = res.get("scorecard") or ((it.get("baseline") or {}).get("scorecard") or {})
    return {"baseline": it.get("baseline"), "iterations": it.get("iterations", []), "final": final}


def run_dirs(roots: list[Path]) -> list[Path]:
    """Every folder under the given roots that looks like a ForgeAI run (has a status file or an agent definition)."""
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.glob("*")):
            if p.is_dir() and ((p / "run_status.json").exists() or (p / "agent_definition.json").exists()):
                out.append(p)
        for p in sorted(root.glob("*/run")):  # workspace/jobs/<id>/run
            if p.is_dir() and (p / "run_status.json").exists():
                out.append(p)
    return out


def summarize_run(d: Path) -> dict:
    st = _read(d / "run_status.json", {}) or {}
    sc = effective_scorecard(d)
    req = _read(d / "03_requirements.final.json", {}) or _read(d / "03_requirements.json", {}) or {}
    cal = _read(d / "07_judge_calibration.json", {}) or {}
    ev = _read(d / "06_eval_set.json", {}) or {}
    defn = _read(d / "agent_definition.json", {}) or {}
    it = _read(d / "iteration_log.json", {}) or {}
    fin = (sc.get("final") or {})
    overall, dev, hold = (fin.get("overall") or {}), (fin.get("dev") or {}), (fin.get("holdout") or {})
    base = ((sc.get("baseline") or {}).get("scorecard") or {}).get("overall", {})
    job = d.parent.parent.name if d.name == "run" else None  # workspace/jobs/<id>/run
    name = req.get("title") or defn.get("name") or (job or d.name)
    ev_total = ev.get("total")
    mtime = max((f.stat().st_mtime for f in d.glob("*.json")), default=0)
    iters = len((it.get("iterations") or []))
    return {
        "key": d.name if job is None else f"{job}/run", "path": str(d), "name": name,
        "status": st.get("status") or ("finished" if defn else "unknown"),
        "warnings": st.get("warnings") or [], "errors": st.get("errors") or [],
        "language": req.get("language") or "English", "language_code": req.get("language_code") or "en",
        "pass": _pct(overall.get("pass_rate")), "dev": _pct(dev.get("pass_rate")), "holdout": _pct(hold.get("pass_rate")),
        "baseline": _pct(base.get("pass_rate")), "iterations": iters,
        "minutes": ((st.get("budget") or {}).get("wall_clock_minutes")), "calls": ((st.get("budget") or {}).get("calls")),
        "cases": ev.get("total"), "reliability": cal.get("reliability"), "tools": defn.get("tools") or [],
        "retrieval": ((defn.get("knowledge") or {}).get("retrieval") or {}).get("method"),
        "documents": len((defn.get("knowledge") or {}).get("documents") or []), "mtime": mtime,
        "has_agent": bool(defn), "known_gaps": req.get("known_gaps") or [],
    }


def list_runs(roots: list[Path]) -> list[dict]:
    runs = [summarize_run(d) for d in run_dirs(roots)]
    return sorted(runs, key=lambda r: -r["mtime"])


def kpis(runs: list[dict]) -> dict:
    done = [r for r in runs if r["pass"] is not None]
    rel = [r["reliability"] for r in runs if r["reliability"] is not None]
    langs = {r["language"] for r in runs if r["has_agent"]}
    hold = [r["holdout"] for r in done if r["holdout"] is not None]
    avg = round(sum(r["pass"] for r in done) / len(done)) if done else None
    return {
        "agents": sum(1 for r in runs if r["has_agent"]),
        "avg_pass": avg,
        "avg_holdout": round(sum(hold) / len(hold)) if hold else None,
        "reliability": round(sum(rel) / len(rel), 2) if rel else None,
        "languages": sorted(langs),
        "attention": [r for r in runs if r["status"] in ("failed", "blocked") or (r["dev"] is not None and r["holdout"] is not None and r["dev"] - r["holdout"] > 15)],
    }


def run_detail(d: str | Path) -> dict:
    d = Path(d)
    return {
        "summary": summarize_run(d),
        "status": _read(d / "run_status.json", {}) or {},
        "scorecard": effective_scorecard(d),
        "iteration_log": _read(d / "iteration_log.json", {}) or {},
        "failures": _read(d / "final_failures.json", []) or [],
        "evalset": _read(d / "06_eval_set.json", {}) or {},
        "requirements": _read(d / "03_requirements.final.json", {}) or {},
        "groundability": _read(d / "04_groundability.json", {}) or {},
        "calibration": _read(d / "07_judge_calibration.json", {}) or {},
        "definition": _read(d / "agent_definition.json", {}) or {},
        "report_md": (d / "REPORT.md").read_text(encoding="utf-8") if (d / "REPORT.md").exists() else "",
        "artifacts": sorted(p.name for p in d.iterdir() if p.is_file() and not p.name.startswith(".")) if d.exists() else [],
    }


def scorecard_rows(detail: dict) -> list[dict]:
    """Per-iteration rows: baseline first, then every optimizer iteration with its decision."""
    sc = detail["scorecard"]
    rows = []
    b = ((sc.get("baseline") or {}).get("scorecard") or {})
    if b:
        rows.append({"label": "0 · Baseline", "outcome": "Untouched", "dev": _pct((b.get("dev") or {}).get("pass_rate")), "holdout": _pct((b.get("holdout") or {}).get("pass_rate")), "overall": _pct((b.get("overall") or {}).get("pass_rate")), "changes": []})
    for h in sc.get("iterations") or []:
        a = h.get("score_after") or {}
        rows.append({"label": f"{h['iteration']} · " + (", ".join(c.get("target", "").split(".")[-1].split("[")[0] for c in h.get("changes", [])[:2]) or "no change"),
                     "outcome": "Accepted" if h.get("accepted") else "Reverted", "dev": _pct(a.get("dev")), "holdout": _pct(a.get("holdout")), "overall": _pct(a.get("overall")), "changes": h.get("changes", []), "decision": h.get("decision", "")})
    return rows


def categories(detail: dict) -> list[dict]:
    by = ((detail["scorecard"].get("final") or {}).get("by_category") or {})
    return [{"name": k, "n": v.get("n"), "pass": _pct(v.get("pass_rate"))} for k, v in by.items()]


def checks(detail: dict) -> list[dict]:
    """Honest 'can you trust this' notes, derived from the data (never hard-coded claims)."""
    s = detail["summary"]
    out = []
    ev = detail["evalset"]
    rej = sum(v for k, v in (ev.get("stats") or {}).items() if k.startswith("rejected:"))
    if ev:
        out.append({"ok": True, "text": f"Expected facts are checked verbatim against the source text; {rej} weak candidate cases were rejected."})
    if s["reliability"] is not None:
        out.append({"ok": s["reliability"] >= 0.75, "text": f"Judge reliability measured on known-answer canaries: {s['reliability']}." + ("" if s["reliability"] >= 0.75 else " Below 0.75: treat LLM-rescued verdicts as weak evidence.")})
    if s["dev"] is not None and s["holdout"] is not None:
        gap = s["dev"] - s["holdout"]
        out.append({"ok": gap <= 15, "text": f"Dev {s['dev']}% vs hold-out {s['holdout']}%: a {gap:+d} point gap" + (" (the optimizer likely overfit its dev cases)." if gap > 15 else ".")})
    if s["cases"]:
        out.append({"ok": s["cases"] >= 20, "text": f"{s['cases']} eval cases: one case moves the score {100 / s['cases']:.0f} points. The same small model writes, answers and judges."})
    for w in s["warnings"][:3]:
        out.append({"ok": False, "text": w})
    return out


def final_results(d: str | Path) -> list[dict]:
    """Per-case results of the final (best accepted) agent: iterations/iter_K_results.json, K = last accepted iteration or 0."""
    d = Path(d)
    it = _read(d / "iteration_log.json", {}) or {}
    accepted = [h["iteration"] for h in it.get("iterations", []) if h.get("accepted")]
    res = _read(d / "iterations" / f"iter_{max(accepted) if accepted else 0}_results.json", {}) or {}
    return res.get("results") or []


def dataset_split(evalset: dict) -> dict:
    """Eval-set composition for the Evaluate page: answerable questions, out-of-scope (must refuse) and edge cases (everything else)."""
    cases = (evalset or {}).get("cases") or []
    ans = sum(1 for c in cases if c.get("category") == "answerable")
    off = sum(1 for c in cases if c.get("category") == "off_topic")
    return {"total": len(cases), "answerable": ans, "out_of_scope": off, "edge": len(cases) - ans - off}


def result_counts(results: list[dict]) -> dict:
    p = sum(1 for r in results if r.get("passed"))
    return {"passed": p, "failed": len(results) - p, "total": len(results)}


def final_metrics(detail: dict) -> dict:
    """Headline metrics of the final agent as percentages (None when the run has no scorecard)."""
    o = (detail["scorecard"].get("final") or {}).get("overall") or {}
    return {k: _pct(o.get(k)) for k in ("pass_rate", "correct_rate", "grounded_rate", "refusal_recall", "retrieval_hit_rate", "contract_violation_rate", "wrong_refusal_rate")}
