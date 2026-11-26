"""Orchestrator: brief + folder -> requirements -> groundability -> agent -> eval set -> judged improvement loop -> report.

Production behaviours
  * preflight: model server + models are checked BEFORE any work starts
  * every stage is checkpointed (.ckpt/<stage>.json keyed by a hash of its inputs); `resume=True` skips finished stages
  * run_status.json is rewritten after every stage, so a crash always leaves a truthful record (status, stages, warnings, errors)
  * stage policy: fatal stages raise a typed ForgeError; degradable stages record a warning and the run continues
  * unexpected exceptions are wrapped, logged with traceback to error.log, and never reach the user as a raw stack trace
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Callable

from .docs import inventory, load_folder
from .embeddings import OllamaEmbedder
from .errors import Blocked, BudgetExceeded, ForgeError, InputError, ModelOutputError, ModelUnavailable
from . import evalcache
from .evalgen import DEFAULT_QUOTAS, generate_eval_set
from .fsutil import atomic_write
from .generator import generate_agent
from .groundability import groundability_guard, groundability_markdown
from .intake import build_requirements, collect_answers, propose_questions, requirements_markdown
from .judge import calibrate
from .lang import corpus_language, detect_language
from .ocr import LANG_MODELS, UNSUPPORTED
from .llm import LLM, Budget
from .optimizer import optimise
from .report import build_report, scorecard_markdown
from .schemas import validate_definition, validate_evalset, validate_requirements

PIPELINE_VERSION = "2"
GroundabilityBlocked = Blocked  # backwards-compatible name


def pick_embed_model(docs, host: str, explicit: str | None = None) -> str | None:
    """Choose an installed Ollama embedding model. The multilingual bge-m3 is always preferred: in our benchmark it retrieved the
    right passage 36/36 for Indian-language questions over English documents and vice versa, where nomic-embed-text managed 21/36
    and 3/36 and BM25 3/36 (users of an English document set will still ask in Hindi, Tamil, ...). Smaller English-only models are
    the fallback; an explicit --embed-model always wins."""
    if explicit:
        return explicit if OllamaEmbedder(explicit, host).available() else None
    return next((c for c in ("bge-m3", "nomic-embed-text", "mxbai-embed-large") if OllamaEmbedder(c, host).available()), None)


def _w(path: Path, obj) -> None:
    atomic_write(path, obj)


def _h(*parts) -> str:
    return hashlib.sha1("|".join(json.dumps(p, sort_keys=True, default=str) for p in parts).encode()).hexdigest()[:16]


class Run:
    def __init__(self, out: Path, root_key: str, resume: bool, log: Callable):
        self.out, self.root_key, self.resume, self.log = out, root_key, resume, log
        self.stages: list[dict] = []
        self.warnings: list[str] = []
        self.errors: list[dict] = []
        self.status = "running"
        self.started = time.time()

    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)
            self.log(f"      warning: {msg}")

    def save_status(self, extra: dict | None = None) -> None:
        _w(self.out / "run_status.json", {
            "status": self.status, "pipeline_version": PIPELINE_VERSION, "stages": self.stages,
            "warnings": self.warnings, "errors": self.errors, "elapsed_seconds": round(time.time() - self.started, 1), **(extra or {}),
        })

    def stage(self, name: str, key_parts: tuple, fn: Callable, *, fatal: bool = True, default=None):
        key = _h(self.root_key, name, *key_parts)
        ckpt = self.out / ".ckpt" / f"{name}.json"
        rec = {"name": name, "status": "ok", "seconds": 0.0}
        if self.resume and ckpt.exists():
            try:
                saved = json.loads(ckpt.read_text(encoding="utf-8"))
                if saved.get("key") == key:
                    rec.update(status="resumed")
                    self.stages.append(rec)
                    self.log(f"      (resumed '{name}' from checkpoint)")
                    self.save_status()
                    return saved["data"]
            except Exception:  # noqa: BLE001
                pass  # corrupt checkpoint -> recompute
        t0 = time.time()
        try:
            data = fn()
        except (ModelUnavailable, Blocked, InputError, BudgetExceeded):
            rec.update(status="failed", seconds=round(time.time() - t0, 1))
            self.stages.append(rec)
            raise
        except Exception as e:  # noqa: BLE001
            err = e if isinstance(e, ForgeError) else ForgeError(f"Internal error in stage '{name}': {type(e).__name__}: {e}", stage=name)
            if not isinstance(e, ForgeError):
                _w(self.out / "error.log", traceback.format_exc())
            err.stage = err.stage or name
            rec.update(status="failed", seconds=round(time.time() - t0, 1), error=err.to_dict())
            self.errors.append(err.to_dict())
            self.stages.append(rec)
            if fatal:
                raise err from e
            self.warn(f"stage '{name}' failed and was skipped: {err} ({err.hint})")
            self.save_status()
            return default
        rec["seconds"] = round(time.time() - t0, 1)
        self.stages.append(rec)
        _w(ckpt, {"key": key, "data": data})
        self.save_status()
        return data


def run_factory(
    brief: str,
    docs_dir: str | Path,
    out_dir: str | Path,
    *,
    model: str = "qwen2.5:3b-instruct",
    judge_model: str | None = None,
    fallback_models: list[str] | None = None,
    host: str = "http://localhost:11434",
    backend: str = "ollama",
    answers: dict | list | None = None,
    interactive: bool = False,
    approve: bool = False,
    force: bool = False,
    resume: bool = False,
    fresh_evalset: bool = False,
    embed_model: str | None = None,
    ocr: str = "auto",
    ocr_langs: list[str] | None = None,
    max_iters: int = 3,
    target: float = 0.9,
    max_calls: int | None = None,
    max_minutes: float | None = None,
    log=print,
    llm: LLM | None = None,
    judge_llm: LLM | None = None,
) -> dict:
    brief = (brief or "").strip()
    if len(brief.split()) < 4:
        raise InputError("The brief is too short to build an agent from (need at least a sentence).")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    budget = Budget(max_calls=max_calls, max_seconds=max_minutes * 60 if max_minutes else None)
    trace = out / "llm_trace.jsonl"
    if trace.exists() and not resume:
        trace.unlink()
    llm = llm or LLM(model=model, host=host, backend=backend, trace_path=trace, budget=budget, fallback_models=fallback_models)
    if judge_llm is None:
        judge_llm = llm if not judge_model else LLM(model=judge_model, host=host, backend=backend, trace_path=trace, budget=budget)
    budget = llm.budget  # one shared budget for agent + judge calls, whoever constructed the clients
    if max_calls is not None:
        budget.max_calls = max_calls
    if max_minutes:
        budget.max_seconds = max_minutes * 60
    budget.started = time.time()
    judge_llm.budget = budget

    if ocr_langs is None:  # scanned pages are read in the language of the brief (the best available hint before any text exists)
        code = detect_language(brief)["code"]
        ocr_langs = [code] if code in LANG_MODELS or code in UNSUPPORTED else ["en"]
    docs, problems = load_folder(docs_dir, ocr=ocr, ocr_langs=ocr_langs)  # raises InputError for an unusable folder
    if not docs:
        raise InputError(f"No readable documents in {docs_dir}. Problems: {problems or 'folder is empty'}")
    doc_digest = [(d.name, hashlib.sha1(d.text.encode("utf-8", "ignore")).hexdigest()[:12]) for d in docs]
    root_key = _h(PIPELINE_VERSION, brief, doc_digest, llm.model, judge_llm.model, answers, max_iters, target, force, ocr, ocr_langs)
    run = Run(out, root_key, resume, log)
    try:
        for p in problems:
            run.warn(f"document issue: {p}")
        run.save_status()
        llm.preflight()
        if judge_llm is not llm:
            judge_llm.preflight()

        docs_out = out / "docs"
        if docs_out.exists():
            shutil.rmtree(docs_out)
        docs_out.mkdir()
        for d in docs:
            shutil.copy(d.path, docs_out / d.name)
        _w(out / "00_brief.txt", brief)
        _w(out / "01_inventory.json", {"documents": inventory(docs), "load_problems": problems})
        log(f"[1/7] intake: {len(docs)} documents loaded" + (f" ({len(problems)} issue(s))" if problems else ""))

        # ---- 1. intake -----------------------------------------------------------
        def _intake():
            questions = propose_questions(brief, docs, llm)
            qa = collect_answers(questions, mode="interactive" if interactive else "auto", answers=answers)
            req = build_requirements(brief, docs, qa, llm)
            return {"questions": questions, "qa": qa, "req": req}

        intake = run.stage("intake", (), _intake)
        qa, req = intake["qa"], intake["req"]
        _w(out / "02_questions_and_answers.json", {"max_questions": 5, "asked": len(intake["questions"]), "qa": qa})
        _w(out / "03_requirements.json", req)
        _w(out / "03_requirements.md", requirements_markdown(req))
        if approve:  # human checkpoint
            print("\n" + requirements_markdown(req))
            ans = input("Approve requirements? [Y]es / [e]dit requirements.json first / [n]o abort > ").strip().lower()
            if ans.startswith("n"):
                raise InputError("Aborted by the requester at the requirements checkpoint.")
            if ans.startswith("e"):
                input(f"Edit {out / '03_requirements.json'} then press Enter... ")
                req = validate_requirements(json.loads((out / "03_requirements.json").read_text(encoding="utf-8")))
        log(f"[2/7] requirements written ({len(intake['questions'])} clarifying question(s) asked)")

        # ---- 2. groundability ----------------------------------------------------
        def _guard():
            rep, narrowed = groundability_guard(req, docs, llm)
            # a BLOCKED report legitimately has no supported topics left; only validate when an agent will be generated
            return {"report": rep, "req": narrowed if rep["status"] == "blocked" else validate_requirements(narrowed)}

        g = run.stage("groundability", (req,), _guard)
        report, req = g["report"], g["req"]
        _w(out / "04_groundability.json", report)
        _w(out / "04_groundability.md", groundability_markdown(report))
        _w(out / "03_requirements.final.json", req)
        _w(out / "03_requirements.final.md", requirements_markdown(req))
        log(f"[3/7] groundability: {report['status'].upper()} - {report['summary']}")
        if report["status"] == "warn":
            run.warn(f"groundability: {report['message']}")
        if report["status"] == "blocked" and not force:
            run.status = "blocked"
            _w(out / "REPORT.md", f"# BLOCKED by the groundability guard\n\n{report['message']}\n\n" + groundability_markdown(report))
            run.save_status()
            raise Blocked(report["message"])

        # ---- 3. generate agent ---------------------------------------------------
        emb = None
        if backend == "ollama" and llm.__class__.__name__ == "LLM":
            emb = pick_embed_model(docs, host, embed_model)
            if embed_model and not emb:
                run.warn(f"embedding model '{embed_model}' is not installed; using BM25 retrieval (run: ollama pull {embed_model})")
        log(f"      retrieval: {'hybrid BM25 + ' + emb if emb else 'BM25'}")
        defn0 = run.stage("generate", (req, emb), lambda: generate_agent(req, docs, llm, embed_model=emb, ocr_langs=ocr_langs))
        validate_definition(defn0)
        _w(out / "05_agent_definition.v0.json", defn0)
        log(f"[4/7] agent generated; tools attached: {defn0['tools'] or 'none'}")

        # ---- 4. eval set ---------------------------------------------------------
        def make_evalset() -> dict:
            k = evalcache.key(docs, req, llm, DEFAULT_QUOTAS)
            hit = None if fresh_evalset else evalcache.load(k)
            if hit:
                return evalcache.stamp_reused(hit)
            fresh = generate_eval_set(docs, req, llm)
            evalcache.save(k, fresh)
            return fresh

        ev = run.stage("evalset", (req, defn0["knowledge"], fresh_evalset), make_evalset)
        validate_evalset(ev)
        _w(out / "06_eval_set.json", ev)
        for w in ev.get("warnings", []):
            run.warn(f"eval set: {w}")
        reused = (ev.get("cache") or {}).get("reused")
        log(f"[5/7] eval set: {ev['total']} cases {ev['counts']} (rejected {sum(v for k, v in ev['stats'].items() if k.startswith('rejected:'))} unverifiable candidates)" + (" - REUSED from an earlier identical build (--fresh-evalset to regenerate)" if reused else ""))

        # ---- 5. judge calibration (degradable) ----------------------------------
        cal = run.stage("calibration", (ev["cases"],), lambda: calibrate(ev["cases"], judge_llm), fatal=False, default={"ran": False, "reason": "calibration failed"})
        _w(out / "07_judge_calibration.json", cal)
        log(f"[6/7] judge calibration: reliability={cal.get('reliability')}")
        if cal.get("ran") and (cal.get("reliability") or 0) < 0.75:
            run.warn(f"judge reliability is low ({cal.get('reliability')}); LLM-rescued verdicts are weak evidence")
        if not cal.get("ran"):
            run.warn("judge calibration did not run; only the deterministic checks can be trusted")

        # ---- 6. critic / optimizer ----------------------------------------------
        def _opt():
            o = optimise(defn0, docs, ev["cases"], llm, judge_llm, out / "iterations", max_iters=max_iters, target=target, log=log)
            return {k: v for k, v in o.items()}

        opt = run.stage("optimize", (defn0, ev["cases"], max_iters, target), _opt)
        final = validate_definition(opt["definition"])
        _w(out / "agent_definition.json", final)
        _w(out / "scorecard.json", {"baseline": opt["baseline"], "iterations": opt["history"], "final": opt["final_scorecard"]})
        _w(out / "scorecard.md", scorecard_markdown(opt["baseline"], opt["history"], opt["final_scorecard"]))
        _w(out / "iteration_log.json", {"baseline": opt["baseline"], "iterations": opt["history"], "stop_reason": opt["stop_reason"]})
        _w(out / "final_failures.json", opt["final_failures"])
        if opt["stop_reason"].startswith("budget"):
            run.warn(f"optimizer stopped early: {opt['stop_reason']}")
        sys_errs = [r for r in opt["final_results"] if any(str(v).startswith("agent_error") for v in r.get("violations", []))]
        if sys_errs:
            run.warn(f"{len(sys_errs)} case(s) hit a model error and counted as failures: {[r['id'] for r in sys_errs]}")
        rate = sum(1 for r in opt["final_results"] if "sources_auto_attributed" in r.get("violations", [])) / max(1, len(opt["final_results"]))
        if rate > 0.5:
            run.warn(f"the model left `sources` empty in {rate:.0%} of answers; citations were attributed by code")

        # ---- 7. report -----------------------------------------------------------
        b = budget.summary()
        b["wall_clock_minutes"] = round((time.time() - run.started) / 60, 1)
        b["agent_model"], b["judge_model"] = llm.model, judge_llm.model
        if b["retries"] or b["fallbacks_used"] or b["failed_calls"] or b["truncated_prompts"]:
            run.warn(f"model reliability events: {b['retries']} retries, {b['failed_calls']} failed calls, {b['fallbacks_used']} fallback uses, {b['truncated_prompts']} truncated prompts")
        run.status = "succeeded_with_warnings" if run.warnings else "succeeded"
        health = {"status": run.status, "warnings": run.warnings, "stages": run.stages}
        try:
            rep = build_report({"requirements": req, "groundability": report, "definition": final, "optimizer": opt, "evalset": ev,
                                "calibration": cal, "budget": json.dumps(b), "health": health})
        except Exception as e:  # noqa: BLE001 - a report bug must never lose a finished run
            _w(out / "error.log", traceback.format_exc())
            rep = f"# Report generation failed ({type(e).__name__}: {e})\n\nRaw data is in scorecard.json and iteration_log.json.\n"
            run.warn("report generation failed; raw scorecard and logs are intact")
            run.status = "succeeded_with_warnings"
        _w(out / "REPORT.md", rep)
        run.save_status({"budget": b})
        log(f"[7/7] done in {b['wall_clock_minutes']} min [{run.status}] -> {out / 'REPORT.md'}")
        return {"definition": final, "optimizer": opt, "report": rep, "out": str(out), "status": run.status, "warnings": run.warnings}
    except Blocked:
        raise
    except KeyboardInterrupt:
        run.status = "interrupted"
        run.save_status()
        raise
    except BaseException as e:
        if run.status != "blocked":
            run.status = "failed"
            if isinstance(e, ForgeError):
                run.errors.append(e.to_dict())
            elif not isinstance(e, SystemExit):
                run.errors.append({"type": type(e).__name__, "message": str(e)})
            run.save_status()
        raise
