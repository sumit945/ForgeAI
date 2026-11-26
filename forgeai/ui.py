"""ForgeAI console (Streamlit).  streamlit run forgeai/ui.py   or   python -m forgeai ui

"Lavender glass" console: Dashboard, Build Agent (5-step wizard: Intake, Requirements, Agent Config, Evaluate, Optimize), Evaluate, Optimize,
Agent Runner, Projects, Reports, Settings. Every number on screen is read from real run folders (forgeai/dashboard_data.py).

The pipeline runs unattended, so the wizard steps fill in live as stages finish; Back / Next only change which step is in view.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402

from forgeai import dashboard_data as dd  # noqa: E402
from forgeai import uikit as ui  # noqa: E402
from forgeai.docs import SUPPORTED, load_folder  # noqa: E402
from forgeai.errors import ForgeError  # noqa: E402
from forgeai.lang import LANGS, detect_language  # noqa: E402
from forgeai.llm import LLM  # noqa: E402
from forgeai.runner import Agent  # noqa: E402
from forgeai.service import JobManager  # noqa: E402

ROOT = Path(os.environ.get("FORGEAI_ROOT", Path(__file__).resolve().parents[1]))
st.set_page_config(page_title="ForgeAI Console", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

S = st.session_state
S.setdefault("page", "dashboard")
S.setdefault("theme", "light")
S.setdefault("model", "qwen2.5:3b-instruct")
S.setdefault("host", "http://localhost:11434")
S.setdefault("workspace", str(ROOT / "workspace"))
S.setdefault("msgs", {})
S.setdefault("wiz", 0)  # step in view in the Build Agent wizard
S.setdefault("brief_text", "")

WIZ = ["Intake", "Requirements", "Agent Config", "Evaluate", "Optimize"]
STAGES = [("intake", "Intake", "Clarifying questions and requirements"), ("groundability", "Groundability guard", "Can the documents support the brief?"), ("generate", "Generate agent", "Prompt, retrieval and tools"),
          ("evalset", "Generate eval set", "Verified test cases from the documents"), ("calibration", "Calibrate judge", "Known-answer canaries"), ("optimize", "Improve (max 3 iterations)", "Classify failures, targeted fixes"),
          ("report", "Honest report", "Scorecard, change log, failures")]


def html(s: str) -> None:
    st.markdown(s, unsafe_allow_html=True)


def go(page: str) -> None:
    S["page"] = page
    st.rerun()


@st.cache_resource
def manager(model: str, host: str, workspace: str) -> JobManager:
    return JobManager(workspace, llm_factory=lambda p: LLM(model=p.get("model") or model, host=p.get("host") or host))


@st.cache_resource(max_entries=4)
def load_agent(defn_path: str, model: str, host: str) -> Agent:
    return Agent.load(defn_path, LLM(model=model, host=host))


@st.cache_data(ttl=20, show_spinner=False)
def health(host: str, model: str) -> dict:
    try:
        names = [m["name"] for m in requests.get(f"{host.rstrip('/')}/api/tags", timeout=2).json().get("models", [])]
        has = lambda m: m in names or any(n.split(":")[0] == m for n in names)  # noqa: E731
        return {"ok": True, "model_ok": has(model), "embed": next((e for e in ("bge-m3", "nomic-embed-text", "mxbai-embed-large") if has(e)), None), "models": names}
    except Exception:  # noqa: BLE001
        return {"ok": False, "model_ok": False, "embed": None, "models": []}


def runs_list() -> list[dict]:
    return dd.list_runs([ROOT / "runs", Path(S["workspace"]) / "jobs"])


def pct(v) -> str:
    return f"{v:g}%" if v is not None else "—"


html(ui.stylesheet(S["theme"], S["page"]))

# =============================================================================================== sidebar
with st.sidebar:
    html(ui.sidebar_logo())
    for key, label in ui.PAGES:
        if st.button(label, key=f"nav_{key}", width="stretch"):
            go(key)
    h = health(S["host"], S["model"])
    html('<div style="height:26vh"></div>' + ui.model_card(h["ok"] and h["model_ok"], S["model"], h["embed"]))
html(ui.topbar("U"))


def title_row(title: str, subtitle: str, *buttons: tuple[str, str, str]) -> None:
    left, right = st.columns([3, 2 if buttons else 0.01], vertical_alignment="center")
    with left:
        html(ui.header(title, subtitle))
    if buttons:
        with right:
            cols = st.columns(len(buttons))
            for c, (label, target, kind) in zip(cols, buttons):
                if c.button(label, type=kind, key=f"act_{label}_{S['page']}", width="stretch"):
                    go(target)
    st.write("")


def pick_run(label: str = "Project", only_agents: bool = False, key: str = "sel_run", need: str | None = None) -> dict | None:
    """need: "cases" keeps projects with an eval set, "scores" keeps projects that were scored."""
    runs = [r for r in runs_list() if (r["has_agent"] or not only_agents) and (need != "cases" or r["cases"]) and (need != "scores" or r["pass"] is not None)]
    if not runs:
        html(ui.alert("info", "No projects yet. Build your first agent from Build Agent.", "info"))
        if st.button("Build Agent", type="primary", key=f"empty_{S['page']}"):
            go("build")
        return None
    names = [f"{r['name']}  ·  {r['key']}" for r in runs]
    idx = names.index(S[key]) if S.get(key) in names else 0
    pick = st.selectbox(label, names, index=idx, key=f"{key}_box_{S['page']}")
    S[key] = pick
    return runs[names.index(pick)]


def card(inner: str, pad: str = "") -> None:
    html(f'<div class="fx-card"{f" style=padding:{pad}" if pad else ""}>{inner}</div>')


# =============================================================================================== Dashboard
def page_dashboard() -> None:
    runs = runs_list()
    k = dd.kpis(runs)
    with st.container(border=True):
        a, b = st.columns([3, 2], vertical_alignment="center")
        with a:
            html('<div class="fx-h1" style="font-size:32px;line-height:40px;font-weight:800">Welcome to ForgeAI</div>'
                 '<div class="fx-h3" style="font-size:20px;margin:6px 0 10px;background:var(--brand-grad);-webkit-background-clip:text;background-clip:text;color:transparent">Build. Evaluate. Optimize.</div>'
                 '<div class="fx-sub" style="max-width:430px;margin-bottom:14px">Create document-grounded AI agents, test them against verified questions, and let the optimizer fix what fails. Every number is measured, not claimed.</div>')
            if st.button("Build New Agent", type="primary", key="dash_build"):
                S["wiz"] = 0
                S.pop("wiz_run", None)
                go("build")
        with b:
            html('<div style="display:flex;justify-content:center">' + ui.hero_art() + "</div>")
    st.write("")
    iters = sum(r["iterations"] for r in runs)
    best = max((r["holdout"] if r["holdout"] is not None else r["pass"] for r in runs if r["pass"] is not None), default=None)
    html(ui.tiles([("layers", str(k["agents"]), "Agents Built", "brand"), ("bars", str(sum(1 for r in runs if r["pass"] is not None)), "Evaluations Run", "info"),
                   ("optimize", str(iters), "Optimization Runs", "warn"), ("trophy", pct(best), "Best Score (hold-out)", "ok")]))
    st.write("")
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        trs = []
        for r in runs[:5]:
            trs.append([ui.name_cell(r["name"], f"{r['documents']} documents · {r['language']}"), ui.status_badge(r["status"]), ui.pct_bar(r["pass"], 80), f"{r['minutes']} min" if r["minutes"] else "—"])
        html('<div class="fx-card" style="padding:20px 0 6px"><div style="padding:0 20px 10px;display:flex;justify-content:space-between"><div class="fx-h3">Recent Projects</div></div>'
             + (ui.table(["Agent", "Status", "Score", "Time"], trs, ["40%", "22%", "26%", "12%"]) if trs else '<div class="fx-muted" style="padding:20px">No projects yet.</div>') + "</div>")
        if st.button("View all projects", key="dash_all"):
            go("projects")
    with c2:
        mgr = manager(S["model"], S["host"], S["workspace"])
        active = sum(1 for j in mgr.snapshot() if j["status"] in ("queued", "running"))
        ok = h["ok"] and h["model_ok"]
        health_rows = [("Model server", S["model"], "ok" if ok else "bad", "Connected" if ok else "Unavailable"),
                       ("Embeddings", f"{h['embed'] or 'none'} · " + ("hybrid retrieval" if h["embed"] else "BM25 only"), "ok" if h["embed"] else "warn", "Ready" if h["embed"] else "Not installed"),
                       ("Job queue", f"{active} running or queued", "info" if active else "neutral", "Active" if active else "Idle")]
        body = "".join(f'<div style="display:flex;justify-content:space-between;align-items:center;margin-top:14px"><div><div style="font-weight:600;color:var(--text)">{ui.esc(a)}</div>'
                       f'<div class="fx-muted">{ui.esc(b)}</div></div>{ui.badge(c, d)}</div>' for a, b, c, d in health_rows)
        html(f'<div class="fx-card"><div class="fx-h3">System health</div>{body}</div>')
        att = k["attention"]
        notes = "".join(f'<div style="display:flex;gap:10px;margin-top:12px"><div style="width:4px;border-radius:4px;background:var(--{"bad" if r["status"] in ("failed", "blocked") else "warn"})"></div><div>'
                        f'<div style="font-weight:600;color:var(--text)">{ui.esc(r["name"])}</div><div class="fx-muted">'
                        + (f"Status: {ui.esc(r['status'].replace('_', ' '))}" if r["status"] in ("failed", "blocked") else f"Dev {r['dev']}% vs hold-out {r['holdout']}%: likely overfit")
                        + "</div></div></div>" for r in att[:3])
        st.write("")
        html(f'<div class="fx-card"><div class="fx-h3">Needs attention</div>{notes or "<div class=fx-muted style=margin-top:12px>Nothing needs attention.</div>"}</div>')


# =============================================================================================== Build Agent (wizard)
def _job_view(mgr: JobManager):
    jid = S.get("job")
    return mgr._get(jid) if jid and jid in mgr.jobs else None


def _wiz_run_path(mgr: JobManager) -> str | None:
    if S.get("wiz_run"):
        return S["wiz_run"]
    job = _job_view(mgr)
    return str(mgr.run_dir(job["id"])) if job else None


def _nav(prev_to: int | None, next_to: int | None, next_label: str = "Next", next_disabled: bool = False) -> None:
    st.write("")
    a, _, c = st.columns([1, 4, 1.4])
    if prev_to is not None and a.button("Back", key=f"back_{S['wiz']}", icon=":material/arrow_back:", width="stretch"):
        S["wiz"] = prev_to
        st.rerun()
    if next_to is not None and c.button(next_label, key=f"next_{S['wiz']}", type="primary", width="stretch", disabled=next_disabled):
        S["wiz"] = next_to
        st.rerun()


def _wizard_states(stages: dict, view: int, has_run: bool) -> list[str]:
    done = [has_run, stages.get("groundability") in ("ok", "resumed"), stages.get("generate") in ("ok", "resumed"), stages.get("calibration") in ("ok", "resumed"), stages.get("report") in ("ok", "resumed") or stages.get("optimize") in ("ok", "resumed")]
    return ["on" if i == view else "done" if done[i] else "todo" for i in range(5)]


def _no_run(what: str) -> None:
    html(ui.alert("info", f"{what} appears here once the build has produced it. Start a build in step 1, or open a finished project from Projects.", "info"))


def page_build() -> None:
    title_row("Build Agent", "Create a document-grounded agent in five steps. The pipeline runs on its own; steps fill in as each stage finishes.")
    mgr = manager(S["model"], S["host"], S["workspace"])
    path = _wiz_run_path(mgr)
    d = dd.run_detail(path) if path and Path(path).exists() else None
    stages = {s["name"]: s["status"] for s in ((d or {}).get("status") or {}).get("stages", [])}
    view = S["wiz"]
    html(ui.wizard_steps(WIZ, _wizard_states(stages, view, bool(d))))
    if view == 0:
        _step_intake(mgr, d)
    elif view == 1:
        _step_requirements(d)
    elif view == 2:
        _step_config(d)
    elif view == 3:
        _step_evaluate(d)
    else:
        _step_optimize(d)


def _step_intake(mgr: JobManager, d: dict | None) -> None:
    job = _job_view(mgr)
    running = bool(job and job["status"] in ("queued", "running"))
    brief = S.get("brief_text", "")
    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.container(border=True):
            lang_badge = ui.badge("brand", f"Detected: {detect_language(brief)['name']}") if brief.strip() else ""
            html(f'<div style="display:flex;justify-content:space-between;align-items:center"><div class="fx-h3">Describe your agent</div>{lang_badge}</div>')
            S["brief_text"] = st.text_area("Brief", value=brief, height=130, label_visibility="collapsed", max_chars=4000,
                                           placeholder="I need an assistant that answers HR policy questions for shore staff…  (Hindi, Tamil, Bengali and 8 more Indian languages are supported.)")
            html('<div class="fx-h3" style="margin-top:6px">Upload documents</div>')
            ups = st.file_uploader("Documents", accept_multiple_files=True, type=[e.lstrip(".") for e in sorted(SUPPORTED)], label_visibility="collapsed")
            if ups:
                with tempfile.TemporaryDirectory() as td:
                    for u in ups:
                        (Path(td) / Path(u.name).name).write_bytes(u.getvalue())
                    docs, problems = load_folder(td, ocr="detect")  # preview only: scanned pages are reported, OCR runs at build time
                loaded = {x.name: x for x in docs}
                rows = ""
                for u in ups:
                    nm = Path(u.name).name
                    prob = next((p for p in problems if p.startswith(nm + ":")), None)
                    kb = f"{u.size / 1024:.0f} KB" if u.size < 1_048_576 else f"{u.size / 1_048_576:.1f} MB"
                    if nm in loaded:
                        lg, st_html = detect_language(loaded[nm].text[:3000])["name"], (ui.badge("warn", "Loaded, injection removed") if prob and "injection" in prob else ui.badge("ok", "Loaded"))
                    elif prob and "OCR when" in prob:
                        lg, st_html = "—", ui.badge("info", "Scanned · OCR at build")
                    else:
                        lg, st_html = "—", ui.badge("neutral" if prob and "duplicate" in prob else "warn", (prob.split(":", 1)[1].strip()[:28] if prob else "Skipped"))
                    rows += ui.file_row(nm, f"{nm.rsplit('.', 1)[-1].upper()} · {kb}", ui.esc(lg), st_html)
                html(f'<div style="margin-top:8px">{rows}</div><div class="fx-muted" style="margin-top:8px">{len(loaded)} of {len(ups)} files usable</div>')
            with st.expander("Advanced settings"):
                a, b, c = st.columns(3)
                S["iters"] = a.number_input("Max improvement iterations", 0, 3, S.get("iters", 3))
                S["target"] = b.slider("Stop at dev pass rate", 0.5, 1.0, S.get("target", 0.9), 0.05)
                S["embed"] = c.text_input("Embedding model (blank = auto)", S.get("embed", ""))
                S["fresh"] = st.checkbox("Generate a fresh eval set (ignore the cache)", S.get("fresh", False), help="By default a verified eval set from an identical earlier build is reused, which saves several minutes.")
        ready = len(brief.split()) >= 4 and bool(ups) and not running
        a, b = st.columns([3, 1.4])
        if b.button("Start Build", type="primary", width="stretch", disabled=not ready, key="start_build"):
            try:
                j = mgr.create(S["brief_text"], {"model": S["model"], "host": S["host"], "max_iters": int(S["iters"]), "target": float(S["target"]), "fresh_evalset": bool(S.get("fresh")), **({"embed_model": S["embed"]} if S["embed"] else {})})
                for u in ups:
                    mgr.add_file(j["id"], u.name, u.getvalue())
                mgr.start(j["id"])
                S["job"] = j["id"]
                S.pop("wiz_run", None)
                st.rerun()
            except ForgeError as e:
                st.error(f"{e}  \n{e.hint}")
        if not ready and not running:
            a.caption("Add a brief (at least four words) and at least one document to start.")
    with right:
        @st.fragment(run_every=2 if running else None)
        def pipeline() -> None:
            jb = _job_view(mgr)
            pub = mgr.public(jb, 25) if jb else None
            stg = {s["name"]: s for s in ((pub or {}).get("run_status") or {}).get("stages", [])}
            status = pub["status"] if pub else "idle"
            parts = [f'<div class="fx-card" style="padding:20px 8px"><div style="display:flex;justify-content:space-between;align-items:center;padding:0 12px 8px"><div class="fx-h3">Pipeline</div>'
                     f'{ui.status_badge(status) if pub else ui.badge("neutral", "Idle")}</div>']
            seen = False
            for key, title, meta in STAGES:
                s = stg.get(key)
                if s and s["status"] in ("ok", "resumed"):
                    parts.append(ui.stage("done", title, meta, f"{s['seconds']}s"))
                elif s and s["status"] == "failed":
                    parts.append(ui.stage("fail", title, "Failed - see run health", f"{s['seconds']}s"))
                elif pub and pub["status"] in ("running", "queued") and not seen:
                    seen = True
                    parts.append(ui.stage("run", title, meta, "running"))
                elif key == "report" and pub and pub["status"] in ("succeeded", "succeeded_with_warnings"):
                    parts.append(ui.stage("done", title, meta, "done"))
                else:
                    parts.append(ui.stage("pend", title, meta, "—"))
            parts.append("</div>")
            html("".join(parts))  # one block: an HTML card cannot be opened and closed across separate markdown calls
            if pub:
                st.write("")
                html(ui.log_block(pub.get("log", [])))
                warns = (pub.get("run_status") or {}).get("warnings") or []
                err = pub.get("error")
                body = "".join(ui.alert("warn", w) for w in warns[:4]) + (ui.alert("bad", f"{err['type']}: {err['message']}  {err.get('hint', '')}") if err else "")
                if body:
                    html(f'<div class="fx-card" style="margin-top:20px"><div class="fx-h3" style="margin-bottom:10px">Run health</div>{body}</div>')
                if pub["status"] in ("interrupted", "failed", "blocked") and st.button("Resume this job", key="resume"):
                    mgr.start(S["job"], resume=True)
                    st.rerun()
                if pub["status"] in ("succeeded", "succeeded_with_warnings"):
                    st.success("Agent built. Continue to review its requirements, configuration and scores.")
        pipeline()
    _nav(None, 1, "Next", next_disabled=d is None)


def _step_requirements(d: dict | None) -> None:
    if not d or not d["requirements"]:
        _no_run("The requirements checklist")
        _nav(0, None)
        return
    req, g = d["requirements"], d["groundability"]
    left, right = st.columns([3, 2], gap="large")
    topics = req.get("must_answer_topics") or []
    verdicts = {t["topic"]: t["verdict"] for t in g.get("topics", [])}
    oks = [verdicts.get(t, "supported") in ("supported", "not_requested") for t in topics]
    with left:
        html(f'<div class="fx-card"><div class="fx-h3">Agent purpose</div><div class="fx-sub" style="margin-bottom:14px">{ui.esc(req.get("purpose", ""))}</div>'
             f'<div class="fx-caption">Audience</div><div style="margin:4px 0 12px;color:var(--text)">{ui.esc(req.get("audience", ""))}</div>'
             f'<div class="fx-caption">Tone · language</div><div style="margin:4px 0 16px;color:var(--text)">{ui.esc(req.get("tone", ""))} · {ui.esc(req.get("language", ""))}</div>'
             f'<div class="fx-h3" style="margin-bottom:6px">Must answer</div>{ui.check_list(topics, oks) if topics else "<div class=fx-muted>None recorded.</div>"}'
             + (f'<div class="fx-h3" style="margin:16px 0 8px">Must refuse</div>' + "".join(ui.chip(t) for t in req.get("must_refuse_topics", [])) if req.get("must_refuse_topics") else "") + "</div>")
    with right:
        sm = g.get("summary") or {}
        total = sum(sm.get(k, 0) for k in ("supported", "partial", "unsupported"))
        score = round(100 * sm.get("supported", 0) / total) if total else None
        gaps = g.get("known_gaps") or req.get("known_gaps") or []
        html(f'<div class="fx-card"><div class="fx-h3" style="margin-bottom:14px">Groundability check</div>{ui.ring(score, "Supported")}'
             f'<div style="margin-top:16px">{ui.metric("Supported by documents", str(sm.get("supported", 0)))}{ui.metric("Partially supported", str(sm.get("partial", 0)))}{ui.metric("Unsupported", str(sm.get("unsupported", 0)))}</div>'
             + "".join(ui.note(False, x if isinstance(x, str) else str(x)) for x in gaps[:4]) + f'<div class="fx-muted" style="margin-top:8px">{ui.esc(g.get("message", ""))}</div></div>')
    _nav(0, 2)


def _step_config(d: dict | None) -> None:
    if not d or not d["definition"]:
        _no_run("The agent configuration")
        _nav(1, None)
        return
    defn = d["definition"]
    sp, kn = defn.get("system_prompt") or {}, defn.get("knowledge") or {}
    left, right = st.columns([3, 2], gap="large")
    with left:
        t1, t2, t3, t4 = st.tabs(["System Prompt", "Retrieval", "Tools", "Output Contract"])
        with t1:
            rules = "\n".join(f"{i + 1}. {r['text']}" for i, r in enumerate(sp.get("rules", [])))
            html(ui.code_block(sp.get("role", "") + "\n\n" + rules + "\n\nRefusal: " + sp.get("refusal_message", "")))
        with t2:
            rt, ck = kn.get("retrieval", {}), kn.get("chunking", {})
            html(f'<div class="fx-card">{ui.metric("Method", str(rt.get("method")))}{ui.metric("Embedder", str(rt.get("embedder") or "—"))}{ui.metric("Top-k passages", str(rt.get("top_k")))}'
                 f'{ui.metric("Chunk size / overlap", f"{ck.get('size_words')} / {ck.get('overlap_words')} words")}{ui.metric("Documents", str(len(kn.get("documents", []))))}</div>')
        with t3:
            sel = defn.get("tool_selection") or {}
            rows = [[ui.name_cell(k), ui.badge("ok" if v.get("attached") else "neutral", "Attached" if v.get("attached") else "Not attached"), ui.esc("; ".join(v.get("reasons", []))[:150])] for k, v in sel.items()]
            html('<div class="fx-card" style="padding:16px 0 4px">' + (ui.table(["Tool", "State", "Why (evidence, not vote alone)"], rows, ["22%", "18%", "60%"]) if rows else '<div class="fx-muted" style="padding:16px">No tools considered.</div>') + "</div>")
        with t4:
            oc = defn.get("output_contract", {})
            html(ui.code_block("\n".join(f"{k}: {v}" for k, v in (oc.get("fields") or {}).items()) + f"\n\ncitation_required: {oc.get('citation_required')}"))
    with right:
        s = d["summary"]
        html('<div class="fx-card"><div class="fx-h3" style="margin-bottom:6px">Configuration Summary</div>'
             + ui.cfg_row("runner", "Model", f"{defn.get('model', {}).get('name', '—')}, temperature {defn.get('model', {}).get('temperature', '—')}")
             + ui.cfg_row("db", "Retrieval", f"{s['retrieval'] or '—'} · top-{kn.get('retrieval', {}).get('top_k', '—')}")
             + ui.cfg_row("tool", "Tools", ", ".join(defn.get("tools", [])) or "None")
             + ui.cfg_row("file", "Documents", f"{len(kn.get('documents', []))} files")
             + ui.cfg_row("globe", "Language", s["language"]) + "</div>")
    _nav(1, 3)


def _step_evaluate(d: dict | None) -> None:
    if not d or not d["evalset"]:
        _no_run("The evaluation results")
        _nav(2, None)
        return
    _evaluation_body(d)
    _nav(2, 4)


def _step_optimize(d: dict | None) -> None:
    if not d or not d["scorecard"]:
        _no_run("The optimization history")
        _nav(3, None)
        return
    _optimization_body(d)
    _nav(3, None)
    a, _, c = st.columns([1, 4, 1.4])
    if c.button("Open in Agent Runner", type="primary", key="to_runner", width="stretch", disabled=not d["definition"]):
        S["sel_agent"] = None
        go("runner")


# =============================================================================================== shared evaluation / optimization bodies
def _evaluation_body(d: dict) -> None:
    ds = dd.dataset_split(d["evalset"])
    results = dd.final_results(d["summary"]["path"])
    cnt = dd.result_counts(results)
    m = dd.final_metrics(d)
    html(ui.tiles([("file", str(ds["total"]), "Test Cases", "brand"), ("check", str(ds["answerable"]), "Answerable", "ok"), ("x", str(ds["out_of_scope"]), "Out of Scope", "warn"), ("alert", str(ds["edge"]), "Edge Cases", "info")]))
    st.write("")
    a, b = st.columns([2, 3], gap="large")
    with a:
        html(f'<div class="fx-card"><div class="fx-h3" style="margin-bottom:14px">Evaluation Results</div>{ui.ring(m["pass_rate"], "Overall Score")}'
             f'<div style="margin-top:14px">{ui.metric("Passed", str(cnt["passed"]))}{ui.metric("Failed", str(cnt["failed"]))}</div></div>')
    with b:
        rows = [("Correctness", m["correct_rate"], "How often the stated facts match the expected ones"), ("Groundedness", m["grounded_rate"], "Numbers and claims traced to retrieved evidence"),
                ("Refusal recall", m["refusal_recall"], "Out-of-scope questions correctly declined"), ("Retrieval hit rate", m["retrieval_hit_rate"], "Needed passage was in the retrieved context")]
        body = "".join(f'<div style="display:flex;align-items:center;justify-content:space-between;gap:14px;padding:10px 0;border-bottom:1px solid var(--border)"><div style="flex:1;min-width:0"><div style="font-weight:600;color:var(--text)">{n}</div>'
                       f'<div class="fx-muted">{ui.esc(t)}</div></div><div style="white-space:nowrap">{ui.pct_bar(v, 70)}</div></div>' for n, v, t in rows)
        html(f'<div class="fx-card"><div class="fx-h3">Metrics</div>{body}'
             + (f'<div class="fx-muted" style="margin-top:10px">Pass = correct AND grounded AND declined when it should. Contract violations: {pct(m["contract_violation_rate"])} of answers.</div>') + "</div>")
    st.write("")
    fails = d["failures"]
    rows = [[ui.name_cell(f["id"], f["category"].replace("_", " ")), ui.esc(f["question"][:110]), ui.badge("bad", f["category"].replace("_", " ").title()), ui.esc(f["why"][:100])] for f in fails]
    html('<div class="fx-card" style="padding:20px 0 4px"><div style="padding:0 20px 10px;display:flex;justify-content:space-between"><div class="fx-h3">Failed Cases</div>'
         + ui.badge("bad" if fails else "ok", f"{len(fails)} case(s)") + "</div>"
         + (ui.table(["Case", "Question", "Diagnosis", "Why"], rows, ["14%", "38%", "16%", "32%"]) if rows else '<div class="fx-muted" style="padding:0 20px 20px">No failing cases on this eval set. That does not mean the agent is perfect.</div>') + "</div>")
    with st.expander("All eval cases"):
        _cases_table(d, results)
    st.write("")
    html('<div class="fx-card"><div class="fx-h3">Why you can (and cannot) trust this</div>' + "".join(ui.note(c["ok"], c["text"]) for c in dd.checks(d)) + "</div>")


def _cases_table(d: dict, results: list[dict] | None = None) -> None:
    cases = d["evalset"].get("cases") or []
    res = {r["id"]: r for r in (results or [])}
    if not cases:
        html(ui.alert("info", "No eval set for this run.", "info"))
        return
    df = pd.DataFrame([{"id": c["id"], "category": c["category"], "split": c.get("split"), "question": c["question"], "expected facts": "; ".join(c.get("expected_facts", [])),
                        "must refuse": c["should_refuse"], "result": ("pass" if res[c["id"]].get("passed") else "fail") if c["id"] in res else "—"} for c in cases])
    st.dataframe(df, hide_index=True, width="stretch")


def _optimization_body(d: dict) -> None:
    s = d["summary"]
    its = d["iteration_log"].get("iterations") or []
    base = ((d["scorecard"].get("baseline") or {}).get("scorecard") or {}).get("overall", {}).get("pass_rate")
    if not its:
        html(ui.alert("info", f"No optimizer iterations ran. Stop reason: {d['iteration_log'].get('stop_reason', 'the agent already met its dev target')}.", "info"))
        if s["pass"] is not None:
            html('<div class="fx-card" style="margin-top:16px"><div class="fx-h3">Final score</div>' + ui.ring(s["pass"], "Overall") + "</div>")
        return
    chain = [("Baseline", pct(round(base * 100)) if base is not None else "—", "")]
    for h in its:
        chain.append((f"Iteration {h['iteration']}", pct(round((h.get("score_after") or {}).get("dev", 0) * 100)), "on" if h.get("accepted") else "bad"))
    labels = [f"Iteration {h['iteration']}" for h in its]
    tabs = st.tabs(labels)
    for tab, h in zip(tabs, its):
        with tab:
            b, a = h.get("score_before") or {}, h.get("score_after") or {}
            left, right = st.columns([3, 2], gap="large")
            with left:
                fc = h.get("failures_classified") or []
                rows = [[ui.name_cell(f["id"]), ui.badge("info" if f["category"] == "retrieval_miss" else "bad", f["category"].replace("_", " ").title()), ui.esc(f["why"][:110])] for f in fc[:8]]
                html('<div class="fx-card" style="padding:20px 0 4px"><div style="padding:0 20px 10px"><div class="fx-h3">Critic analysis</div><div class="fx-muted">What failed on the dev split and why.</div></div>'
                     + (ui.table(["Case", "Diagnosis", "Why"], rows, ["14%", "22%", "64%"]) if rows else '<div class="fx-muted" style="padding:0 20px 20px">No failures to classify.</div>') + "</div>")
                st.write("")
                changes = "".join(f'<div class="fx-item"><b>{ui.esc(c["type"])}</b> <span class="fx-muted">{ui.esc(c["target"])}</span><div style="margin-top:4px;color:var(--text-2)">{ui.esc(str(c.get("before")))} → {ui.esc(str(c.get("after"))[:200])}</div>'
                                  f'<div class="fx-muted">why: {ui.esc(c.get("why", ""))}</div></div>' for c in h.get("changes", []))
                html(f'<div class="fx-card"><div class="fx-h3">Applied changes</div>{changes or "<div class=fx-muted style=margin-top:8px>No change was proposed.</div>"}</div>')
            with right:
                html(f'<div class="fx-card"><div class="fx-h3" style="margin-bottom:12px">Decision</div>{ui.badge("ok" if h.get("accepted") else "bad", "Accepted" if h.get("accepted") else "Reverted")}'
                     f'<div class="fx-muted" style="margin:10px 0">{ui.esc(h.get("decision", ""))}</div>'
                     f'{ui.metric("Dev score", f"{pct(round(b.get('dev', 0) * 100))} → {pct(round(a.get('dev', 0) * 100))}")}{ui.metric("Hold-out (never used to decide)", f"{pct(round(b.get('holdout', 0) * 100))} → {pct(round(a.get('holdout', 0) * 100))}")}'
                     f'{ui.metric("Fixed / regressed", f"{len(h.get('fixed') or [])} / {len(h.get('regressed') or [])}")}</div>')
    st.write("")
    html('<div class="fx-card"><div class="fx-h3">Optimization progress</div>' + ui.iter_chain(chain) + "</div>")
    if s["dev"] is not None and s["holdout"] is not None and s["dev"] - s["holdout"] > 15:
        html('<div style="margin-top:16px">' + ui.alert("warn", f"Dev {s['dev']}% vs hold-out {s['holdout']}% is a {s['dev'] - s['holdout']}-point gap. The optimizer likely overfit its dev cases: treat hold-out as the honest estimate.") + "</div>")


# =============================================================================================== Evaluate / Optimize pages
def page_evaluate() -> None:
    title_row("Evaluate", "The generated test set, how it was verified, and how the agent scored.")
    r = pick_run(need="cases")
    if not r:
        return
    d = dd.run_detail(r["path"])
    if not d["evalset"]:
        html(ui.alert("info", "This project has no eval set.", "info"))
        return
    st.write("")
    _evaluation_body(d)
    cal = d["calibration"]
    bk = cal.get("by_kind") or {}
    if bk:
        st.write("")
        rows = [[ui.name_cell(k.replace("_", " ")), str(v)] for k, v in bk.items()]
        html(f'<div class="fx-card"><div class="fx-h3">Judge calibration · reliability {cal.get("reliability", "—")}</div><div class="fx-muted" style="margin-bottom:8px">Answers with a known verdict, run through the same judge.</div>{ui.table(["Canary", "Correct"], rows)}</div>')


def page_optimize() -> None:
    title_row("Optimize", "How the critic classified failures, what it changed, and whether each change was kept.")
    r = pick_run(need="scores")
    if not r:
        return
    st.write("")
    _optimization_body(dd.run_detail(r["path"]))


# =============================================================================================== Agent Runner
def page_runner() -> None:
    title_row("Agent Runner", "Chat with any agent the factory produced. Answers come back in your language, with sources and guardrail results.")
    r = pick_run("Agent", only_agents=True, key="sel_agent_run")
    if not r:
        return
    key = r["key"]
    S["msgs"].setdefault(key, [])
    chat, side = st.columns([2, 1], gap="large")
    with side:
        last = next((m for m in reversed(S["msgs"][key]) if m["role"] == "assistant" and m.get("meta")), None)
        meta = last["meta"] if last else None
        src = "".join(f'<div class="fx-src"><b>{ui.esc(x["id"])}</b> · score {x["score"]}</div>' for x in (meta or {}).get("retrieved", [])[:4]) or '<div class="fx-muted">Ask a question to see its sources.</div>'
        tools = ", ".join(t["tool"] for t in (meta or {}).get("tool_calls", [])) or "none used"
        flags = (meta or {}).get("violations", [])
        html(f'<div class="fx-card"><div class="fx-h3">Inspector</div><div class="fx-caption" style="margin-top:14px">Agent</div><div style="font-weight:600">{ui.esc(r["name"])}</div>'
             f'<div class="fx-muted">{ui.esc(r["language"])} · {ui.esc(r["retrieval"] or "bm25")} retrieval · tools: {ui.esc(", ".join(r["tools"]) or "none")}</div>'
             f'<div class="fx-caption" style="margin-top:16px">Answered in</div><div>{ui.esc(LANGS.get((meta or {}).get("lang", ""), {}).get("name", "—"))}</div>'
             f'<div class="fx-caption" style="margin-top:16px">Sources retrieved</div>{src}<div class="fx-caption" style="margin-top:16px">Tools</div><div class="fx-muted">{ui.esc(tools)}</div>'
             f'<div class="fx-caption" style="margin-top:16px">Guardrails</div>' + (ui.badge("warn", ", ".join(flags)) if flags else ui.badge("ok", "No contract violations")) + "</div>")
    with chat:
        if st.button("Clear conversation", key="clear_chat"):
            S["msgs"][key] = []
            st.rerun()
        for m in S["msgs"][key]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("caption"):
                    st.caption(m["caption"])
        q = st.chat_input("Ask in English, Hindi, Tamil, Bengali…")
        if q:
            S["msgs"][key].append({"role": "user", "content": q})
            try:
                ag = load_agent(str(Path(r["path"]) / "agent_definition.json"), S["model"], S["host"])
                hist = [{"q": a["content"], "a": b["content"]} for a, b in zip(S["msgs"][key][::2], S["msgs"][key][1::2])]
                with st.spinner("Thinking…"):
                    res = ag.answer(q, hist[-3:])
                cap = ("Declined · " if res["refused"] else "") + ("Sources: " + ", ".join(res["sources"]) if res["sources"] else "No sources") + (f" · Language: {LANGS[res['lang']]['name']}" if res.get("lang") in LANGS else "")
                S["msgs"][key].append({"role": "assistant", "content": res["answer"], "caption": cap, "meta": {k: res.get(k) for k in ("retrieved", "tool_calls", "violations", "lang")}})
            except ForgeError as e:
                S["msgs"][key].append({"role": "assistant", "content": f"⚠️ {e} — {e.hint}"})
            st.rerun()


# =============================================================================================== Projects
def page_projects() -> None:
    runs = runs_list()
    title_row("Projects", "Every agent the factory has built, with its status and measured scores.", ("New Agent", "build", "primary"))
    if not runs:
        html(ui.alert("info", "No projects yet.", "info"))
        return
    trs = [[ui.name_cell(r["name"], f"{r['key']} · {r['documents']} documents · {r['cases'] or '—'} cases"), ui.chip(r["language"]), ui.status_badge(r["status"]), ui.pct_bar(r["pass"], 90),
            pct(r["holdout"]), f"{r['iterations']}", f"{r['minutes']} min" if r["minutes"] else "—"] for r in runs]
    html('<div class="fx-card" style="padding:20px 0 6px">' + ui.table(["Project", "Language", "Status", "Score", "Hold-out", "Iterations", "Time"], trs, ["32%", "11%", "15%", "17%", "9%", "8%", "8%"]) + "</div>")
    st.write("")
    names = [f"{r['name']}  ·  {r['key']}" for r in runs]
    c1, c2, c3 = st.columns([3, 1, 1])
    pick = c1.selectbox("Open a project", names, key="proj_pick", label_visibility="collapsed")
    r = runs[names.index(pick)]
    if c2.button("Open in wizard", key="proj_wiz", width="stretch"):
        S["wiz_run"], S["wiz"], S["page"] = r["path"], 1, "build"
        st.rerun()
    if c3.button("Chat", key="proj_chat", type="primary", width="stretch", disabled=not r["has_agent"]):
        S["sel_agent_run"] = pick
        go("runner")


# =============================================================================================== Reports
def page_reports() -> None:
    title_row("Reports", "The honest report for each project: scorecard, change log, what still fails and why.")
    r = pick_run()
    if not r:
        return
    d = dd.run_detail(r["path"])
    s = d["summary"]
    st.write("")
    html(f'<div style="display:flex;gap:12px;align-items:center;margin:0 0 4px"><span class="fx-h1">{ui.esc(s["name"])}</span>{ui.status_badge(s["status"])}{ui.chip(s["language"])}</div>'
         f'<div class="fx-sub" style="margin-bottom:16px">{s["documents"]} documents · {s["cases"] or "—"} eval cases · {s["retrieval"] or "—"} retrieval · {s["minutes"] or "—"} min · {s["calls"] or "—"} model calls</div>')
    html(ui.kpi_cards([("Overall pass rate", pct(s["pass"]), f"Baseline {s['baseline']}%" if s["baseline"] is not None else ""), ("Dev split", pct(s["dev"]), "The optimizer tunes against this"),
                       ("Hold-out split", pct(s["holdout"]), "Never used for any decision"), ("Judge reliability", str(s["reliability"] if s["reliability"] is not None else "—"), "Known-answer canaries"),
                       ("Eval cases", str(s["cases"] or "—"), f"{s['iterations']} optimizer iteration(s)")], cols=5))
    st.write("")
    a, b = st.columns([3, 2], gap="large")
    with a:
        rows = dd.scorecard_rows(d)
        pts = [(x["label"].split(" · ")[0], x["dev"]) for x in rows]
        cats = "".join(f'<div style="display:flex;align-items:center;gap:12px;margin-top:12px"><div style="width:160px;font-size:14px;color:var(--text-2)">{ui.esc(c["name"].replace("_", " "))}</div>{ui.pct_bar(c["pass"], 200)}<span class="fx-muted">n={c["n"]}</span></div>' for c in dd.categories(d))
        html('<div class="fx-card"><div class="fx-h3">Dev score by iteration</div>' + ui.trend_chart(pts) + "</div>")
        st.write("")
        html(f'<div class="fx-card"><div class="fx-h3">Pass rate by category</div>{cats or "<div class=fx-muted>—</div>"}</div>')
    with b:
        html('<div class="fx-card"><div class="fx-h3">Why you can (and cannot) trust this</div>' + "".join(ui.note(c["ok"], c["text"]) for c in dd.checks(d)) + "</div>")
        st.write("")
        if d["report_md"]:
            st.download_button("Download REPORT.md", d["report_md"], file_name=f"{r['key'].replace('/', '_')}_REPORT.md", icon=":material/download:", type="primary", width="stretch")
    if d["report_md"]:
        with st.expander("Read REPORT.md"):
            st.markdown(d["report_md"])
    with st.expander("Artifacts"):
        st.dataframe(pd.DataFrame({"artifact": d["artifacts"]}), hide_index=True, width="stretch")


# =============================================================================================== Settings
def page_settings() -> None:
    title_row("Settings", "Model endpoint, workspace, OCR, languages and appearance.")
    with st.container(border=True):
        html('<div class="fx-h3">Model server</div>')
        a, b = st.columns(2)
        S["model"] = a.text_input("Chat model (Ollama)", S["model"])
        S["host"] = b.text_input("Ollama host", S["host"])
        hh = health(S["host"], S["model"])
        html(ui.badge("ok" if hh["ok"] and hh["model_ok"] else "bad", "Connected, model installed" if hh["ok"] and hh["model_ok"] else ("Model not installed" if hh["ok"] else "Server not reachable"))
             + f' <span class="fx-muted">embedding model: {ui.esc(hh["embed"] or "none installed")} · installed: {ui.esc(", ".join(hh["models"]) or "—")}</span>')
        from forgeai import ocr as ocr_mod

        oc = ocr_mod.status(["hi"])
        have = [c for c, ok in oc.get("models", {}).items() if ok and c != "hi-Latn"]
        html('<div class="fx-caption" style="margin-top:16px">OCR (scanned PDFs and images)</div>'
             + ui.badge("ok" if have else "warn", f"Models installed: {', '.join(have)}" if have else "No OCR models installed")
             + '<span class="fx-muted"> EasyOCR on CPU · English, Hindi, Marathi, Bengali, Tamil, Telugu, Kannada, Urdu · install more with <code>python -m forgeai ocr-setup --langs ta</code></span>')
        if st.button("Re-check now"):
            health.clear()
            st.rerun()
    with st.container(border=True):
        html('<div class="fx-h3">Workspace and appearance</div>')
        S["workspace"] = st.text_input("Workspace folder (jobs are stored here)", S["workspace"])
        dark = st.toggle("Dark theme", value=S["theme"] == "dark")
        if dark != (S["theme"] == "dark"):
            S["theme"] = "dark" if dark else "light"
            st.rerun()
    tiles = "".join(f'<span class="fx-lang">{ui.esc(v["native"])}<span>{ui.esc(v["name"])}</span></span>' for c, v in LANGS.items() if c != "hi-Latn")
    html('<div class="fx-card" style="margin-top:16px"><div class="fx-h3">Supported languages</div><div class="fx-muted" style="margin-bottom:14px">Documents, briefs and questions in any of these are detected automatically (offline, by script). '
         f'The model still sets the quality ceiling.</div>{tiles}</div>')
    html('<div class="fx-muted" style="margin-top:12px">REST API: <code>python -m forgeai serve</code> · CLI: <code>python -m forgeai build --brief ... --docs ... --out ...</code></div>')


PAGE_FN = {"dashboard": page_dashboard, "build": page_build, "evaluate": page_evaluate, "optimize": page_optimize, "runner": page_runner, "projects": page_projects, "reports": page_reports, "settings": page_settings}
PAGE_FN.get(S["page"], page_dashboard)()
