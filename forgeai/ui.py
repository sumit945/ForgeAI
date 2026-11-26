"""Streamlit UI:  streamlit run forgeai/ui.py   (or: python -m forgeai ui)

Tabs: Build (brief + documents -> live progress), Runs (reports, scorecards, eval sets of any run), Chat (talk to a produced agent).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from forgeai.docs import SUPPORTED  # noqa: E402
from forgeai.errors import ForgeError  # noqa: E402
from forgeai.llm import LLM  # noqa: E402
from forgeai.runner import Agent  # noqa: E402
from forgeai.service import JobManager  # noqa: E402

ROOT = Path(os.environ.get("FORGEAI_ROOT", Path(__file__).resolve().parents[1]))
st.set_page_config(page_title="ForgeAI", page_icon="🛠", layout="wide")


@st.cache_resource
def manager(model: str, host: str, workspace: str) -> JobManager:
    return JobManager(workspace, llm_factory=lambda p: LLM(model=p.get("model") or model, host=p.get("host") or host))


@st.cache_resource(max_entries=4)
def load_agent(defn_path: str, model: str, host: str) -> Agent:
    return Agent.load(defn_path, LLM(model=model, host=host))


def run_folders() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for base in (ROOT / "runs", ROOT / "workspace" / "jobs"):
        if base.exists():
            for p in sorted(base.glob("**/run_status.json")):
                found[f"{p.parent.parent.name + '/' if base.name == 'jobs' else ''}{p.parent.name}"] = p.parent
    return found


def status_badge(s: str) -> str:
    return {"succeeded": "🟢", "succeeded_with_warnings": "🟡", "blocked": "🟠", "failed": "🔴", "running": "🔵", "queued": "🔵", "interrupted": "🟣"}.get(s, "⚪")


with st.sidebar:
    st.title("🛠 ForgeAI")
    st.caption("The agent that builds, tests and reports on agents.")
    model = st.text_input("Model (Ollama)", "qwen2.5:3b-instruct")
    host = st.text_input("Ollama host", "http://localhost:11434")
    workspace = st.text_input("Workspace folder", str(ROOT / "workspace"))
    mgr = manager(model, host, workspace)
    if st.button("Check model server"):
        try:
            LLM(model=model, host=host).preflight()
            st.success("Model server and model are ready.")
        except ForgeError as e:
            st.error(f"{e}\n\n{e.hint}")

tab_build, tab_runs, tab_chat = st.tabs(["Build an agent", "Runs & reports", "Chat with an agent"])

# ------------------------------------------------------------------------------------------------ build
with tab_build:
    st.subheader("1. Describe the assistant")
    brief = st.text_area("One-paragraph brief", height=130, placeholder="I need an assistant that answers HR policy questions for shore staff...")
    files = st.file_uploader("2. Upload the documents", accept_multiple_files=True, type=[e.lstrip(".") for e in sorted(SUPPORTED)])
    with st.expander("Advanced"):
        c1, c2, c3 = st.columns(3)
        max_iters = c1.number_input("Max improvement iterations", 0, 3, 3)
        target = c2.slider("Stop when dev pass-rate reaches", 0.5, 1.0, 0.9, 0.05)
        embed = c3.text_input("Embedding model (optional)", "")
    if st.button("Build agent", type="primary", disabled=not (brief.strip() and files)):
        try:
            job = mgr.create(brief, {"model": model, "host": host, "max_iters": int(max_iters), "target": float(target), **({"embed_model": embed} if embed else {})})
            for f in files:
                mgr.add_file(job["id"], f.name, f.getvalue())
            mgr.start(job["id"])
            st.session_state["job"] = job["id"]
        except ForgeError as e:
            st.error(f"{e}  \n{e.hint}")

    @st.fragment(run_every=2)
    def progress():
        jid = st.session_state.get("job")
        if not jid:
            return
        j = mgr.public(mgr._get(jid), 25)
        st.markdown(f"### {status_badge(j['status'])} Job `{jid}` - **{j['status']}**")
        rs = j.get("run_status", {})
        if rs.get("stages"):
            st.write(" → ".join(f"{status_badge('succeeded' if s['status'] in ('ok', 'resumed') else 'failed')} {s['name']}" for s in rs["stages"]))
        st.code("\n".join(j["log"][-12:]) or "waiting...", language="text")
        if j.get("error"):
            e = j["error"]
            st.error(f"**{e['type']}**: {e['message']}  \n{e.get('hint', '')}")
        if j["status"] in ("succeeded", "succeeded_with_warnings"):
            st.success("Agent built. Open the *Runs & reports* tab for the full report, or chat with it in the *Chat* tab.")
            for w in rs.get("warnings", []):
                st.warning(w)
        if j["status"] in ("interrupted", "failed") and st.button("Resume this job"):
            mgr.start(jid, resume=True)

    progress()

# ------------------------------------------------------------------------------------------------ runs
with tab_runs:
    runs = run_folders()
    if not runs:
        st.info("No runs yet. Build an agent first, or run `python -m forgeai build ...` into the `runs/` folder.")
    else:
        pick = st.selectbox("Run", list(runs), key="run_pick")
        d = runs[pick]
        rs = json.loads((d / "run_status.json").read_text(encoding="utf-8"))
        st.markdown(f"### {status_badge(rs['status'])} {pick} - {rs['status']}")
        for w in rs.get("warnings", []):
            st.warning(w)
        for e in rs.get("errors", []):
            st.error(f"{e.get('type')}: {e.get('message')}")
        if (d / "scorecard.json").exists():
            sc = json.loads((d / "scorecard.json").read_text(encoding="utf-8"))
            rows = [{"iteration": 0, "outcome": "baseline", **{k: sc["baseline"]["scorecard"][k]["pass_rate"] for k in ("dev", "holdout", "overall")}}]
            rows += [{"iteration": h["iteration"], "outcome": "accepted" if h["accepted"] else "reverted", **h["score_after"]} for h in sc["iterations"]]
            st.markdown("**Score per iteration** (pass-rate)")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            fin = sc["final"]["by_category"]
            st.bar_chart(pd.DataFrame({"pass_rate": {k: v["pass_rate"] for k, v in fin.items()}}))
        sub = st.tabs(["Report", "Eval set", "Agent definition", "Requirements"])
        with sub[0]:
            st.markdown((d / "REPORT.md").read_text(encoding="utf-8") if (d / "REPORT.md").exists() else "_no report yet_")
        with sub[1]:
            if (d / "06_eval_set.json").exists():
                ev = json.loads((d / "06_eval_set.json").read_text(encoding="utf-8"))
                st.dataframe(pd.DataFrame([{k: c.get(k) for k in ("id", "category", "split", "question", "expected_facts", "should_refuse")} for c in ev["cases"]]), hide_index=True, width="stretch")
        with sub[2]:
            if (d / "agent_definition.json").exists():
                st.json(json.loads((d / "agent_definition.json").read_text(encoding="utf-8")), expanded=False)
        with sub[3]:
            if (d / "03_requirements.final.md").exists():
                st.markdown((d / "03_requirements.final.md").read_text(encoding="utf-8"))

# ------------------------------------------------------------------------------------------------ chat
with tab_chat:
    ready = {k: v for k, v in run_folders().items() if (v / "agent_definition.json").exists()}
    if not ready:
        st.info("No finished agent yet.")
    else:
        pick = st.selectbox("Agent", list(ready), key="chat_pick")
        if st.session_state.get("chat_for") != pick:
            st.session_state["chat_for"], st.session_state["msgs"] = pick, []
        agent_path = str(ready[pick] / "agent_definition.json")
        for m in st.session_state["msgs"]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("meta"):
                    st.caption(m["meta"])
        q = st.chat_input("Ask the agent a question")
        if q:
            st.session_state["msgs"].append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            try:
                ag = load_agent(agent_path, model, host)
                hist = [{"q": a["content"], "a": b["content"]} for a, b in zip(st.session_state["msgs"][::2], st.session_state["msgs"][1::2])]
                with st.spinner("Thinking..."):
                    r = ag.answer(q, hist[-3:])
                meta = ("declined · " if r["refused"] else "") + ("sources: " + ", ".join(r["sources"]) if r["sources"] else "no sources")
                if r["tool_calls"]:
                    meta += " · tools: " + ", ".join(t["tool"] for t in r["tool_calls"])
                st.session_state["msgs"].append({"role": "assistant", "content": r["answer"], "meta": meta})
            except ForgeError as e:
                st.session_state["msgs"].append({"role": "assistant", "content": f"⚠️ {e} — {e.hint}"})
            st.rerun()
