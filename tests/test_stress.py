"""Stress tests on 100-document corpora: concurrency, fault injection and hostile inputs (scripted model -> fast, deterministic).
The real-model equivalents are run by `python stress/run_stress.py --mode real`."""
from __future__ import annotations

import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tests", ROOT / "stress"):
    sys.path.insert(0, str(p))

import psutil  # noqa: E402
import pytest  # noqa: E402
from make_corpus import make_corpus  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402
from test_forgeai import scripted  # noqa: E402

from forgeai.errors import Blocked, ForgeError, InputError  # noqa: E402
from forgeai.factory import run_factory  # noqa: E402
from forgeai.llm import ScriptedLLM  # noqa: E402
from forgeai.runner import Agent  # noqa: E402
from forgeai.service import JobManager, create_app  # noqa: E402

BRIEF = "I need an assistant that answers questions about our standard operating procedures (SOPs): deadlines in hours, retention and limits."


def quiet(*_a, **_k):
    return None


@pytest.fixture(scope="module")
def corpora(tmp_path_factory):
    base = tmp_path_factory.mktemp("corpora")
    return {lv: (base / lv, make_corpus(lv, base / lv)) for lv in ("L1", "L2", "L3")}


@pytest.fixture(scope="module")
def built_agent(corpora, tmp_path_factory):
    out = tmp_path_factory.mktemp("built") / "run"
    run_factory(BRIEF, corpora["L2"][0], out, llm=ScriptedLLM(scripted, model="scripted"), max_iters=0, log=quiet)
    return out, corpora["L2"][1]


def test_100_doc_build_stays_within_time_and_memory_budget(corpora, tmp_path):
    proc, peak = psutil.Process(), [0]
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.1)

    threading.Thread(target=watch, daemon=True).start()
    t0 = time.time()
    run_factory(BRIEF, corpora["L2"][0], tmp_path / "o", llm=ScriptedLLM(scripted, model="scripted"), max_iters=1, log=quiet)
    stop.set()
    assert time.time() - t0 < 90, "100-document build is too slow"
    assert peak[0] < 1.5 * 1024**3, f"peak memory {peak[0] / 1024**2:.0f} MB"


def test_200_parallel_questions_on_one_agent_all_answered_correctly(built_agent):
    out, man = built_agent
    agent = Agent.load(out / "agent_definition.json", ScriptedLLM(scripted, model="scripted"))
    qa = (man["qa"] * 20)[:200]

    def ask(item):
        r = agent.answer(item["q"])
        return item, r

    with ThreadPoolExecutor(32) as ex:
        results = list(ex.map(ask, qa))
    assert len(results) == 200
    errors = [r for _, r in results if not isinstance(r.get("answer"), str)]
    assert not errors
    # answers must belong to their own question (no cross-talk between threads): the SOP id of the question appears in the evidence
    wrong = [i["q"] for i, r in results if not r["refused"] and i["q"].split(",")[0].replace("Under ", "") not in " ".join(h["text"] for h in r["retrieved"])]
    assert not wrong, wrong[:3]


def test_six_simultaneous_jobs_with_polling_are_isolated_and_race_free(corpora, tmp_path):
    mgr = JobManager(tmp_path / "ws", llm_factory=lambda p: ScriptedLLM(scripted, model="scripted"))
    c = TestClient(create_app(mgr, api_key=None))
    docs = sorted(p for p in corpora["L1"][0].iterdir() if p.is_file())
    jobs = []
    for k in range(6):
        jid = c.post("/jobs", json={"brief": BRIEF + f" (tenant {k})", "max_iters": 0}).json()["id"]
        for f in docs:  # 100 documents per job
            assert c.put(f"/jobs/{jid}/files/{f.name}", content=f.read_bytes()).status_code == 200
        jobs.append(jid)
    stop, codes = threading.Event(), []

    def poll():
        while not stop.is_set():
            for jid in jobs:
                codes.append(c.get(f"/jobs/{jid}").status_code)
            codes.append(c.get("/jobs").status_code)

    pollers = [threading.Thread(target=poll, daemon=True) for _ in range(4)]
    for t in pollers:
        t.start()
    for jid in jobs:  # all started at the same moment
        assert c.post(f"/jobs/{jid}/start", json={}).status_code == 202
    t0 = time.time()
    while time.time() - t0 < 300 and any(c.get(f"/jobs/{j}").json()["status"] in ("created", "queued", "running") for j in jobs):
        time.sleep(0.3)
    stop.set()
    assert set(codes) == {200}, set(codes)  # polling during writes never saw an error
    for k, jid in enumerate(jobs):
        j = c.get(f"/jobs/{jid}").json()
        assert j["status"] in ("succeeded", "succeeded_with_warnings"), (jid, j["status"], j.get("error"))
        assert f"(tenant {k})" in (mgr.run_dir(jid) / "00_brief.txt").read_text(encoding="utf-8")  # no cross-job contamination
        assert len(list((mgr.run_dir(jid) / "docs").iterdir())) >= 99
        assert c.post(f"/jobs/{jid}/chat", json={"question": "Under SOP-001, how many hours does the fire warden have to evacuate the floor?"}).status_code == 200


@pytest.mark.parametrize("seed", [11, 12])
def test_random_model_faults_on_the_hostile_100_doc_corpus(corpora, tmp_path, seed):
    rnd = random.Random(seed)

    def faulty(task, system, user):
        r = rnd.random()
        if r < 0.2:
            raise ForgeError("injected failure")
        if r < 0.3:
            return "not json at all {{{"
        return scripted(task, system, user)

    out = tmp_path / f"chaos{seed}"
    try:
        res = run_factory(BRIEF, corpora["L3"][0], out, llm=ScriptedLLM(faulty, model="scripted"), max_iters=1, log=quiet)
        assert res["status"] in ("succeeded", "succeeded_with_warnings")
    except ForgeError:
        assert json.loads((out / "run_status.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_crash_midway_then_resume_on_100_docs(corpora, tmp_path):
    out = tmp_path / "resume"
    n = {"k": 0}

    def dies(task, system, user):
        if task == "agent_answer":
            n["k"] += 1
            if n["k"] > 5:
                raise RuntimeError("power cut")
        return scripted(task, system, user)

    with pytest.raises(ForgeError):
        run_factory(BRIEF, corpora["L2"][0], out, llm=ScriptedLLM(dies, model="scripted"), max_iters=1, log=quiet)
    res = run_factory(BRIEF, corpora["L2"][0], out, llm=ScriptedLLM(scripted, model="scripted"), max_iters=1, resume=True, log=quiet)
    assert res["status"] in ("succeeded", "succeeded_with_warnings")
    st = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    assert {"intake", "groundability", "generate", "evalset"} <= {s["name"] for s in st["stages"] if s["status"] == "resumed"}


def test_100_unreadable_files_fail_cleanly_with_input_error(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    for k in range(100):
        (d / f"f{k}.pdf").write_bytes(b"%PDF-1.4 garbage" + bytes(range(k % 200, 200)))
    with pytest.raises(InputError) as ei:
        run_factory(BRIEF, d, tmp_path / "o", llm=ScriptedLLM(scripted), log=quiet)
    assert ei.value.exit_code == 3 and "No readable documents" in str(ei.value)


def test_100_documents_that_cannot_support_the_brief_are_blocked(corpora, tmp_path):
    crypto = ScriptedLLM(lambda t, s, u: json.dumps(
        {"title": "Crypto", "purpose": "Advise on cryptocurrency staking yields", "scope": "staking", "audience": "investors", "tone": "x", "language": "English",
         "must_answer_topics": ["staking yields", "validator slashing", "token vesting"], "must_refuse_topics": [], "escalation": "support"})
        if t == "requirements" else scripted(t, s, u))
    with pytest.raises(Blocked) as ei:
        run_factory("Assistant advising on cryptocurrency staking yields and validator slashing.", corpora["L2"][0], tmp_path / "o", llm=crypto, log=quiet)
    assert ei.value.exit_code == 2
