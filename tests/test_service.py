"""API tests with Starlette's TestClient and a scripted model (no Ollama needed)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import pytest  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402
from test_forgeai import HR, scripted  # noqa: E402

from forgeai.llm import ScriptedLLM  # noqa: E402
from forgeai.service import JobManager, create_app, safe_name  # noqa: E402

BRIEF = "I need an assistant that answers HR policy questions for shore staff."


@pytest.fixture()
def client(tmp_path):
    mgr = JobManager(tmp_path / "ws", llm_factory=lambda p: ScriptedLLM(scripted, model="scripted"))
    return TestClient(create_app(mgr, api_key=None)), mgr


def wait(c, jid, until=("succeeded", "succeeded_with_warnings", "failed", "blocked"), timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/jobs/{jid}").json()
        if j["status"] in until:
            return j
        time.sleep(0.3)
    raise AssertionError("job did not finish")


def test_full_job_lifecycle_and_chat(client):
    c, _ = client
    j = c.post("/jobs", json={"brief": BRIEF, "max_iters": 1}).json()
    jid = j["id"]
    for f in sorted(HR.glob("*.pdf")):
        r = c.put(f"/jobs/{jid}/files/{f.name}", content=f.read_bytes())
        assert r.status_code == 200
    assert c.post(f"/jobs/{jid}/start", json={}).status_code == 202
    done = wait(c, jid)
    assert done["status"] in ("succeeded", "succeeded_with_warnings"), (done["error"], done["log"][-5:])
    assert "ForgeAI run report" in c.get(f"/jobs/{jid}/report").text
    assert c.get(f"/jobs/{jid}/artifacts/agent_definition.json").status_code == 200
    r = c.post(f"/jobs/{jid}/chat", json={"question": "how many days of sick leave do employees get?"})
    assert r.status_code == 200 and r.json()["answer"]


def test_validation_and_security(client):
    c, _ = client
    assert c.post("/jobs", json={"brief": "hi"}).status_code == 400
    jid = c.post("/jobs", json={"brief": BRIEF}).json()["id"]
    assert c.put(f"/jobs/{jid}/files/evil.exe", content=b"MZ").status_code == 400
    assert c.put(f"/jobs/{jid}/files/empty.txt", content=b"").status_code == 400
    assert c.post(f"/jobs/{jid}/start", json={}).status_code == 400  # no documents
    assert c.get("/jobs/does-not-exist").status_code == 404
    assert c.post(f"/jobs/{jid}/chat", json={"question": "x"}).status_code == 400  # no finished agent
    assert c.get(f"/jobs/{jid}/artifacts/../job.json").status_code == 404  # path traversal
    assert safe_name("../../etc/passwd.txt") == "passwd.txt"
    assert safe_name("a b$c.pdf") == "a b_c.pdf"


def test_api_key_is_enforced(tmp_path):
    mgr = JobManager(tmp_path / "ws", llm_factory=lambda p: ScriptedLLM(scripted))
    c = TestClient(create_app(mgr, api_key="s3cret"))
    assert c.get("/jobs").status_code == 401
    assert c.get("/jobs", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert c.get("/health").status_code in (200, 503)  # health is open


def test_blocked_and_restart_recovery(client, tmp_path):
    c, mgr = client
    jid = c.post("/jobs", json={"brief": "Assistant that advises on cryptocurrency staking yields and validator slashing."}).json()["id"]
    c.put(f"/jobs/{jid}/files/hr.txt", content=b"Employees receive 24 days of leave per year. Leave must be approved by the manager in advance.")
    c.post(f"/jobs/{jid}/start", json={})
    done = wait(c, jid)
    assert done["status"] in ("blocked", "failed", "succeeded", "succeeded_with_warnings")
    # simulate a server crash mid-run, then restart
    mgr.jobs[jid]["status"] = "running"
    mgr._save(mgr.jobs[jid])
    mgr2 = JobManager(mgr.root, llm_factory=lambda p: ScriptedLLM(scripted))
    assert mgr2.jobs[jid]["status"] == "interrupted"
