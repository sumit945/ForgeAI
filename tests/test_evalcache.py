"""Eval-set cache: reuse on identical rebuilds, never on any change, never for weak sets, --fresh-evalset forces regeneration."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from test_forgeai import scripted  # noqa: E402

from forgeai import evalcache  # noqa: E402
from forgeai.factory import run_factory  # noqa: E402
from forgeai.llm import ScriptedLLM  # noqa: E402

BRIEF = "I need an assistant that answers HR policy questions for shore staff."
DOCS = ROOT / "demos" / "hr_policy" / "docs"


def build(tmp_path, name, docs=DOCS, **kw):
    calls = {"evalgen": 0}

    def counting(task, system, user):
        if task.startswith("evalgen"):
            calls["evalgen"] += 1
        return scripted(task, system, user)

    out = tmp_path / name
    logs: list[str] = []
    run_factory(BRIEF, docs, out, llm=ScriptedLLM(counting, model="scripted"), log=logs.append, **kw)
    return out, calls["evalgen"], "\n".join(logs)


def test_identical_rebuild_reuses_the_verified_eval_set(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGEAI_CACHE", str(tmp_path / "cache"))
    first, n1, _ = build(tmp_path, "a")
    second, n2, log2 = build(tmp_path, "b")
    assert n1 > 0 and n2 == 0  # no eval-generation calls at all the second time
    assert "REUSED" in log2
    e1 = json.loads((first / "06_eval_set.json").read_text(encoding="utf-8"))
    e2 = json.loads((second / "06_eval_set.json").read_text(encoding="utf-8"))
    assert e1["cases"] == e2["cases"] and e1["stats"] == e2["stats"]  # every detail identical
    assert e2["cache"]["reused"] is True and not (e1.get("cache") or {}).get("reused")
    assert "reused" in (second / "REPORT.md").read_text(encoding="utf-8").lower()
    assert (second / "scorecard.json").exists()  # the agent was still scored fresh


def test_fresh_evalset_forces_regeneration(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGEAI_CACHE", str(tmp_path / "cache"))
    build(tmp_path, "a")
    _, n, log = build(tmp_path, "b", fresh_evalset=True)
    assert n > 0 and "REUSED" not in log


def test_changing_the_documents_is_a_miss(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGEAI_CACHE", str(tmp_path / "cache"))
    build(tmp_path, "a")
    docs = tmp_path / "docs2"
    shutil.copytree(DOCS, docs)
    (docs / "extra.txt").write_text("Remote work is allowed up to 2 days per week with manager approval. " * 4, encoding="utf-8")  # one more document
    _, n, log = build(tmp_path, "b", docs=docs)
    assert n > 0 and "REUSED" not in log


def test_weak_or_corrupt_entries_are_never_used(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGEAI_CACHE", str(tmp_path / "cache"))
    assert evalcache.save("k1", {"total": 5, "warnings": [], "cases": []}) is False  # too few cases
    assert evalcache.save("k2", {"total": 25, "warnings": ["w"], "cases": []}) is False  # carries a warning
    d = evalcache.cache_dir()
    d.mkdir(parents=True)
    (d / "bad.json").write_text("{not json", encoding="utf-8")
    assert evalcache.load("bad") is None and evalcache.load("missing") is None


def test_key_depends_on_every_input(tmp_path):
    from forgeai.docs import Document

    llm = ScriptedLLM(scripted, model="m1")
    docs = [Document("a.txt", "a.txt", "text", "hello world")]
    req = {"purpose": "p", "must_answer_topics": ["t"], "must_refuse_topics": [], "known_gaps": [], "language_code": "en", "language": "English"}
    q = {"answerable": 9}
    base = evalcache.key(docs, req, llm, q)
    assert evalcache.key(docs, req, llm, q) == base
    assert evalcache.key([Document("a.txt", "a.txt", "text", "hello world!")], req, llm, q) != base
    assert evalcache.key(docs, {**req, "must_answer_topics": ["t2"]}, llm, q) != base
    assert evalcache.key(docs, req, ScriptedLLM(scripted, model="m2"), q) != base
    assert evalcache.key(docs, req, llm, {"answerable": 10}) != base
