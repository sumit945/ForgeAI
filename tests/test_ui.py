"""Headless tests of the Streamlit console and its data layer (no Ollama needed)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "forgeai" / "ui.py"
PAGES = ["dashboard", "build", "evaluate", "optimize", "runner", "projects", "reports", "settings"]


def make_run(base: Path, name: str, *, with_scorecard: bool = True, dev=1.0, hold=0.6) -> Path:
    d = base / "runs" / name
    (d / "iterations").mkdir(parents=True)

    def w(f, o):
        (d / f).write_text(json.dumps(o), encoding="utf-8")

    sc = lambda dv, hd, ov: {"dev": {"pass_rate": dv}, "holdout": {"pass_rate": hd}, "overall": {"pass_rate": ov, "n": 20}}  # noqa: E731
    w("run_status.json", {"status": "succeeded_with_warnings", "warnings": ["a warning"], "stages": [], "budget": {"wall_clock_minutes": 3.2, "calls": 40}})
    w("03_requirements.final.json", {"title": f"Agent {name}", "language": "Hindi", "language_code": "hi", "known_gaps": []})
    w("agent_definition.json", {"name": f"Agent {name}", "tools": ["calculator"], "knowledge": {"documents": ["a.txt"], "retrieval": {"method": "hybrid"}}})
    w("06_eval_set.json", {"total": 20, "counts": {"answerable": 20}, "stats": {"rejected:x": 4}, "cases": [{"id": "ans-01", "category": "answerable", "split": "dev", "question": "q?", "expected_facts": ["24"], "should_refuse": False}]})
    w("07_judge_calibration.json", {"reliability": 0.9, "by_kind": {"gold": "3/3"}})
    w("final_failures.json", [{"id": "ans-01", "category": "retrieval_miss", "why": "w", "question": "q?", "answer": "a", "notes": []}])
    w("iteration_log.json", {"baseline": {"scorecard": sc(0.5, 0.4, 0.47)}, "iterations": [{"iteration": 1, "accepted": True, "score_after": {"dev": dev, "holdout": hold, "overall": 0.9}, "changes": [{"type": "set_param", "target": "knowledge.retrieval.top_k", "before": 1, "after": 3, "why": "w", "cases": []}], "decision": "accepted"}], "stop_reason": "x"})
    w("iterations/iter_1_results.json", {"scorecard": {**sc(dev, hold, 0.9), "by_category": {"answerable": {"n": 20, "pass_rate": 0.9}}},
                                          "results": [{"id": "ans-01", "passed": True}, {"id": "ans-02", "passed": False}]})
    if with_scorecard:
        w("scorecard.json", {"baseline": {"scorecard": sc(0.5, 0.4, 0.47)}, "iterations": [], "final": {**sc(dev, hold, 0.9), "by_category": {"answerable": {"n": 20, "pass_rate": 0.9}}}})
    return d


def test_dashboard_data_reads_real_artifacts_and_survives_missing_files(tmp_path):
    from forgeai import dashboard_data as dd

    make_run(tmp_path, "with_sc")
    make_run(tmp_path, "no_sc", with_scorecard=False)  # scorecard rebuilt from iteration_log + iter results
    (tmp_path / "runs" / "empty_dir").mkdir()  # not a run
    runs = dd.list_runs([tmp_path / "runs"])
    assert {r["key"] for r in runs} == {"with_sc", "no_sc"}
    for r in runs:
        assert (r["pass"], r["dev"], r["holdout"], r["baseline"], r["iterations"], r["language_code"]) == (90, 100, 60, 47, 1, "hi")
    k = dd.kpis(runs)
    assert k["agents"] == 2 and k["avg_pass"] == 90 and k["reliability"] == 0.9 and k["languages"] == ["Hindi"] and len(k["attention"]) == 2  # 40-point dev/hold-out gap
    detail = dd.run_detail(runs[0]["path"])
    assert [x["outcome"] for x in dd.scorecard_rows(detail)] == ["Untouched", "Accepted"]
    assert any("40" in c["text"] or "gap" in c["text"] for c in dd.checks(detail))


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_with_real_run_data(tmp_path, monkeypatch, page):
    make_run(tmp_path, "demo")
    monkeypatch.setenv("FORGEAI_ROOT", str(tmp_path))
    at = AppTest.from_file(str(UI), default_timeout=60)
    at.session_state["page"] = page
    at.session_state["workspace"] = str(tmp_path / "workspace")
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]


def test_every_page_renders_with_no_runs_at_all(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGEAI_ROOT", str(tmp_path))
    for page in PAGES:
        at = AppTest.from_file(str(UI), default_timeout=60)
        at.session_state["page"] = page
        at.session_state["workspace"] = str(tmp_path / "workspace")
        at.run()
        assert not at.exception, (page, [str(e.value) for e in at.exception])


def test_navigation_buttons_switch_pages(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGEAI_ROOT", str(tmp_path))
    at = AppTest.from_file(str(UI), default_timeout=60)
    at.session_state["workspace"] = str(tmp_path / "workspace")
    at.run()
    nav = {b.key: b for b in at.sidebar.button}
    assert set(nav) == {f"nav_{p}" for p in PAGES}
    nav["nav_settings"].click().run()
    assert at.session_state["page"] == "settings" and not at.exception
    assert any("Supported languages" in m.value for m in at.markdown)


@pytest.mark.parametrize("step", range(5))
def test_every_wizard_step_renders_for_a_finished_run(tmp_path, monkeypatch, step):
    d = make_run(tmp_path, "demo")
    monkeypatch.setenv("FORGEAI_ROOT", str(tmp_path))
    at = AppTest.from_file(str(UI), default_timeout=60)
    at.session_state["page"], at.session_state["wiz"], at.session_state["wiz_run"] = "build", step, str(d)
    at.session_state["workspace"] = str(tmp_path / "workspace")
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]


def test_wizard_back_and_next_move_between_steps(tmp_path, monkeypatch):
    d = make_run(tmp_path, "demo")
    monkeypatch.setenv("FORGEAI_ROOT", str(tmp_path))
    at = AppTest.from_file(str(UI), default_timeout=60)
    at.session_state["page"], at.session_state["wiz"], at.session_state["wiz_run"] = "build", 1, str(d)
    at.session_state["workspace"] = str(tmp_path / "workspace")
    at.run()
    next(b for b in at.button if b.key == "next_1").click().run()
    assert at.session_state["wiz"] == 2 and not at.exception
    next(b for b in at.button if b.key == "back_2").click().run()
    assert at.session_state["wiz"] == 1


def test_result_helpers_count_cases_from_final_iteration(tmp_path):
    from forgeai import dashboard_data as dd

    d = make_run(tmp_path, "demo")
    assert dd.result_counts(dd.final_results(d)) == {"passed": 1, "failed": 1, "total": 2}
    assert dd.dataset_split(json.loads((d / "06_eval_set.json").read_text(encoding="utf-8"))) == {"total": 1, "answerable": 1, "out_of_scope": 0, "edge": 0}
    assert dd.final_metrics(dd.run_detail(d))["pass_rate"] == 90


def test_uikit_escapes_untrusted_text():
    from forgeai import uikit as ui

    evil = '<img src=x onerror=alert(1)>'
    for out in (ui.badge("ok", evil), ui.name_cell(evil, evil), ui.chip(evil), ui.alert("warn", evil), ui.note(True, evil), ui.stage("done", evil, evil, evil),
                ui.check_list([evil]), ui.cfg_row("db", evil, evil), ui.metric(evil, evil), ui.code_block(evil), ui.bubble("user", evil, evil), ui.iter_chain([(evil, evil, "")]),
                ui.file_row(evil, evil, ui.esc(evil), "x"), ui.header(evil, evil), ui.kpi_cards([(evil, evil, evil)]), ui.model_card(True, evil, evil), ui.ring(50, evil), ui.log_block([evil])):
        assert "<img" not in out.replace("&lt;img", "")


def test_demo_mode_is_read_only_and_shows_recorded_answers(tmp_path, monkeypatch):
    d = make_run(tmp_path, "demo")
    (d / "manual_test_answers.json").write_text(json.dumps([{"n": 1, "question": "How much?", "expected": "Rs. 799", "answer": "Rs. 799 per month.", "refused": False,
                                                              "sources": ["a.pdf"], "auto_pass": True}]), encoding="utf-8")
    monkeypatch.setenv("FORGEAI_ROOT", str(tmp_path))
    monkeypatch.setenv("FORGEAI_DEMO", "1")
    at = AppTest.from_file(str(UI), default_timeout=60)
    at.session_state["page"] = "build"
    at.session_state["workspace"] = str(tmp_path / "workspace")
    at.run()
    assert not at.exception and next(b for b in at.button if b.key == "start_build").disabled
    assert any("Demo mode" in m.value for m in at.markdown)
    at = AppTest.from_file(str(UI), default_timeout=60)
    at.session_state["page"] = "runner"
    at.session_state["workspace"] = str(tmp_path / "workspace")
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any("Recorded answers" in m.value for m in at.markdown)
    assert any(e.label.startswith("1. How much?") for e in at.expander)
