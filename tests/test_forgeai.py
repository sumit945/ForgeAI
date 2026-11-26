"""Offline tests. A scripted 'LLM' stands in for Ollama so the whole pipeline can be exercised without a model."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from forgeai.docs import chunk_documents, load_folder  # noqa: E402
from forgeai.generator import generate_agent, select_tools  # noqa: E402
from forgeai.groundability import check_topic, groundability_guard  # noqa: E402
from forgeai.judge import calibrate, judge_case  # noqa: E402
from forgeai.llm import ScriptedLLM, extract_json  # noqa: E402
from forgeai.retrieval import Retriever, tokenize  # noqa: E402
from forgeai.tools import calculator, run_tool  # noqa: E402

HR = ROOT / "demos" / "hr_policy" / "docs"
SHIP = ROOT / "demos" / "ship_spares" / "docs"


def sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.replace("\n", " ")) if len(s.split()) > 4]


def scripted(task, system, user):
    if task == "intake_questions":
        return json.dumps({"questions": [{"id": f"q{i}", "question": f"Question number {i} about audience?", "why": "w", "default": "d"} for i in range(8)]})
    if task == "requirements" and "Spares" in user:
        return json.dumps({"title": "Spares Helper", "purpose": "Look up spare parts and supplier terms for superintendents", "scope": "catalogue, suppliers, approvals",
                           "audience": "superintendents", "tone": "precise", "language": "English",
                           "must_answer_topics": ["catalogue part prices", "supplier payment terms", "purchase order approval limits"],
                           "must_refuse_topics": ["legal advice"], "escalation": "Fleet Procurement"})
    if task == "requirements":
        return json.dumps({"title": "HR Helper", "purpose": "Answer HR policy questions for shore staff", "scope": "leave, hours, conduct",
                           "audience": "shore staff", "tone": "friendly", "language": "English",
                           "must_answer_topics": ["annual leave entitlement", "overtime pay rates", "grievance procedure steps", "pension contribution rates"],
                           "must_refuse_topics": ["legal advice"], "escalation": "HR"})
    if task == "groundability_check":
        topic = re.search(r"TOPIC: (.*)", user).group(1)
        ex = user.split("EXCERPTS:")[1]
        toks = [t for t in tokenize(topic) if len(t) > 2]
        have = sum(1 for t in toks if t in set(tokenize(ex)))
        if toks and have / len(toks) >= 0.6:
            return json.dumps({"verdict": "yes", "quote": sentences(ex)[0]})
        return json.dumps({"verdict": "no", "quote": ""})
    if task == "tool_vote":
        return json.dumps({"tools": {n: {"needed": True, "reason": "sure"} for n in ("csv_lookup", "calculator", "open_meteo_weather", "current_date")}})
    if task == "agent_role":
        return json.dumps({"role": "You are an HR helper for shore staff. You answer questions about HR policies in a friendly tone."})
    if task == "evalgen_answerable":
        passage = user.split("):", 1)[1].split("QUESTIONS ALREADY")[0]
        avoid = user.split("QUESTIONS ALREADY")[1]
        for s in sentences(passage):
            m = re.search(r"\d[\d,.]*(?: [a-zA-Z%]+)?", s)
            if not m:
                continue
            words = s.split()
            q = f"What does the policy say about {' '.join(words[:6])}?"
            if m.group(0) in q or q in avoid:
                continue
            return json.dumps({"question": q, "key_facts": [m.group(0)], "answer": s})
        return "{}"
    if task == "evalgen_paraphrase":
        q = re.search(r"PLAIN QUESTION: (.*)", user).group(1)
        return json.dumps({"question": "hey quick one, " + q.lower().replace("what is the rule about", "how does it work for")})
    if task == "evalgen_multidoc":
        a = user.split("PASSAGE A")[1].split("PASSAGE B")[0]
        b = user.split("PASSAGE B")[1]
        sa, sb = sentences(a.split("):", 1)[1])[0], sentences(b.split("):", 1)[1])[0]
        return json.dumps({"question": "How do these two policy points combine together?", "fact_from_a": sa, "fact_from_b": sb, "answer": sa + " " + sb})
    if task == "evalgen_nearmiss":
        return json.dumps({"questions": ["What is the company dental insurance premium for dependants?", "What is the monthly gym membership subsidy amount?"]})
    if task == "evalgen_offtopic":
        return json.dumps({"questions": ["Write python code to sort a list.", "Should I take ibuprofen for a headache?", "Who won the 2018 world cup?"]})
    if task == "agent_answer":
        ctx = user.split("CONTEXT:")[1].split("QUESTION:")[0]
        q = user.split("QUESTION:")[1].split("Reply with")[0]
        qt = set(tokenize(q))
        best, score, doc = "", 0, ""
        for block in ctx.split("\n\n["):
            block = block.lstrip("[")
            m = re.match(r"([^\]]+)\]\n(.*)", block, re.S)
            if not m:
                continue
            for s in sentences(m.group(2)):
                sc = len(qt & set(tokenize(s)))
                if sc > score:
                    best, score, doc = s, sc, m.group(1)
        if score < 2:
            return json.dumps({"answer": "I can't answer that from the documents I have.", "sources": [], "evidence": [], "refused": True})
        return json.dumps({"answer": best, "sources": [doc], "evidence": [best], "refused": False})
    if task == "judge_facts":
        ans = user.split("ANSWER:")[1].split("EXPECTED FACTS:")[0].strip()
        facts = [ln.split(". ", 1)[1] for ln in user.split("EXPECTED FACTS:")[1].strip().split("\n") if ". " in ln]
        out = [{"n": i + 1, "stated": f.lower() in ans.lower(), "answer_quote": f if f.lower() in ans.lower() else ""} for i, f in enumerate(facts)]
        return json.dumps({"facts": out, "unsupported_claim": False, "unsupported_quote": "", "contradicts_facts": False})
    if task == "fix_prompt_rule":
        return json.dumps({"rule": "Always quote the exact numbers from the context and cite the source file."})
    if task == "fix_glossary":
        return json.dumps({"pairs": []})
    return "{}"


@pytest.fixture()
def llm(tmp_path):
    return ScriptedLLM(scripted, model="scripted", trace_path=tmp_path / "t.jsonl")


# ---------------------------------------------------------------- unit tests
def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1,}\n```') == {"a": 1}
    assert extract_json('noise {"a": [1, 2]} tail') == {"a": [1, 2]}
    assert extract_json('{"a": "x", "b": [1,')["a"] == "x"


def test_calculator_is_safe():
    assert calculator({"expression": "12 * 4.5 + 3"}, {})["result"] == 57.0
    assert "error" in calculator({"expression": "__import__('os').system('echo hi')"}, {})
    assert "error" in calculator({"expression": "9**99999"}, {})


def test_pdf_loading_and_retrieval():
    docs, problems = load_folder(HR)
    assert not problems and len(docs) == 3
    retr = Retriever(chunk_documents(docs, 140, 30))
    top = retr.search("how many days of annual leave do I get", 3)
    assert top and top[0][0].doc == "leave_policy.pdf"
    assert not retr.search("capital of australia", 3)  # no lexical overlap -> nothing retrieved


def test_csv_lookup_tool():
    docs, _ = load_folder(SHIP)
    r = run_tool("csv_lookup", {"query": "turbocharger cartridge"}, {"documents": docs})
    assert r["matches"][0]["row"]["part_no"] == "MP-1003"


def test_tools_need_llm_vote_and_evidence(llm):
    docs, _ = load_folder(HR)
    req = {"brief": "HR policy assistant for shore staff", "purpose": "answer HR policy questions", "scope": "leave and conduct", "must_answer_topics": ["annual leave", "grievance steps"]}
    sel = select_tools(req, docs, llm)  # scripted LLM votes YES for every tool
    assert sel["attached"] == [], sel  # ...but no evidence in the brief and no csv -> nothing attached
    docs2, _ = load_folder(SHIP)
    req2 = {"brief": "Spares assistant: look up part prices in the catalogue and calculate the total order cost", "purpose": "spares", "scope": "x", "must_answer_topics": ["part prices"]}
    sel2 = select_tools(req2, docs2, llm)
    assert "csv_lookup" in sel2["attached"] and "calculator" in sel2["attached"]
    assert "open_meteo_weather" not in sel2["attached"]


def test_groundability_flags_unsupported_and_checks_quotes(llm):
    docs, _ = load_folder(HR)
    req = {"purpose": "Answer HR policy questions", "must_answer_topics": ["annual leave entitlement", "pension contribution rates", "overtime pay rates"], "must_refuse_topics": []}
    rep, narrowed = groundability_guard(req, docs, llm)
    verdicts = {t["topic"]: t["verdict"] for t in rep["topics"]}
    assert verdicts["pension contribution rates"] == "unsupported"
    assert verdicts["annual leave entitlement"] in ("supported", "partial")
    assert "pension contribution rates" in narrowed["known_gaps"]
    assert "pension contribution rates" not in narrowed["must_answer_topics"]


def test_groundability_blocks_when_nothing_is_supported(llm):
    docs, _ = load_folder(HR)
    req = {"purpose": "Advise on cryptocurrency staking yields", "must_answer_topics": ["staking yields", "validator slashing", "token vesting"], "must_refuse_topics": []}
    rep, _ = groundability_guard(req, docs, llm)
    assert rep["status"] == "blocked"


def test_llm_lie_about_support_is_caught():
    docs, _ = load_folder(HR)
    retr = Retriever(chunk_documents(docs, 120, 30))
    liar = ScriptedLLM(lambda t, s, u: json.dumps({"verdict": "yes", "quote": "Employees receive free dental cover for their whole family."}))
    r = check_topic("free dental cover for family members", retr, liar)
    assert r["quote_verified"] is False and r["verdict"] != "supported"


def test_judge_deterministic_paths():
    case = {"id": "t1", "category": "answerable", "question": "How many annual leave days?", "should_refuse": False,
            "expected_facts": ["24 working days"], "expected_sources": ["leave_policy.pdf"]}
    ctx = [{"id": "leave_policy.pdf#0", "doc": "leave_policy.pdf", "text": "Staff are entitled to 24 working days of paid annual leave."}]
    good = {"answer": "You get 24 working days of paid annual leave.", "sources": ["leave_policy.pdf"], "evidence": ["Staff are entitled to 24 working days of paid annual leave."], "refused": False, "retrieved": ctx, "tool_calls": []}
    assert judge_case(case, good, None)["passed"]
    wrong_number = {**good, "answer": "You get 25 working days of paid annual leave."}
    j = judge_case(case, wrong_number, None)
    assert not j["passed"] and not j["correct"]
    invented = {**good, "answer": "You get 24 working days, plus 6 bonus days on your birthday."}
    j = judge_case(case, invented, None)
    assert not j["grounded"] and not j["passed"]
    refusal_case = {"id": "t2", "category": "off_topic", "question": "capital of Australia?", "should_refuse": True, "expected_facts": [], "expected_sources": []}
    assert judge_case(refusal_case, {"answer": "Canberra.", "refused": False, "retrieved": [], "tool_calls": []}, None)["passed"] is False
    assert judge_case(refusal_case, {"answer": "I can't answer that.", "refused": True, "retrieved": [], "tool_calls": []}, None)["passed"] is True


def test_llm_judge_cannot_invent_a_quote(llm):
    case = {"id": "t1", "category": "answerable", "question": "How many days?", "should_refuse": False, "expected_facts": ["24 working days"], "expected_sources": []}
    res = {"answer": "Plenty of days.", "sources": ["x"], "evidence": [], "refused": False, "retrieved": [{"text": "24 working days"}], "tool_calls": []}
    cheat = ScriptedLLM(lambda t, s, u: json.dumps({"facts": [{"n": 1, "stated": True, "answer_quote": "24 working days"}], "unsupported_claim": False, "contradicts_facts": False}))
    assert judge_case(case, res, cheat)["correct"] is False  # the 'quote' is not in the answer -> verdict discarded


# ---------------------------------------------------------------- end-to-end (offline)
@pytest.mark.parametrize("demo,brief", [
    ("hr_policy", "I need an assistant that answers HR policy questions for shore staff."),
    ("ship_spares", "Spares assistant for superintendents: look up catalogue prices and calculate the total order cost and approvals."),
])
def test_full_pipeline_offline(tmp_path, demo, brief):
    from forgeai.factory import run_factory

    out = tmp_path / demo
    res = run_factory(brief, ROOT / "demos" / demo / "docs", out, llm=ScriptedLLM(scripted, model="scripted", trace_path=out / "t.jsonl"), log=lambda *_: None)
    for f in ["03_requirements.md", "04_groundability.json", "05_agent_definition.v0.json", "06_eval_set.json", "07_judge_calibration.json",
              "iteration_log.json", "scorecard.md", "agent_definition.json", "REPORT.md"]:
        assert (out / f).exists(), f
    ev = json.loads((out / "06_eval_set.json").read_text(encoding="utf-8"))
    assert ev["total"] >= (20 if demo == "hr_policy" else 15)  # scripted stand-in cannot write CSV questions well
    assert (ev["total"] >= 20) == (not ev["warnings"])
    assert {"answerable", "off_topic", "near_miss"} <= set(ev["counts"])
    qa = json.loads((out / "02_questions_and_answers.json").read_text(encoding="utf-8"))
    assert qa["asked"] <= 5
    assert len(json.loads((out / "iteration_log.json").read_text(encoding="utf-8"))["iterations"]) <= 3
    # the produced definition must run in the generic runner
    from forgeai.runner import Agent

    agent = Agent.load(out / "agent_definition.json", ScriptedLLM(scripted, model="scripted"))
    r = agent.answer("how many days of sick leave do employees get?" if demo == "hr_policy" else "what are the approval limits for purchase orders?")
    assert r["answer"]


def test_short_exact_quotes_are_accepted_but_fuzzy_short_ones_are_not():
    from forgeai.textutil import quote_in

    assert quote_in("48 hours", "The officer must check the delivery within 48 hours of arrival.")
    assert not quote_in("72 hours", "The officer must check the delivery within 48 hours of arrival.")
    assert not quote_in("overdue hours", "delivery within 48 hours of arrival")  # short AND not verbatim -> rejected


# ---------------------------------------------------------------- hybrid retrieval
def test_hybrid_retrieval_keeps_off_topic_empty_and_still_finds_answers():
    from forgeai.embeddings import HashingEmbedder
    from forgeai.retrieval import HybridRetriever

    docs, _ = load_folder(HR)
    chunks = chunk_documents(docs, 100, 20)
    hy = HybridRetriever(chunks, HashingEmbedder())
    for q in ("what is the capital of australia", "Write me a recipe for pasta", "Who won the football world cup?"):
        assert not hy.search(q, 4), q  # calibrated floor: unrelated questions retrieve nothing -> deterministic refusal
    top = hy.search("how many days of annual leave do I get", 3)
    assert top and top[0][0].doc == "leave_policy.pdf"


def test_agent_degrades_to_bm25_when_embedder_is_down():
    from forgeai.errors import ModelUnavailable
    from forgeai.runner import Agent

    class Dead:
        floor = 0.1

        def embed(self, texts):
            raise ModelUnavailable("embedding server down")

    docs, _ = load_folder(HR)
    defn = generate_agent({"brief": "HR bot", "purpose": "HR", "scope": "HR", "audience": "staff", "tone": "kind", "language": "English",
                           "title": "HR Bot", "must_answer_topics": ["leave"], "must_refuse_topics": [], "escalation": "HR", "known_gaps": []},
                          docs, None, embed_model="whatever")
    a = Agent(defn, ScriptedLLM(scripted), docs=docs, embedder=Dead())
    assert "BM25 only" in a.retrieval_note and a.retr.search("annual leave days", 2)


def test_strong_evidence_attaches_tool_even_if_model_votes_no():
    docs, _ = load_folder(SHIP)
    no_votes = ScriptedLLM(lambda t, s, u: json.dumps({"tools": {n: {"needed": False, "reason": "no"} for n in ("csv_lookup", "calculator", "open_meteo_weather", "current_date")}}))
    req = {"brief": "Look up parts in the spares catalogue and calculate the total order cost.", "purpose": "spares", "scope": "x", "must_answer_topics": ["stock levels"]}
    sel = select_tools(req, docs, no_votes)
    assert "csv_lookup" in sel["attached"] and "calculator" in sel["attached"]
    assert "open_meteo_weather" not in sel["attached"] and "current_date" not in sel["attached"]


def test_escalation_is_reduced_to_a_contact():
    from forgeai.textutil import clean_escalation

    assert clean_escalation("Point users to HR for any requests that cannot be answered within the documents.") == "HR"
    assert clean_escalation("Contact the HR helpdesk") == "the HR helpdesk"
    assert clean_escalation("") == "the relevant team"
    assert clean_escalation("a very long rambling set of words going on and on and on forever") == "the relevant team"


def test_judge_recognises_a_refusal_written_in_other_words():
    from forgeai.judge import agent_refused

    assert agent_refused({"refused": False, "answer": "The documents do not cover pension contributions. Please contact HR."})
    assert not agent_refused({"refused": False, "answer": "Employees receive 24 working days of paid annual leave per calendar year."})


def test_compound_questions_get_passages_for_every_part():
    from forgeai.runner import Agent, split_question

    q = "How many days of bereavement leave do employees receive? Also, how long is a written warning valid for?"
    assert len(split_question(q)) == 2 and split_question("How many leave days?") == ["How many leave days?"]
    docs, _ = load_folder(HR)
    defn = generate_agent({"brief": "HR bot", "purpose": "HR", "scope": "HR", "audience": "staff", "tone": "kind", "language": "English",
                           "title": "HR Bot", "must_answer_topics": ["leave"], "must_refuse_topics": [], "escalation": "HR", "known_gaps": []}, docs, None)
    got = {c.doc for c, _ in Agent(defn, ScriptedLLM(scripted), docs=docs).retrieve(q)}
    assert {"leave_policy.pdf", "conduct_grievance_and_travel.pdf"} <= got
