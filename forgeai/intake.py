"""Stage 1 - Intake interviewer: brief + folder inspection -> <=5 clarifying questions -> requirements document."""
from __future__ import annotations

import json
import re
from typing import Callable

from .docs import Document, inventory, inventory_for_prompt
from .errors import BudgetExceeded, ModelUnavailable
from .lang import corpus_language
from .llm import LLM
from .prompts import render
from .textutil import clean_escalation
from .schemas import QuestionsOut, RequirementsOut, validate_requirements

MAX_QUESTIONS = 5


def propose_questions(brief: str, docs: list[Document], llm: LLM) -> list[dict]:
    sysm, user = render("intake_questions", max_q=str(MAX_QUESTIONS), brief=brief, inventory=inventory_for_prompt(docs), language=corpus_language([d.text for d in docs] + [brief])["name"])
    try:
        data = llm.json(sysm, user, task="intake_questions", schema=QuestionsOut, max_tokens=600)
        qs = data.get("questions", []) if isinstance(data, dict) else data
    except (BudgetExceeded, ModelUnavailable):  # never swallow a budget stop or an outage
        raise
    except Exception:  # noqa: BLE001
        qs = []
    out, seen = [], set()
    for i, q in enumerate(qs if isinstance(qs, list) else []):
        if not isinstance(q, dict) or not str(q.get("question", "")).strip():
            continue
        key = re.sub(r"\W+", " ", q["question"].lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "id": f"q{len(out) + 1}",
                "question": str(q["question"]).strip(),
                "why": str(q.get("why", "")).strip(),
                "default": str(q.get("default", "")).strip() or "Use sensible defaults for the stated audience.",
            }
        )
        if len(out) >= MAX_QUESTIONS:  # hard cap enforced in code, not trusted to the model
            break
    return out


def collect_answers(
    questions: list[dict],
    mode: str = "auto",
    answers: dict | None = None,
    ask_fn: Callable[[str], str] | None = None,
) -> list[dict]:
    """mode: 'interactive' (CLI dialogue), 'answers' (pre-supplied dict by id or question text), 'auto' (assume defaults)."""
    ask_fn = ask_fn or input
    qa = []
    for i, q in enumerate(questions):
        ans, source = None, "assumed-default"
        if isinstance(answers, list):  # positional answers: ["first", "second", ...] ("" = take the default)
            ans = (answers[i] if i < len(answers) else "") or None
            if ans:
                source = "requester"
        elif answers:
            ans = answers.get(q["id"]) or answers.get(q["question"])
            if ans:
                source = "requester"
        if ans is None and mode == "interactive":
            r = ask_fn(f"\n[{q['id']}] {q['question']}\n    (why: {q['why']})\n    default: {q['default']}\n  > ").strip()
            if r:
                ans, source = r, "requester"
        if ans is None:
            ans = q["default"]
        qa.append({**q, "answer": ans, "source": source})
    return qa


def _as_list(v, cap=10) -> list[str]:
    if isinstance(v, str):
        v = [x.strip() for x in re.split(r"[;\n]", v) if x.strip()]
    if not isinstance(v, list):
        return []
    out = []
    for x in v:
        s = str(x.get("topic") if isinstance(x, dict) and "topic" in x else x).strip()
        if s and s.lower() not in [o.lower() for o in out]:
            out.append(s)
    return out[:cap]


def build_requirements(brief: str, docs: list[Document], qa: list[dict], llm: LLM) -> dict:
    qa_txt = "\n".join(
        f"- {q['question']} -> {q['answer']}" + ("  [ASSUMED DEFAULT]" if q["source"] != "requester" else "") for q in qa
    ) or "(no questions were needed)"
    sysm, user = render("requirements", brief=brief, qa=qa_txt, inventory=inventory_for_prompt(docs), language=corpus_language([d.text for d in docs] + [brief])["name"])
    try:
        d = llm.json(sysm, user, task="requirements", schema=RequirementsOut, max_tokens=900)
        if not isinstance(d, dict):
            raise ValueError("not an object")
    except (BudgetExceeded, ModelUnavailable):  # never swallow a budget stop or an outage
        raise
    except Exception:  # noqa: BLE001
        d = {}
    lang = corpus_language([doc.text for doc in docs] + [brief])
    language_name = str(d.get("language") or "").strip() or lang["name"]
    if lang["code"] != "en":  # the documents decide: a small model often answers "English" regardless
        language_name = lang["name"]
    heads = []
    for doc in docs:
        heads += inventory([doc])[0].get("headings", [])[:3]
    req = {
        "title": str(d.get("title") or "Document assistant")[:80],
        "brief": brief.strip(),
        "purpose": str(d.get("purpose") or brief.strip())[:300],
        "scope": str(d.get("scope") or brief.strip())[:400],
        "audience": str(d.get("audience") or "general users"),
        "tone": str(d.get("tone") or "clear, professional, concise"),
        "language": language_name,
        "language_code": lang["code"],
        "language_shares": lang["shares"],
        "must_answer_topics": _as_list(d.get("must_answer_topics")) or heads[:6] or [brief.strip()[:80]],  # never empty: the brief itself is the last resort
        "must_refuse_topics": _as_list(d.get("must_refuse_topics"), 8)
        or ["Anything not stated in the provided documents", "Legal, medical or financial advice"],
        "escalation": clean_escalation(str(d.get("escalation") or "")),
        "clarifications": [{"question": q["question"], "answer": q["answer"], "source": q["source"]} for q in qa],
        "assumptions": [f"{q['question']} -> assumed: {q['answer']}" for q in qa if q["source"] != "requester"],
        "documents": [x.name for x in docs],
        "known_gaps": [],
    }
    return validate_requirements(req)


def requirements_markdown(req: dict) -> str:
    def bl(xs):
        return "\n".join(f"- {x}" for x in xs) or "- (none)"

    return f"""# Requirements - {req['title']}

**Brief:** {req['brief']}

| | |
|---|---|
| Purpose | {req['purpose']} |
| Scope | {req['scope']} |
| Audience | {req['audience']} |
| Tone | {req['tone']} |
| Language | {req['language']} |
| Escalation | {req['escalation']} |

## Must answer
{bl(req['must_answer_topics'])}

## Must refuse
{bl(req['must_refuse_topics'])}

## Known gaps (documents cannot support these - flagged by the groundability guard)
{bl(req.get('known_gaps', []))}

## Clarifications
{bl([f"{c['question']} -> {c['answer']} ({c['source']})" for c in req['clarifications']])}

## Assumptions made because nobody answered
{bl(req['assumptions'])}

## Documents
{bl(req['documents'])}
"""


def dumps(o) -> str:
    return json.dumps(o, indent=2, ensure_ascii=False)
