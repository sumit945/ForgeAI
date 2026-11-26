"""Typed contracts. Two families:

  *Out  - what we accept from an LLM call (lenient: coerces the usual small-model sloppiness such as a string where a
          list belongs, but refuses missing/garbage fields, so the model is re-asked with the exact validation error)
  Stage artifacts (Requirements, AgentDefinition, EvalSet) - strict: validated at every stage boundary so a malformed
          object can never flow downstream. `validate_*` raise SchemaError with a field-level message.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .errors import SchemaError
from .tools import REGISTRY


def _as_list(v: Any) -> list:
    if v is None:
        return []
    if isinstance(v, str):
        return [x.strip() for x in v.replace("\n", ";").split(";") if x.strip()] if v.strip() else []
    if isinstance(v, dict):
        return list(v.values())
    return list(v)


def _str_list(v: Any) -> list[str]:
    out = []
    for x in _as_list(v):
        if isinstance(x, dict):
            x = x.get("topic") or x.get("question") or x.get("text") or next(iter(x.values()), "")
        s = str(x).strip()
        if s:
            out.append(s)
    return out


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


# ---------------------------------------------------------------- LLM output schemas
class QuestionItem(_Lenient):
    question: str = Field(min_length=5)
    why: str = ""
    default: str = ""
    id: str = ""


class QuestionsOut(_Lenient):
    questions: list[QuestionItem] = []

    @field_validator("questions", mode="before")
    @classmethod
    def _q(cls, v):
        return [{"question": x} if isinstance(x, str) else x for x in _as_list(v)]


class RequirementsOut(_Lenient):
    title: str = ""
    purpose: str = ""
    scope: str = ""
    audience: str = ""
    tone: str = ""
    language: str = ""
    must_answer_topics: list[str] = []
    must_refuse_topics: list[str] = []
    escalation: str = ""

    _l1 = field_validator("must_answer_topics", "must_refuse_topics", mode="before")(lambda cls, v: _str_list(v))

    @model_validator(mode="after")
    def _need_something(self):
        if not (self.purpose or self.scope or self.must_answer_topics):
            raise ValueError("requirements need at least a purpose, scope or must_answer_topics")
        return self


class GroundCheckOut(_Lenient):
    quote: str = ""
    verdict: Literal["yes", "partial", "no"]

    @field_validator("verdict", mode="before")
    @classmethod
    def _v(cls, v):
        return str(v).strip().lower()


class ToolVoteOut(_Lenient):
    tools: dict[str, Any] = {}


class AnswerableOut(_Lenient):
    question: str = Field(min_length=5)
    key_facts: list[str] = Field(min_length=1)
    answer: str = ""

    _l = field_validator("key_facts", mode="before")(lambda cls, v: _str_list(v))


class QuestionListOut(_Lenient):
    questions: list[str] = []

    _l = field_validator("questions", mode="before")(lambda cls, v: _str_list(v))


class TranslationOut(_Lenient):
    text: str = Field(min_length=2)


class RuleOut(_Lenient):
    rule: str = Field(min_length=8)


class GlossaryOut(_Lenient):
    pairs: list[dict] = []

    @field_validator("pairs", mode="before")
    @classmethod
    def _p(cls, v):
        return [x for x in _as_list(v) if isinstance(x, dict)]


class JudgeFact(_Lenient):
    n: int
    stated: bool = False
    answer_quote: str = ""

    @field_validator("answer_quote", mode="before")
    @classmethod
    def _q(cls, v):
        return "" if v is None else str(v)


class JudgeOut(_Lenient):
    facts: list[JudgeFact] = []
    unsupported_claim: bool = False
    unsupported_quote: str = ""
    contradicts_facts: bool = False

    @field_validator("unsupported_quote", mode="before")
    @classmethod
    def _q(cls, v):
        return "" if v is None else str(v)


# ---------------------------------------------------------------- stage artifacts (strict)
class Requirements(BaseModel):
    model_config = ConfigDict(extra="allow")
    title: str = Field(min_length=1)
    brief: str = Field(min_length=3)
    purpose: str = Field(min_length=3)
    scope: str = Field(min_length=3)
    audience: str
    tone: str
    language: str
    must_answer_topics: list[str] = Field(min_length=1)
    must_refuse_topics: list[str]
    escalation: str
    clarifications: list[dict]
    assumptions: list[str]
    documents: list[str] = Field(min_length=1)
    known_gaps: list[str]

    @field_validator("clarifications")
    @classmethod
    def _max5(cls, v):
        if len(v) > 5:
            raise ValueError("more than 5 clarifying questions")
        return v


class Rule(BaseModel):
    id: str
    text: str = Field(min_length=8)
    origin: str


class SystemPrompt(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str = Field(min_length=20)
    rules: list[Rule] = Field(min_length=3)
    must_refuse: list[str]
    known_gaps: list[str]
    refusal_message: str = Field(min_length=10)


class Chunking(BaseModel):
    size_words: int = Field(ge=40, le=400)
    overlap_words: int = Field(ge=0, le=150)

    @model_validator(mode="after")
    def _ov(self):
        if self.overlap_words >= self.size_words:
            raise ValueError("overlap_words must be smaller than size_words")
        return self


class Retrieval(BaseModel):
    model_config = ConfigDict(extra="allow")
    method: str
    top_k: int = Field(ge=1, le=20)
    min_score: float = Field(ge=0)
    neighbors: int = Field(ge=0, le=3)
    glossary: dict[str, list[str]]


class Knowledge(BaseModel):
    documents_dir: str
    documents: list[str] = Field(min_length=1)
    chunking: Chunking
    retrieval: Retrieval


class OutputContract(BaseModel):
    model_config = ConfigDict(extra="allow")
    format: Literal["json"]
    fields: dict[str, str]
    citation_required: bool

    @field_validator("fields")
    @classmethod
    def _need(cls, v):
        for k in ("answer", "refused"):
            if k not in v:
                raise ValueError(f"output contract must define '{k}'")
        return v


class ModelCfg(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    temperature: float = Field(ge=0.0, le=1.0)


class AgentDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")
    schema_: str = Field(alias="schema")
    name: str = Field(min_length=1)
    model: ModelCfg
    system_prompt: SystemPrompt
    knowledge: Knowledge
    tools: list[str]
    output_contract: OutputContract

    @field_validator("tools")
    @classmethod
    def _tools(cls, v):
        bad = [t for t in v if t not in REGISTRY]
        if bad:
            raise ValueError(f"unknown tools: {bad}")
        return v

    @field_validator("schema_")
    @classmethod
    def _s(cls, v):
        if not v.startswith("forgeai.agent/"):
            raise ValueError("unsupported definition schema")
        return v


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    category: Literal["answerable", "ambiguous", "multi_doc", "near_miss", "off_topic"]
    question: str = Field(min_length=5)
    should_refuse: bool
    expected_facts: list[str]
    expected_sources: list[str]
    split: Literal["dev", "holdout"] = "dev"

    @model_validator(mode="after")
    def _consistent(self):
        if self.should_refuse and self.expected_facts:
            raise ValueError(f"{self.id}: a must-refuse case cannot have expected facts")
        if not self.should_refuse and not self.expected_facts:
            raise ValueError(f"{self.id}: an answerable case needs expected facts")
        return self


class EvalSet(BaseModel):
    model_config = ConfigDict(extra="allow")
    total: int
    cases: list[EvalCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self):
        ids = [c.id for c in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate eval case ids")
        if not any(c.should_refuse for c in self.cases) or not any(not c.should_refuse for c in self.cases):
            raise ValueError("eval set needs both answerable and must-refuse cases")
        return self


# ---------------------------------------------------------------- validation entry points
def _validate(model, data: dict, what: str):
    try:
        return model.model_validate(data)
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()[:8])
        raise SchemaError(f"{what} failed validation: {problems}", stage=what) from e


def validate_requirements(d: dict) -> dict:
    return _validate(Requirements, d, "requirements").model_dump()


def validate_definition(d: dict) -> dict:
    _validate(AgentDefinition, d, "agent_definition")
    return d  # keep the original dict (extra keys such as tool_selection/generated_from are preserved)


def validate_evalset(d: dict) -> dict:
    _validate(EvalSet, d, "eval_set")
    return d
