"""LLM access layer: Ollama (default) or any OpenAI-compatible endpoint - built to fail loudly and recover quietly.

Reliability features
  * retries with exponential backoff on connection errors, timeouts and HTTP 5xx (Ollama returns 500 on transient OOM)
  * ordered fallback models (`fallback_models`) after the primary exhausts its retries
  * circuit breaker: N consecutive failed calls -> ModelUnavailable (fail fast instead of hanging for hours)
  * preflight(): is the server up, is the model installed? clear, actionable message if not
  * prompt guard: over-long prompts are shrunk (middle cut) instead of being silently truncated by the server
  * json(schema=...): output is parsed AND validated; the model is re-asked with the exact validation error
  * every call is traced (JSONL) with the task tag, so any decision can be audited
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests

from .errors import BudgetExceeded, ModelOutputError, ModelUnavailable

__all__ = ["LLM", "ScriptedLLM", "Budget", "BudgetExceeded", "extract_json"]


@dataclass
class Budget:
    max_calls: int | None = None
    max_seconds: float | None = None
    calls: int = 0
    started: float = field(default_factory=time.time)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0
    failed_calls: int = 0
    fallbacks_used: int = 0
    truncated_prompts: int = 0

    def check(self) -> None:
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise BudgetExceeded(f"LLM call budget exhausted ({self.max_calls})")
        if self.max_seconds is not None and time.time() - self.started > self.max_seconds:
            raise BudgetExceeded(f"time budget exhausted ({self.max_seconds}s)")

    def summary(self) -> dict:
        return {
            "calls": self.calls, "seconds": round(time.time() - self.started, 1),
            "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
            "retries": self.retries, "failed_calls": self.failed_calls,
            "fallbacks_used": self.fallbacks_used, "truncated_prompts": self.truncated_prompts,
        }


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in `text`, tolerating fences, trailing commas and truncated output."""
    if text is None:
        raise ValueError("empty response")
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    try:
        return json.loads(t)
    except Exception:  # noqa: BLE001
        pass
    start = None
    for i, ch in enumerate(t):
        if ch in "{[":
            start = i
            break
    if start is None:
        raise ValueError("no JSON found")
    opener = t[start]
    closer = "}" if opener == "{" else "]"
    depth, in_str, esc = 0, False, False
    for j in range(start, len(t)):
        c = t[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                frag = re.sub(r",\s*([}\]])", r"\1", t[start : j + 1])
                return json.loads(frag)
    frag = t[start:]  # truncated output: close what is open
    stack, in_str, esc = [], False, False
    for c in frag:
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            stack.append("}" if c == "{" else "]")
        elif c in "}]" and stack:
            stack.pop()
    if in_str:
        frag += '"'
    frag = re.sub(r"[,:]\s*$", "", frag.rstrip())
    frag += "".join(reversed(stack))
    return json.loads(re.sub(r",\s*([}\]])", r"\1", frag))


def _approx_tokens(s: str) -> int:
    return int(len(s) / 3.3) + 4


class _Transient(Exception):
    """Internal: a failure worth retrying."""


class LLM:
    def __init__(
        self,
        model: str = "qwen2.5:3b-instruct",
        host: str = "http://localhost:11434",
        backend: str = "ollama",
        api_key: str | None = None,
        temperature: float = 0.2,
        seed: int = 7,
        num_ctx: int = 4096,
        timeout: int = 600,
        trace_path: str | Path | None = None,
        budget: Budget | None = None,
        retries: int = 3,
        backoff: float = 2.0,
        fallback_models: list[str] | None = None,
        breaker_threshold: int = 6,
    ):
        self.model, self.host, self.backend = model, host.rstrip("/"), backend
        self.api_key = api_key
        self.temperature, self.seed, self.num_ctx, self.timeout = temperature, seed, num_ctx, timeout
        self.trace_path = Path(trace_path) if trace_path else None
        self.budget = budget or Budget()
        self.retries, self.backoff = retries, backoff
        self.fallback_models = list(fallback_models or [])
        self.breaker_threshold = breaker_threshold
        self._consecutive_failures = 0
        self._lock = threading.Lock()  # counters and the trace file are shared by parallel calls

    # -- tracing ---------------------------------------------------------------------
    def _trace(self, rec: dict) -> None:
        if not self.trace_path:
            return
        try:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, open(self.trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass  # tracing must never break a run

    # -- health ----------------------------------------------------------------------
    def installed_models(self) -> list[str]:
        r = requests.get(f"{self.host}/api/tags", timeout=10)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    def preflight(self) -> dict:
        """Raise ModelUnavailable with an actionable message unless every configured model is reachable."""
        if self.backend != "ollama":
            return {"ok": True, "backend": self.backend}
        try:
            names = self.installed_models()
        except Exception as e:  # noqa: BLE001
            raise ModelUnavailable(f"Cannot reach Ollama at {self.host} ({type(e).__name__}).") from e
        missing = [m for m in [self.model, *self.fallback_models] if not self._has(names, m)]
        if missing:
            raise ModelUnavailable(
                f"Model(s) not installed: {', '.join(missing)}. Installed: {', '.join(names) or 'none'}.",
                hint=f"Run: ollama pull {missing[0]}",
            )
        return {"ok": True, "models": names}

    @staticmethod
    def _has(names: list[str], model: str) -> bool:
        return model in names or (":" not in model and any(n.split(":")[0] == model for n in names))

    # -- prompt guard ----------------------------------------------------------------
    def _fit(self, messages: list[dict], max_tokens: int) -> list[dict]:
        budget = self.num_ctx - max_tokens - 64
        total = sum(_approx_tokens(m["content"]) for m in messages)
        if total <= budget:
            return messages
        msgs = [dict(m) for m in messages]
        with self._lock:
            self.budget.truncated_prompts += 1
        while total > budget:
            i = max(range(len(msgs)), key=lambda k: len(msgs[k]["content"]))
            c = msgs[i]["content"]
            keep = max(400, int(len(c) * 0.75))
            if keep >= len(c):
                break
            cut = len(c) - keep
            head = c[: keep // 2]
            tail = c[len(c) - (keep - keep // 2) :]
            msgs[i]["content"] = head + f"\n[... {cut} characters omitted to fit the context window ...]\n" + tail
            total = sum(_approx_tokens(m["content"]) for m in msgs)
        return msgs

    # -- one HTTP call ---------------------------------------------------------------
    def _call(self, model: str, messages: list[dict], json_mode: bool, temp: float, max_tokens: int) -> tuple[str, int, int]:
        try:
            if self.backend == "ollama":
                payload = {
                    "model": model, "messages": messages, "stream": False,
                    "options": {"temperature": temp, "seed": self.seed, "num_predict": max_tokens, "num_ctx": self.num_ctx},
                }
                if json_mode:
                    payload["format"] = "json"
                r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
            else:
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                payload = {"model": model, "messages": messages, "temperature": temp, "max_tokens": max_tokens, "seed": self.seed}
                if json_mode:
                    payload["response_format"] = {"type": "json_object"}
                r = requests.post(f"{self.host}/v1/chat/completions", json=payload, headers=headers, timeout=self.timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            raise _Transient(f"{type(e).__name__}: {e}") from e
        if r.status_code == 404:
            raise ModelUnavailable(f"Model '{model}' not found on {self.host}.", hint=f"Run: ollama pull {model}")
        if r.status_code >= 500 or r.status_code == 429:
            raise _Transient(f"HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code >= 400:
            raise ModelOutputError(f"HTTP {r.status_code} from model server: {r.text[:300]}")
        try:
            data = r.json()
            if self.backend == "ollama":
                return data["message"]["content"], data.get("prompt_eval_count", 0), data.get("eval_count", 0)
            u = data.get("usage", {})
            return data["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise _Transient(f"malformed server response: {e}") from e

    # -- public chat -----------------------------------------------------------------
    def chat(self, messages: list[dict], task: str = "chat", json_mode: bool = False, temperature: float | None = None, max_tokens: int = 700) -> str:
        self.budget.check()
        temp = self.temperature if temperature is None else temperature
        msgs = self._fit(messages, max_tokens)
        t0 = time.time()
        last_err = "unknown"
        models = [self.model, *self.fallback_models]
        for mi, model in enumerate(models):
            for attempt in range(self.retries + 1):
                try:
                    out, pt, ct = self._call(model, msgs, json_mode, temp, max_tokens)
                except _Transient as e:
                    last_err = str(e)
                    with self._lock:
                        self.budget.retries += 1
                    if attempt < self.retries:
                        time.sleep(self.backoff * (2**attempt))
                    continue
                with self._lock:
                    self._consecutive_failures = 0
                    if mi > 0:
                        self.budget.fallbacks_used += 1
                    self.budget.prompt_tokens += pt
                    self.budget.completion_tokens += ct
                    self.budget.calls += 1
                self._trace({"task": task, "model": model, "seconds": round(time.time() - t0, 2), "messages": msgs, "response": out})
                return out
        with self._lock:
            self.budget.failed_calls += 1
            self._consecutive_failures += 1
            tripped = self._consecutive_failures >= self.breaker_threshold
        self._trace({"task": task, "model": self.model, "error": last_err})
        if tripped:
            raise ModelUnavailable(f"Circuit breaker open: {self._consecutive_failures} consecutive model failures (last: {last_err}).")
        raise ModelOutputError(f"{task}: model call failed after retries ({last_err}).")

    # -- JSON helper -----------------------------------------------------------------
    def json(
        self,
        system: str,
        user: str,
        task: str = "json",
        retries: int = 2,
        temperature: float | None = None,
        max_tokens: int = 700,
        schema: Any = None,
    ) -> Any:
        """Ask for JSON, parse it, optionally validate it with a pydantic model, and re-ask with the exact error."""
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_err: Exception | None = None
        for _ in range(retries + 1):
            raw = self.chat(msgs, task=task, json_mode=True, temperature=temperature, max_tokens=max_tokens)
            try:
                data = extract_json(raw)
                if schema is not None:
                    data = schema.model_validate(data).model_dump()
                return data
            except Exception as e:  # noqa: BLE001
                last_err = e
                why = "That was not valid JSON." if not isinstance(e, ValueError) or "JSON" in str(e)[:40] else "That JSON did not match the required format."
                detail = str(e)[:300] if schema is not None else ""
                msgs = msgs + [
                    {"role": "assistant", "content": raw[:1500]},
                    {"role": "user", "content": f"{why} {detail}\nReply again with ONLY the corrected JSON object."},
                ]
        raise ModelOutputError(f"{task}: no valid JSON after {retries + 1} attempts ({last_err})")

    def alive(self) -> bool:
        try:
            self.preflight()
            return True
        except ModelUnavailable:
            return False


class ScriptedLLM(LLM):
    """Deterministic stand-in for tests: handler(task, system, user) -> str."""

    def __init__(self, handler: Callable[[str, str, str], str], **kw):
        kw.setdefault("backoff", 0.0)
        super().__init__(**kw)
        self.handler = handler

    def chat(self, messages, task="chat", json_mode=False, temperature=None, max_tokens=700):  # noqa: D401
        self.budget.check()
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        out = self.handler(task, system, user)
        self.budget.calls += 1
        self._trace({"task": task, "model": "scripted", "messages": messages, "response": out})
        return out

    def preflight(self) -> dict:
        return {"ok": True, "backend": "scripted"}
