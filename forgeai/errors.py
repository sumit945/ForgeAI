"""Error taxonomy. Every failure in ForgeAI is one of these, carries a human-readable fix hint, and maps to an exit code.

    InputError        - the requester can fix it (no readable documents, empty brief)          exit 3
    ModelUnavailable  - the LLM backend is down / model not installed / circuit breaker open    exit 4
    ModelOutputError  - the LLM answered but the output is unusable after retries               (stage degrades)
    SchemaError       - a stage produced data that violates its contract                        (stage degrades or fails)
    BudgetExceeded    - call / time budget used up                                              (run finishes early, flagged)
    Blocked           - the groundability guard refused to generate an agent                     exit 2
    ForgeError        - anything else we raised on purpose                                       exit 1
"""
from __future__ import annotations


class ForgeError(Exception):
    exit_code = 1
    hint = ""

    def __init__(self, message: str, hint: str | None = None, stage: str | None = None):
        super().__init__(message)
        self.hint = hint or self.hint
        self.stage = stage

    def to_dict(self) -> dict:
        return {"type": type(self).__name__, "message": str(self), "hint": self.hint, "stage": self.stage}


class InputError(ForgeError):
    exit_code = 3
    hint = "Check the brief and the document folder, then run again."


class ModelUnavailable(ForgeError):
    exit_code = 4
    hint = "Start Ollama (`ollama serve`) and install the model (`ollama pull <model>`), or pass --host/--backend for another endpoint."


class ModelOutputError(ForgeError):
    exit_code = 1
    hint = "The model returned unusable output repeatedly; try a larger model or lower --max-iters."


class SchemaError(ForgeError):
    exit_code = 1
    hint = "A stage produced data that failed validation; see run_status.json for the field list."


class BudgetExceeded(ForgeError):
    exit_code = 5
    hint = "The configured call/time budget was reached before the run could finish; raise --max-calls/--max-minutes or use --resume."


class Blocked(ForgeError):
    exit_code = 2
    hint = "The documents cannot support this brief. Add the missing documents or narrow the brief."
