"""REST backend (Starlette) + job manager.

  POST /jobs                     {brief, model?, answers?, ...}  -> create a job (status "created")
  PUT  /jobs/{id}/files/{name}   raw bytes of one document        -> upload (validated, size-limited, sanitised name)
  POST /jobs/{id}/start          {resume?: bool}                  -> queue the factory run (one run at a time: single GPU)
  GET  /jobs, /jobs/{id}         status, log tail, run_status, errors
  GET  /jobs/{id}/report         REPORT.md       GET /jobs/{id}/artifacts/{path}   any file of the run folder
  POST /jobs/{id}/chat           {question, history?} -> answer from the generated agent
  GET  /health                   model server preflight

Auth: set FORGEAI_API_KEY and send it as `X-API-Key`. Jobs survive restarts (job.json); a job that was running when the
server died comes back as "interrupted" and can be restarted with {"resume": true}.
"""
from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.routing import Route

from .docs import SUPPORTED
from .errors import Blocked, ForgeError, InputError
from .factory import run_factory
from .fsutil import atomic_write
from .llm import LLM
from .runner import Agent

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_FILES_PER_JOB = 100
MAX_BRIEF_CHARS = 4000
RUN_KWARGS = {"judge_model", "host", "backend", "max_iters", "target", "max_calls", "max_minutes", "embed_model", "force", "ocr", "ocr_langs", "fresh_evalset"}
ALLOWED_PARAMS = {"model", "judge_model", "host", "backend", "max_iters", "target", "max_calls", "max_minutes", "embed_model", "force", "ocr", "ocr_langs", "fresh_evalset"}


def safe_name(name: str) -> str:
    base = Path(name.replace("\\", "/")).name
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    if not base or base.startswith("."):
        raise InputError("Invalid file name.")
    if Path(base).suffix.lower() not in SUPPORTED:
        raise InputError(f"Unsupported file type '{Path(base).suffix}'. Allowed: {', '.join(sorted(SUPPORTED))}.")
    return base


class JobManager:
    def __init__(self, workspace: str | Path, llm_factory: Callable[[dict], LLM] | None = None):
        self.root = Path(workspace)
        (self.root / "jobs").mkdir(parents=True, exist_ok=True)
        self.llm_factory = llm_factory or (lambda p: LLM(model=p.get("model", "qwen2.5:3b-instruct"), host=p.get("host", "http://localhost:11434")))
        self.lock = threading.RLock()
        self.q: queue.Queue = queue.Queue()
        self.jobs: dict[str, dict] = {}
        self.agents: dict[str, Agent] = {}
        for jf in sorted((self.root / "jobs").glob("*/job.json")):
            try:
                j = json.loads(jf.read_text(encoding="utf-8"))
                if j["status"] in ("running", "queued"):
                    j["status"] = "interrupted"
                    j["log"].append("server restarted while this job was running; start it again with resume=true")
                self.jobs[j["id"]] = j
            except Exception:  # noqa: BLE001
                continue
        threading.Thread(target=self._worker, daemon=True, name="forgeai-worker").start()

    # -- persistence -----------------------------------------------------------------
    def _dir(self, jid: str) -> Path:
        return self.root / "jobs" / jid

    def _save(self, j: dict) -> None:
        with self.lock:
            atomic_write(self._dir(j["id"]) / "job.json", j)

    def _get(self, jid: str) -> dict:
        if not re.fullmatch(r"[0-9a-f]{12}", jid or "") or jid not in self.jobs:
            raise KeyError(jid)
        return self.jobs[jid]

    # -- API -------------------------------------------------------------------------
    def create(self, brief: str, params: dict | None = None) -> dict:
        brief = (brief or "").strip()
        if len(brief.split()) < 4:
            raise InputError("The brief is too short (write at least a sentence).")
        if len(brief) > MAX_BRIEF_CHARS:
            raise InputError(f"The brief is too long (max {MAX_BRIEF_CHARS} characters).")
        params = {k: v for k, v in (params or {}).items() if k in ALLOWED_PARAMS}
        jid = uuid.uuid4().hex[:12]
        j = {"id": jid, "brief": brief, "params": params, "status": "created", "created": time.time(), "log": [], "error": None, "files": []}
        (self._dir(jid) / "docs").mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.jobs[jid] = j
            self._save(j)
        return j

    def add_file(self, jid: str, name: str, data: bytes) -> dict:
        j = self._get(jid)
        if j["status"] not in ("created", "interrupted", "failed", "blocked"):
            raise InputError("Files can only be added before a run starts.")
        if len(data) > MAX_UPLOAD_BYTES:
            raise InputError(f"File too large (max {MAX_UPLOAD_BYTES // 1_048_576} MB).")
        if not data:
            raise InputError("Empty upload.")
        name = safe_name(name)
        with self.lock:
            if name not in j["files"] and len(j["files"]) >= MAX_FILES_PER_JOB:
                raise InputError(f"Too many files (max {MAX_FILES_PER_JOB}).")
            (self._dir(jid) / "docs" / name).write_bytes(data)
            if name not in j["files"]:
                j["files"].append(name)
            self._save(j)
        return j

    def start(self, jid: str, resume: bool = False) -> dict:
        j = self._get(jid)
        with self.lock:
            if j["status"] in ("queued", "running"):
                raise InputError("This job is already queued or running.")
            if not j["files"]:
                raise InputError("Upload at least one document before starting.")
            j.update(status="queued", error=None, resume=bool(resume))
            j["log"].append("queued" + (" (resume)" if resume else ""))
            self._save(j)
        self.q.put(jid)
        return j

    def _worker(self) -> None:
        while True:
            jid = self.q.get()
            j = self.jobs.get(jid)
            if not j:
                continue
            try:
                self._run(j)
            except Exception as e:  # noqa: BLE001 - the worker must never die
                with self.lock:
                    j.update(status="failed", error={"type": type(e).__name__, "message": str(e), "hint": ""})
                    self._save(j)

    def _run(self, j: dict) -> None:
        jid = j["id"]
        with self.lock:
            j["status"] = "running"
            self._save(j)
        p = dict(j["params"])

        def log(m):
            with self.lock:
                j["log"].append(str(m))
                del j["log"][:-400]  # keep the last 400 lines

        try:
            llm = self.llm_factory(p)
            kwargs = {k: v for k, v in p.items() if k in RUN_KWARGS}
            run_factory(j["brief"], self._dir(jid) / "docs", self._dir(jid) / "run", llm=llm, resume=j.get("resume", False), log=log, **kwargs)
            st = json.loads((self._dir(jid) / "run" / "run_status.json").read_text(encoding="utf-8"))["status"]
            with self.lock:
                j["status"] = st
        except Blocked as e:
            with self.lock:
                j.update(status="blocked", error=e.to_dict())
        except ForgeError as e:
            with self.lock:
                j.update(status="failed", error=e.to_dict())
        finally:
            self.agents.pop(jid, None)
            with self.lock:
                self._save(j)

    def snapshot(self) -> list[dict]:
        with self.lock:
            return list(self.jobs.values())

    def public(self, j: dict, log_tail: int = 40) -> dict:
        with self.lock:  # the worker thread mutates the job while API threads read it
            out = {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v) for k, v in j.items() if k != "log"}
            out["log"] = list(j["log"][-log_tail:]) if log_tail else []
        rs = self._dir(j["id"]) / "run" / "run_status.json"
        for _ in range(5):  # the file may be mid-replace (Windows sharing violation) or half-visible
            try:
                out["run_status"] = json.loads(rs.read_text(encoding="utf-8"))
                break
            except FileNotFoundError:
                break
            except (OSError, ValueError):
                time.sleep(0.02)
        return out

    def run_dir(self, jid: str) -> Path:
        self._get(jid)
        return self._dir(jid) / "run"

    def chat(self, jid: str, question: str, history: list | None = None) -> dict:
        j = self._get(jid)
        if j["status"] not in ("succeeded", "succeeded_with_warnings"):
            raise InputError(f"No finished agent for this job (status: {j['status']}).")
        question = (question or "").strip()
        if not question:
            raise InputError("Empty question.")
        if len(question) > 4000:
            raise InputError("Question too long (max 4000 characters).")
        with self.lock:
            ag = self.agents.get(jid)
            if ag is None:
                ag = Agent.load(self.run_dir(jid) / "agent_definition.json", self.llm_factory(j["params"]))
                self.agents[jid] = ag
        r = ag.answer(question, (history or [])[-6:])
        return {k: r[k] for k in ("answer", "refused", "sources", "evidence", "tool_calls", "violations")} | {"retrieved": [{"id": h["id"], "score": h["score"]} for h in r["retrieved"]]}


# =============================================================================== HTTP
class ApiKey(BaseHTTPMiddleware):
    def __init__(self, app, key: str | None):
        super().__init__(app)
        self.key = key

    async def dispatch(self, request, call_next):
        if self.key and request.url.path != "/health" and request.headers.get("x-api-key") != self.key:
            return JSONResponse({"error": "unauthorized", "hint": "send the X-API-Key header"}, status_code=401)
        return await call_next(request)


def create_app(mgr: JobManager, api_key: str | None = None) -> Starlette:
    api_key = api_key if api_key is not None else os.environ.get("FORGEAI_API_KEY")

    def err(e: Exception, code: int = 400):
        if isinstance(e, ForgeError):
            return JSONResponse({"error": e.to_dict()}, status_code=code)
        return JSONResponse({"error": {"type": type(e).__name__, "message": str(e)}}, status_code=code)

    async def guarded(fn, request):
        try:
            return await fn(request)
        except KeyError:
            return JSONResponse({"error": {"type": "NotFound", "message": "job not found"}}, status_code=404)
        except ForgeError as e:
            return err(e, 400 if isinstance(e, InputError) else 422)
        except ValueError as e:
            return err(e, 400)
        except Exception as e:  # noqa: BLE001
            return err(e, 500)

    async def health(request):
        try:
            llm = mgr.llm_factory({})
            return JSONResponse({"ok": True, **llm.preflight()})
        except ForgeError as e:
            return JSONResponse({"ok": False, "error": e.to_dict()}, status_code=503)

    async def jobs_list(request):
        return JSONResponse([mgr.public(j, 0) for j in sorted(mgr.snapshot(), key=lambda x: -x["created"])])

    async def jobs_create(request):
        body = await request.json()
        j = mgr.create(body.get("brief", ""), body)
        return JSONResponse(mgr.public(j), status_code=201)

    async def job_get(request):
        return JSONResponse(mgr.public(mgr._get(request.path_params["jid"])))

    async def job_upload(request):
        if int(request.headers.get("content-length", 0) or 0) > MAX_UPLOAD_BYTES:
            raise InputError(f"File too large (max {MAX_UPLOAD_BYTES // 1_048_576} MB).")
        data = await request.body()
        return JSONResponse(mgr.public(mgr.add_file(request.path_params["jid"], request.path_params["name"], data)))

    async def job_start(request):
        try:
            body = await request.json()
        except ValueError:
            body = {}
        return JSONResponse(mgr.public(mgr.start(request.path_params["jid"], bool(body.get("resume")))), status_code=202)

    async def job_report(request):
        p = mgr.run_dir(request.path_params["jid"]) / "REPORT.md"
        if not p.exists():
            return JSONResponse({"error": {"type": "NotReady", "message": "report not available yet"}}, status_code=404)
        return PlainTextResponse(p.read_text(encoding="utf-8"), media_type="text/markdown; charset=utf-8")

    async def job_artifact(request):
        base = mgr.run_dir(request.path_params["jid"]).resolve()
        p = (base / request.path_params["path"]).resolve()
        if not p.is_relative_to(base) or not p.is_file() or ".ckpt" in p.parts:
            return JSONResponse({"error": {"type": "NotFound", "message": "no such artifact"}}, status_code=404)
        return FileResponse(p)

    async def job_chat(request):
        body = await request.json()
        return JSONResponse(mgr.chat(request.path_params["jid"], body.get("question", ""), body.get("history")))

    def wrap(fn):
        async def h(request):
            return await guarded(fn, request)

        return h

    routes = [
        Route("/health", health),
        Route("/jobs", wrap(jobs_list), methods=["GET"]),
        Route("/jobs", wrap(jobs_create), methods=["POST"]),
        Route("/jobs/{jid}", wrap(job_get), methods=["GET"]),
        Route("/jobs/{jid}/files/{name}", wrap(job_upload), methods=["PUT"]),
        Route("/jobs/{jid}/start", wrap(job_start), methods=["POST"]),
        Route("/jobs/{jid}/report", wrap(job_report), methods=["GET"]),
        Route("/jobs/{jid}/artifacts/{path:path}", wrap(job_artifact), methods=["GET"]),
        Route("/jobs/{jid}/chat", wrap(job_chat), methods=["POST"]),
    ]
    return Starlette(routes=routes, middleware=[Middleware(ApiKey, key=api_key)])


def serve(workspace: str = "workspace", host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(JobManager(workspace)), host=host, port=port)
