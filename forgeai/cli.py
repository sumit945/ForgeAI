"""CLI:  python -m forgeai build ... | python -m forgeai chat <agent_definition.json>

Exit codes: 0 ok (possibly with warnings) | 1 internal/other error | 2 blocked by the groundability guard |
            3 bad input | 4 model unavailable | 5 budget exhausted | 130 interrupted (re-run with --resume)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import ForgeError

DEFAULT_MODEL = "qwen2.5:3b-instruct"


def _fail(e: ForgeError, out: str | None = None) -> int:
    print(f"\nERROR [{type(e).__name__}]: {e}", file=sys.stderr)
    if e.hint:
        print(f"  -> {e.hint}", file=sys.stderr)
    if out:
        print(f"  details: {Path(out) / 'run_status.json'}", file=sys.stderr)
    return e.exit_code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="forgeai", description="ForgeAI - the agent that builds (and tests) agents")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="brief + document folder -> tested agent")
    b.add_argument("--brief", help="one-paragraph brief (or use --brief-file)")
    b.add_argument("--brief-file")
    b.add_argument("--docs", required=True, help="folder of source documents (pdf/txt/md/csv/docx/xlsx/html)")
    b.add_argument("--out", required=True, help="output run folder")
    b.add_argument("--model", default=DEFAULT_MODEL)
    b.add_argument("--judge-model", default=None, help="separate judge model (reduces self-preference)")
    b.add_argument("--fallback-model", action="append", default=[], help="model to use if the primary keeps failing (repeatable)")
    b.add_argument("--host", default="http://localhost:11434")
    b.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    b.add_argument("--answers", help="JSON file: a list of answers in order, or {q1: answer,...}")
    b.add_argument("--interactive", action="store_true", help="ask the clarifying questions on the terminal")
    b.add_argument("--approve", action="store_true", help="human checkpoint: approve/edit requirements before generation")
    b.add_argument("--force", action="store_true", help="generate even if the groundability guard says BLOCKED")
    b.add_argument("--resume", action="store_true", help="continue an interrupted run: finished stages and evaluated cases are reused")
    b.add_argument("--fresh-evalset", action="store_true", help="ignore the eval-set cache and generate a new eval set (default: reuse a verified set from an identical earlier build)")
    b.add_argument("--embed-model", default=None, help="Ollama embedding model for hybrid retrieval (auto-detects nomic-embed-text / bge-m3 if installed)")
    b.add_argument("--ocr", choices=["auto", "off", "force"], default="auto", help="read scanned PDF pages / image files with OCR (auto: only pages without a text layer)")
    b.add_argument("--ocr-lang", default=None, help="OCR language codes, e.g. hi or ta (default: language of the brief)")
    b.add_argument("--max-iters", type=int, default=3)
    b.add_argument("--target", type=float, default=0.9, help="stop when dev pass-rate reaches this")
    b.add_argument("--max-calls", type=int, help="LLM call budget")
    b.add_argument("--max-minutes", type=float, help="wall-clock budget")

    c = sub.add_parser("chat", help="chat with any produced agent definition")
    c.add_argument("definition")
    c.add_argument("--docs", help="documents folder (default: ./docs next to the definition)")
    c.add_argument("--model", default=DEFAULT_MODEL)
    c.add_argument("--host", default="http://localhost:11434")
    c.add_argument("-v", "--verbose", action="store_true")

    sv = sub.add_parser("serve", help="start the REST API (set FORGEAI_API_KEY to require an X-API-Key header)")
    sv.add_argument("--workspace", default="workspace")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sub.add_parser("ui", help="start the Streamlit web UI")
    o = sub.add_parser("ocr-setup", help="download OCR models for the given languages (explicit, never automatic)")
    o.add_argument("--langs", default="hi,en", help="comma-separated codes: en hi mr bn ta te kn ur")
    sub.add_parser("ocr-status", help="show which OCR models are installed")

    a = ap.parse_args(argv)
    try:
        if a.cmd == "chat":
            from .runner import chat

            chat(a.definition, a.model, a.docs, a.verbose, a.host)
            return 0

        if a.cmd in ("ocr-status", "ocr-setup"):
            from . import ocr as ocr_mod

            codes = a.langs.split(",") if a.cmd == "ocr-setup" else None
            st = ocr_mod.setup(codes) if codes else ocr_mod.status()
            print(json.dumps({k: v for k, v in st.items() if k != "models"}, indent=2))
            print("models installed:", ", ".join(k for k, v in st.get("models", {}).items() if v) or "none")
            return 0 if st["available"] or not codes else 1
        if a.cmd == "serve":
            from .service import serve

            serve(a.workspace, a.host, a.port)
            return 0
        if a.cmd == "ui":
            import subprocess

            return subprocess.call([sys.executable, "-m", "streamlit", "run", str(Path(__file__).with_name("ui.py")), "--server.maxUploadSize", "25"])

        from .factory import run_factory

        brief = a.brief or (Path(a.brief_file).read_text(encoding="utf-8") if a.brief_file else None)
        if not brief:
            ap.error("provide --brief or --brief-file")
        answers = None
        if a.answers:
            try:
                answers = json.loads(Path(a.answers).read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                print(f"ERROR: cannot read --answers file: {e}", file=sys.stderr)
                return 3
        res = run_factory(
            brief, a.docs, a.out, model=a.model, judge_model=a.judge_model, fallback_models=a.fallback_model, host=a.host,
            backend=a.backend, answers=answers, interactive=a.interactive, approve=a.approve, force=a.force, resume=a.resume, fresh_evalset=a.fresh_evalset, embed_model=a.embed_model, ocr=a.ocr, ocr_langs=a.ocr_lang.split(",") if a.ocr_lang else None,
            max_iters=a.max_iters, target=a.target, max_calls=a.max_calls, max_minutes=a.max_minutes,
        )
        if res["warnings"]:
            print(f"\nFinished with {len(res['warnings'])} warning(s) - see {a.out}/REPORT.md section 0.")
        return 0
    except ForgeError as e:
        return _fail(e, getattr(a, "out", None))
    except KeyboardInterrupt:
        print("\nInterrupted. Progress is saved; continue with the same command plus --resume.", file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001 - last line of defence: never show a raw traceback to the user
        print(f"\nUNEXPECTED ERROR: {type(e).__name__}: {e}\nA full traceback was saved to <out>/error.log when possible.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
