# ForgeAI - the agent that builds (and tests) agents

Give ForgeAI **one paragraph** and **a folder of documents**. It interviews the requester, checks that the documents can
actually support the request, generates a declarative agent, generates its own test set *from the documents*, judges the
agent, diagnoses every failure and applies targeted fixes (max 3 iterations), then hands back the agent **and an honest report**.

Everything runs on small local open models through Ollama (`qwen2.5:3b-instruct` for the agent, `bge-m3` for multilingual hybrid
retrieval) - fits a 4 GB GPU. No cloud, no API keys. Works in 11 Indian languages + Hinglish + English.

```
brief + docs ─► 1 Intake ─► 2 Groundability guard ─► 3 Agent generator ─► 4 Eval generator
                (<=5 Qs)     (flag / narrow / block)   (prompt, retrieval,    (>=20 verified cases,
                                                         tools, contract)      dev / hold-out split)
                                                              │                      │
                         7 Honest report  ◄── 6 Critic ─► optimizer ◄── 5 Judge (deterministic + guarded LLM)
                                              classify → targeted fix → re-run (<=3, revert if no gain)
```

## Measured results

Every number below was measured on a GTX 1650 (4 GB) with the local models above. Sample sizes are small (about 21 eval cases per
run, 3-15 questions per probe): treat them as evidence the machinery works, not as accuracy guarantees. Raw artifacts are in `runs/`
and `stress/results/`.

### 1. The two required demo runs (`runs/hr_policy`, `runs/ship_spares`) - unattended, 11-13 min each

| | HR policy assistant | Ship-spares procurement assistant |
|---|---|---|
| Documents | 3 policy PDFs | catalogue CSV + 2 PDFs |
| Groundability guard | WARN: removed *health insurance* and *pension contributions* (not in the documents) | OK |
| Tools attached | none (no evidence for any) | `csv_lookup`, `calculator` |
| Final pass rate | **100%** (21/21), dev 100%, hold-out 100% | **95%** (20/21), dev 93%, hold-out 100% |
| Judge reliability (known-answer canaries) | 0.83 | 0.92 |

The one ship-spares failure is a judge-strictness case (correct answer, but the model paraphrased its evidence quote of a CSV row).

### 2. Does the improvement loop actually work?

In both demos the generated agent already passed its dev split, so the loop had nothing to fix (also in two re-runs with a 100% target:
`runs/hr_policy_optimizer_demo` dev 100% / hold-out 83%, `runs/ship_spares_optimizer_demo` dev 100% / **hold-out 57%** - a large gap
that the report flags). To test the loop itself, `demos/optimizer_repair_demo.py` starts from a **deliberately sabotaged** agent
(top_k=1, 40-word chunks, key rules removed, temperature 0.7) on the same eval set (`runs/optimizer_repair_demo`):

| Iteration | Outcome | Dev | Hold-out | Change |
|---|---|---|---|---|
| 0 baseline | - | 50% | 43% | - |
| 1 | accepted | 81% | 57% | `top_k` 1 -> 3 (fixed 7 retrieval misses) + one prompt rule |
| 2 | reverted | 81% | 43% | one more prompt rule: no gain, undone |

Three cases were still failing afterwards. This is a labelled fault-injection test, not a claim about first-pass quality.

### 3. Stress test: 100 documents at four severities (real model, `stress/`)

| Level | What it contains | Time | Peak memory | Direct questions | Attacks followed |
|---|---|---|---|---|---|
| L1 low | 100 clean documents | 12.4 min | 75 MB | 15/15 | 0 |
| L2 moderate | 7 file formats, long docs, Unicode, near-duplicates | 15.1 min | 301 MB | 15/15 | 0 |
| L3 high | + 22 hostile files (corrupt, empty, encrypted, duplicate, 40k-row CSV) | 11.8 min | 501 MB | 15/15 | 0 |
| L4 severe | + prompt-injection docs, contradictory pairs, noise, 55 MB file, 520 extra files | 11.5 min | 329 MB | 15/15 | 0 |

At L4 the agent followed none of the 6 injected documents. **Known weakness:** on the 4 contradictory document pairs it silently
picked one answer every time and never surfaced the conflict. The same four levels also pass on a scripted model in 1-12 s, plus 8
concurrency/chaos tests (200 parallel questions, 6 simultaneous jobs with API polling, random model faults, crash-then-resume).

### 4. Indian languages

* **Micro-benchmark** (`stress/multilingual_bench.py --part micro`, real model, 3 fact questions + 1 off-topic + 1 injection per
  language): Hindi, Bengali, Marathi, Telugu, Tamil, Gujarati, Urdu, Kannada, Malayalam, Odia, Punjabi, Hinglish and English all scored
  3/3, answered in the right script, and refused every off-topic and injection attempt. Tiny sample, short single-paragraph documents.
* **Cross-language retrieval** (36 questions per setup, top-2 passages):

  | Setup | BM25 | nomic-embed-text | **bge-m3** |
  |---|---|---|---|
  | same-language question | 36/36 | 36/36 | 36/36 |
  | Indian-language question over English documents | 3/36 | 21/36 | **36/36** |
  | English question over Indian-language documents | 3/36 | 3/36 | **36/36** |

* **Hindi end to end** (Hindi brief + 3 Hindi documents, 40.8 min): the groundability guard supported 10/10 topics (an earlier run was
  wrongly *blocked* because the requirements model wrote English topics for Hindi documents - fixed by asking for requirements in the
  documents' language and by translating mismatched topics for the guard). The finished agent answered **9/10** direct Hindi questions
  correctly, all in Devanagari. Misses: one wrongly refused question, and the health-insurance question was answered with unrelated
  sick-leave text instead of refused (the requirements model had dropped that topic, so the gap was never recorded). Its own eval set
  had only 18 verified cases (below the target of 20, flagged) and scored 67% (correct 55%, grounded 82%, refusal recall 86%), so **Hindi
  quality is clearly below English** on this 3B model. The optimizer's fixes were reverted (no gain).
* The translations in `forgeai/i18n.py` and the multilingual test documents were written by an AI assistant and have **not** been
  reviewed by native speakers.

### 5. Retrieval

17 deliberately paraphrased questions, top-4 passages: BM25 9/12 -> hybrid **11/12** (HR), 3/5 -> **5/5** (ship). A dependency-free
hashing embedder gave no gain, so dense retrieval only switches on when an embedding model is installed.

### 6. OCR for scanned documents (`forgeai/ocr.py`, `stress/ocr_bench.py`)

Scanned PDF pages and image files are read with EasyOCR (CPU). OCR only runs on pages that have **no text layer and contain an image**
(a text PDF that merely ends with a near-empty page is not OCR'd), pages carry a confidence score, and results are cached by file hash.
Measured on a 2-page Hindi policy rendered as a scan (real Hindi model, this CPU):

| Scan | Time per page | OCR confidence | Numbers recovered | Words recovered | Retrieval (5 Hindi questions) |
|---|---|---|---|---|---|
| clean (about 170 DPI) | 21.5 s | 0.35-0.43 (flagged as low) | 16/16 | 68% | 5/5 with BM25 and hybrid - same as the clean text |
| poor (half resolution, blur, noise, tilt) | - | 0.04-0.07 | - | 7-10% of tokens real | pages **dropped as unreadable** (never used) |

The confidence thresholds (drop below 0.20, flag below 0.50) were set from that gap: clean pages scored 0.35-0.43 with about 70% correct
words, the poor scan scored 0.04-0.07 with about 10%. They come from one document and one font, so treat them as a starting point.
EasyOCR covers English, Hindi, Marathi, Bengali, Tamil, Telugu, Kannada and Urdu - **not** Gujarati, Malayalam, Odia or Punjabi (those scans are
reported as unreadable). OCR models are never downloaded automatically: `python -m forgeai ocr-setup --langs ta` installs more.
Only the Hindi/Marathi (Devanagari) model is installed here; English OCR falls back to that model (reads Latin text, lower accuracy, may turn
digits into Devanagari numerals - the pipeline normalises those back).

## Quick start

```powershell
pip install -r requirements.txt
ollama pull qwen2.5:3b-instruct
ollama pull bge-m3                               # recommended: multilingual hybrid retrieval (auto-detected)
python demos/build_demo_docs.py                  # (re)builds the demo PDFs from demos/*/sources
python -m forgeai build --brief-file demos/hr_policy/brief.txt --docs demos/hr_policy/docs --out runs/hr_policy --answers demos/hr_policy/answers.json
python -m forgeai chat runs/hr_policy/agent_definition.json      # generic chat runner: loads ANY produced definition
python -m forgeai ui                             # web console: overview, build, runs, agents (chat), evaluations, languages, settings
python -m forgeai serve                          # REST API on :8000 (set FORGEAI_API_KEY to require X-API-Key)
python stress/run_stress.py --mode real          # the 100-document stress test (use --mode scripted for a fast run)
python -m forgeai ocr-status                     # which OCR models are installed (ocr-setup --langs ta,en installs more)
python -m pytest tests -q                        # 195 offline tests (scripted model, no Ollama needed)
```

**Unseen brief (review mode)**, fully unattended - unanswered clarifying questions become recorded assumptions:

```powershell
python -m forgeai build --brief "..." --docs path\to\folder --out runs\review
```
Flags: `--interactive` (answer the <=5 questions in the terminal) | `--approve` (human checkpoint on the requirements) |
`--resume` (continue an interrupted run) | `--fresh-evalset` (ignore the eval-set cache; by default a verified eval set from an identical earlier build is reused) | `--judge-model` | `--embed-model` | `--fallback-model` | `--max-calls/--max-minutes` (budget) |
`--ocr auto|off|force` and `--ocr-lang hi` (scanned pages; default: language of the brief) | `--force` (generate even if the guard says BLOCKED). Exit codes: 0 ok, 2 blocked by the guard, 3 bad input, 4 model unavailable,
5 budget exhausted, 130 interrupted.

## What each requirement maps to

| # | Requirement | Where | How it is made trustworthy |
|---|---|---|---|
| 1 | Intake interviewer | `intake.py`, `prompts/intake_questions.txt`, `requirements.txt` | 5-question cap enforced in code; unanswered questions become *recorded assumptions*; written in the documents' language. |
| 2 | Groundability guard | `groundability.py` | Per topic: lexical coverage + a deterministic evidence sentence + an LLM verdict whose quote is verified verbatim. Topics must trace back to the brief; topics in another language are translated before matching. Unsupported topics become refusals; nothing supportable -> `BLOCKED`. |
| 3 | Agent generator | `generator.py` | Declarative JSON (`forgeai.agent/1`), schema-validated. Tools need deterministic evidence in the brief (an HR agent cannot get a weather tool); every decision is logged with reasons. |
| 4 | Eval generator | `evalgen.py`, `prompts/evalgen_*.txt` | Generated, not hand-written. Facts must be verbatim in the source *and* the reference answer; weak candidates are rejected and counted. Answerable, ambiguous, multi-document, near-miss, off-topic/injection. Dev / hold-out split. |
| 5 | Judge | `judge.py` | Deterministic checks first; the LLM judge is pointwise, per-fact, needs a *verified* quote, and is calibrated with known-answer canaries (see the module docstring for position / self-preference / verbosity bias). |
| 6 | Critic-optimizer | `optimizer.py` | Failures classified as retrieval_miss / prompt_defect / hallucination / wrong_refusal; single-parameter or single-rule patches; kept only if dev strictly improves, else reverted and retried one category at a time; max 3; everything logged. |
| 7 | Honest reporting | `report.py` | Assembled from measurements by code (no model prose): per-iteration scorecard, dev vs hold-out gap, Wilson interval, every remaining failure, judge reliability, run health. |
| 8 | Runnable product | `runner.py`, `ui.py`, `service.py` | CLI chat, web console and REST API; BM25/hybrid retrieval, compound-question retrieval, tool loop, JSON output contract, input prompt-injection guard, answers and refusals in the user's language. |
| 9 | Two demos | `runs/hr_policy`, `runs/ship_spares` | Full artifacts committed, plus every model prompt and response (`llm_trace.jsonl`). |

Stretch goals implemented: human checkpoint (`--approve`), budget enforcement (calls / minutes), web UI, REST API.
Not done: prompt tournament, factory-vs-hand-written baseline, regression mode, hosting.

## Production hardening
* **Typed errors** with fix hints and exit codes; **pydantic schemas** on every model reply and stage artifact.
* **Resilient model client**: retries with backoff, fallback models, circuit breaker, preflight, over-long prompts shrunk not silently cut.
* **Resumable**: stage checkpoints keyed by input hashes + per-case cache; `run_status.json` after every stage; atomic, Windows-safe writes.
* **Robust ingestion**: corrupt / empty / duplicate / encrypted / oversized files are skipped and reported; instruction-like text inside documents is removed and reported; scanned PDFs and images are read with OCR (confidence-gated) or reported when OCR is unavailable.
* **Security**: user-side prompt-injection guard, document-side sanitising, API key, upload limits, path-traversal protection, escaped HTML in the UI.
* **Chaos and stress tests** found real bugs (ignored budget limit, swallowed outages, a Windows file-replace race, an oversized prompt inventory at 100 documents, unlocked job state under API polling).

## Honest limitations
* A 3B model writes the eval set, answers the questions and (by default) judges: shared blind spots and self-preference are possible. Use `--judge-model` with a larger model for a stronger judge.
* "No errors" is not achievable with LLMs; the goal is *no silent errors*: failures are typed, explained, recorded, and quality is measured.
* Non-English quality is below English on this model (Hindi end to end: 9/10 direct questions, 67% on its own eval set). Languages other than Hindi were only micro-tested.
* OCR is slow on CPU (about 20 s per dense page), only as good as the scan (clean Hindi scan: about 70% of words correct), and unavailable for Gujarati, Malayalam, Odia and Punjabi. Eval sets are small (about 21 cases).
* Contradictory documents are not surfaced; the agent silently picks one answer.
* Retrieval index is in memory per agent (no persistent vector database); above 2,500 chunks dense retrieval is skipped and BM25 is used.

## Layout
`forgeai/` package | `forgeai/prompts/` every generation prompt | `demos/` briefs, synthetic sources and the prompts that produced them (`SYNTHESIS_PROMPTS.md`), Hindi corpus, optimizer repair demo | `runs/` committed artifacts | `stress/` 100-document corpora, stress and multilingual benchmarks | `tests/` 195 offline tests | `.github/workflows/tests.yml` CI.
