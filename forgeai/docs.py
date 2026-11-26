"""Document loading and chunking. Handles .pdf .txt .md .csv .docx; unreadable files are reported, not fatal."""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED = {".pdf", ".txt", ".md", ".csv", ".docx", ".xlsx", ".html", ".htm"}
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_TEXT_CHARS = 3_000_000
MAX_FILES = 500


INJECTION_RE = [re.compile(p, re.I) for p in (
    r"ignore (?:all |any |the |your )?(?:previous|prior|above|earlier|preceding) (?:instructions?|rules|prompts?)",
    r"disregard (?:all |any |the |your )?(?:previous|prior|above|earlier) (?:instructions?|rules)",
    r"\b(?:system|developer|admin) (?:instruction|prompt|message|override)\b\s*:",
    r"(?:print|reveal|show|output|repeat) (?:your |the )?(?:full |entire |hidden )?(?:system )?(?:prompt|instructions)",
    r"from now on,? (?:you|answer|reply|respond)",
    r"you (?:must|should|will) (?:now )?(?:answer|reply|respond) (?:only )?(?:with )?['\"]?[A-Z]{3,}",
)]
# Extra patterns that are only checked on USER input (too common in ordinary document prose, e.g. "you are now eligible").
USER_ONLY_RE = [re.compile(p, re.I) for p in (
    r"\byou are (?:now|no longer) (?:a|an|the|my|dan|free|unrestricted|jailbroken|evil|bound by no)\b", r"\bpretend (?:to be|you are|that you)\b", r"\bjailbreak\b", r"\bdeveloper mode\b", r"\bact as (?:if|though) you (?:have|had) no\b",
    r"\bforget (?:all |everything |your )(?:previous |prior |the )?(?:instructions|rules|training)\b",
)]
_SENT = re.compile(r"(?<=[.!?।॥۔؟])\s+")  # includes the danda of Hindi/Bengali/Marathi and the Urdu full stop


def is_user_injection(text: str) -> bool:
    return is_injection(text) or any(p.search(text) for p in USER_ONLY_RE)


def is_injection(sentence: str) -> bool:
    from .i18n import has_multilingual_injection

    return any(p.search(sentence) for p in INJECTION_RE) or has_multilingual_injection(sentence)


def sanitize_injections(text: str) -> tuple[str, int]:
    """Remove instruction-like sentences ("ignore all previous instructions...") from document text. Documents are DATA;
    text that tries to give the assistant orders never reaches the model. Returns (clean_text, number_removed)."""
    removed = 0
    out_lines = []
    for line in text.split(chr(10)):
        if not is_injection(line):
            out_lines.append(line)
            continue
        kept = []
        for sent in _SENT.split(line):
            if is_injection(sent):
                removed += 1
            else:
                kept.append(sent)
        out_lines.append(" ".join(kept))
    return chr(10).join(out_lines), removed


@dataclass
class Document:
    name: str
    path: str
    kind: str  # "text" | "csv"
    text: str
    columns: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class Chunk:
    id: str
    doc: str
    idx: int
    text: str


def _read_text_file(p: Path) -> str:
    raw = p.read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _read_pdf(p: Path) -> tuple[str, list[str]]:
    from pypdf import PdfReader

    notes: list[str] = []
    r = PdfReader(str(p))
    if r.is_encrypted:
        try:
            ok = r.decrypt("")
        except Exception:  # noqa: BLE001
            ok = 0
        if not ok:
            raise ValueError("PDF is password-protected")
        notes.append("encrypted PDF opened with an empty password")
    pages, bad = [], 0
    for pg in r.pages:
        try:
            pages.append(pg.extract_text() or "")
        except Exception:  # noqa: BLE001
            bad += 1
    if bad:
        notes.append(f"{bad} page(s) could not be read")
    return "\n\n".join(pages), notes


def _read_docx(p: Path) -> str:
    import docx  # type: ignore

    d = docx.Document(str(p))
    parts = [par.text for par in d.paragraphs if par.text.strip()]
    for t in d.tables:  # tables carry facts too
        for row in t.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def _read_html(p: Path) -> str:
    from html.parser import HTMLParser

    class _P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.out, self.skip = [], 0

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style", "nav", "footer"):
                self.skip += 1
            if tag in ("p", "br", "li", "tr", "h1", "h2", "h3", "h4", "div"):
                self.out.append("\n")

        def handle_endtag(self, tag):
            if tag in ("script", "style", "nav", "footer") and self.skip:
                self.skip -= 1

        def handle_data(self, data):
            if not self.skip:
                self.out.append(data)

    parser = _P()
    parser.feed(_read_text_file(p))
    return re.sub(r"\n\s*\n+", "\n\n", "".join(parser.out))


def _read_xlsx(p: Path) -> tuple[list[str], list[dict]]:
    import openpyxl  # type: ignore

    wb = openpyxl.load_workbook(str(p), read_only=True, data_only=True)
    ws = wb.worksheets[0]
    it = ws.iter_rows(values_only=True)
    header = [str(c).strip() if c is not None else "" for c in next(it, [])]
    cols = [h or f"col{i + 1}" for i, h in enumerate(header)]
    rows = [{cols[i]: ("" if v is None else str(v).strip()) for i, v in enumerate(r) if i < len(cols)} for r in it if any(v is not None for v in r)]
    return cols, rows


def clean_text(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"(?<=[a-z,;])\n(?=[a-z])", " ", t)  # re-join hard-wrapped lines
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()



def reflow(text: str) -> str:
    """Rebuild paragraphs from PDF text where every visual line ends in a newline and blank lines are lost."""
    lines = [ln.strip() for ln in text.replace(chr(13), chr(0)[:0]).split(chr(10))]
    paras: list[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf:
            paras.append(buf)
        buf = ""

    for i, l in enumerate(lines):
        if not l:
            flush()
            continue
        nxt = next((x for x in lines[i + 1 :] if x), "")
        if not buf:
            buf = l
        elif re.fullmatch(r"\d+(\.\d+)*\.?", buf):  # bare section number: glue to its title
            buf += " " + l
        elif not re.search(r"[.!?:;\u0964\u0965\u06d4\u061f]$", buf) and (l[0].islower() or (l[0].isalpha() and not l[0].isupper()) or l[0].isdigit() or l[0] in "($\"'" or len(buf.split()) > 8):
            buf += " " + l
        elif re.search(r"[.!?:;\u0964\u0965\u06d4\u061f]$", buf) and l[0].islower():
            buf += " " + l
        else:
            flush()
            buf = l
        # a short title-like line followed by a sentence starts its own paragraph
        if buf and len(buf.split()) <= 9 and not re.search(r"[.!?:;,]$", buf) and nxt[:1].isupper() and re.match(r"\d|[A-Z]", buf):
            flush()
    flush()
    return "\n\n".join(paras)


def load_folder(folder: str | Path) -> tuple[list[Document], list[str]]:
    """Load every readable document. Never raises for a bad FILE: it is skipped and reported in `problems`.
    Raises InputError only when the folder itself is unusable."""
    import hashlib

    from .errors import InputError

    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        raise InputError(f"Document folder not found: {folder}")
    docs: list[Document] = []
    problems: list[str] = []
    seen: dict[str, str] = {}
    files = [p for p in sorted(folder.rglob("*")) if p.is_file()]
    for p in files:
        if p.suffix.lower() not in SUPPORTED and not p.name.startswith("."):
            problems.append(f"{p.name}: unsupported file type '{p.suffix}' (skipped)")
    candidates = [p for p in files if p.suffix.lower() in SUPPORTED]
    if len(candidates) > MAX_FILES:
        problems.append(f"{len(candidates)} files found; only the first {MAX_FILES} were loaded")
        candidates = candidates[:MAX_FILES]
    for p in candidates:
        ext = p.suffix.lower()
        try:
            size = p.stat().st_size
            if size == 0:
                problems.append(f"{p.name}: empty file (skipped)")
                continue
            if size > MAX_FILE_BYTES:
                problems.append(f"{p.name}: {size // 1_048_576} MB exceeds the {MAX_FILE_BYTES // 1_048_576} MB limit (skipped)")
                continue
            rows: list[dict] = []
            cols: list[str] = []
            if ext == ".csv":
                raw = _read_text_file(p)
                rdr = csv.DictReader(io.StringIO(raw))
                rows = [{k.strip(): (v or "").strip() for k, v in r.items() if k} for r in rdr]
                cols = [c.strip() for c in (rdr.fieldnames or [])]
                if not rows:
                    problems.append(f"{p.name}: CSV has no data rows (skipped)")
                    continue
                kind, text = "csv", raw
            elif ext == ".xlsx":
                cols, rows = _read_xlsx(p)
                if not rows:
                    problems.append(f"{p.name}: spreadsheet has no data rows (skipped)")
                    continue
                kind = "csv"
                text = "\n".join([",".join(cols)] + [",".join(r.get(c, "") for c in cols) for r in rows])
            else:
                kind = "text"
                if ext == ".pdf":
                    raw_text, notes = _read_pdf(p)
                    problems += [f"{p.name}: {n}" for n in notes]
                    text = reflow(raw_text)
                elif ext == ".docx":
                    text = _read_docx(p)
                elif ext in (".html", ".htm"):
                    text = _read_html(p)
                else:
                    text = _read_text_file(p)
                text = clean_text(text)
                if len(text.split()) < 5:
                    why = "no extractable text (scanned/image-only PDF? OCR is not enabled)" if ext == ".pdf" else "no extractable text"
                    problems.append(f"{p.name}: {why} (skipped)")
                    continue
            if len(re.findall(r"[^\W\d_]{2,}", text)) < 5:  # needs real words, not just symbols/numbers
                problems.append(f"{p.name}: no readable text content (symbols or numbers only) (skipped)")
                continue
            text, n_inj = sanitize_injections(text)
            if n_inj:
                problems.append(f"{p.name}: contains instruction-like text (possible prompt injection); {n_inj} sentence(s) were removed before use")
                for r in rows:
                    for k, v in r.items():
                        if is_injection(v):
                            r[k] = "[removed]"
            if len(text) > MAX_TEXT_CHARS:
                text = text[:MAX_TEXT_CHARS]
                problems.append(f"{p.name}: text truncated to {MAX_TEXT_CHARS:,} characters")
            digest = hashlib.sha1(re.sub(r"\s+", " ", text).encode("utf-8", "ignore")).hexdigest()
            if digest in seen:
                problems.append(f"{p.name}: duplicate of {seen[digest]} (skipped)")
                continue
            seen[digest] = p.name
            name = p.name
            if any(d.name == name for d in docs):  # same file name in two sub-folders
                name = f"{p.parent.name}__{p.name}"
            docs.append(Document(name, str(p), kind, text, cols, rows))
        except Exception as e:  # noqa: BLE001
            problems.append(f"{p.name}: could not be read ({type(e).__name__}: {str(e)[:120]}) (skipped)")
    return docs, problems


def _split_units(text: str) -> list[str]:
    """Paragraphs, with over-long paragraphs split into sentences."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para.split()) > 90:
            units.extend(s.strip() for s in re.split(r"(?<=[.!?\u0964\u0965\u06d4\u061f])\s+(?=[A-Z0-9(\u0900-\u0d7f\u0600-\u06ff])", para) if s.strip())
        else:
            units.append(para)
    return units


def chunk_text(doc: Document, size_words: int, overlap_words: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    if doc.kind == "csv":
        per = max(1, size_words // 14)
        step = max(1, per - max(0, overlap_words // 14))
        for i in range(0, max(1, len(doc.rows)), step):
            rows = doc.rows[i : i + per]
            if not rows:
                break
            body = "\n".join("; ".join(f"{k}: {v}" for k, v in r.items()) for r in rows)
            chunks.append(Chunk(f"{doc.name}#{len(chunks)}", doc.name, len(chunks), body))
            if i + per >= len(doc.rows):
                break
        return chunks
    units = _split_units(doc.text)
    cur: list[str] = []
    cur_w = 0
    for u in units:
        w = len(u.split())
        if cur and cur_w + w > size_words:
            chunks.append(Chunk(f"{doc.name}#{len(chunks)}", doc.name, len(chunks), "\n".join(cur)))
            # overlap: carry trailing units totalling <= overlap_words
            carry: list[str] = []
            cw = 0
            for prev in reversed(cur):
                pw = len(prev.split())
                if cw + pw > overlap_words:
                    break
                carry.insert(0, prev)
                cw += pw
            cur, cur_w = carry, cw
        cur.append(u)
        cur_w += w
    if cur:
        chunks.append(Chunk(f"{doc.name}#{len(chunks)}", doc.name, len(chunks), "\n".join(cur)))
    return chunks


def chunk_documents(docs: list[Document], size_words: int = 140, overlap_words: int = 30) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_text(d, size_words, overlap_words))
    return out


def inventory(docs: list[Document]) -> list[dict]:
    """Compact, LLM-friendly description of the folder (used by intake + groundability)."""
    inv = []
    for d in docs:
        entry = {"name": d.name, "kind": d.kind, "words": d.words}
        if d.kind == "csv":
            entry["columns"] = d.columns
            entry["rows"] = len(d.rows)
            entry["sample_rows"] = d.rows[:3]
        else:
            heads = [
                ln.strip()
                for ln in d.text.split("\n")
                if 3 < len(ln.strip()) < 80 and not ln.strip().endswith((".", ",", ";", "\u0964", "\u06d4"))
            ][:14]
            entry["headings"] = heads
            entry["opening"] = " ".join(d.text.split()[:70])
        inv.append(entry)
    return inv


def inventory_for_prompt(docs: list[Document], max_chars: int = 4500) -> list[dict] | dict:
    """Inventory sized for a small model's context window: full detail for small folders, a statistical summary for big ones
    (counts by kind, name samples, the most frequent headings, a few evenly spaced openings, and every CSV's columns)."""
    full = inventory(docs)
    import json as _json
    from collections import Counter

    if len(_json.dumps(full, ensure_ascii=False)) <= max_chars:
        return full
    kinds = Counter(d.name.rsplit(".", 1)[-1].lower() for d in docs)
    heads: Counter = Counter()
    for e in full:
        for h in e.get("headings", [])[:6]:
            heads[re.sub(r"\d+", "#", h)] += 1
    step = max(1, len(docs) // 5)
    summary = {
        "note": f"{len(docs)} documents - this is a summary, not a full listing",
        "documents": len(docs),
        "total_words": sum(d.words for d in docs),
        "by_file_type": dict(kinds),
        "example_names": [d.name for d in docs[:: max(1, len(docs) // 12)]][:12],
        "common_headings": [h for h, _ in heads.most_common(14)],
        "sample_openings": [{"name": docs[i].name, "opening": " ".join(docs[i].text.split()[:35])} for i in range(0, len(docs), step)][:5],
        "tables": [{"name": d.name, "columns": d.columns[:10], "rows": len(d.rows)} for d in docs if d.kind == "csv"][:6],
    }
    while len(_json.dumps(summary, ensure_ascii=False)) > max_chars and summary["sample_openings"]:
        summary["sample_openings"].pop()
    return summary
