"""OCR for scanned PDFs and image files (EasyOCR, CPU).

Design rules
  * optional: nothing here is imported unless a page actually has no text layer; if EasyOCR / its models are missing, documents are
    reported ("OCR is not available: ...") exactly as before - never a crash;
  * no surprise downloads: models are loaded with download_enabled=False. `python -m forgeai ocr-setup --langs hi,ta` downloads them on request;
  * honest quality: every page carries a confidence score; pages below DROP_BELOW are not used (they would feed wrong facts into an agent),
    pages below WARN_BELOW are used but flagged; the document-level mean is reported;
  * bounded: page and time budgets per run, results cached on disk by file hash so a resumed run or a chat session does not OCR again.

EasyOCR covers English, Hindi, Marathi, Bengali, Tamil, Telugu, Kannada and Urdu. It does NOT cover Gujarati, Malayalam, Odia or Punjabi:
for those, scanned pages are reported as unreadable.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

# language code -> (EasyOCR code, recogniser weight file)
LANG_MODELS = {
    "en": ("en", "english_g2.pth"), "hi": ("hi", "devanagari.pth"), "mr": ("mr", "devanagari.pth"), "bn": ("bn", "bengali.pth"),
    "ta": ("ta", "tamil.pth"), "te": ("te", "telugu.pth"), "kn": ("kn", "kannada.pth"), "ur": ("ur", "arabic.pth"), "hi-Latn": ("en", "english_g2.pth"),
}
UNSUPPORTED = {"gu": "Gujarati", "ml": "Malayalam", "or": "Odia", "pa": "Punjabi"}
DETECTOR = "craft_mlt_25k.pth"

# Calibrated on real EasyOCR Hindi output: clean scans scored 0.35-0.43 with ~70% correct words (usable), a poor scan scored 0.04-0.07 with
# 7-10% correct words (garbage). The drop line sits in that gap; anything below the warn line is used but flagged.
DROP_BELOW = 0.20  # page confidence under this is not used
WARN_BELOW = 0.50  # used, but flagged as low confidence
RENDER_LONG_SIDE = 2000  # pixels: every page is rendered to about this size (about 170 DPI for A4) regardless of its PDF page size
MAX_PAGES_PER_RUN = int(os.environ.get("FORGEAI_OCR_MAX_PAGES", "60"))
MAX_SECONDS_PER_RUN = float(os.environ.get("FORGEAI_OCR_MAX_SECONDS", "1500"))
MIN_TEXT_CHARS = 25  # a PDF page with fewer extracted characters than this has no usable text layer
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def model_dir() -> Path:
    return Path(os.environ.get("EASYOCR_MODULE_PATH", Path.home() / ".EasyOCR")) / "model"


def cache_dir() -> Path:
    d = Path(os.environ.get("FORGEAI_CACHE", Path.home() / ".cache" / "forgeai")) / "ocr"
    return d


def reader_langs(codes: list[str] | None) -> list[str]:
    """EasyOCR language list for a request such as ['hi']: the script language plus English (they combine), de-duplicated."""
    out: list[str] = []
    for c in codes or ["en"]:
        e = LANG_MODELS.get(c, (None,))[0]
        if e and e not in out:
            out.append(e)
    if not out:
        out = ["en"]
    if "en" not in out:
        out.append("en")
    # EasyOCR cannot mix two non-Latin scripts in one reader: keep the first
    non_latin = [x for x in out if x != "en"]
    return ([non_latin[0]] if non_latin else []) + ["en"] if non_latin else ["en"]


def status(codes: list[str] | None = None) -> dict:
    """Is OCR usable for these languages right now, and if not, why? Cheap: no model is loaded."""
    try:
        import easyocr  # noqa: F401
        import pypdfium2  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"OCR libraries missing ({type(e).__name__}: {e}); pip install easyocr pypdfium2 pillow", "models": {}}
    md = model_dir()
    models = {c: (md / f).exists() for c, (_, f) in LANG_MODELS.items()}
    codes = codes or ["en"]
    bad = [UNSUPPORTED[c] for c in codes if c in UNSUPPORTED]
    if bad:
        return {"available": False, "reason": f"EasyOCR does not support {', '.join(bad)}", "models": models, "detector": (md / DETECTOR).exists()}
    need = [f for c in codes if c in LANG_MODELS for f in [LANG_MODELS[c][1]]]
    missing = [f for f in need if not (md / f).exists()] + ([DETECTOR] if not (md / DETECTOR).exists() else [])
    if missing:
        return {"available": False, "reason": f"OCR models not installed ({', '.join(sorted(set(missing)))}); run: python -m forgeai ocr-setup --langs {','.join(codes)}",
                "models": models, "detector": (md / DETECTOR).exists()}
    return {"available": True, "reason": "", "models": models, "detector": True}


@dataclass
class Line:
    text: str
    conf: float
    top: float
    left: float
    height: float


class EasyEngine:
    """Thin wrapper so tests (and other engines) can be swapped in: anything with .recognize(PIL.Image) -> list[Line]."""

    def __init__(self, langs: list[str]):
        import easyocr

        self.langs = langs
        self.reader = easyocr.Reader(langs, gpu=False, verbose=False, download_enabled=False)

    def recognize(self, image) -> list[Line]:
        import numpy as np

        out = []
        for box, text, conf in self.reader.readtext(np.array(image.convert("RGB")), detail=1, paragraph=False):
            ys = [p[1] for p in box]
            xs = [p[0] for p in box]
            if text.strip():
                out.append(Line(text.strip(), float(conf), min(ys), min(xs), max(ys) - min(ys)))
        return out


_ENGINES: dict[tuple, object] = {}
_OVERRIDE = None


def set_engine(engine) -> None:
    """Install a custom engine (tests); None restores EasyOCR."""
    global _OVERRIDE
    _OVERRIDE = engine


def get_engine(langs: list[str]):
    if _OVERRIDE is not None:
        return _OVERRIDE
    key = tuple(langs)
    if key not in _ENGINES:
        _ENGINES[key] = EasyEngine(list(langs))
    return _ENGINES[key]


def lines_to_text(lines: list[Line]) -> tuple[str, float]:
    """Rebuild reading order: cluster regions into visual lines by vertical position, left-to-right inside a line.
    Returns (text, confidence) where confidence is weighted by text length."""
    if not lines:
        return "", 0.0
    hs = sorted(l.height for l in lines)
    tol = max(6.0, 0.6 * hs[len(hs) // 2])
    rows: list[list[Line]] = []
    for l in sorted(lines, key=lambda x: x.top):
        if rows and abs(l.top - rows[-1][0].top) <= tol:
            rows[-1].append(l)
        else:
            rows.append([l])
    text = "\n".join(" ".join(x.text for x in sorted(r, key=lambda x: x.left)) for r in rows)
    w = sum(len(l.text) for l in lines)
    conf = sum(l.conf * len(l.text) for l in lines) / w if w else 0.0
    return text, round(conf, 3)


def _cache_key(file_hash: str, page: int, langs: list[str]) -> Path:
    return cache_dir() / f"{file_hash}_{page}_{'-'.join(langs)}.json"


def file_hash(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:20]


class Budget:
    """Page / time limits shared by every file of one load_folder call."""

    def __init__(self, max_pages: int = MAX_PAGES_PER_RUN, max_seconds: float = MAX_SECONDS_PER_RUN):
        self.pages_left, self.deadline = max_pages, time.time() + max_seconds
        self.skipped = 0

    def take(self) -> bool:
        if self.pages_left <= 0 or time.time() > self.deadline:
            self.skipped += 1
            return False
        self.pages_left -= 1
        return True


def ocr_image(image, langs: list[str], cache_file: Path | None = None) -> dict:
    if cache_file is not None and cache_file.exists():
        try:
            return {**json.loads(cache_file.read_text(encoding="utf-8")), "cached": True}
        except (OSError, ValueError):
            pass
    text, conf = lines_to_text(get_engine(langs).recognize(image))
    res = {"text": text, "conf": conf}
    if cache_file is not None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass
    return res


def ocr_pdf_pages(path: Path, pages: list[int], langs: list[str], budget: Budget) -> dict[int, dict]:
    """OCR the given 0-based pages of a PDF. Returns {page: {text, conf, cached?}}; pages beyond the budget are absent."""
    import pypdfium2 as pdfium

    fh = file_hash(path)
    out: dict[int, dict] = {}
    pdf = pdfium.PdfDocument(str(path))
    try:
        for i in pages:
            cf = _cache_key(fh, i, langs)
            if not cf.exists() and not budget.take():
                continue
            img = None
            if not cf.exists():
                w, h = pdf[i].get_size()
                img = pdf[i].render(scale=min(3.0, max(1.0, RENDER_LONG_SIDE / max(w, h, 1)))).to_pil()
            out[i] = ocr_image(img, langs, cf)
    finally:
        pdf.close()
    return out


def ocr_image_file(path: Path, langs: list[str], budget: Budget) -> dict | None:
    from PIL import Image

    cf = _cache_key(file_hash(path), 0, langs)
    if not cf.exists() and not budget.take():
        return None
    img = None if cf.exists() else Image.open(path)
    return ocr_image(img, langs, cf)


def setup(codes: list[str]) -> dict:
    """Download the models for these languages (explicit, on request). Returns the new status."""
    import easyocr

    langs = reader_langs(codes)
    easyocr.Reader(langs, gpu=False, verbose=True, download_enabled=True)
    return status(codes)
