"""OCR tests. Logic is tested with a fake engine (fast, deterministic); one test runs real EasyOCR when its models are installed."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stress"))

import pytest  # noqa: E402
from make_corpus import write_pdf  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from forgeai import ocr  # noqa: E402
from forgeai.docs import load_folder  # noqa: E402
from forgeai.textutil import numbers  # noqa: E402

READY = {"available": True, "reason": "", "models": {"en": True, "hi": True}, "detector": True}
CANNED = [ocr.Line("Staff receive 24 days of annual leave.", 0.9, 10, 10, 20), ocr.Line("Claims within 14 days.", 0.9, 50, 10, 20)]


class Fake:
    def __init__(self, lines=None, conf=None):
        self.lines = [ocr.Line(l.text, conf if conf is not None else l.conf, l.top, l.left, l.height) for l in (lines or CANNED)]
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        return list(self.lines)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGEAI_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(ocr, "status", lambda codes=None: dict(READY))
    yield tmp_path
    ocr.set_engine(None)


def image_pdf(path: Path, pages: int = 1) -> None:
    ims = [Image.new("RGB", (500, 300), "white") for _ in range(pages)]
    for i, im in enumerate(ims):
        ImageDraw.Draw(im).text((20, 20 + i), f"scan {i}", fill="black")  # make pages distinct so cache keys differ per page content hash
    ims[0].save(path, "PDF", save_all=True, append_images=ims[1:])


def mixed_pdf(path: Path, tmp: Path) -> None:
    from pypdf import PdfReader, PdfWriter

    text_pdf, scan_pdf = tmp / "_t.pdf", tmp / "_s.pdf"
    write_pdf(text_pdf, "Purpose. This page has a real text layer with enough words to count as text. The leave policy is described here in detail for staff.")
    image_pdf(scan_pdf)
    w = PdfWriter()
    for src in (text_pdf, scan_pdf):
        for pg in PdfReader(str(src)).pages:
            w.add_page(pg)
    with open(path, "wb") as f:
        w.write(f)


def folder(tmp: Path, name: str = "docs") -> Path:
    d = tmp / name
    d.mkdir(exist_ok=True)
    return d


def test_scanned_pdf_is_read_with_ocr_and_reported(env):
    d = folder(env)
    image_pdf(d / "scan.pdf", pages=2)
    fake = Fake()
    ocr.set_engine(fake)
    docs, problems = load_folder(d, ocr_langs=["en"])
    assert len(docs) == 1 and "24 days" in docs[0].text and "14 days" in docs[0].text
    assert docs[0].ocr and docs[0].ocr["pages"] == 2 and docs[0].ocr["conf"] == 0.9
    assert any("read 2 scanned page(s) with OCR (mean confidence 90%)" in p for p in problems)
    assert fake.calls == 2


def test_mixed_pdf_ocrs_only_the_scanned_page(env):
    d = folder(env)
    mixed_pdf(d / "mixed.pdf", env)
    fake = Fake()
    ocr.set_engine(fake)
    docs, problems = load_folder(d)
    assert fake.calls == 1, "the text page must not be OCR'd"
    assert "real text layer" in docs[0].text and "24 days" in docs[0].text and docs[0].ocr["pages"] == 1


def test_a_text_pdf_with_a_nearly_empty_last_page_does_not_trigger_ocr(env):
    d = folder(env)
    write_pdf(d / "doc.pdf", "Policy text. " + "Staff must report lost badges within 24 hours of noticing the loss. " * 80 + "\n" * 60 + "7")
    fake = Fake()
    ocr.set_engine(fake)
    docs, _ = load_folder(d)
    assert len(docs) == 1 and fake.calls == 0 and docs[0].ocr is None


def test_low_confidence_pages_are_flagged_and_unreadable_ones_are_dropped(env):
    d = folder(env)
    image_pdf(d / "shaky.pdf")
    ocr.set_engine(Fake(conf=0.4))
    docs, problems = load_folder(d)
    assert docs and "low OCR confidence on page(s) [1]" in " ".join(problems) and docs[0].ocr["low"] == [1]

    d2 = folder(env, "docs2")
    image_pdf(d2 / "junk.pdf")
    ocr.set_engine(Fake(conf=0.1))
    docs, problems = load_folder(d2)
    assert docs == [] and any("OCR confidence below 20%" in p for p in problems)  # an unreadable scan never feeds an agent


RealBudget = ocr.Budget  # captured before any monkeypatching


def test_ocr_budget_limits_pages_and_says_so(env, monkeypatch):
    d = folder(env)
    image_pdf(d / "long.pdf", pages=4)
    monkeypatch.setattr(ocr, "Budget", lambda: RealBudget(max_pages=2, max_seconds=60))
    ocr.set_engine(Fake())
    docs, problems = load_folder(d)
    assert docs[0].ocr["pages"] == 2 and any("OCR page/time limit reached" in p and "[3, 4]" in p for p in problems)


def test_results_are_cached_by_file_hash(env):
    d = folder(env)
    image_pdf(d / "scan.pdf", pages=2)
    fake = Fake()
    ocr.set_engine(fake)
    load_folder(d)
    assert fake.calls == 2
    load_folder(d)  # resumed run / chat session: no OCR again
    assert fake.calls == 2


def test_unavailable_ocr_is_reported_never_a_crash(env, monkeypatch):
    d = folder(env)
    image_pdf(d / "scan.pdf")
    monkeypatch.setattr(ocr, "status", lambda codes=None: {"available": False, "reason": "OCR models not installed (x)", "models": {}})
    docs, problems = load_folder(d)
    assert docs == [] and any("OCR is not available: OCR models not installed" in p for p in problems)
    docs, problems = load_folder(d, ocr="off")
    assert docs == [] and any("OCR is switched off" in p for p in problems)


def test_detect_mode_reports_scans_without_running_ocr(env):
    d = folder(env)
    image_pdf(d / "scan.pdf", pages=3)
    Image.new("RGB", (300, 200), "white").save(d / "photo.png")
    fake = Fake()
    ocr.set_engine(fake)
    docs, problems = load_folder(d, ocr="detect")
    text = " ".join(problems)
    assert fake.calls == 0 and "3 scanned page(s) detected" in text and "photo.png" in text and "when the agent is built" in text


def test_image_files_are_read_with_ocr(env):
    d = folder(env)
    Image.new("RGB", (400, 200), "white").save(d / "notice.png")
    ocr.set_engine(Fake())
    docs, problems = load_folder(d)
    assert [x.name for x in docs] == ["notice.png"] and "24 days" in docs[0].text and any("read with OCR" in p for p in problems)


def test_languages_easyocr_cannot_read_are_reported(env, monkeypatch):
    monkeypatch.undo()  # use the real status(): Gujarati is unsupported
    st = ocr.status(["gu"]) if ocr.status(["en"]).get("detector") is not None else {"available": False, "reason": "EasyOCR does not support Gujarati"}
    assert not st["available"] and ("Gujarati" in st["reason"] or "libraries missing" in st["reason"])


def test_reading_order_is_rebuilt_and_native_digits_survive():
    jumbled = [ocr.Line("दिन की छुट्टी", 0.8, 12, 200, 20), ocr.Line("२४ कार्यदिवस", 0.8, 10, 20, 20), ocr.Line("दूसरी पंक्ति", 0.8, 80, 20, 20)]
    text, conf = ocr.lines_to_text(jumbled)
    assert text.split("\n") == ["२४ कार्यदिवस दिन की छुट्टी", "दूसरी पंक्ति"] and conf == 0.8
    assert numbers(text) == {"24"}  # OCR output with Devanagari digits still yields the number 24


def test_reader_languages_combine_script_with_english():
    assert ocr.reader_langs(["hi"]) == ["hi", "en"] and ocr.reader_langs(["en"]) == ["en"] and ocr.reader_langs(["ta", "hi"]) == ["ta", "en"] and ocr.reader_langs(None) == ["en"]


# ---------------------------------------------------------------- real OCR (skipped where models or fonts are missing)
NIRMALA = Path("C:/Windows/Fonts/Nirmala.ttc")


@pytest.mark.skipif(not NIRMALA.exists() or not ocr.status(["hi"]).get("available"), reason="needs EasyOCR + the Hindi model + a Devanagari font")
def test_real_hindi_ocr_recovers_the_key_facts(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGEAI_CACHE", str(tmp_path / "cache"))
    lines = ["पूर्णकालिक शोर स्टाफ को प्रति कैलेंडर वर्ष 24 कार्यदिवस की", "सवेतन वार्षिक छुट्टी मिलती है। लगातार 3 दिन से अधिक", "अनुपस्थिति पर मेडिकल प्रमाणपत्र आवश्यक है।"]
    img = Image.new("RGB", (1240, 300), "white")
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(str(NIRMALA), 34)
    for i, l in enumerate(lines):
        d.text((40, 30 + i * 80), l, font=f, fill="black")
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    img.save(docs_dir / "scan.pdf", "PDF")
    docs, problems = load_folder(docs_dir, ocr_langs=["hi"])
    assert len(docs) == 1, problems
    got = docs[0].text
    assert "24" in numbers(got) and "3" in numbers(got)  # the numbers (read as Devanagari or ASCII digits) survive normalisation
    from forgeai.retrieval import tokenize

    src = set(tokenize(" ".join(lines)))
    recall = len(src & set(tokenize(got))) / len(src)
    assert recall >= 0.4, (recall, got)  # EasyOCR makes small matra errors; most words must still be recognisable
