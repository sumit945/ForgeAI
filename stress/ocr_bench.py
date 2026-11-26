"""OCR benchmark: scanned Hindi policy (clean and degraded scans) -> OCR -> facts and retrieval.

  python stress/ocr_bench.py

Measures, on the real EasyOCR Hindi model (CPU): seconds per page, mean confidence, how many pages the confidence gate keeps, whether the
numbers of the source survive, word recall, and whether BM25 / hybrid retrieval still finds the right passage for Hindi questions
compared with the clean text. Requires the Hindi OCR model, a Devanagari font (Nirmala) and, for the hybrid column, bge-m3 in Ollama.
"""
from __future__ import annotations

import json
import random
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402

from forgeai import ocr  # noqa: E402
from forgeai.docs import Document, chunk_documents, load_folder  # noqa: E402
from forgeai.embeddings import OllamaEmbedder  # noqa: E402
from forgeai.lang import normalize_text  # noqa: E402
from forgeai.retrieval import HybridRetriever, Retriever, tokenize  # noqa: E402
from forgeai.textutil import numbers  # noqa: E402

FONT = "C:/Windows/Fonts/Nirmala.ttc"
SRC = ROOT / "demos" / "hindi_hr" / "docs" / "avkash_neeti.txt"
QA = [("पूर्णकालिक शोर स्टाफ को साल में कितने दिन की वार्षिक छुट्टी मिलती है?", "24"), ("परिवीक्षा अवधि में अधिकतम कितने दिन की छुट्टी ली जा सकती है?", "5"),
      ("बीमारी की छुट्टी के लिए मेडिकल प्रमाणपत्र कब आवश्यक है?", "3"), ("पितृत्व अवकाश कितने सप्ताह का होता है?", "4"), ("शोर स्टाफ को साल में कितने सार्वजनिक अवकाश मिलते हैं?", "11")]


def render(text: str, path: Path, degrade: bool) -> int:
    f = ImageFont.truetype(FONT, 30)
    W, H, M = 1240, 1754, 70
    lines: list[str] = []
    for para in text.split("\n"):
        cur = ""
        for w in para.split():
            t = (cur + " " + w).strip()
            if ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(t, font=f) > W - 2 * M:
                lines.append(cur)
                cur = w
            else:
                cur = t
        lines.append(cur)
    per = (H - 2 * M) // 52
    pages = [lines[i : i + per] for i in range(0, len(lines), per)]
    ims = []
    rnd = random.Random(3)
    for pl in pages:
        im = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(im)
        for i, l in enumerate(pl):
            d.text((M, M + i * 52), l, font=f, fill="black")
        if degrade:  # a poor scan: low resolution, blur, noise, slight tilt
            im = im.resize((W // 2, H // 2)).filter(ImageFilter.GaussianBlur(0.9)).rotate(0.8, fillcolor="white")
            px = im.load()
            for _ in range(W * H // 60):
                x, y = rnd.randrange(im.width), rnd.randrange(im.height)
                v = rnd.randrange(60, 200)
                px[x, y] = (v, v, v)
        ims.append(im)
    ims[0].save(path, "PDF", save_all=True, append_images=ims[1:])
    return len(ims)


def retrieval(text: str, hybrid: bool) -> int:
    chunks = chunk_documents([Document("d", "d", "text", text)], 40, 8)
    r = HybridRetriever(chunks, OllamaEmbedder("bge-m3")) if hybrid else Retriever(chunks)
    return sum(num in normalize_text(" ".join(c.text for c, _ in r.search(q, 3))) for q, num in QA)


def main() -> None:
    src_text = SRC.read_text(encoding="utf-8")
    src_nums = numbers(src_text)
    src_words = set(tokenize(src_text))
    base = {"bm25": retrieval(src_text, False)}
    try:
        base["hybrid"] = retrieval(src_text, True)
    except Exception as e:  # noqa: BLE001
        base["hybrid"] = f"n/a ({type(e).__name__})"
    print("clean-text retrieval baseline (of 5 questions):", base, flush=True)
    results = {"baseline_retrieval": base, "source_numbers": len(src_nums)}
    for name, deg in (("clean_scan", False), ("poor_scan", True)):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            n = render(src_text, td / "scan.pdf", deg)
            t0 = time.time()
            docs, problems = load_folder(td, ocr_langs=["hi"])
            secs = time.time() - t0
            if not docs:
                results[name] = {"pages": n, "seconds": round(secs, 1), "kept": 0, "problems": problems}
                print(name, "-> no usable text:", problems, flush=True)
                continue
            got = docs[0].text
            gn = numbers(got)
            row = {"pages": n, "seconds": round(secs, 1), "seconds_per_page": round(secs / n, 1), "mean_confidence": docs[0].ocr["conf"], "pages_used": docs[0].ocr["pages"],
                   "low_confidence_pages": docs[0].ocr["low"], "numbers_recovered": f"{len(src_nums & gn)}/{len(src_nums)}",
                   "word_recall": round(len(src_words & set(tokenize(got))) / len(src_words), 2), "retrieval_bm25": retrieval(got, False), "messages": problems}
            try:
                row["retrieval_hybrid"] = retrieval(got, True)
            except Exception as e:  # noqa: BLE001
                row["retrieval_hybrid"] = f"n/a ({type(e).__name__})"
            results[name] = row
            print(name, {k: v for k, v in row.items() if k != "messages"}, flush=True)
            for m in problems:
                print("   ", m, flush=True)
    out = ROOT / "stress" / "results" / "ocr"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ocr_bench.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
