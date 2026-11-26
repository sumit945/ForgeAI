"""Turns demos/*/sources/*.txt into real text PDFs (pure python, no dependencies) and copies CSVs, into demos/*/docs/."""
import shutil, sys, textwrap
from pathlib import Path

def esc(s): return s.replace("\\", "\\\\").replace("(", "\(").replace(")", "\)")

def make_pdf(text: str, out: Path):
    lines = []
    for para in text.split("\n"):
        if not para.strip():
            lines.append("")
        else:
            lines += textwrap.wrap(para, 92) or [""]
    per_page = 50
    pages = [lines[i:i + per_page] for i in range(0, len(lines), per_page)] or [[]]
    objs = []
    def add(b): objs.append(b); return len(objs)
    add(b"<< /Type /Catalog /Pages 2 0 R >>")
    add(b"")  # pages placeholder
    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    kids = []
    for pg in pages:
        stream = ["BT", "/F1 10 Tf", "13 TL", "50 790 Td"]
        for ln in pg:
            stream.append(f"({esc(ln.encode('cp1252', 'replace').decode('cp1252'))}) Tj T*")
        stream.append("ET")
        data = "\n".join(stream).encode("cp1252", "replace")
        c = add(b"<< /Length %d >>\nstream\n" % len(data) + data + b"\nendstream")
        p = add(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents {c} 0 R /Resources << /Font << /F1 {font} 0 R >> >> >>".encode())
        kids.append(p)
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>".encode()
    buf = bytearray(b"%PDF-1.4\n"); offs = []
    for i, o in enumerate(objs, 1):
        offs.append(len(buf)); buf += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    x = len(buf)
    buf += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for o in offs: buf += f"{o:010d} 00000 n \n".encode()
    buf += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{x}\n%%EOF\n".encode()
    out.write_bytes(bytes(buf))

root = Path(__file__).parent
for demo in ("hr_policy", "ship_spares"):
    src, dst = root / demo / "sources", root / demo / "docs"
    if dst.exists(): shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in sorted(src.iterdir()):
        if f.suffix == ".txt": make_pdf(f.read_text(encoding="utf-8"), dst / (f.stem + ".pdf"))
        else: shutil.copy(f, dst / f.name)
    print(demo, sorted(p.name for p in dst.iterdir()))
