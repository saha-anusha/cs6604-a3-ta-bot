#!/usr/bin/env python3
"""
A3 · The Course TA-Bot — corpus builder.

Builds the reading-list corpus from manifest_seed.csv:

  1. verify   : resolve every arXiv id against the arXiv API; fill title / authors /
                abstract / dates; flag title mismatches  -> manifest.json, manifest.csv
  2. fetch    : download each PDF to pdfs/<paper_id>.pdf (skips files already present)
  3. extract  : pdftotext -layout  -> text/<paper_id>.txt   (+ light cleanup)
  4. chunk    : section-aware chunking, ~400 tokens, 15% overlap -> chunks.jsonl
  5. db       : SQLite  corpus.db  with  papers, chunks, and an FTS5 index (BM25 baseline)

Usage
    python3 build_corpus.py all                 # everything, in order
    python3 build_corpus.py verify fetch        # any subset, in order
    python3 build_corpus.py chunk db --chunk-tokens 250 --overlap 0.10

Requires: python >= 3.9, `requests`, poppler's `pdftotext` on PATH. Optional: `tiktoken`
(exact token counts; otherwise a whitespace×1.3 approximation is used).

Network: only https://export.arxiv.org (API) and https://arxiv.org (PDFs). arXiv asks for
<= 1 request / 3 s; this script sleeps accordingly (a full build is ~10 minutes).
"""

from __future__ import annotations

import argparse, csv, hashlib, json, os, re, sqlite3, subprocess, sys, time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = HERE / "manifest_seed.csv"
MANIFEST_JSON = HERE / "manifest.json"
MANIFEST_CSV = HERE / "manifest.csv"
PDF_DIR = HERE / "pdfs"
TXT_DIR = HERE / "text"
CHUNKS = HERE / "chunks.jsonl"
DB = HERE / "corpus.db"

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_PDF = "https://arxiv.org/pdf/{id}"
ARXIV_ABS = "https://arxiv.org/abs/{id}"
UA = {"User-Agent": "ConvAI-A3-corpus-builder/1.0 (course use; contact: instructor)"}
SLEEP = 3.0  # arXiv rate-limit courtesy


# ----------------------------------------------------------------------------- helpers
def log(msg: str) -> None:
    print(msg, flush=True)


def read_seed() -> list[dict]:
    with SEED.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("paper_id")]
    return rows


def load_manifest() -> list[dict]:
    if MANIFEST_JSON.exists():
        return json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    return read_seed()


def save_manifest(rows: list[dict]) -> None:
    MANIFEST_JSON.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    keys = [
        "paper_id",
        "lecture",
        "arxiv_id",
        "title",
        "authors",
        "year",
        "published",
        "url_abs",
        "url_pdf",
        "title_match",
        "pdf_sha256",
        "n_pages",
        "n_chars",
        "n_chunks",
        "abstract",
    ]
    with MANIFEST_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def title_similar(a: str, b: str) -> bool:
    a, b = norm_title(a), norm_title(b)
    if a == b or a in b or b in a:
        return True
    wa, wb = set(a.split()), set(b.split())
    return len(wa & wb) / max(1, len(wa | wb)) >= 0.6


# ----------------------------------------------------------------------------- 1 verify
def verify(rows: list[dict]) -> list[dict]:
    import requests

    ns = {"a": "http://www.w3.org/2005/Atom", "ar": "http://arxiv.org/schemas/atom"}
    ids = [r["arxiv_id"] for r in rows if r.get("arxiv_id")]
    meta: dict[str, dict] = {}
    for i in range(0, len(ids), 25):
        batch = ids[i : i + 25]
        resp = requests.get(
            ARXIV_API,
            params={"id_list": ",".join(batch), "max_results": len(batch)},
            headers=UA,
            timeout=60,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        for e in root.findall("a:entry", ns):
            aid = e.find("a:id", ns).text.rsplit("/abs/", 1)[-1]
            base = re.sub(r"v\d+$", "", aid)
            meta[base] = {
                "title": re.sub(r"\s+", " ", e.find("a:title", ns).text or "").strip(),
                "authors": "; ".join(
                    a.find("a:name", ns).text for a in e.findall("a:author", ns)
                ),
                "abstract": re.sub(
                    r"\s+", " ", e.find("a:summary", ns).text or ""
                ).strip(),
                "published": (e.find("a:published", ns).text or "")[:10],
                "latest_version": aid,
            }
        log(f"[verify] {min(i+25, len(ids))}/{len(ids)} resolved")
        time.sleep(SLEEP)

    bad = []
    for r in rows:
        m = meta.get(r["arxiv_id"])
        if not m:
            r["title_match"] = "NOT FOUND"
            bad.append(r)
            continue
        r.update(
            {
                "title": m["title"],
                "authors": m["authors"],
                "abstract": m["abstract"],
                "published": m["published"],
                "arxiv_version": m["latest_version"],
                "url_abs": ARXIV_ABS.format(id=r["arxiv_id"]),
                "url_pdf": ARXIV_PDF.format(id=r["arxiv_id"]),
            }
        )
        r["title_match"] = (
            "ok"
            if title_similar(m["title"], r.get("title_expected", ""))
            else "MISMATCH"
        )
        if r["title_match"] != "ok":
            bad.append(r)
    for r in bad:
        log(
            f"[verify] !! {r['paper_id']} {r['arxiv_id']} {r['title_match']}: expected '{r.get('title_expected')}' got '{r.get('title','')}'"
        )
    log(f"[verify] {len(rows)-len(bad)} ok, {len(bad)} to check")
    save_manifest(rows)
    return rows


# ----------------------------------------------------------------------------- 2 fetch
def fetch(rows: list[dict]) -> list[dict]:
    import requests

    PDF_DIR.mkdir(exist_ok=True)
    for r in rows:
        if r.get("title_match") in ("NOT FOUND", "MISMATCH"):
            log(f"[fetch] skip {r['paper_id']} ({r['title_match']})")
            continue
        out = PDF_DIR / f"{r['paper_id']}.pdf"
        if out.exists() and out.stat().st_size > 10_000:
            r["pdf_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
            continue
        url = r.get("url_pdf") or ARXIV_PDF.format(id=r["arxiv_id"])
        for attempt in range(3):
            try:
                resp = requests.get(url, headers=UA, timeout=120)
                if resp.status_code == 200 and resp.content[:4] == b"%PDF":
                    out.write_bytes(resp.content)
                    r["pdf_sha256"] = hashlib.sha256(resp.content).hexdigest()
                    log(f"[fetch] {r['paper_id']} {len(resp.content)/1e6:.1f} MB")
                    break
                log(
                    f"[fetch] {r['paper_id']} HTTP {resp.status_code} (attempt {attempt+1})"
                )
            except Exception as exc:  # noqa: BLE001
                log(f"[fetch] {r['paper_id']} error {exc} (attempt {attempt+1})")
            time.sleep(SLEEP * (attempt + 1))
        time.sleep(SLEEP)
    save_manifest(rows)
    return rows


# ----------------------------------------------------------------------------- 3 extract
HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
MULTI_BLANK = re.compile(r"\n{3,}")


def clean_text(t: str) -> str:
    t = t.replace("\f", "\n")
    t = HYPHEN_BREAK.sub(r"\1\2", t)  # re-join words broken across lines
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = MULTI_BLANK.sub("\n\n", t)
    return t.strip() + "\n"


def extract(rows: list[dict]) -> list[dict]:
    TXT_DIR.mkdir(exist_ok=True)
    for r in rows:
        pdf = PDF_DIR / f"{r['paper_id']}.pdf"
        if not pdf.exists():
            continue
        txt = TXT_DIR / f"{r['paper_id']}.txt"
        if not txt.exists():
            res = subprocess.run(
                ["pdftotext", "-enc", "UTF-8", str(pdf), "-"],
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                log(f"[extract] {r['paper_id']} pdftotext failed: {res.stderr[:200]}")
                continue
            txt.write_text(clean_text(res.stdout), encoding="utf-8")
        body = txt.read_text(encoding="utf-8")
        info = subprocess.run(
            ["pdfinfo", str(pdf)], capture_output=True, text=True
        ).stdout
        m = re.search(r"Pages:\s+(\d+)", info)
        r["n_pages"] = int(m.group(1)) if m else None
        r["n_chars"] = len(body)
        r["text_path"] = f"text/{r['paper_id']}.txt"
    log(f"[extract] {sum(1 for r in rows if r.get('n_chars'))} texts")
    save_manifest(rows)
    return rows


# ----------------------------------------------------------------------------- 4 chunk
try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def n_tokens(s: str) -> int:
        return len(_ENC.encode(s, disallowed_special=()))

except Exception:  # noqa: BLE001

    def n_tokens(s: str) -> int:
        return int(len(s.split()) * 1.3)


SECTION_WORDS = (
    r"(Abstract|Introduction|Related Work|Background|Preliminaries|Method(?:s|ology)?|Approach|Model|"
    r"Experiments?|Experimental Setup|Setup|Results|Analysis|Discussion|Limitations|Conclusions?|"
    r"Acknowledg(?:e)?ments|References|Bibliography|Appendix|Appendices|Ethics Statement|Broader Impact)"
)
# A heading is a SHORT line (<= 60 chars) that is either numbered ("3 Method", "4.1 Setup") or consists of the
# keyword alone (optionally followed by a few title-case words). Lines of running text never qualify.
SECTION_RE = re.compile(
    r"^[ \t]{0,3}(?:(?P<num>\d{1,2}(?:\.\d{1,2})?|[A-H]|[IVX]{1,4})[.)]?[ \t]+)?(?P<name>"
    + SECTION_WORDS
    + r")(?:[ \t]+(?:and|&|of|the|[A-Z][\w-]*)){0,5}[ \t]*$",
    re.MULTILINE,
)


def split_sections(text: str) -> list[tuple[str, str]]:
    """Returns [(section_name, section_text)], cutting at the References heading."""
    hits = []
    for m in SECTION_RE.finditer(text):
        line = text[
            m.start() : (
                text.find("\n", m.start())
                if text.find("\n", m.start()) != -1
                else len(text)
            )
        ]
        if len(line) > 60:
            continue
        name = m.group("name")
        # unnumbered single-word headings that are also common sentence openers need to stand alone on the line
        if not m.group("num"):
            standalone = line.strip().lower() == name.lower()
            if (
                name.lower()
                in {"approach", "setup", "results", "analysis", "background"}
                and not standalone
            ):
                continue  # running text that happens to start with the keyword
            if name.lower() == "model" and standalone:
                continue  # a bare "Model" line is almost always a table header, not a heading
        hits.append((m.start(), name.title()))
    if not hits:
        return [("Body", text)]
    secs, prev_pos, prev_name = [], 0, "Front matter"
    for pos, name in hits:
        if pos - prev_pos > 200:
            secs.append((prev_name, text[prev_pos:pos]))
            prev_pos, prev_name = pos, name
        else:
            prev_name = name
    secs.append((prev_name, text[prev_pos:]))
    # drop the bibliography (not useful retrieval text) but keep any appendix that follows it
    out = [
        (name, body)
        for name, body in secs
        if name.lower() not in ("references", "bibliography")
    ]
    return out or [("Body", text)]


def quality(text: str) -> float:
    """Fraction of whitespace tokens that are ordinary words (letters with a vowel). ~0.7+ is prose;
    tables, equations and garbled figure fonts score low. Stored so students can filter, not applied.
    """
    toks = text.split()
    words = [
        w for w in re.findall(r"[A-Za-z]{2,}", text) if re.search(r"[aeiouAEIOU]", w)
    ]
    return round(len(words) / max(1, len(toks)), 3)


def chunk_text(text: str, target: int, overlap: float) -> list[dict]:
    """Paragraph-packed windows of ~target tokens with ~overlap fraction carried over."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur, cur_tok = [], [], 0
    for p in paras:
        pt = n_tokens(p)
        if pt > target * 1.5:  # very long paragraph: split by sentences
            sents = re.split(r"(?<=[.!?])\s+", p)
            buf, bt = [], 0
            for s in sents:
                st = n_tokens(s)
                if bt + st > target and buf:
                    paras_chunk = " ".join(buf)
                    chunks.append(paras_chunk)
                    buf, bt = [], 0
                buf.append(s)
                bt += st
            if buf:
                chunks.append(" ".join(buf))
            continue
        if cur_tok + pt > target and cur:
            chunks.append("\n\n".join(cur))
            # carry overlap: keep trailing paragraphs worth ~overlap*target tokens
            keep, kt = [], 0
            for q in reversed(cur):
                qt = n_tokens(q)
                if kt + qt > overlap * target:
                    break
                keep.insert(0, q)
                kt += qt
            cur, cur_tok = keep, kt
        cur.append(p)
        cur_tok += pt
    if cur:
        chunks.append("\n\n".join(cur))
    final = []
    for c in chunks:
        if (
            n_tokens(c) > target * 1.5
        ):  # no paragraph/sentence boundaries (tables, lists): hard-split by lines
            lines, buf, bt = c.split("\n"), [], 0
            for ln in lines:
                lt = n_tokens(ln)
                if bt + lt > target and buf:
                    final.append("\n".join(buf))
                    buf, bt = [], 0
                buf.append(ln)
                bt += lt
            if buf:
                final.append("\n".join(buf))
        else:
            final.append(c)
    return [{"text": c, "n_tokens": n_tokens(c)} for c in final if n_tokens(c) >= 30]


def chunk(rows: list[dict], target: int, overlap: float) -> list[dict]:
    n_total = 0
    with CHUNKS.open("w", encoding="utf-8") as out:
        for r in rows:
            txt = TXT_DIR / f"{r['paper_id']}.txt"
            if not txt.exists():
                continue
            text = txt.read_text(encoding="utf-8")
            idx = 0
            for sec_name, sec_text in split_sections(text):
                for c in chunk_text(sec_text, target, overlap):
                    rec = {
                        "chunk_id": f"{r['paper_id']}#{idx:03d}",
                        "paper_id": r["paper_id"],
                        "chunk_index": idx,
                        "section": sec_name,
                        "title": r.get("title", ""),
                        "lecture": r["lecture"],
                        "n_tokens": c["n_tokens"],
                        "quality": quality(c["text"]),
                        "text": c["text"],
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    idx += 1
            r["n_chunks"] = idx
            n_total += idx
    for r in rows:
        r["tokenizer"] = "cl100k_base" if "_ENC" in globals() else "whitespace*1.3"
    log(
        f"[chunk] {n_total} chunks (~{target} tokens, {int(overlap*100)}% overlap) -> {CHUNKS.name}"
    )
    save_manifest(rows)
    return rows


# ----------------------------------------------------------------------------- 5 db
SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
  paper_id TEXT PRIMARY KEY, lecture TEXT, arxiv_id TEXT, arxiv_version TEXT,
  title TEXT, authors TEXT, year INTEGER, published TEXT, abstract TEXT,
  url_abs TEXT, url_pdf TEXT, pdf_sha256 TEXT, n_pages INTEGER, n_chars INTEGER, n_chunks INTEGER,
  text_path TEXT);
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY, paper_id TEXT REFERENCES papers(paper_id), chunk_index INTEGER,
  section TEXT, n_tokens INTEGER, quality REAL, text TEXT);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  chunk_id UNINDEXED, paper_id UNINDEXED, title, section, text, tokenize='porter unicode61');
"""


def build_db(rows: list[dict]) -> None:
    if DB.exists():
        DB.unlink()
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    con.executemany(
        "INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                r["paper_id"],
                r["lecture"],
                r.get("arxiv_id"),
                r.get("arxiv_version"),
                r.get("title"),
                r.get("authors"),
                int(r["year"]) if r.get("year") else None,
                r.get("published"),
                r.get("abstract"),
                r.get("url_abs"),
                r.get("url_pdf"),
                r.get("pdf_sha256"),
                r.get("n_pages"),
                r.get("n_chars"),
                r.get("n_chunks"),
                r.get("text_path"),
            )
            for r in rows
            if r.get("n_chars")
        ],
    )
    n = 0
    if CHUNKS.exists():
        with CHUNKS.open(encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                con.execute(
                    "INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                    (
                        c["chunk_id"],
                        c["paper_id"],
                        c["chunk_index"],
                        c["section"],
                        c["n_tokens"],
                        c.get("quality"),
                        c["text"],
                    ),
                )
                con.execute(
                    "INSERT INTO chunks_fts VALUES (?,?,?,?,?)",
                    (c["chunk_id"], c["paper_id"], c["title"], c["section"], c["text"]),
                )
                n += 1
    con.commit()
    np_ = con.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    con.close()
    log(
        f"[db] {DB.name}: {np_} papers, {n} chunks (+ FTS5 index; try: "
        "SELECT chunk_id, bm25(chunks_fts) AS s FROM chunks_fts WHERE chunks_fts MATCH 'query rewriting' ORDER BY s LIMIT 5)"
    )


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "steps", nargs="+", choices=["all", "verify", "fetch", "extract", "chunk", "db"]
    )
    ap.add_argument("--chunk-tokens", type=int, default=400)
    ap.add_argument("--overlap", type=float, default=0.15)
    args = ap.parse_args()
    steps = (
        ["verify", "fetch", "extract", "chunk", "db"]
        if "all" in args.steps
        else args.steps
    )
    rows = load_manifest()
    for s in steps:
        if s == "verify":
            rows = verify(rows)
        elif s == "fetch":
            rows = fetch(rows)
        elif s == "extract":
            rows = extract(rows)
        elif s == "chunk":
            rows = chunk(rows, args.chunk_tokens, args.overlap)
        elif s == "db":
            build_db(rows)


if __name__ == "__main__":
    main()
