# A3 corpus · The Course TA-Bot

The reading list of ECE 6256 / CS 6604 (Fall 2026) as a retrieval corpus: 108 open-access
papers, chosen from the readings cited in Lectures 1–24, with extracted text, a default
chunking, and a SQLite database. Everything here is derived from arXiv; the PDFs are not
committed (fetch them with the script below).

## What is in the box

| Path | What | Notes |
|---|---|---|
| `manifest.csv` / `manifest.json` | one row per paper: `paper_id`, `lecture`, `arxiv_id`, `title`, `authors`, `year`, `published`, `abstract`, `url_abs`, `url_pdf`, `pdf_sha256`, `n_pages`, `n_chars`, `n_chunks` | metadata resolved from the arXiv API; `paper_id` is `L<lecture>-<nn>` |
| `text/<paper_id>.txt` | full text, one file per paper | `pdftotext` (reading-order mode), hyphenation re-joined, form feeds removed; references section kept in the text file (dropped at chunking) |
| `chunks.jsonl` | the default chunking | ~400 tokens, 15 % overlap, section-aware; fields: `chunk_id` (`<paper_id>#<nnn>`), `paper_id`, `chunk_index`, `section`, `title`, `lecture`, `n_tokens`, `quality`, `text` |
| `corpus.db` | SQLite | tables `papers`, `chunks`; virtual table `chunks_fts` (FTS5, porter stemming) — a BM25 lexical baseline for free |
| `manifest_seed.csv` | the hand-curated seed (id, lecture, expected title) the build starts from | edit this to add or remove papers |
| `manifest_seed_full.csv` | the longer 161-paper candidate list the 108 were pruned from | extra distractors / readings if you want a bigger corpus |
| `build_corpus.py` | the whole pipeline | `python3 build_corpus.py all` rebuilds everything |

## Rebuilding (TA / students)

```bash
pip install requests tiktoken          # tiktoken optional: exact cl100k token counts
python3 build_corpus.py all            # verify → fetch (≈10 min, arXiv rate limit) → extract → chunk → db
python3 build_corpus.py chunk db --chunk-tokens 250 --overlap 0.10   # re-chunk for the R4 ablation
```

`fetch` only downloads PDFs that are missing, and the manifest records each PDF's SHA-256, so a
rebuild on another machine can be checked against this one. `pdftotext` (poppler) must be on PATH.

## Using the DB

```python
import sqlite3
con = sqlite3.connect("corpus.db")

# BM25 baseline over chunks (R1 lexical retriever; also a good hybrid partner for dense retrieval)
q = "conversational query rewriting"
rows = con.execute("""
  SELECT c.chunk_id, p.title, c.section, bm25(chunks_fts) AS score, substr(c.text, 1, 200)
  FROM chunks_fts JOIN chunks c USING(chunk_id) JOIN papers p USING(paper_id)
  WHERE chunks_fts MATCH ? ORDER BY score LIMIT 5""", (q,)).fetchall()

# everything a dense retriever needs: (chunk_id, text) pairs to embed with the course endpoint
pairs = con.execute("SELECT chunk_id, text FROM chunks").fetchall()
```

Citations in generated answers should name the paper (`papers.title`, `papers.arxiv_id`) and
may name the chunk (`chunk_id`) so the harness can check retrieval hit-rate.

## Chunking rules (defaults in the kit)

Section headings matching the usual paper skeleton (Abstract, Introduction, Related Work, Method,
Experiments, Results, Discussion, Conclusion, Appendix, …) start a new section; the References
section and anything after it is excluded. Within a section, paragraphs are packed into windows
of about 400 tokens; when a window closes, the trailing paragraphs worth ~15 % of the budget are
carried into the next window as overlap. Paragraphs longer than 1.5× the budget are split at
sentence boundaries. Chunks under 30 tokens are dropped. Nothing else is filtered: tables, equations and the occasional garbled figure font are kept, and each chunk carries a `quality` score (fraction of tokens that are ordinary words; prose ≈ 0.7+, tables/equations ≈ 0–0.4) so you can filter them yourself — e.g. `WHERE quality >= 0.5` for a prose-only index, which is itself a reasonable R4 ablation axis. Token counts use `cl100k_base` when
`tiktoken` is importable and a whitespace × 1.3 estimate otherwise (the `n_tokens` column says
which — see `tokenizer` in `manifest.json`).

## Coverage by lecture

L1–L17 carry 4–10 papers each (the material taught before A3 is due); L18–L24 carry one anchor
paper each so the TA-bot can answer about the rest of the semester too. The full candidate list in
`manifest_seed_full.csv` has 161 entries if a larger corpus is wanted.

## Licensing

All papers are arXiv e-prints distributed under arXiv's non-exclusive license or a CC license;
the corpus is for course use only. Each `manifest` row carries the `url_abs` to cite.
