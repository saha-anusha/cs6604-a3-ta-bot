# Assignment 3 (A3): The Course TA-Bot — Conversational RAG with Memory

Build a teaching-assistant chatbot that retrieves from a corpus of course papers, answers
questions with grounded citations, and remembers facts about each user across sessions.

| | | | |
|---|---|---|---|
| **Due** | Mon, November 2, 11:59 pm ET |  |  |
| **Work mode** | Paired (teams of 2) | **Effort** | ≈25–30 hours |
| **Code** | Canvas (zip) | **Report** | Gradescope (PDF, ≤3 pages) |

> In Assignment 2 you taught a model *when* to ask versus answer. Here you build a full
> conversational system that answers from evidence, cites its sources, and learns what each
> user cares about — without memorizing things it should forget. The pipeline (chunking,
> indexing, retrieval, generation, memory) is yours to design; the interface contract and
> evaluation harness are fixed.

---

## Setup

```bash
pip install -r requirements.txt
```

> **On PACE-ICE:** your home directory is small. Point your Hugging Face cache at the shared
> read-only copy and run offline — no download needed:
>
> ```bash
> export HF_HOME=/storage/ice1/0/4/lheck6/CS6604_ECE6256/hf_cache
> export HF_HUB_OFFLINE=1
> export TRANSFORMERS_OFFLINE=1
> ```
>
> Available models: `Llama-3.2-1B-Instruct`, `Llama-3.2-3B-Instruct`, `Llama-3.1-8B-Instruct`,
> `openai/gpt-oss-20b`.
> The generation and embedding endpoint is the **course vLLM server** (see "Endpoint" below) —
> you do **not** load the model yourself for inference.

---

## Endpoint

All generation and embedding calls go through a single **vLLM-compatible OpenAI API** endpoint.

- **For graded runs** on PACE-ICE the TAs start a shared vLLM server. Your code receives the URL
  via `config["endpoint"]` — never hard-code it.
- **For local development** use the bundled mock endpoint:

```bash
python mock_endpoint.py                    # starts on a random port, prints URL
# or use eval_dev.py --mock (starts it for you)
```

The mock returns canned completions and deterministic embeddings (SHA-256-seeded, 128-dim).
It is good enough to verify your pipeline wiring — retrieval, memory, output format — before
you test on the real model.

---

## Starter Kit Contents

| File | Description |
|---|---|
| `ta_bot.py` | **Your main file.** Implements `TABot` — the class the harness imports. |
| `mock_endpoint.py` | Local mock vLLM server for offline testing. |
| `eval_dev.py` | Dev-set self-check script (see "Self-Check" below). |
| `corpus/` | Course paper corpus — `corpus.db` (SQLite with papers, chunks, FTS5 index). |
| `dev_eval/` | Small public dev set (single-turn, multi-turn, memory probes). |
| `requirements.txt` | Pinned dependencies. |
| `README.md` | This file. |

---

## What to Build

You implement the four methods of `TABot` in `ta_bot.py`. Do **not** rename the class or
change the method signatures — the autograder imports your file by name and calls these
methods directly.

### The interface contract

```python
class TABot:
    def __init__(self, config: dict) -> None:
        """
        Required config keys (set by the harness — do not hard-code):
            endpoint    (str)   — vLLM-compatible base URL
            model       (str)   — model name served at the endpoint
            corpus_path (str)   — path to the paper corpus directory
            temperature (float) — 0 for graded runs
            seed        (int)   — fixed seed for reproducibility
        You may add your own config keys for custom settings.
        """

    def chat(self, session_id: str, user_id: str, message: str) -> dict:
        """
        Returns {"answer": str, "citations": [paper_id, ...]}
        - answer:    natural-language response
        - citations: paper_id strings from the corpus papers table (paper-level, never chunk ids)
        """

    def reset_memory(self, user_id: str) -> None:
        """Empty the persistent memory store for this user."""

    def get_memory(self, user_id: str) -> list[str]:
        """Return stored facts about this user as readable strings (read-only view)."""
```

### Session vs. user semantics

- `session_id` scopes **conversational state** (history, anaphora). A new `session_id` starts a
  fresh conversation.
- `user_id` scopes **cross-session memory**. The same `user_id` across different sessions shares
  memory; a new `user_id` is a stranger.

### Error contract

- If the endpoint is unreachable or slow, raise `TimeoutError`. The harness catches it and marks
  that turn as failed — it does not kill your process.
- Any other unhandled exception scores zero for that turn.

---

## Your Design Space

The following are **your decisions** — there is no single correct answer:

1. **Chunking strategy** — how you split papers into pieces for indexing. Chunk size, overlap,
   and boundaries are free variables. Scoring is at paper level (INV-8), so your chunk design
   won't be penalised directly — only retrieval effectiveness matters.

2. **Retrieval method** — BM25, dense embeddings via the endpoint, hybrid, re-ranking, or
   anything else. The endpoint serves `/v1/embeddings` alongside `/v1/chat/completions`.

3. **Memory architecture** — how you store, update, and filter cross-session facts. The
   memory probes test both what you store (positive recall) and what you refuse to store
   (negative probes). Design accordingly.

4. **Prompt engineering** — system prompt, few-shot examples, chain-of-thought — all up to you,
   as long as they go through `config["endpoint"]`.

5. **Additional dependencies** — add to `requirements.txt`. Stick to standard PyPI packages.

---

## Self-Check (Dev Set)

A small public dev set ships in `dev_eval/`. Run the self-check locally:

```bash
# With the mock endpoint (offline, no GPU):
python eval_dev.py --mock

# With a real vLLM endpoint:
python eval_dev.py --endpoint http://localhost:8000/v1
```

The script reports:
- **Single-turn retrieval hit-rate** — did your citations include the gold paper?
- **Multi-turn retrieval hit-rate** — same, across multi-turn dialogues.
- **Memory probe pass/fail** — did your memory store the expected facts?

> **The dev set is small on purpose.** High dev scores do not guarantee high graded scores.
> The hidden test set covers different papers, conversation patterns, and memory scenarios
> than the dev set. Build a system that generalises, not one that overfits to these examples.

---

## Grading

### Pre-submission sanity check

Before submitting, run `eval_dev.py --mock` — it includes preflight checks that verify
your interface contract (import, `chat()` shape, `reset_memory()`, `get_memory()` return
type) before running the dev eval. Fix any preflight failures before submitting.

### Graded evaluation (PACE-ICE)

The TAs run your submission against the hidden test set with the real vLLM endpoint and
an LLM judge. Scores are uploaded to Canvas manually.

**Headline metrics:**

| Metric | What it measures |
|---|---|
| **Retrieval hit-rate** | % of questions where your citations include the gold paper |
| **Answer quality** | LLM-judge 1–5 score (accuracy, relevance, completeness, grounding) |
| **Memory recall** | % of positive probes where the planted fact is recalled |
| **Negative probe pass** | Must-not-store facts do not leak into memory or output |

**Thresholds** _(calibrated 2026-10-23)_:

| Metric | Passing Threshold | Notes |
|---|---|---|
| retrieval_hit_rate | **0.70** | Basic retrieval bar (FTS alone easily clears) |
| answer_quality | **0.76** | Primary discriminator — grounded, accurate answers |
| memory_recall | **0.80** | Allows 1 probe miss; negative_probe is the isolation gate |
| negative_probe | **pass** | Binary — must-not-store facts must not leak |

A naive baseline that does minimal retrieval and no memory filtering sets the floor; a
well-built system sets the ceiling. You need to meaningfully beat naive — not be perfect.

**Reproducibility:** with `temperature=0` and the fixed seed, your scores must reproduce
within tolerance across two runs. Large variance suggests non-determinism in your pipeline
(random chunking, unseeded sampling, etc.) — fix it.

**Judge caveat (on purpose):** you are being graded by an LLM judge two weeks before
Lecture 12 teaches you why that is dangerous. Up to **5 bonus points** for documenting a
case where the judge scored your system wrongly — with evidence.

---

## Grading Rubric

| Component | What full credit looks like | Points |
|---|---|---|
| **Harness scores** | Hit-rate and judged answer quality above posted thresholds; memory probes pass. | 35 |
| **Ablation + analysis** | One axis, honest measurement, a plot, and an interpretation that would survive a skeptical reviewer. | 35 |
| **Memory design writeup** | What you store, when you write, what you refuse to store and why. Privacy reasoning counts. | 15 |
| **Reproducibility** | Clean clone + endpoint reproduces harness scores within tolerance. | 10 |
| **AI-Use Disclosure** | Complete and candid. | 5 |
| **Total** | | **100** |

---

## Submission Checklist

Before submitting, verify:

- [ ] `python eval_dev.py --mock` runs without errors
- [ ] `chat()` returns the correct schema (`{"answer": str, "citations": [paper_id, ...]}`)
- [ ] Citations are `paper_id` strings from `corpus.db` (papers table), not chunk ids or filenames
- [ ] `reset_memory()` + `get_memory()` work correctly
- [ ] `requirements.txt` lists all your dependencies
- [ ] No hard-coded endpoints, file paths, or model names — everything comes from `config`

### What to submit

**1. Code (Canvas)** — Upload your project as a zip. Must include `ta_bot.py` at the root,
plus `requirements.txt` with all dependencies. The TAs run your code on PACE-ICE with the
hidden test set and real vLLM endpoint.

**2. Report (Gradescope, PDF, ≤3 pages)** — Must include:

- **Ablation study** (R4): one axis (rewriting on/off, k, chunk size, embedding model, etc.),
  honest measurement, a plot, and an interpretation that would survive a skeptical reviewer.
- **Memory design writeup** (R3): what you store, when you write, what you deliberately refuse
  to store and why. Privacy reasoning counts.
- **AI-Use Disclosure** (≤half page, included in the 3-page limit): which AI tools you used,
  for what, and how you verified the results. A candid disclosure never costs you points on
  the work itself; an empty or evasive disclosure on work that plainly used AI is an
  honor-code matter.

---

## Academic Integrity

- Build your own pipeline. You may use any libraries, but the retrieval, memory, and
  prompting logic must be your own work.
- Do **not** reverse-engineer the hidden test set or the judge prompt.
- Do **not** hard-code answers or paper ids — the grader detects this.
- Do **not** use external APIs (OpenAI, Anthropic, etc.) — all calls go through the
  course vLLM endpoint only.
