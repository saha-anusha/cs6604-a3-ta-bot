"""
CS 6604 / ECE 6256 — Assignment 3: The Course TA-Bot
Conversational RAG with Memory

Starter stub. Implement the methods below.
Do NOT rename the class or change the method signatures.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests


# ============================================================================================
# PART 1 — RETRIEVAL  (Anusha)
# ============================================================================================

STOP = set("""
a an the of to in on for by with and or is are was were be been being it its this that these those
what which who whom how why when where does do did done can could would should will shall may might
about from as at into than then there their them they we you your i me my our us he she his her
explain describe tell please give show summarize summarise between vs versus compare comparison
paper papers approach approaches method methods work works use used using uses also just really
some any all each other more most such only own same so very s t can't don't let lets let's
""".split())

ANAPHORA = re.compile(
    r"\b(it|its|that|this|these|those|they|them|their|above|previous|earlier|former|latter|"
    r"same|both|all (two|three|four)|we discussed|you mentioned|the paper|that paper|"
    r"that approach|this approach|that method|this method)\b", re.I)
TOPIC_SWITCH = re.compile(
    r"\b(switch(ing)? (topics?|gears)|change (of )?(topics?|subject)|changing (topics?|subject)|"
    r"different (topic|question|subject)|new (topic|question)|unrelated|moving on|on another note)\b",
    re.I)
SUMMARY_ALL = re.compile(
    r"\b(all (two|three|four|five|of them|of these|of those|the (approaches|methods|papers|strategies))|"
    r"we('ve| have)? discussed|so far|everything|each of (them|these|those)|compare (them|these|those)|"
    r"these (two|three|four) )\b", re.I)
NUMBER_WORDS = {"two": 2, "both": 2, "three": 3, "four": 4, "five": 5}

RRF_K = 60


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 1]


class SessionContext:
    """Per-session retrieval state (owned by the bot, one per session_id)."""

    def __init__(self) -> None:
        self.user_turns: list[str] = []      # raw user messages
        self.queries: list[str] = []         # contextualised queries actually used
        self.primary: list[str] = []         # top-1 paper of each turn (what the turn was "about")
        self.cited: list[str] = []           # every paper cited so far, in order


class Retriever:
    def __init__(self, corpus_path: str | Path, config: dict | None = None) -> None:
        cfg = dict(config or {})
        self.cfg = cfg
        db_path = Path(corpus_path) / "corpus.db"
        if not db_path.exists():
            raise FileNotFoundError(f"corpus.db not found under {corpus_path}")
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)

        # --- knobs (all overridable from the bot config) ---------------------------------
        self.k_papers = int(cfg.get("k_papers", 3))            # citations returned
        # generator context is packed by TOKEN BUDGET (not a fixed chunk count) so that
        # different chunk sizes are compared at equal context cost
        self.context_budget = int(cfg.get("context_budget", 1600))
        self.max_chunks_per_paper = int(cfg.get("max_chunks_per_paper", 0))   # 0 = no cap, budget only
        self.min_quality = float(cfg.get("min_quality", 0.4))  # drop tables/equations
        self.bm25_pool = int(cfg.get("bm25_pool", 200))        # chunks pulled from FTS per query
        self.w_title, self.w_section, self.w_text = cfg.get("bm25_weights", (4.0, 1.0, 1.0))
        self.ctx_weight = float(cfg.get("ctx_weight", 0.7))    # weight of the contextualised query
        self.use_dense = bool(cfg.get("dense", False))
        self.dense_weight = float(cfg.get("dense_weight", 1.0))
        self.rewrite_mode = cfg.get("rewrite", "heuristic")    # "none" | "heuristic" | "llm"
        if self.rewrite_mode is True:
            self.rewrite_mode = "heuristic"
        elif self.rewrite_mode is False:
            self.rewrite_mode = "none"

        self.titles: dict[str, str] = dict(self.db.execute("SELECT paper_id, title FROM papers"))

        # endpoint is only needed for dense retrieval and LLM rewriting
        self.endpoint = cfg.get("endpoint")
        self.model = cfg.get("model")
        self.embed_model = cfg.get("embed_model") or self.model
        self.timeout = float(cfg.get("timeout", 60))
        self.seed = int(cfg.get("seed", 42))
        self._dense = None
        if self.use_dense:
            self._dense = self._load_or_build_dense(Path(corpus_path))

    # =================================================================== public API
    def contextualise(self, message: str, sess: SessionContext, llm=None) -> str:
        """Turn a follow-up into a standalone retrieval query."""
        if self.rewrite_mode == "none" or not sess.user_turns:
            return message
        if TOPIC_SWITCH.search(message):
            return message
        if self.rewrite_mode == "llm" and llm is not None:
            return self._llm_rewrite(message, sess, llm)
        # heuristic: only borrow context when the turn leans on it
        needs_ctx = bool(ANAPHORA.search(message)) or len(_tokens(message)) <= 3
        if not needs_ctx:
            return message
        # previous *raw* turn (not the previous rewritten query, which would snowball) plus the
        # title of the paper that turn was about, so "it"/"that approach" resolves concretely
        prev = sess.user_turns[-1]
        prev_title = self.titles.get(sess.primary[-1], "") if sess.primary else ""
        return f"{message} {prev} {prev_title}".strip()

    def retrieve(self, query: str, message: str | None = None) -> tuple[list[dict], list[str]]:
        """Return (chunks for the generator, ranked paper ids).

        `query` is the contextualised query; `message` is the raw user turn. When they
        differ, both are searched and fused, with the raw turn weighted higher — so context
        resolves "it"/"that" without drowning out what the user just asked about.
        """
        rankings: list[tuple[list[str], float]] = []

        bm25_chunks = self._bm25(query)
        bm25_papers = list(dict.fromkeys(c["paper_id"] for c in bm25_chunks))
        if message and message != query:
            cur_chunks = self._bm25(message)
            rankings.append((list(dict.fromkeys(c["paper_id"] for c in cur_chunks)), 1.0))
            rankings.append((bm25_papers, self.ctx_weight))
            bm25_chunks = cur_chunks + bm25_chunks
        else:
            rankings.append((bm25_papers, 1.0))

        dense_chunks: list[dict] = []
        if self._dense is not None:
            dense_chunks = self._dense_search(query)
            rankings.append((list(dict.fromkeys(c["paper_id"] for c in dense_chunks)), self.dense_weight))

        scores: dict[str, float] = defaultdict(float)
        for ranked, w in rankings:
            for r, pid in enumerate(ranked):
                scores[pid] += w / (RRF_K + r + 1)
        papers = sorted(scores, key=lambda p: (-scores[p], p))   # tie-break on id -> deterministic

        chunks = self._chunks_for_papers(message or query, papers[: self.k_papers],
                                         bm25_chunks + dense_chunks)
        return chunks, papers

    def select_citations(self, message: str, papers: list[str], sess: SessionContext) -> list[str]:
        """Pick the paper ids to cite for this turn and update the session state."""
        cites = papers[: self.k_papers]
        m = SUMMARY_ALL.search(message)
        if m and sess.primary:
            n = len(sess.primary)
            for word, val in NUMBER_WORDS.items():
                if re.search(rf"\b{word}\b", message, re.I):
                    n = val
                    break
            earlier = list(dict.fromkeys(sess.primary))[-n:]
            cites = list(dict.fromkeys(earlier + cites))[: max(self.k_papers, len(earlier))]
        return cites

    def update_session(self, sess: SessionContext, message: str, query: str,
                       papers: list[str], cites: list[str]) -> None:
        sess.user_turns.append(message)
        sess.queries.append(query)
        if cites or papers:
            # what this turn was "about": the best-ranked cited paper not already a primary
            # (so "compare all three" gets three distinct papers, not one paper three times)
            pool = cites or papers
            fresh = [p for p in pool if p not in sess.primary]
            sess.primary.append(fresh[0] if fresh else pool[0])
        sess.cited.extend(p for p in cites if p not in sess.cited)

    # =================================================================== BM25
    def _fts_query(self, text: str) -> str:
        toks = list(dict.fromkeys(_tokens(text)))
        return " OR ".join(f'"{t}"' for t in toks)

    def _bm25(self, query: str) -> list[dict]:
        q = self._fts_query(query)
        if not q:
            return []
        # bm25() weights follow the FTS column order: chunk_id, paper_id, title, section, text
        rows = self.db.execute(
            f"""SELECT c.chunk_id, c.paper_id, c.section, c.text, c.n_tokens,
                       bm25(chunks_fts, 0, 0, {self.w_title}, {self.w_section}, {self.w_text}) AS s
                FROM chunks_fts JOIN chunks c USING(chunk_id)
                WHERE chunks_fts MATCH ? AND c.quality >= ?
                ORDER BY s, c.chunk_id LIMIT ?""",
            (q, self.min_quality, self.bm25_pool)).fetchall()
        return [{"chunk_id": cid, "paper_id": pid, "section": sec, "text": txt, "n_tokens": nt,
                 "score": -s} for cid, pid, sec, txt, nt, s in rows]

    def _best_chunk_in_paper(self, query: str, pid: str) -> dict | None:
        q = self._fts_query(query)
        row = None
        if q:
            row = self.db.execute(
                f"""SELECT c.chunk_id, c.section, c.text, c.n_tokens
                    FROM chunks_fts JOIN chunks c USING(chunk_id)
                    WHERE chunks_fts MATCH ? AND c.paper_id = ? AND c.quality >= ?
                    ORDER BY bm25(chunks_fts, 0, 0, {self.w_title}, {self.w_section}, {self.w_text}),
                             c.chunk_id LIMIT 1""", (q, pid, self.min_quality)).fetchone()
        if row is None:   # fall back to the paper's first chunk (usually the abstract)
            row = self.db.execute("SELECT chunk_id, section, text, n_tokens FROM chunks WHERE paper_id=? "
                                  "ORDER BY chunk_index LIMIT 1", (pid,)).fetchone()
        if row is None:
            return None
        return {"chunk_id": row[0], "paper_id": pid, "section": row[1], "text": row[2],
                "n_tokens": row[3], "score": 0.0}

    def _chunks_for_papers(self, query: str, papers: list[str], pool: list[dict]) -> list[dict]:
        """Context for the generator, packed to `context_budget` tokens.

        1. the best chunk of the top-ranked paper (always included, even if it alone exceeds
           the budget),
        2. then chunks from the cited papers in score order, until the token budget is used.
           A chunk that would overflow the budget is skipped. Optional per-paper cap.
        """
        allowed = set(papers)
        ranked, seen = [], set()
        for c in pool:                                   # pool is already in score order
            if c["chunk_id"] not in seen and c["paper_id"] in allowed:
                seen.add(c["chunk_id"])
                ranked.append(c)
        firsts = []
        if papers:
            best = next((c for c in ranked if c["paper_id"] == papers[0]), None) \
                or self._best_chunk_in_paper(query, papers[0])
            if best:
                firsts.append(best)
        cap = self.max_chunks_per_paper or 10 ** 9
        out, used, per = [], 0, defaultdict(int)
        for c in firsts + [c for c in ranked if c not in firsts]:
            if c["chunk_id"] in {o["chunk_id"] for o in out} or per[c["paper_id"]] >= cap:
                continue
            n = int(c.get("n_tokens") or len(c["text"].split()) * 1.3)
            if used + n > self.context_budget and out:
                continue
            out.append(c)
            used += n
            per[c["paper_id"]] += 1
        return out

    # =================================================================== dense (optional)
    def _base(self) -> str:
        ep = (self.endpoint or "").rstrip("/")
        return ep if ep.endswith("/v1") else ep + "/v1"

    def _embed(self, texts: list[str]) -> np.ndarray:
        try:
            r = requests.post(f"{self._base()}/embeddings", timeout=self.timeout,
                              json={"model": self.embed_model, "input": texts})
            r.raise_for_status()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            raise TimeoutError(str(e)) from e
        vecs = np.array([d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])],
                        dtype=np.float32)
        return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)

    def _load_or_build_dense(self, corpus_path: Path):
        rows = self.db.execute("SELECT chunk_id, paper_id, section, text FROM chunks "
                               "WHERE quality >= ? ORDER BY chunk_id", (self.min_quality,)).fetchall()
        key = hashlib.sha256((str(self.embed_model) + "|" + "|".join(r[0] for r in rows))
                             .encode()).hexdigest()[:16]
        cache_dir = Path(self.cfg.get("cache_dir") or Path(__file__).resolve().parent / ".cache")
        cache = cache_dir / f"dense_{key}.npy"
        if cache.exists():
            mat = np.load(cache)
        else:
            parts = [self._embed([r[3][:2000] for r in rows[i:i + 64]]) for i in range(0, len(rows), 64)]
            mat = np.vstack(parts)
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                np.save(cache, mat)
            except OSError:
                pass
        meta = [{"chunk_id": r[0], "paper_id": r[1], "section": r[2], "text": r[3]} for r in rows]
        return mat, meta

    def _dense_search(self, query: str, k: int = 50) -> list[dict]:
        mat, meta = self._dense
        q = self._embed([query])[0]
        sims = mat @ q
        idx = np.argsort(-sims, kind="stable")[:k]
        return [dict(meta[i], score=float(sims[i])) for i in idx]

    # =================================================================== LLM rewrite (optional)
    def _llm_rewrite(self, message: str, sess: SessionContext, llm) -> str:
        history = "\n".join(f"- {t}" for t in sess.user_turns[-3:])
        prompt = [
            {"role": "system", "content":
                "Rewrite the user's latest question into a single standalone search query for a "
                "corpus of NLP research papers. Resolve pronouns and references like 'it', 'that "
                "approach' or 'all three' using the earlier questions. Keep method and paper names. "
                "If the latest question starts a new topic, return it unchanged. "
                "Output only the rewritten query."},
            {"role": "user", "content": f"Earlier questions:\n{history}\n\nLatest question: {message}"},
        ]
        out = (llm(prompt, max_tokens=80) or "").strip().strip('"').splitlines()
        out = out[0].strip() if out else ""
        # guard against junk / canned replies: fall back to the raw message
        if not out or len(out) > 400 or "mock response" in out.lower():
            return message
        return out


# ============================================================================================
# TABot: harness interface + generation  (memory + the prompt in _generate: Yu)
# ============================================================================================


class TABot:
    """Course TA-Bot with RAG retrieval and cross-session memory."""

    def __init__(self, config: dict) -> None:
        """
        Initialise the TA-Bot.

        Parameters
        ----------
        config : dict
            Required keys set by the harness (do not hard-code their values):
                endpoint   : str   – vLLM-compatible base URL
                model      : str   – model name served at the endpoint
                corpus_path: str   – path to the paper corpus directory
                temperature: float – generation temperature (0 for graded runs)
                seed       : int   – random seed for reproducibility
            You may add custom keys for your own use.
        """
        self.config = config
        # TODO: initialise your retrieval pipeline, memory store, etc.
        ep = config["endpoint"].rstrip("/")
        self.base = ep if ep.endswith("/v1") else ep + "/v1"   # mock URL has no /v1, real one does
        self.model = config["model"]
        self.temperature = float(config.get("temperature", 0.0))
        self.seed = int(config.get("seed", 42))
        self.timeout = float(config.get("timeout", 60))

        # retrieval knobs can be overridden via config (see Retriever.__init__ for the list)
        self.retriever = Retriever(config["corpus_path"], config)
        self.sessions: dict[str, SessionContext] = {}
        self.history: dict[str, list[tuple[str, str]]] = {}


    def chat(
        self,
        session_id: str,
        user_id: str,
        message: str,
    ) -> dict:
        """
        Process one user turn and return an answer with citations.

        Parameters
        ----------
        session_id : str
            Conversation scope — a new session_id starts a fresh
            conversation but keeps the user's cross-session memory.
        user_id : str
            Memory scope — identifies the user whose memory store
            to read from and write to.
        message : str
            The user's message.

        Returns
        -------
        dict
            {"answer": str, "citations": list[str]}
            - answer: the bot's response (complete, no streaming)
            - citations: list of paper_id strings from the corpus
              that were used to produce the answer. May be empty
              if no retrieval was needed.

        Raises
        ------
        TimeoutError
            You MUST raise TimeoutError if the endpoint is
            unreachable or a request exceeds your timeout.
            The harness catches it and marks this turn as 0.
            Do NOT let the process hang — raise, don't wait.
        """
        # TODO: implement retrieval, generation, and memory update
        sess = self.sessions.setdefault(session_id, SessionContext())
        hist = self.history.setdefault(session_id, [])

        query = self.retriever.contextualise(message, sess, llm=self._llm)
        chunks, papers = self.retriever.retrieve(query, message=message)
        cites = self.retriever.select_citations(message, papers, sess)
        if set(cites) - {c["paper_id"] for c in chunks}:
            # summary turns ("compare all three") cite earlier papers: make sure they are grounded
            chunks, _ = self.retriever.retrieve(" ".join([query] + [self.retriever.titles[p] for p in cites]),
                                                message=message)
        self.retriever.update_session(sess, message, query, papers, cites)

        answer = self._generate(user_id, message, hist, chunks)
        hist += [("user", message), ("assistant", answer)]
        return {"answer": answer, "citations": cites}

    def reset_memory(self, user_id: str) -> None:
        """
        Clear all stored memory for the given user.

        The harness calls this before each memory-probe sequence
        to ensure a clean slate. After this call, ``get_memory(user_id)``
        must return an empty list.

        Parameters
        ----------
        user_id : str
            The user whose memory to clear.
        """
        # TODO: clear the memory store for this user
        raise NotImplementedError("Implement reset_memory")

    def get_memory(self, user_id: str) -> list[str]:
        """
        Return the current memory store for the given user as a
        list of human-readable fact strings.

        This is read-only — calling it must not change the store.
        The harness uses it to inspect what your bot chose to remember
        (and what it chose not to remember).

        Any internal representation is fine as long as each entry
        serialises to a readable string.

        Parameters
        ----------
        user_id : str
            The user whose memory to inspect.

        Returns
        -------
        list[str]
            Stored facts as readable strings. Empty list if nothing
            is stored (or after ``reset_memory``).
        """
        # TODO: return the contents of the memory store
        raise NotImplementedError("Implement get_memory")

    # ------------------------------------------------------------------ generation (Yu)
    def _generate(self, user_id: str, message: str, hist: list, chunks: list[dict]) -> str:
        titles = self.retriever.titles
        ctx = "\n\n".join(
            f"[{c['paper_id']}] {titles.get(c['paper_id'], '')} — {c['section']}\n{c['text']}"
            for c in chunks) or "(no relevant passages found)"
        system = (
            "You are a teaching assistant for a graduate course on conversational AI. "
            "Answer ONLY from the provided passages. Cite papers inline by their id in brackets, e.g. [L04-04]. "
            "If the passages do not contain the answer, say so briefly. Be accurate, specific and concise "
            "(about 120-200 words)."
        )
        msgs = [{"role": "system", "content": system}]
        for role, text in hist[-6:]:
            msgs.append({"role": role, "content": text})
        msgs.append({"role": "user", "content": f"Passages:\n{ctx}\n\nQuestion: {message}"})
        return self._llm(msgs)

    def _llm(self, messages: list[dict], max_tokens: int = 400) -> str:
        try:
            r = requests.post(f"{self.base}/chat/completions", timeout=self.timeout, json={
                "model": self.model, "messages": messages, "temperature": self.temperature,
                "seed": self.seed, "max_tokens": max_tokens})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            raise TimeoutError(str(e)) from e
