#!/usr/bin/env python3
"""
eval_dev.py — Local dev-set evaluation for A3.

Runs your TABot against the dev eval data and reports:
  - Single-turn retrieval hit-rate (paper-level)
  - Multi-turn retrieval hit-rate
  - Memory probe pass/fail (positive probes only — no Judge)

This is a quick self-check. The real grading (Tier-2) runs on PACE-ICE
with the hidden set and the Llama Judge.

Usage
-----
    # With the mock endpoint (no GPU needed):
    python eval_dev.py --mock

    # With the real vLLM endpoint:
    python eval_dev.py --endpoint http://... --model llama-3.1-...

    # Specify corpus path (default: ./corpus)
    python eval_dev.py --mock --corpus ./my_corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEV_DIR = SCRIPT_DIR / "dev_eval"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping.")
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def hit_rate(predicted: list[str], gold: list[str]) -> float:
    """Paper-level hit rate: fraction of gold papers found in predicted."""
    if not gold:
        return 1.0
    gold_set = set(gold)
    return sum(1 for p in gold_set if p in predicted) / len(gold_set)


def run_single_turn(bot, data: list[dict], session_prefix: str = "eval-st") -> float:
    """Run single-turn questions, return mean hit-rate."""
    if not data:
        return 0.0
    scores = []
    for item in data:
        qid = item["id"]
        try:
            result = bot.chat(f"{session_prefix}-{qid}", "eval_user", item["question"])
            citations = result.get("citations", [])
            hr = hit_rate(citations, item.get("gold_paper_ids", []))
            scores.append(hr)
            gold = item.get('gold_paper_ids', [])
            status = "✓" if hr == 1.0 else f"partial ({hr:.0%})" if hr > 0 else "✗"
            print(f"  {qid}: {status}  cited={citations}  gold={gold}")
        except Exception as exc:
            scores.append(0.0)
            print(f"  {qid}: ERROR — {type(exc).__name__}: {exc}")
    mean = sum(scores) / len(scores) if scores else 0.0
    return mean


def run_multi_turn(bot, data: list[dict], session_prefix: str = "eval-mt") -> float:
    """Run multi-turn dialogues, return mean hit-rate on last turn."""
    if not data:
        return 0.0
    scores = []
    for item in data:
        qid = item["id"]
        try:
            result = None
            for turn in item["turns"]:
                result = bot.chat(
                    f"{session_prefix}-{qid}", "eval_user", turn["content"]
                )
            citations = result.get("citations", []) if result else []
            # Gold papers come from the last turn (per-turn schema)
            last_gold = (
                item["turns"][-1].get("gold_paper_ids", []) if item["turns"] else []
            )
            hr = hit_rate(citations, last_gold)
            scores.append(hr)
            gold = item.get('gold_paper_ids', [])
            status = "✓" if hr == 1.0 else f"partial ({hr:.0%})" if hr > 0 else "✗"
            print(f"  {qid}: {status}  cited={citations}  gold={gold}")
        except Exception as exc:
            scores.append(0.0)
            print(f"  {qid}: ERROR — {type(exc).__name__}: {exc}")
    return sum(scores) / len(scores) if scores else 0.0


def run_memory_probes(bot, data: list[dict]) -> tuple[int, int]:
    """Run memory probes. Returns (passed, total)."""
    if not data:
        return 0, 0
    passed = 0
    total = 0
    for item in data:
        pid = item["id"]
        probe_type = item.get("type", "positive")
        total += 1

        try:
            plant = item["plant_session"]
            test = item["test_session"]
            expected = item.get("expected_fact", "")

            # 1. Reset memory
            bot.reset_memory(plant["user_id"])

            # 2. Plant facts in session 1
            for turn in plant.get("turns", []):
                bot.chat(plant["session_id"], plant["user_id"], turn["content"])

            # 3. Inspect store
            memory = bot.get_memory(plant["user_id"])
            memory_text = " ".join(str(m) for m in memory).lower()

            # 4. Test in session 2
            test_turns = test.get("turns", [])
            answer = ""
            for t in test_turns:
                result = bot.chat(test["session_id"], test["user_id"], t["content"])
                answer = result.get("answer", "")
            answer = answer.lower()

            if probe_type == "positive":
                in_store = expected.lower() in memory_text
                in_output = expected.lower() in answer
                ok = in_store and in_output
                detail = f"store={'✓' if in_store else '✗'} output={'✓' if in_output else '✗'}"
            else:  # negative
                not_in_store = expected.lower() not in memory_text
                not_in_output = expected.lower() not in answer
                ok = not_in_store and not_in_output
                detail = f"store={'✓' if not_in_store else '✗'} output={'✓' if not_in_output else '✗'}"

            if ok:
                passed += 1
            print(f"  {pid} ({probe_type}): {'PASS' if ok else 'FAIL'} — {detail}")

        except NotImplementedError:
            print(f"  {pid} ({probe_type}): SKIP — not yet implemented")
        except Exception as exc:
            print(f"  {pid} ({probe_type}): ERROR — {type(exc).__name__}: {exc}")

    return passed, total


def run_preflight(bot) -> bool:
    """Quick structural checks (replaces the old Gradescope Tier-1 gate)."""
    print("=" * 60)
    print("PREFLIGHT — structural checks")
    print("-" * 60)
    ok = True

    # T1-2: chat() shape
    try:
        result = bot.chat("preflight-s", "preflight-u", "test")
        if not isinstance(result, dict):
            print(
                f"  chat() shape:    FAIL — returned {type(result).__name__}, expected dict"
            )
            ok = False
        elif "answer" not in result or "citations" not in result:
            print(f"  chat() shape:    FAIL — missing keys. Got: {list(result.keys())}")
            ok = False
        elif not isinstance(result["answer"], str):
            print(
                f"  chat() shape:    FAIL — answer is {type(result['answer']).__name__}, expected str"
            )
            ok = False
        elif not isinstance(result["citations"], list):
            print(
                f"  chat() shape:    FAIL — citations is {type(result['citations']).__name__}, expected list"
            )
            ok = False
        else:
            print("  chat() shape:    ✓  {answer: str, citations: list}")
    except TimeoutError:
        print("  chat() shape:    ✓  (TimeoutError — acceptable)")
    except Exception as exc:
        print(f"  chat() shape:    FAIL — {type(exc).__name__}: {exc}")
        ok = False

    # T1-3: reset_memory
    try:
        bot.reset_memory("preflight-u")
        print("  reset_memory():  ✓")
    except Exception as exc:
        print(f"  reset_memory():  FAIL — {type(exc).__name__}: {exc}")
        ok = False

    # T1-5: get_memory
    try:
        mem = bot.get_memory("preflight-u")
        if not isinstance(mem, list):
            print(
                f"  get_memory():    FAIL — returned {type(mem).__name__}, expected list"
            )
            ok = False
        elif any(not isinstance(m, str) for m in mem):
            print("  get_memory():    FAIL — list contains non-str items")
            ok = False
        else:
            print(f"  get_memory():    ✓  list[str] ({len(mem)} items)")
    except Exception as exc:
        print(f"  get_memory():    FAIL — {type(exc).__name__}: {exc}")
        ok = False

    if ok:
        print("  All preflight checks passed.")
    else:
        print("  ⚠ Some checks failed — fix these before submitting.")
    print()
    return ok


def main():
    parser = argparse.ArgumentParser(description="A3 dev-set evaluation")
    parser.add_argument(
        "--mock", action="store_true", help="Use the bundled mock endpoint"
    )
    parser.add_argument("--endpoint", type=str, default=None, help="vLLM endpoint URL")
    parser.add_argument("--model", type=str, default="mock-model", help="Model name")
    parser.add_argument(
        "--corpus",
        type=str,
        default=str(SCRIPT_DIR / "corpus"),
        help="Path to corpus directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Resolve endpoint
    server = None
    if args.mock:
        sys.path.insert(0, str(SCRIPT_DIR))
        from mock_endpoint import start_mock_endpoint

        endpoint, server = start_mock_endpoint()
        print(f"Mock endpoint started at {endpoint}\n")
    elif args.endpoint:
        endpoint = args.endpoint
    else:
        print("ERROR: specify --mock or --endpoint <url>", file=sys.stderr)
        sys.exit(1)

    config = {
        "endpoint": endpoint,
        "model": args.model,
        "corpus_path": args.corpus,
        "temperature": 0.0,
        "seed": args.seed,
    }

    # Import and instantiate
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        from ta_bot import TABot

        bot = TABot(config)
    except NotImplementedError:
        print("TABot.__init__ raises NotImplementedError — implement the stub first.")
        sys.exit(1)

    # Preflight checks
    run_preflight(bot)

    # Load dev data
    st_data = load_jsonl(DEV_DIR / "single_turn.jsonl")
    mt_data = load_jsonl(DEV_DIR / "multi_turn.jsonl")
    mp_data = load_jsonl(DEV_DIR / "memory_probes.jsonl")

    # Run
    print("=" * 60)
    print("SINGLE-TURN RETRIEVAL")
    print("-" * 60)
    st_hr = run_single_turn(bot, st_data)

    print()
    print("=" * 60)
    print("MULTI-TURN RETRIEVAL")
    print("-" * 60)
    mt_hr = run_multi_turn(bot, mt_data)

    print()
    print("=" * 60)
    print("MEMORY PROBES")
    print("-" * 60)
    mp_passed, mp_total = run_memory_probes(bot, mp_data)

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY (dev set — not your final grade)")
    print("-" * 60)
    print(f"  Single-turn hit-rate:  {st_hr:.1%}  ({len(st_data)} questions)")
    print(f"  Multi-turn hit-rate:   {mt_hr:.1%}  ({len(mt_data)} dialogues)")
    print(f"  Memory probes:         {mp_passed}/{mp_total} passed")
    print()
    print("Note: This uses dev data only. Your grade comes from")
    print("the hidden set on PACE-ICE with the Llama Judge.")
    print("=" * 60)

    if server:
        from mock_endpoint import stop_mock_endpoint

        stop_mock_endpoint(server)


if __name__ == "__main__":
    main()
