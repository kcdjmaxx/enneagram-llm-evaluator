#!/usr/bin/env python3
"""
Run Enneagram tests (Likert + paired) multiple times with an Ollama model
and aggregate all results into ONE markdown file.

- Uses:
    tests/enneagram_likert.json
    tests/enneagram_test.json

- For each test:
    * Runs N times (default N=3)
    * Records Enneagram-type scores (1–9)
    * Computes mean and standard deviation per type across runs
    * Computes Center-of-Intelligence scores (Head/Heart/Gut)

- Output:
    results/enneagram-multi_<model>_<YYYY-MM-DD>.md
"""

import argparse
import datetime as dt
import json
import math
import pathlib
import re
import textwrap
from typing import Dict, List, Tuple

import requests

# Ollama HTTP endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"

# Enneagram centers
CENTER_MAP = {
    "head": [5, 6, 7],
    "heart": [2, 3, 4],
    "gut": [8, 9, 1],
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-")


def stddev(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var)


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def ollama_generate(model: str, prompt: str) -> str:
    """
    Call Ollama's /api/generate endpoint (non-streaming) and return the response text.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


def ask_choice_ab(model: str, question_text: str) -> str:
    """
    Ask the model to choose A or B.
    Returns normalized "A" or "B".
    """
    prompt = textwrap.dedent(
        f"""
        You are taking a two-choice (A/B) personality test.

        For each item you will be given two statements, labeled A and B.
        Pick whichever statement fits you better OVER MOST OF YOUR LIFE.

        Respond with ONLY a single letter:
        - 'A' if statement A fits better
        - 'B' if statement B fits better

        Do NOT include any explanation or extra text.

        {question_text}

        Your answer (A or B only):
        """
    ).strip()

    raw = ollama_generate(model, prompt).upper()
    match = re.search(r"\b([AB])\b", raw)
    if match:
        return match.group(1)

    if raw.startswith("A"):
        return "A"
    if raw.startswith("B"):
        return "B"
    return "A"


def ask_likert_1_to_5(model: str, question_text: str) -> int:
    """
    Ask the model to rate from 1 to 5.
    Returns an integer 1..5.
    """
    prompt = textwrap.dedent(
        f"""
        You are taking a personality test that uses a 1–5 Likert scale.

        For each statement, answer with a number from 1 to 5:
        1 = Almost Never
        2 = Rarely
        3 = Sometimes
        4 = Frequently
        5 = Almost Always

        Respond with ONLY the digit 1, 2, 3, 4, or 5.
        Do NOT include any explanation or extra text.

        Statement:
        {question_text}

        Your answer (1–5 only):
        """
    ).strip()

    raw = ollama_generate(model, prompt)
    match = re.search(r"\b([1-5])\b", raw)
    if match:
        return int(match.group(1))

    for ch in raw:
        if ch in "12345":
            return int(ch)

    return 3


# ---------------------------------------------------------------------------
# Single-run implementations (Likert + Paired)
# ---------------------------------------------------------------------------

def run_likert_once(model: str, json_path: pathlib.Path) -> Dict:
    """
    Run the Likert-style Enneagram test ONCE and return a dict:

    {
        "test_name": str,
        "scores_by_enneagram_type": {1: score, ..., 9: score},
        "top_types": [(etype, score), ...],  # sorted desc
    }
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    test_name: str = data["test_name"]
    types = data["types"]

    # score per A–I and then aggregate into Enneagram 1–9
    type_scores_A_I: Dict[str, int] = {k: 0 for k in types.keys()}
    scores_by_enneagram: Dict[int, int] = {}

    for type_key, tinfo in types.items():
        stmts = tinfo["statements"]
        for idx, stmt in enumerate(stmts, start=1):
            question_text = f"[Type {type_key}] Item {idx}:\n{stmt}"
            rating = ask_likert_1_to_5(model, question_text)
            type_scores_A_I[type_key] += rating

    # aggregate into Enneagram types
    for type_key, score in type_scores_A_I.items():
        e_type = types[type_key]["maps_to_enneagram_type"]
        scores_by_enneagram[e_type] = scores_by_enneagram.get(e_type, 0) + score

    top_types = sorted(
        scores_by_enneagram.items(), key=lambda x: x[1], reverse=True
    )

    return {
        "test_name": test_name,
        "scores_by_enneagram_type": scores_by_enneagram,
        "top_types": top_types,
    }


def run_paired_once(model: str, json_path: pathlib.Path) -> Dict:
    """
    Run the paired-question Enneagram test ONCE and return a dict:

    {
        "test_name": str,
        "counts_by_enneagram_type": {1: count, ..., 9: count},
        "top_types": [(etype, count), ...],  # sorted desc
    }
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    test_name: str = data["test_name"]
    columns = data["columns"]
    items = data["items"]

    counts_by_ennea: Dict[int, int] = {}

    for item in items:
        qid = item["id"]
        pair = item["pair"]

        a = next(p for p in pair if p["side"].upper() == "A")
        b = next(p for p in pair if p["side"].upper() == "B")

        question_text = textwrap.dedent(
            f"""
            Question {qid}:

            A) {a['text']}
            B) {b['text']}
            """
        ).strip()

        choice = ask_choice_ab(model, question_text)
        chosen = a if choice == "A" else b
        col = chosen["column"]
        e_type = columns[col]["type"]

        counts_by_ennea[e_type] = counts_by_ennea.get(e_type, 0) + 1

    top_types = sorted(
        counts_by_ennea.items(), key=lambda x: x[1], reverse=True
    )

    return {
        "test_name": test_name,
        "counts_by_enneagram_type": counts_by_ennea,
        "top_types": top_types,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_enneagram_stats(runs: List[Dict[int, int]]) -> Dict[int, Dict]:
    """
    Given a list of dicts: [{type: score, ...}, ...] for multiple runs,
    compute mean and stddev per type.

    Returns:
    {
        e_type: {
            "values": [...],
            "mean": float,
            "std": float
        },
        ...
    }
    """
    all_types = sorted({t for r in runs for t in r.keys()})
    result: Dict[int, Dict] = {}
    for t in all_types:
        vals = [r.get(t, 0) for r in runs]
        mean = sum(vals) / len(vals)
        sd = stddev(vals)
        result[t] = {
            "values": vals,
            "mean": mean,
            "std": sd,
        }
    return result


def center_scores_for_run(scores: Dict[int, int]) -> Dict[str, int]:
    """
    Compute Head / Heart / Gut totals for a single run, given an {type: score} dict.
    """
    center_totals = {}
    for center, types in CENTER_MAP.items():
        center_totals[center] = sum(scores.get(t, 0) for t in types)
    return center_totals


def aggregate_centers(runs: List[Dict[int, int]]) -> Dict[str, Dict]:
    """
    Compute center-of-intelligence stats across runs:
    {
        "head": {"values": [...], "mean": float, "std": float},
        "heart": {...},
        "gut": {...}
    }
    """
    center_vals = {c: [] for c in CENTER_MAP.keys()}
    for r in runs:
        cs = center_scores_for_run(r)
        for c in center_vals.keys():
            center_vals[c].append(cs[c])

    out = {}
    for c, vals in center_vals.items():
        mean = sum(vals) / len(vals)
        sd = stddev(vals)
        out[c] = {"values": vals, "mean": mean, "std": sd}
    return out


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------

def write_multi_markdown(
    model: str,
    n_runs: int,
    likert_runs: List[Dict],
    paired_runs: List[Dict],
    outdir: pathlib.Path,
) -> pathlib.Path:
    today = dt.date.today().isoformat()
    slug_model = slugify(model)
    out_path = outdir / f"enneagram-multi_{slug_model}_{today}.md"

    # Extract raw score dicts for stats
    likert_score_runs = [
        r["scores_by_enneagram_type"] for r in likert_runs
    ]
    paired_count_runs = [
        r["counts_by_enneagram_type"] for r in paired_runs
    ]

    likert_stats = aggregate_enneagram_stats(likert_score_runs)
    paired_stats = aggregate_enneagram_stats(paired_count_runs)

    likert_center_stats = aggregate_centers(likert_score_runs)
    paired_center_stats = aggregate_centers(paired_count_runs)

    # Top types per run
    def top3_str(top_list: List[Tuple[int, int]]) -> str:
        return ", ".join(
            [f"Type {t} ({v})" for t, v in top_list[:3]]
        )

    lines: List[str] = []
    lines.append(f"# Enneagram LLM Multi-Run Report")
    lines.append("")
    lines.append(f"- **Model:** `{model}`")
    lines.append(f"- **Date:** {today}")
    lines.append(f"- **Runs per test:** {n_runs}")
    lines.append("")
    lines.append(
        "This file aggregates multiple runs of two Enneagram tests "
        "for the same LLM model to analyze consistency, variability, and centers."
    )
    lines.append("")

    # ------------------------------------------------------------------
    # 1. LIKERT TEST SUMMARY
    # ------------------------------------------------------------------
    likert_name = likert_runs[0]["test_name"] if likert_runs else "Likert Test"
    lines.append("## 1. Likert Test – Multi-Run Summary")
    lines.append("")
    lines.append(f"**Test name:** {likert_name}")
    lines.append("")
    lines.append("### 1.1 Primary Type Tally per Run (Likert)")
    lines.append("")
    for i, run in enumerate(likert_runs, start=1):
        lines.append(
            f"- Run {i}: top types → {top3_str(run['top_types'])}"
        )
    lines.append("")

    # Table of scores per type per run
    lines.append("### 1.2 Scores by Enneagram Type Across Runs (Likert)")
    lines.append("")
    header = "| Type | " + " | ".join(
        [f"Run {i}" for i in range(1, n_runs + 1)]
    ) + " | Mean | σ |"
    sep = "|------|" + "|".join(["------" for _ in range(n_runs + 2)]) + "|"
    lines.append(header)
    lines.append(sep)

    for t in sorted(likert_stats.keys()):
        vals = likert_stats[t]["values"]
        mean = likert_stats[t]["mean"]
        sd = likert_stats[t]["std"]
        vals_str = " | ".join(f"{v:.0f}" for v in vals)
        lines.append(
            f"| {t} | {vals_str} | {mean:.2f} | {sd:.2f} |"
        )
    lines.append("")

    # Centers
    lines.append("### 1.3 Centers of Intelligence per Run (Likert)")
    lines.append("")
    lines.append(
        "Head = Types 5, 6, 7 &nbsp;&nbsp; "
        "Heart = Types 2, 3, 4 &nbsp;&nbsp; "
        "Gut = Types 8, 9, 1"
    )
    lines.append("")
    header = "| Center | " + " | ".join(
        [f"Run {i}" for i in range(1, n_runs + 1)]
    ) + " | Mean | σ |"
    lines.append(header)
    lines.append(sep)

    for center in ["head", "heart", "gut"]:
        cs = likert_center_stats[center]
        vals_str = " | ".join(f"{v:.0f}" for v in cs["values"])
        lines.append(
            f"| {center.capitalize()} | {vals_str} | {cs['mean']:.2f} | {cs['std']:.2f} |"
        )
    lines.append("")

    # ------------------------------------------------------------------
    # 2. PAIRED TEST SUMMARY
    # ------------------------------------------------------------------
    paired_name = paired_runs[0]["test_name"] if paired_runs else "Paired Test"
    lines.append("## 2. Paired A/B Test – Multi-Run Summary")
    lines.append("")
    lines.append(f"**Test name:** {paired_name}")
    lines.append("")
    lines.append("### 2.1 Primary Type Tally per Run (Paired)")
    lines.append("")
    for i, run in enumerate(paired_runs, start=1):
        lines.append(
            f"- Run {i}: top types → {top3_str(run['top_types'])}"
        )
    lines.append("")

    # Table of counts per type per run
    lines.append("### 2.2 Selections by Enneagram Type Across Runs (Paired)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for t in sorted(paired_stats.keys()):
        vals = paired_stats[t]["values"]
        mean = paired_stats[t]["mean"]
        sd = paired_stats[t]["std"]
        vals_str = " | ".join(f"{v:.0f}" for v in vals)
        lines.append(
            f"| {t} | {vals_str} | {mean:.2f} | {sd:.2f} |"
        )
    lines.append("")

    # Centers
    lines.append("### 2.3 Centers of Intelligence per Run (Paired)")
    lines.append("")
    lines.append(
        "Head = Types 5, 6, 7 &nbsp;&nbsp; "
        "Heart = Types 2, 3, 4 &nbsp;&nbsp; "
        "Gut = Types 8, 9, 1"
    )
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for center in ["head", "heart", "gut"]:
        cs = paired_center_stats[center]
        vals_str = " | ".join(f"{v:.0f}" for v in cs["values"])
        lines.append(
            f"| {center.capitalize()} | {vals_str} | {cs['mean']:.2f} | {cs['std']:.2f} |"
        )
    lines.append("")

    # ------------------------------------------------------------------
    # 3. Quick Interpretation Hooks (for your analysis)
    # ------------------------------------------------------------------
    lines.append("## 3. How to Use These Stats (Cheat Sheet)")
    lines.append("")
    lines.append("- **High consistency (low σ) for a type** → stable trait in the model.")
    lines.append("- **High variability (high σ) for a type** → volatile or prompt-sensitive trait.")
    lines.append("- **Dominant center across runs** → primary processing mode:")
    lines.append("  - Head (5/6/7) → thinking, anticipating, planning")
    lines.append("  - Heart (2/3/4) → relating, identity, image")
    lines.append("  - Gut (8/9/1) → instinct, control, anger")
    lines.append("")
    lines.append(
        "You can now compare this file across models, or rerun the same model with "
        "different prompts or temperatures and see how the Enneagram profile shifts."
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Enneagram Likert + Paired tests multiple times with an Ollama model and aggregate into one markdown file."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model name, e.g. 'mistral', 'llama3', 'qwen2:7b', etc.",
    )
    parser.add_argument(
        "--tests-dir",
        default="tests",
        help="Directory containing enneagram_likert.json and enneagram_test.json.",
    )
    parser.add_argument(
        "--outdir",
        default="results",
        help="Directory to write the aggregated markdown file into.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="How many times to run each test (default: 3).",
    )
    args = parser.parse_args()

    tests_dir = pathlib.Path(args.tests_dir)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    likert_path = tests_dir / "enneagram_likert.json"
    paired_path = tests_dir / "enneagram_test.json"

    if not likert_path.exists():
        raise FileNotFoundError(f"Likert test file not found: {likert_path}")
    if not paired_path.exists():
        raise FileNotFoundError(f"Paired test file not found: {paired_path}")

    likert_runs = []
    paired_runs = []

    print(f"Model: {args.model}")
    print(f"Runs per test: {args.runs}")

    for i in range(1, args.runs + 1):
        print(f"\n=== RUN {i} / {args.runs} – Likert test ===")
        likert_res = run_likert_once(args.model, likert_path)
        likert_runs.append(likert_res)

        print(f"\n=== RUN {i} / {args.runs} – Paired test ===")
        paired_res = run_paired_once(args.model, paired_path)
        paired_runs.append(paired_res)

    out_path = write_multi_markdown(
        args.model,
        args.runs,
        likert_runs,
        paired_runs,
        outdir,
    )
    print(f"\nAggregated multi-run markdown written to: {out_path}")


if __name__ == "__main__":
    main()
