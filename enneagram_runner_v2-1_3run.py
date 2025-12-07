#!/usr/bin/env python3
"""
Run Enneagram tests (Likert + Paired) multiple times with an Ollama LLM,
and write a single markdown file containing:

- Multi-run stats per Enneagram type (mean, stdev, per-run scores)
- Dominant types per run
- Full transcripts: every question and how the LLM answered it

JSON files expected in tests/:
- enneagram_likert.json
- enneagram_test.json
"""

import argparse
import datetime as dt
import json
import pathlib
import re
import statistics
import textwrap
from typing import Dict, List, Tuple, Any

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-")


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


def ask_choice_ab(model: str, item_prompt: str) -> Tuple[str, str]:
    """
    Ask the model to choose A or B.
    Returns (normalized_choice, raw_answer_text).
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

        {item_prompt}

        Your answer (A or B only):
        """
    ).strip()

    raw = ollama_generate(model, prompt).strip()
    upper = raw.upper()
    match = re.search(r"\b([AB])\b", upper)
    if match:
        return match.group(1), raw

    if upper.startswith("A"):
        return "A", raw
    if upper.startswith("B"):
        return "B", raw
    # very defensive fallback
    return "A", raw


def ask_likert_1_to_5(model: str, item_prompt: str) -> Tuple[int, str]:
    """
    Ask the model to rate from 1 to 5.
    Returns (rating_int, raw_answer_text).
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
        {item_prompt}

        Your answer (1–5 only):
        """
    ).strip()

    raw = ollama_generate(model, prompt).strip()
    match = re.search(r"\b([1-5])\b", raw)
    if match:
        return int(match.group(1)), raw

    for ch in raw:
        if ch in "12345":
            return int(ch), raw

    return 3, raw  # fallback


# ---------------------------------------------------------------------------
# Likert-style test, single run
# ---------------------------------------------------------------------------

def run_likert_once(
    model: str,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the Likert-style Enneagram test once and return a dict with:
    - 'enneagram_scores': {ennea_type -> score}
    - 'type_scores': {type_key -> score}
    - 'answers': list of question-level dicts including raw answers
    - 'dominant_types': list of (ennea_type, score) sorted desc
    """
    types = data["types"]

    type_scores: Dict[str, int] = {k: 0 for k in types.keys()}
    enneagram_scores: Dict[int, int] = {}
    answers: List[Dict[str, Any]] = []

    for type_key in sorted(types.keys()):
        tinfo = types[type_key]
        e_type = tinfo.get("maps_to_enneagram_type")
        stmts = tinfo["statements"]

        for idx, stmt in enumerate(stmts, start=1):
            item_prompt = f"[Type {type_key}] Item {idx}:\n{stmt}"
            rating, raw_answer = ask_likert_1_to_5(model, item_prompt)

            type_scores[type_key] += rating

            if e_type is not None:
                enneagram_scores[e_type] = enneagram_scores.get(e_type, 0) + rating

            answers.append(
                {
                    "type_key": type_key,
                    "item_index": idx,
                    "statement": stmt,
                    "rating": rating,
                    "raw_answer": raw_answer,
                    "enneagram_type": e_type,
                }
            )

    dominant_types = sorted(
        enneagram_scores.items(), key=lambda x: x[1], reverse=True
    )

    return {
        "enneagram_scores": enneagram_scores,
        "type_scores": type_scores,
        "answers": answers,
        "dominant_types": dominant_types,
    }


# ---------------------------------------------------------------------------
# Paired test, single run
# ---------------------------------------------------------------------------

def run_paired_once(
    model: str,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the paired-question Enneagram test once and return a dict with:
    - 'enneagram_counts': {ennea_type -> count}
    - 'column_counts': {column -> count}
    - 'answers': list of question-level dicts including raw answers
    - 'dominant_types': list of (ennea_type, count) sorted desc
    """
    columns = data["columns"]
    items = data["items"]

    column_counts: Dict[str, int] = {c: 0 for c in columns.keys()}
    enneagram_counts: Dict[int, int] = {}
    answers: List[Dict[str, Any]] = []

    for item in items:
        qid = item["id"]
        pair = item["pair"]

        a = next(p for p in pair if p["side"].upper() == "A")
        b = next(p for p in pair if p["side"].upper() == "B")

        item_prompt = textwrap.dedent(
            f"""
            Question {qid}:

            A) {a['text']}
            B) {b['text']}
            """
        ).strip()

        choice, raw_answer = ask_choice_ab(model, item_prompt)
        chosen = a if choice == "A" else b
        col = chosen["column"]
        column_counts[col] += 1

        e_type = columns[col]["type"]
        enneagram_counts[e_type] = enneagram_counts.get(e_type, 0) + 1

        answers.append(
            {
                "id": qid,
                "choice": choice,
                "raw_answer": raw_answer,
                "chosen_side": choice,
                "chosen_text": chosen["text"],
                "column": col,
                "enneagram_type": e_type,
                "a_text": a["text"],
                "b_text": b["text"],
                "a_column": a["column"],
                "b_column": b["column"],
            }
        )

    dominant_types = sorted(
        enneagram_counts.items(), key=lambda x: x[1], reverse=True
    )

    return {
        "enneagram_counts": enneagram_counts,
        "column_counts": column_counts,
        "answers": answers,
        "dominant_types": dominant_types,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_numeric_runs(
    runs: List[Dict[int, int]]
) -> Dict[int, Dict[str, Any]]:
    """
    Given a list of dicts like [{1: 10, 2: 3, ...}, {1: 11, ...}, ...],
    return {type: {"scores": [...], "mean": float, "stdev": float}}
    """
    all_types = set()
    for r in runs:
        all_types.update(r.keys())

    agg: Dict[int, Dict[str, Any]] = {}
    for t in sorted(all_types):
        scores = [r.get(t, 0) for r in runs]
        if len(scores) > 1:
            stdev = statistics.stdev(scores)
        else:
            stdev = 0.0
        agg[t] = {
            "scores": scores,
            "mean": statistics.mean(scores),
            "stdev": stdev,
            "min": min(scores),
            "max": max(scores),
        }
    return agg


# ---------------------------------------------------------------------------
# Markdown building
# ---------------------------------------------------------------------------

def build_markdown(
    model: str,
    likert_json: Dict[str, Any],
    paired_json: Dict[str, Any],
    likert_runs: List[Dict[str, Any]],
    paired_runs: List[Dict[str, Any]],
) -> str:
    today = dt.date.today().isoformat()
    lines: List[str] = []

    lines.append(f"# Enneagram Multi-Run Results – Model `{model}`")
    lines.append("")
    lines.append(f"_Date: {today}_")
    lines.append("")
    lines.append(
        "This document contains results for running **two Enneagram tests** "
        "multiple times with the same LLM:"
    )
    lines.append("")
    lines.append("- **Likert-style Enneagram Assessment** (1–5 scale)")
    lines.append("- **Paired A/B Enneagram Test** (forced choice)")
    lines.append("")
    lines.append(
        "Each test was run multiple times (N runs), to evaluate consistency, variability, "
        "and the stability of the model's Enneagram-like 'personality' profile."
    )
    lines.append("")

    # ------------------------------------------------------------------
    # Likert section
    # ------------------------------------------------------------------
    lines.append("## 1. Likert-Style Enneagram Assessment")
    lines.append("")
    lines.append(f"Test name: **{likert_json.get('test_name', '')}**")
    lines.append("")
    n_runs = len(likert_runs)
    lines.append(f"Number of runs: **{n_runs}**")
    lines.append("")

    # Scores by Enneagram type, per run
    lines.append("### 1.1 Per-Run Scores by Enneagram Type")
    lines.append("")
    # collect run-level enneagram_scores
    likert_ennea_runs: List[Dict[int, int]] = [
        r["enneagram_scores"] for r in likert_runs
    ]
    all_types = sorted(
        set().union(*[r.keys() for r in likert_ennea_runs])
    )

    header = "| Run | " + " | ".join(f"Type {t}" for t in all_types) + " |"
    sep = "|" + "---|" * (len(all_types) + 1)
    lines.append(header)
    lines.append(sep)
    for i, run in enumerate(likert_ennea_runs, start=1):
        row = [str(i)]
        for t in all_types:
            row.append(str(run.get(t, 0)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Aggregated stats
    lines.append("### 1.2 Averages and Standard Deviations (σ) by Enneagram Type")
    lines.append("")
    likert_agg = aggregate_numeric_runs(likert_ennea_runs)
    lines.append("| Enneagram type | Scores (per run) | Mean | σ (stdev) | Min | Max |")
    lines.append("|----------------|------------------|------|----------|-----|-----|")
    for t in sorted(likert_agg.keys()):
        info = likert_agg[t]
        scores_str = ", ".join(str(s) for s in info["scores"])
        lines.append(
            f"| {t} | {scores_str} | "
            f"{info['mean']:.2f} | {info['stdev']:.2f} | "
            f"{info['min']} | {info['max']} |"
        )
    lines.append("")

    # Dominant type per run
    lines.append("### 1.3 Dominant Type per Run (Likert)")
    lines.append("")
    lines.append("| Run | Top type(s) |")
    lines.append("|-----|-------------|")
    for i, run in enumerate(likert_runs, start=1):
        tops = run["dominant_types"]
        # top types (maybe multiple if tied at same score)
        if not tops:
            txt = "n/a"
        else:
            max_score = tops[0][1]
            top_list = [f"Type {t} ({score})" for t, score in tops if score == max_score]
            txt = ", ".join(top_list)
        lines.append(f"| {i} | {txt} |")
    lines.append("")

    # Full transcript
    lines.append("### 1.4 Full Question & Answer Transcript (Likert)")
    lines.append("")
    lines.append(
        "_For each run, every statement, the LLM's raw answer, and the normalized "
        "rating (1–5) are listed below._"
    )
    lines.append("")

    for i, run in enumerate(likert_runs, start=1):
        lines.append(f"#### Likert Run {i}")
        lines.append("")
        answers = run["answers"]
        lines.append("| # | Type | Enneagram | Statement | Raw answer | Rating (1–5) |")
        lines.append("|---|------|-----------|-----------|------------|--------------|")
        for q_idx, ans in enumerate(answers, start=1):
            stmt = ans["statement"].replace("|", "\\|")
            raw = ans["raw_answer"].replace("|", "\\|").replace("\n", "\\n")
            e_type = ans["enneagram_type"]
            lines.append(
                f"| {q_idx} | {ans['type_key']} | {e_type} | {stmt} | {raw} | {ans['rating']} |"
            )
        lines.append("")

    # ------------------------------------------------------------------
    # Paired section
    # ------------------------------------------------------------------
    lines.append("## 2. Paired A/B Enneagram Test")
    lines.append("")
    lines.append(f"Test name: **{paired_json.get('test_name', '')}**")
    lines.append("")
    n_runs_paired = len(paired_runs)
    lines.append(f"Number of runs: **{n_runs_paired}**")
    lines.append("")

    # Scores by Enneagram type, per run
    lines.append("### 2.1 Per-Run Selections by Enneagram Type")
    lines.append("")
    paired_ennea_runs: List[Dict[int, int]] = [
        r["enneagram_counts"] for r in paired_runs
    ]
    all_p_types = sorted(
        set().union(*[r.keys() for r in paired_ennea_runs])
    )

    header = "| Run | " + " | ".join(f"Type {t}" for t in all_p_types) + " |"
    sep = "|" + "---|" * (len(all_p_types) + 1)
    lines.append(header)
    lines.append(sep)
    for i, run in enumerate(paired_ennea_runs, start=1):
        row = [str(i)]
        for t in all_p_types:
            row.append(str(run.get(t, 0)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Aggregated stats
    lines.append("### 2.2 Averages and Standard Deviations (σ) by Enneagram Type")
    lines.append("")
    paired_agg = aggregate_numeric_runs(paired_ennea_runs)
    lines.append("| Enneagram type | Selections (per run) | Mean | σ (stdev) | Min | Max |")
    lines.append("|----------------|----------------------|------|----------|-----|-----|")
    for t in sorted(paired_agg.keys()):
        info = paired_agg[t]
        scores_str = ", ".join(str(s) for s in info["scores"])
        lines.append(
            f"| {t} | {scores_str} | "
            f"{info['mean']:.2f} | {info['stdev']:.2f} | "
            f"{info['min']} | {info['max']} |"
        )
    lines.append("")

    # Dominant type per run
    lines.append("### 2.3 Dominant Type per Run (Paired A/B)")
    lines.append("")
    lines.append("| Run | Top type(s) |")
    lines.append("|-----|-------------|")
    for i, run in enumerate(paired_runs, start=1):
        tops = run["dominant_types"]
        if not tops:
            txt = "n/a"
        else:
            max_score = tops[0][1]
            top_list = [f"Type {t} ({count})" for t, count in tops if count == max_score]
            txt = ", ".join(top_list)
        lines.append(f"| {i} | {txt} |")
    lines.append("")

    # Full transcript
    lines.append("### 2.4 Full Question & Answer Transcript (Paired A/B)")
    lines.append("")
    lines.append(
        "_For each run, every A/B question, the LLM's raw answer text, normalized "
        "choice, and the statement selected are listed below._"
    )
    lines.append("")

    for i, run in enumerate(paired_runs, start=1):
        lines.append(f"#### Paired Run {i}")
        lines.append("")
        answers = run["answers"]
        lines.append(
            "| # | Choice | Column | Enneagram | Raw answer | "
            "A text (col) | B text (col) | Chosen text |"
        )
        lines.append(
            "|---|--------|--------|-----------|-----------|----------------|----------------|-------------|"
        )
        for ans in sorted(answers, key=lambda x: x["id"]):
            raw = ans["raw_answer"].replace("|", "\\|").replace("\n", "\\n")
            a_text = ans["a_text"].replace("|", "\\|")
            b_text = ans["b_text"].replace("|", "\\|")
            chosen = ans["chosen_text"].replace("|", "\\|")
            lines.append(
                f"| {ans['id']} | {ans['choice']} | {ans['column']} | {ans['enneagram_type']} "
                f"| {raw} | {a_text} ({ans['a_column']}) | {b_text} ({ans['b_column']}) | {chosen} |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Likert + Paired Enneagram tests multiple times with an Ollama LLM, "
                    "and write a single markdown file with stats + full transcripts."
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
        help="Directory to write markdown result file into.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of times to run each test (default: 3).",
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

    likert_data = json.loads(likert_path.read_text(encoding="utf-8"))
    paired_data = json.loads(paired_path.read_text(encoding="utf-8"))

    # Run tests multiple times
    likert_runs: List[Dict[str, Any]] = []
    paired_runs: List[Dict[str, Any]] = []

    print(f"Running Likert test {args.runs} times...")
    for i in range(args.runs):
        print(f"  Likert run {i+1} / {args.runs}")
        likert_runs.append(run_likert_once(args.model, likert_data))

    print(f"Running Paired test {args.runs} times...")
    for i in range(args.runs):
        print(f"  Paired run {i+1} / {args.runs}")
        paired_runs.append(run_paired_once(args.model, paired_data))

    # Build markdown
    multi_md = build_markdown(
        model=args.model,
        likert_json=likert_data,
        paired_json=paired_data,
        likert_runs=likert_runs,
        paired_runs=paired_runs,
    )

    today = dt.date.today().isoformat()
    slug_model = slugify(args.model)
    out_path = outdir / f"enneagram-multi_{slug_model}_{today}.md"
    out_path.write_text(multi_md, encoding="utf-8")

    print(f"\nAll results written to: {out_path}")


if __name__ == "__main__":
    main()
