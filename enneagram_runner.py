#!/usr/bin/env python3
"""
Have a local Ollama LLM take two Enneagram tests:

1. Likert-style Enneagram Assessment (tests/enneagram_likert.json)
2. Paired-question Enneagram Test Assessment (tests/enneagram_test.json)

It writes separate markdown result files for each test, named:
    <slug-test-name>_<slug-model-name>_<YYYY-MM-DD>.md
"""

import argparse
import datetime as dt
import json
import pathlib
import re
import textwrap
from typing import Dict, List, Tuple

import requests


# Default Ollama HTTP endpoint
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

    # Fallbacks
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
    # Extract first digit 1–5
    match = re.search(r"\b([1-5])\b", raw)
    if match:
        return int(match.group(1))

    # Fallback: try first char
    for ch in raw:
        if ch in "12345":
            return int(ch)

    # Very defensive fallback
    return 3


# ---------------------------------------------------------------------------
# Likert-style test logic (tests/enneagram_likert.json)
# ---------------------------------------------------------------------------

def run_likert_test(
    model: str,
    json_path: pathlib.Path,
    outdir: pathlib.Path,
) -> pathlib.Path:
    """
    Run the Likert-style Enneagram test and write a markdown report.
    Returns the path to the output .md file.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    test_name: str = data["test_name"]
    instructions: str = data.get("instructions", "")
    types: Dict[str, Dict] = data["types"]

    today = dt.date.today().isoformat()
    slug_test = slugify(test_name)
    slug_model = slugify(model)
    out_path = outdir / f"{slug_test}_{slug_model}_{today}.md"

    # Prepare result containers
    type_scores: Dict[str, int] = {k: 0 for k in types.keys()}
    # Also score by Enneagram type number
    enneagram_scores: Dict[int, int] = {}
    # Detailed answers: (type_key, idx, statement, rating)
    answers: List[Tuple[str, int, str, int]] = []

    print(f"\n=== Running Likert test: {test_name} ===")

    for type_key in sorted(types.keys()):
        tinfo = types[type_key]
        label = tinfo.get("label", "")
        stmts = tinfo["statements"]

        print(f"\nPersonality Type {type_key} ({label}) — {len(stmts)} items")

        for idx, stmt in enumerate(stmts, start=1):
            question_text = f"[Type {type_key}] Item {idx}:\n{stmt}"
            rating = ask_likert_1_to_5(model, question_text)
            type_scores[type_key] += rating
            answers.append((type_key, idx, stmt, rating))

            print(f"Type {type_key} item {idx:02d}: {rating}")

    # aggregate by Enneagram type
    for type_key, score in type_scores.items():
        e_type = types[type_key].get("maps_to_enneagram_type")
        if e_type is None:
            continue
        enneagram_scores[e_type] = enneagram_scores.get(e_type, 0) + score

    # Sort for top 3
    top_types = sorted(
        enneagram_scores.items(), key=lambda x: x[1], reverse=True
    )

    # Build markdown
    lines: List[str] = []
    lines.append(f"# {test_name} – Likert Results")
    lines.append("")
    lines.append(f"- **Model:** `{model}`")
    lines.append(f"- **Date:** {today}")
    lines.append("")
    if instructions:
        lines.append("## Test instructions")
        lines.append("")
        lines.append(instructions)
        lines.append("")

    lines.append("## Scores by Personality Type (A–I)")
    lines.append("")
    lines.append("| Type key | Label | Enneagram type | Total score |")
    lines.append("|----------|-------|----------------|-------------|")
    for type_key in sorted(types.keys()):
        tinfo = types[type_key]
        label = tinfo.get("label", "")
        e_type = tinfo.get("maps_to_enneagram_type", "")
        score = type_scores[type_key]
        lines.append(f"| {type_key} | {label} | {e_type} | {score} |")

    lines.append("")
    lines.append("## Scores by Enneagram Type")
    lines.append("")
    lines.append("| Enneagram type | Total score |")
    lines.append("|----------------|-------------|")
    for e_type in sorted(enneagram_scores.keys()):
        lines.append(f"| {e_type} | {enneagram_scores[e_type]} |")

    lines.append("")
    lines.append("## Top 3 Candidate Types (by score)")
    lines.append("")
    for rank, (e_type, score) in enumerate(top_types[:3], start=1):
        lines.append(f"**#{rank} – Type {e_type}** (score: {score})")
        lines.append("")

    lines.append("")
    lines.append("## Question-by-question ratings")
    lines.append("")
    lines.append("| # | Type | Statement | Rating (1–5) |")
    lines.append("|---|------|-----------|--------------|")
    counter = 1
    for type_key, idx, stmt, rating in answers:
        safe_stmt = stmt.replace("|", "\\|")
        lines.append(
            f"| {counter} | {type_key} | {safe_stmt} | {rating} |"
        )
        counter += 1

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nLikert test results saved to: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Paired-question test logic (tests/enneagram_test.json)
# ---------------------------------------------------------------------------

def run_paired_test(
    model: str,
    json_path: pathlib.Path,
    outdir: pathlib.Path,
) -> pathlib.Path:
    """
    Run the paired-question Enneagram test and write a markdown report.
    Returns the path to the output .md file.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    test_name: str = data["test_name"]
    columns = data["columns"]
    items = data["items"]

    today = dt.date.today().isoformat()
    slug_test = slugify(test_name)
    slug_model = slugify(model)
    out_path = outdir / f"{slug_test}_{slug_model}_{today}.md"

    # Initialize counts
    counts_by_column: Dict[str, int] = {c: 0 for c in columns.keys()}
    counts_by_type: Dict[int, int] = {}

    answers_detail = []  # list of dicts per question

    print(f"\n=== Running paired-question test: {test_name} ===")

    for item in items:
        qid = item["id"]
        pair = item["pair"]

        # Expect exactly two entries in pair
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
        counts_by_column[col] += 1

        e_type = columns[col]["type"]
        counts_by_type[e_type] = counts_by_type.get(e_type, 0) + 1

        answers_detail.append(
            {
                "id": qid,
                "choice": choice,
                "chosen_text": chosen["text"],
                "column": col,
                "enneagram_type": e_type,
                "a_text": a["text"],
                "b_text": b["text"],
                "a_column": a["column"],
                "b_column": b["column"],
            }
        )

        print(f"Q{qid:02d}: choice={choice}, column={col}, type={e_type}")

    # sort types for top 3
    top_types = sorted(
        counts_by_type.items(), key=lambda x: x[1], reverse=True
    )

    # Build markdown
    lines: List[str] = []
    lines.append(f"# {test_name} – Paired-Question Results")
    lines.append("")
    lines.append(f"- **Model:** `{model}`")
    lines.append(f"- **Date:** {today}")
    lines.append("")

    lines.append("## Column → Enneagram type mapping")
    lines.append("")
    lines.append("| Column | Enneagram type | Label | Total selections |")
    lines.append("|--------|----------------|-------|------------------|")
    for col in sorted(columns.keys()):
        e_type = columns[col]["type"]
        label = columns[col].get("label", "")
        count = counts_by_column[col]
        lines.append(f"| {col} | {e_type} | {label} | {count} |")

    lines.append("")
    lines.append("## Scores by Enneagram type")
    lines.append("")
    lines.append("| Enneagram type | Total selections |")
    lines.append("|----------------|------------------|")
    for e_type in sorted(counts_by_type.keys()):
        lines.append(f"| {e_type} | {counts_by_type[e_type]} |")

    lines.append("")
    lines.append("## Top 3 Candidate Types (by count)")
    lines.append("")
    for rank, (e_type, count) in enumerate(top_types[:3], start=1):
        lines.append(f"**#{rank} – Type {e_type}** (selections: {count})")
        lines.append("")

    lines.append("")
    lines.append("## Question-by-question choices")
    lines.append("")
    lines.append(
        "| # | Choice | Column | Enneagram type | Statement chosen | A (column) | B (column) |"
    )
    lines.append(
        "|---|--------|--------|----------------|------------------|-----------|-----------|"
    )
    for q in sorted(answers_detail, key=lambda x: x["id"]):
        short_chosen = q["chosen_text"].replace("|", "\\|")
        lines.append(
            f"| {q['id']} | {q['choice']} | {q['column']} | {q['enneagram_type']} "
            f"| {short_chosen} | {q['a_column']} | {q['b_column']} |"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nPaired test results saved to: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Have an Ollama LLM take two Enneagram tests (Likert + paired) "
                    "and write markdown reports."
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
        help="Directory to write markdown result files into.",
    )
    parser.add_argument(
        "--run",
        choices=["likert", "paired", "both"],
        default="both",
        help="Which tests to run (default: both).",
    )
    args = parser.parse_args()

    tests_dir = pathlib.Path(args.tests_dir)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    likert_path = tests_dir / "enneagram_likert.json"
    paired_path = tests_dir / "enneagram_test.json"

    if args.run in ("likert", "both"):
        if not likert_path.exists():
            raise FileNotFoundError(f"Likert test file not found: {likert_path}")
        run_likert_test(args.model, likert_path, outdir)

    if args.run in ("paired", "both"):
        if not paired_path.exists():
            raise FileNotFoundError(f"Paired test file not found: {paired_path}")
        run_paired_test(args.model, paired_path, outdir)


if __name__ == "__main__":
    main()
