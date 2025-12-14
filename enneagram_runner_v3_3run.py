#!/usr/bin/env python3
"""
Enneagram LLM Multi-Run Runner (v3, unlabeled prompts)

- Talks to a local Ollama instance (HTTP API) to have a model take:
  1. Likert-style Enneagram test (tests/enneagram_likert.json)
  2. Paired Enneagram test (tests/enneagram_test.json)

- Runs each test N times for a given model.
- Produces ONE markdown report per script invocation, e.g.:

    results/enneagram-multi_mistral_2025-12-07_19-32-10_v3_unlabeled.md

Key behavior change from v2-2:
- Likert questions NO LONGER show any type or Enneagram labels to the model.
"""

import argparse
import datetime
import json
import math
import pathlib
import textwrap
from typing import Any, Dict, List, Tuple

import requests


# -----------------------------------------------------------------------------
# Ollama / LLM helpers
# -----------------------------------------------------------------------------

def call_ollama(model: str, prompt: str) -> str:
    """
    Call the Ollama HTTP API with the given model and prompt.
    Returns the 'response' string.
    """
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(url, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


def ask_likert_1_to_5(model: str, question_text: str) -> Tuple[int, str]:
    """
    Ask the model to rate from 1 to 5.
    Returns (rating, raw_response).
    """
    prompt = textwrap.dedent(
        f"""
        You are taking a personality test that uses a 1–5 Likert scale.

        For each statement, answer with a number from 1 to 5:
        1 = Almost Never
        2 = Rarely
        3 = Sometimes
        4 = Often
        5 = Almost Always

        Respond with ONLY a single digit from 1 to 5.
        Do not explain your choice. Do not add any other text.

        Statement:
        {question_text}

        Your answer (just 1–5):
        """
    )
    raw = call_ollama(model, prompt)
    for ch in raw:
        if ch in "12345":
            return int(ch), raw
    # Fallback if the model does something weird
    return 3, raw


def ask_forced_choice_ab(model: str, question_text: str) -> Tuple[str, str]:
    """
    Ask the model to choose A or B.
    Returns (choice, raw_response).
    """
    prompt = textwrap.dedent(
        f"""
        You are taking a personality test.

        For each question, you must choose either option A or option B.

        Rules:
        - Respond with ONLY the single letter 'A' or 'B'.
        - Do not explain your answer.
        - Do not add punctuation or repeat the text.

        Question:
        {question_text}

        Your answer (just A or B):
        """
    )
    raw = call_ollama(model, prompt)
    raw_up = raw.strip().upper()
    for ch in raw_up:
        if ch in ("A", "B"):
            return ch, raw
    # Fallback
    return "A", raw


# -----------------------------------------------------------------------------
# Scoring helpers (centers, wings, tritype)
# -----------------------------------------------------------------------------

def compute_center_scores(enneagram_scores: Dict[int, int]) -> Dict[str, int]:
    """
    Compute center scores from a dict {enneagram_type: score}.
    Head = 5, 6, 7
    Heart = 2, 3, 4
    Gut = 8, 9, 1
    """
    head = sum(enneagram_scores.get(t, 0) for t in (5, 6, 7))
    heart = sum(enneagram_scores.get(t, 0) for t in (2, 3, 4))
    gut = sum(enneagram_scores.get(t, 0) for t in (8, 9, 1))
    return {"head": head, "heart": heart, "gut": gut}


def derive_profile_from_scores(enneagram_scores: Dict[int, int]) -> Dict[str, Any]:
    """
    From {type: score}, derive:
      - core_type
      - primary_wing
      - center_scores
      - tritype (best Gut, Heart, Head)
    """
    if not enneagram_scores:
        return {
            "core_type": None,
            "primary_wing": None,
            "center_scores": {"head": 0, "heart": 0, "gut": 0},
            "tritype": (None, None, None),
        }

    # Core type
    core_type = max(enneagram_scores.items(), key=lambda kv: kv[1])[0]

    # Wing (adjacent types)
    left = 9 if core_type == 1 else core_type - 1
    right = 1 if core_type == 9 else core_type + 1
    left_score = enneagram_scores.get(left, 0)
    right_score = enneagram_scores.get(right, 0)
    primary_wing = left if left_score >= right_score else right

    # Centers
    center_scores = compute_center_scores(enneagram_scores)

    # Tritype: best from Gut / Heart / Head
    gut_candidates = [(t, enneagram_scores.get(t, 0)) for t in (8, 9, 1)]
    heart_candidates = [(t, enneagram_scores.get(t, 0)) for t in (2, 3, 4)]
    head_candidates = [(t, enneagram_scores.get(t, 0)) for t in (5, 6, 7)]

    gut_type = max(gut_candidates, key=lambda kv: kv[1])[0]
    heart_type = max(heart_candidates, key=lambda kv: kv[1])[0]
    head_type = max(head_candidates, key=lambda kv: kv[1])[0]

    return {
        "core_type": core_type,
        "primary_wing": primary_wing,
        "center_scores": center_scores,
        "tritype": (gut_type, heart_type, head_type),
    }


# -----------------------------------------------------------------------------
# Likert test (single run)
# -----------------------------------------------------------------------------

def run_likert_once(
    model: str,
    json_path: pathlib.Path,
    run_index: int,
) -> Dict[str, Any]:
    """
    Run the Likert-style Enneagram test ONCE.
    JSON structure (enneagram_likert.json):

    {
      "test_name": "...",
      "instructions": "...",
      "types": {
        "A": {
          "label": "...",
          "maps_to_enneagram_type": 4,
          "statements": [...]
        },
        ...
      }
    }
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    test_name: str = data["test_name"]
    instructions: str = data.get("instructions", "")
    types = data["types"]

    type_key_scores: Dict[str, int] = {k: 0 for k in types.keys()}
    enneagram_scores: Dict[int, int] = {}
    transcript: List[Dict[str, Any]] = []

    global_index = 1

    print(f"\n[Likert] Run {run_index}: {test_name}")

    for type_key in sorted(types.keys()):
        tinfo = types[type_key]
        e_type = tinfo.get("maps_to_enneagram_type")
        statements = tinfo.get("statements", [])

        for idx, stmt in enumerate(statements, start=1):
            # IMPORTANT: no type or Enneagram labels in the prompt.
            q_text = f"[Item {global_index}] {stmt}"

            rating, raw = ask_likert_1_to_5(model, q_text)

            type_key_scores[type_key] += rating
            if e_type is not None:
                enneagram_scores[e_type] = enneagram_scores.get(e_type, 0) + rating

            transcript.append(
                {
                    "global_index": global_index,
                    "type_key": type_key,
                    "enneagram_type": e_type,
                    "statement_index_within_type": idx,
                    "statement": stmt,
                    "parsed_rating": rating,
                    "raw_response": raw,
                }
            )

            print(
                f"[Likert run {run_index}] Q{global_index:03d} "
                f"(Type {type_key} / {e_type}) → rating={rating}"
            )
            global_index += 1

    profile = derive_profile_from_scores(enneagram_scores)
    center_scores = profile["center_scores"]

    return {
        "run_index": run_index,
        "test_name": test_name,
        "instructions": instructions,
        "type_key_scores": type_key_scores,
        "enneagram_scores": enneagram_scores,
        "center_scores": center_scores,
        "profile": profile,
        "transcript": transcript,
    }


# -----------------------------------------------------------------------------
# Paired test (single run)
# -----------------------------------------------------------------------------

def run_paired_once(
    model: str,
    json_path: pathlib.Path,
    run_index: int,
) -> Dict[str, Any]:
    """
    Run the paired Enneagram test ONCE.

    JSON structure (enneagram_test.json):

    {
      "test_name": "...",
      "columns": {
        "A": { "type": 9, "label": "Nine" },
        ...
      },
      "items": [
        {
          "id": 1,
          "pair": [
            { "side": "A", "text": "...", "column": "E" },
            { "side": "B", "text": "...", "column": "B" }
          ]
        },
        ...
      ]
    }
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    test_name: str = data["test_name"]
    columns = data["columns"]
    items = data["items"]

    counts_by_type: Dict[int, int] = {}
    counts_by_column: Dict[str, int] = {c: 0 for c in columns.keys()}
    transcript: List[Dict[str, Any]] = []

    print(f"\n[Paired] Run {run_index}: {test_name}")

    for item in items:
        qid = item["id"]
        pair_list = item["pair"]
        if len(pair_list) != 2:
            # Defensive, but your data always has 2
            continue

        # Normalize so we know which is A and which is B
        # (in case order is not guaranteed)
        sides = {p["side"]: p for p in pair_list}
        a = sides["A"]
        b = sides["B"]

        question_text = textwrap.dedent(
            f"""
            Question {qid}:

            A) {a['text']}
            B) {b['text']}
            """
        )

        choice, raw = ask_forced_choice_ab(model, question_text)

        if choice == "A":
            chosen = a
        else:
            chosen = b

        chosen_column = chosen["column"]
        col_info = columns[chosen_column]
        chosen_type = col_info["type"]

        counts_by_column[chosen_column] = counts_by_column.get(chosen_column, 0) + 1
        counts_by_type[chosen_type] = counts_by_type.get(chosen_type, 0) + 1

        transcript.append(
            {
                "id": qid,
                "choice": choice,
                "raw_response": raw,
                "chosen_column": chosen_column,
                "chosen_type": chosen_type,
                "chosen_text": chosen["text"],
                "a_text": a["text"],
                "b_text": b["text"],
                "a_column": a["column"],
                "b_column": b["column"],
            }
        )

        print(
            f"[Paired run {run_index}] Q{qid:02d} → choice={choice}, "
            f"column={chosen_column}, type={chosen_type}"
        )

    profile = derive_profile_from_scores(counts_by_type)
    center_scores = profile["center_scores"]

    return {
        "run_index": run_index,
        "test_name": test_name,
        "columns": columns,
        "counts_by_column": counts_by_column,
        "counts_by_type": counts_by_type,
        "center_scores": center_scores,
        "profile": profile,
        "transcript": transcript,
    }


# -----------------------------------------------------------------------------
# Aggregation helpers
# -----------------------------------------------------------------------------

def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def aggregate_likert_runs(likert_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute means, standard deviations, and average centers
    across multiple Likert runs.
    """
    if not likert_runs:
        return {}

    per_run_scores: List[Dict[int, int]] = [r["enneagram_scores"] for r in likert_runs]
    all_types = sorted({t for scores in per_run_scores for t in scores.keys()})

    per_type_values: Dict[int, List[int]] = {t: [] for t in all_types}
    for scores in per_run_scores:
        for t in all_types:
            per_type_values[t].append(scores.get(t, 0))

    mean_scores: Dict[int, float] = {}
    sigma_scores: Dict[int, float] = {}
    for t, vals in per_type_values.items():
        m, s = mean_std(vals)
        mean_scores[t] = m
        sigma_scores[t] = s

    per_center_vals: Dict[str, List[int]] = {"head": [], "heart": [], "gut": []}
    for r in likert_runs:
        centers = r["center_scores"]
        for c in per_center_vals.keys():
            per_center_vals[c].append(centers.get(c, 0))

    avg_center_scores: Dict[str, float] = {}
    for c, vals in per_center_vals.items():
        avg_center_scores[c] = sum(vals) / len(vals)

    return {
        "mean_scores": mean_scores,
        "sigma_scores": sigma_scores,
        "avg_center_scores": avg_center_scores,
    }


def aggregate_paired_runs(paired_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate type and column counts, and center stats, across paired runs.
    """
    from collections import Counter

    combined_type_counts = Counter()
    combined_column_counts = Counter()
    per_run_type_counts: List[Dict[int, int]] = []
    per_run_center_scores: List[Dict[str, int]] = []

    for r in paired_runs:
        tc = r["counts_by_type"]
        combined_type_counts.update(tc)
        combined_column_counts.update(r["counts_by_column"])
        per_run_type_counts.append(tc)
        per_run_center_scores.append(r["center_scores"])

    all_types = sorted(combined_type_counts.keys())
    per_type_values: Dict[int, List[int]] = {t: [] for t in all_types}
    for tc in per_run_type_counts:
        for t in all_types:
            per_type_values[t].append(tc.get(t, 0))

    mean_type_counts: Dict[int, float] = {}
    sigma_type_counts: Dict[int, float] = {}
    for t, vals in per_type_values.items():
        m, s = mean_std(vals)
        mean_type_counts[t] = m
        sigma_type_counts[t] = s

    per_center_vals: Dict[str, List[int]] = {"head": [], "heart": [], "gut": []}
    for cs in per_run_center_scores:
        for c in per_center_vals.keys():
            per_center_vals[c].append(cs.get(c, 0))

    center_agg: Dict[str, Dict[str, Any]] = {}
    for center, vals in per_center_vals.items():
        m, s = mean_std(vals)
        center_agg[center] = {"mean": m, "std": s, "values": vals}

    return {
        "combined_type_counts": dict(combined_type_counts),
        "combined_column_counts": dict(combined_column_counts),
        "mean_type_counts": mean_type_counts,
        "sigma_type_counts": sigma_type_counts,
        "center_agg": center_agg,
    }


# -----------------------------------------------------------------------------
# Markdown formatting
# -----------------------------------------------------------------------------

def format_likert_section_md(
    likert_runs: List[Dict[str, Any]], agg: Dict[str, Any]
) -> str:
    lines: List[str] = []
    lines.append("## 1. Likert Test – Multi-Run Summary\n")

    lines.append("### 1.1 Primary Type / Wing / Tritype per Run\n")
    for r in likert_runs:
        run_idx = r["run_index"]
        profile = r["profile"]
        core = profile["core_type"]
        wing = profile["primary_wing"]
        gut, heart, head = profile["tritype"]
        centers = r["center_scores"]
        lines.append(
            f"- **Run {run_idx}** → Core: Type {core}, Wing: {wing}, "
            f"Tritype (Gut/Heart/Head): ({gut}, {heart}, {head}); "
            f"Centers: Head={centers['head']}, Heart={centers['heart']}, Gut={centers['gut']}"
        )

    lines.append("\n### 1.2 Scores by Enneagram Type Across Runs (Means & σ)\n")
    lines.append("| Type | Mean Score | σ |")
    lines.append("|------|------------|---|")

    mean_scores = agg["mean_scores"]
    sigma_scores = agg["sigma_scores"]
    for t in sorted(mean_scores.keys()):
        m = mean_scores[t]
        s = sigma_scores[t]
        lines.append(f"| {t} | {m:.2f} | {s:.2f} |")

    lines.append("\n### 1.3 Average Centers of Intelligence Across Runs\n")
    lines.append("| Center | Average Score |")
    lines.append("|--------|---------------|")

    avg_centers = agg["avg_center_scores"]
    for c in ("head", "heart", "gut"):
        val = avg_centers.get(c, 0.0)
        lines.append(f"| {c.capitalize()} | {val:.2f} |")

    lines.append("")
    return "\n".join(lines)


def format_paired_section_md(
    paired_runs: List[Dict[str, Any]], agg: Dict[str, Any]
) -> str:
    lines: List[str] = []
    lines.append("## 2. Paired A/B Test – Multi-Run Summary\n")

    lines.append("### 2.1 Type Selection Counts per Run\n")
    for r in paired_runs:
        run_idx = r["run_index"]
        tcounts = r["counts_by_type"]
        if not tcounts:
            lines.append(f"- **Run {run_idx}:** no selections")
        else:
            sorted_counts = sorted(tcounts.items(), key=lambda kv: kv[0])
            row = ", ".join(f"Type {t}: {c}" for t, c in sorted_counts)
            lines.append(f"- **Run {run_idx}:** {row}")
    lines.append("")

    lines.append("### 2.2 Combined Type Selection Counts (All Runs)\n")
    combined = agg["combined_type_counts"]
    if not combined:
        lines.append("- No selections at all.")
    else:
        for t, c in sorted(combined.items(), key=lambda kv: kv[0]):
            lines.append(f"- Type {t}: {c}")
    lines.append("")

    lines.append("### 2.3 Combined Column Selection Counts (All Runs)\n")
    combined_cols = agg["combined_column_counts"]
    if not combined_cols:
        lines.append("- No column selections at all.")
    else:
        for col, c in sorted(combined_cols.items(), key=lambda kv: kv[0]):
            lines.append(f"- Column {col}: {c}")
    lines.append("")

    lines.append("### 2.4 Centers of Intelligence (Paired) – Means & σ\n")
    lines.append("| Center | Mean | σ | Values |")
    lines.append("|--------|------|---|--------|")
    center_agg = agg["center_agg"]
    for center_name, stats in center_agg.items():
        mean = stats["mean"]
        std = stats["std"]
        vals = ", ".join(str(v) for v in stats["values"])
        lines.append(f"| {center_name.capitalize()} | {mean:.2f} | {std:.2f} | {vals} |")

    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Have an Ollama LLM take Enneagram tests (Likert + paired) "
            "multiple times and write a single markdown report (v3, unlabeled prompts)."
        )
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
        "--runs-per-test",
        type=int,
        default=3,
        help="How many times to run each test (Likert and paired).",
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

    likert_runs: List[Dict[str, Any]] = []
    paired_runs: List[Dict[str, Any]] = []

    for i in range(1, args.runs_per_test + 1):
        likert_runs.append(run_likert_once(args.model, likert_path, i))
        paired_runs.append(run_paired_once(args.model, paired_path, i))

    likert_agg = aggregate_likert_runs(likert_runs)
    paired_agg = aggregate_paired_runs(paired_runs)

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    out_path = outdir / f"{args.model}_enneagram-multi_{timestamp}_v3_unlabeled.md"

    lines: List[str] = []
    lines.append(f"# Enneagram LLM Multi-Run Report (Model: {args.model}, v3 Unlabeled)")
    lines.append("")
    lines.append(f"- **Date:** {now.date().isoformat()}")
    lines.append(f"- **Time:** {now.strftime('%H:%M:%S')}")
    lines.append(f"- **Runs per test:** {args.runs_per_test}")
    lines.append("")
    lines.append(format_likert_section_md(likert_runs, likert_agg))
    lines.append(format_paired_section_md(paired_runs, paired_agg))

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDone. Wrote markdown report to: {out_path}")


if __name__ == "__main__":
    main()
