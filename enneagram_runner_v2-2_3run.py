#!/usr/bin/env python3
"""
Enneagram LLM Multi-Run Runner

- Talks to a local Ollama instance (HTTP API) to have a model take:
  1. Likert-style Enneagram test (tests/enneagram_likert.json)
  2. Paired-question Enneagram test (tests/enneagram_test.json)

- Runs each test N times (default: 3) for a given model.
- Produces ONE markdown report per script invocation, e.g.:

    results/enneagram-multi_mistral_2025-12-06_23-01-15.md

  The report includes:
    - Per-run scores by type
    - Per-run centers (Head/Heart/Gut)
    - Derived per-run Enneagram profile:
        * Core type
        * Primary wing and both wing scores
        * Tritype (Gut / Heart / Head)
        * Dominant center
    - Multi-run stats (mean and σ per type and center)
    - Full question transcript for each run of each test:
        * Likert: statement + raw answer + parsed rating
        * Paired: pair text + raw answer + parsed choice
"""

import argparse
import datetime as dt
import json
import math
import pathlib
import re
import textwrap
from typing import Dict, List, Tuple, Any

import requests


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"

HEAD_TYPES = [5, 6, 7]
HEART_TYPES = [2, 3, 4]
GUT_TYPES = [8, 9, 1]


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-")


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / n
    return m, math.sqrt(var)


# -----------------------------------------------------------------------------
# Ollama helpers
# -----------------------------------------------------------------------------

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


def ask_choice_ab(model: str, question_text: str) -> Tuple[str, str]:
    """
    Ask the model to choose A or B.
    Returns (normalized_choice, raw_response).
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

    raw = ollama_generate(model, prompt).strip()
    upper = raw.upper()
    match = re.search(r"\b([AB])\b", upper)
    if match:
        return match.group(1), raw

    # Fallbacks
    if upper.startswith("A"):
        return "A", raw
    if upper.startswith("B"):
        return "B", raw
    return "A", raw  # ultra-defensive default


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
        4 = Frequently
        5 = Almost Always

        Respond with ONLY the digit 1, 2, 3, 4, or 5.
        Do NOT include any explanation or extra text.

        Statement:
        {question_text}

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

    # Very defensive fallback
    return 3, raw


# -----------------------------------------------------------------------------
# Enneagram profile derivation (core, wings, tritype, centers)
# -----------------------------------------------------------------------------

def compute_center_scores(type_scores: Dict[int, int]) -> Dict[str, int]:
    head = sum(type_scores.get(t, 0) for t in HEAD_TYPES)
    heart = sum(type_scores.get(t, 0) for t in HEART_TYPES)
    gut = sum(type_scores.get(t, 0) for t in GUT_TYPES)
    return {"Head": head, "Heart": heart, "Gut": gut}


def derive_profile_from_scores(type_scores: Dict[int, int]) -> Dict[str, Any]:
    """
    Given scores keyed by Enneagram type 1–9, derive:
      - core type
      - wings (both + primary)
      - tritype (Gut / Heart / Head highest)
      - center sums and dominant center
      - top3 types
    """
    if not type_scores:
        return {}

    # Core type and top3
    sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
    core_type, core_score = sorted_types[0]

    top3 = sorted_types[:3]

    # Wings
    if core_type == 1:
        left = 9
        right = 2
    elif core_type == 9:
        left = 8
        right = 1
    else:
        left = core_type - 1
        right = core_type + 1

    left_score = type_scores.get(left, 0)
    right_score = type_scores.get(right, 0)

    if left_score > right_score:
        primary_wing = left
    elif right_score > left_score:
        primary_wing = right
    else:
        primary_wing = None  # tie

    # Tritype
    gut_type = max(GUT_TYPES, key=lambda t: type_scores.get(t, 0))
    heart_type = max(HEART_TYPES, key=lambda t: type_scores.get(t, 0))
    head_type = max(HEAD_TYPES, key=lambda t: type_scores.get(t, 0))

    center_scores = compute_center_scores(type_scores)
    dominant_center = max(center_scores.items(), key=lambda x: x[1])[0]

    return {
        "core_type": core_type,
        "core_score": core_score,
        "top3": top3,
        "left_wing": left,
        "right_wing": right,
        "left_wing_score": left_score,
        "right_wing_score": right_score,
        "primary_wing": primary_wing,  # may be None if tied
        "gut_type": gut_type,
        "heart_type": heart_type,
        "head_type": head_type,
        "center_scores": center_scores,
        "dominant_center": dominant_center,
    }


# -----------------------------------------------------------------------------
# Likert test execution (single run)
# -----------------------------------------------------------------------------

def run_likert_once(
    model: str,
    json_path: pathlib.Path,
    run_index: int,
) -> Dict[str, Any]:
    """
    Run the Likert-style Enneagram test ONCE.
    Returns a dict with scores and a full transcript.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    test_name: str = data["test_name"]
    instructions: str = data.get("instructions", "")
    types = data["types"]  # keys A–I

    # Scores
    type_key_scores: Dict[str, int] = {k: 0 for k in types.keys()}
    enneagram_scores: Dict[int, int] = {}

    transcript: List[Dict[str, Any]] = []
    global_index = 1

    print(f"\n[Likert] Run {run_index}: {test_name}")

    for type_key in sorted(types.keys()):
        tinfo = types[type_key]
        e_type = tinfo.get("maps_to_enneagram_type")
        stmts = tinfo["statements"]

        for idx, stmt in enumerate(stmts, start=1):
            q_text = f"[Type {type_key} → Enneagram {e_type}] Item {idx}:\n{stmt}"
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
    center_scores = profile.get("center_scores", compute_center_scores(enneagram_scores))

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
# Paired test execution (single run)
# -----------------------------------------------------------------------------

def run_paired_once(
    model: str,
    json_path: pathlib.Path,
    run_index: int,
) -> Dict[str, Any]:
    """
    Run the paired-question Enneagram test ONCE.
    Returns a dict with counts and a full transcript.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    test_name: str = data["test_name"]
    columns = data["columns"]  # mapping column → { type, label }
    items = data["items"]      # list of question pairs

    counts_by_column: Dict[str, int] = {c: 0 for c in columns.keys()}
    counts_by_type: Dict[int, int] = {}

    transcript: List[Dict[str, Any]] = []

    print(f"\n[Paired] Run {run_index}: {test_name}")

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

        choice, raw = ask_choice_ab(model, question_text)
        chosen = a if choice == "A" else b
        col = chosen["column"]
        counts_by_column[col] += 1

        e_type = columns[col]["type"]
        counts_by_type[e_type] = counts_by_type.get(e_type, 0) + 1

        transcript.append(
            {
                "id": qid,
                "choice": choice,
                "raw_response": raw,
                "chosen_side": chosen["side"],
                "column": col,
                "enneagram_type": e_type,
                "chosen_text": chosen["text"],
                "a_text": a["text"],
                "b_text": b["text"],
                "a_column": a["column"],
                "b_column": b["column"],
            }
        )

        print(
            f"[Paired run {run_index}] Q{qid:02d} → choice={choice}, "
            f"column={col}, type={e_type}"
        )

    profile = derive_profile_from_scores(counts_by_type)
    center_scores = profile.get("center_scores", compute_center_scores(counts_by_type))

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

def aggregate_type_scores_across_runs(
    runs: List[Dict[str, Any]],
    key: str,
) -> Dict[int, Dict[str, float]]:
    """
    Given a list of run dicts and the name of the dict key that holds scores
    (e.g. 'enneagram_scores' or 'counts_by_type'), return per-type mean & σ.
    """
    per_type_values: Dict[int, List[int]] = {}
    for r in runs:
        scores: Dict[int, int] = r[key]
        for t, val in scores.items():
            per_type_values.setdefault(t, []).append(val)

    result: Dict[int, Dict[str, float]] = {}
    for t in sorted(per_type_values.keys()):
        vals = per_type_values[t]
        m, s = mean_std(vals)
        result[t] = {"mean": m, "std": s, "values": vals}
    return result


def aggregate_center_scores_across_runs(
    runs: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    per_center_values: Dict[str, List[int]] = {"Head": [], "Heart": [], "Gut": []}
    for r in runs:
        centers = r["center_scores"]
        for c in per_center_values.keys():
            per_center_values[c].append(centers.get(c, 0))

    result: Dict[str, Dict[str, float]] = {}
    for center, vals in per_center_values.items():
        m, s = mean_std(vals)
        result[center] = {"mean": m, "std": s, "values": vals}
    return result


# -----------------------------------------------------------------------------
# Markdown report builder
# -----------------------------------------------------------------------------

def build_markdown_report(
    model: str,
    timestamp: dt.datetime,
    runs_per_test: int,
    likert_runs: List[Dict[str, Any]],
    paired_runs: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []

    date_str = timestamp.date().isoformat()
    time_str = timestamp.time().replace(microsecond=0).isoformat()

    lines.append("# Enneagram LLM Multi-Run Report")
    lines.append("")
    lines.append(f"- **Model:** `{model}`")
    lines.append(f"- **Date:** {date_str}")
    lines.append(f"- **Time:** {time_str}")
    lines.append(f"- **Runs per test:** {runs_per_test}")
    lines.append("")
    lines.append(
        "This file aggregates multiple runs of two Enneagram tests for the same "
        "LLM model to analyze consistency, variability, centers, and full transcripts."
    )
    lines.append("")

    # ------------------------------------------------------------------
    # 1. Likert multi-run summary
    # ------------------------------------------------------------------
    if likert_runs:
        test_name = likert_runs[0]["test_name"]
        lines.append("## 1. Likert Test – Multi-Run Summary")
        lines.append("")
        lines.append(f"**Test name:** {test_name}")
        lines.append("")

        # 1.1 primary type per run
        lines.append("### 1.1 Primary Type (and Wings) per Run (Likert)")
        lines.append("")
        for r in likert_runs:
            p = r["profile"]
            core = p["core_type"]
            core_score = p["core_score"]
            left = p["left_wing"]
            right = p["right_wing"]
            lw = p["left_wing_score"]
            rw = p["right_wing_score"]
            primary_wing = p["primary_wing"]
            lines.append(
                f"- **Run {r['run_index']}** → Core: Type {core} "
                f"(score {core_score}); "
                f"Wings: {left} ({lw}), {right} ({rw}); "
                f"Primary wing: {primary_wing if primary_wing else 'tie'}"
            )
        lines.append("")

        # 1.2 scores by type across runs
        agg_types = aggregate_type_scores_across_runs(likert_runs, "enneagram_scores")
        lines.append("### 1.2 Scores by Enneagram Type Across Runs (Likert)")
        lines.append("")
        lines.append("| Type | " + " | ".join(f"Run {i}" for i in range(1, runs_per_test + 1)) + " | Mean | σ |")
        lines.append("|------|" + "------|" * runs_per_test + "------|------|")
        for t in sorted(agg_types.keys()):
            vals = agg_types[t]["values"]
            # pad if fewer runs for some reason
            vals = vals + [""] * (runs_per_test - len(vals))
            m = agg_types[t]["mean"]
            s = agg_types[t]["std"]
            vals_str = " | ".join(str(v) for v in vals)
            lines.append(f"| {t} | {vals_str} | {m:.2f} | {s:.2f} |")
        lines.append("")

        # 1.3 centers across runs
        centers_agg = aggregate_center_scores_across_runs(likert_runs)
        lines.append("### 1.3 Centers of Intelligence per Run (Likert)")
        lines.append("")
        lines.append(
            "Head = Types 5, 6, 7 &nbsp;&nbsp; "
            "Heart = Types 2, 3, 4 &nbsp;&nbsp; "
            "Gut = Types 8, 9, 1"
        )
        lines.append("")
        lines.append("| Center | " + " | ".join(f"Run {i}" for i in range(1, runs_per_test + 1)) + " | Mean | σ |")
        lines.append("|--------|" + "------|" * runs_per_test + "------|------|")
        for center in ["Head", "Heart", "Gut"]:
            vals = centers_agg[center]["values"]
            vals = vals + [""] * (runs_per_test - len(vals))
            m = centers_agg[center]["mean"]
            s = centers_agg[center]["std"]
            vals_str = " | ".join(str(v) for v in vals)
            lines.append(f"| {center} | {vals_str} | {m:.2f} | {s:.2f} |")
        lines.append("")

        # 1.4 derived profile per run (including tritype / dominant center)
        lines.append("### 1.4 Derived Enneagram Profile per Run (Likert)")
        lines.append("")
        for r in likert_runs:
            p = r["profile"]
            lines.append(f"#### Likert – Run {r['run_index']} Profile")
            lines.append("")
            lines.append(f"- **Core type:** {p['core_type']} (score {p['core_score']})")
            t1, t2, t3 = p["top3"]
            lines.append(
                f"- **Top 3 types:** "
                f"{t1[0]} ({t1[1]}), {t2[0]} ({t2[1]}), {t3[0]} ({t3[1]})"
            )
            lines.append(
                f"- **Wings:** {p['left_wing']} ({p['left_wing_score']}), "
                f"{p['right_wing']} ({p['right_wing_score']}); "
                f"primary wing: {p['primary_wing'] if p['primary_wing'] else 'tie'}"
            )
            lines.append(
                f"- **Tritype (Gut / Heart / Head):** "
                f"{p['gut_type']} / {p['heart_type']} / {p['head_type']}"
            )
            cs = p["center_scores"]
            lines.append(
                f"- **Center scores:** Head={cs['Head']}, Heart={cs['Heart']}, Gut={cs['Gut']} "
                f"(dominant center: {p['dominant_center']})"
            )
            lines.append("")

        # 1.5 full transcripts
        lines.append("### 1.5 Full Question Transcripts (Likert)")
        lines.append("")
        for r in likert_runs:
            lines.append(f"#### Likert – Run {r['run_index']} Transcript")
            lines.append("")
            lines.append("| # | Type Key | Enneagram | Statement | Raw Answer | Parsed Rating |")
            lines.append("|---|----------|-----------|-----------|-----------|---------------|")
            for q in r["transcript"]:
                stmt = q["statement"].replace("|", "\\|")
                raw = q["raw_response"].replace("|", "\\|")
                lines.append(
                    f"| {q['global_index']} | {q['type_key']} | {q['enneagram_type']} "
                    f"| {stmt} | {raw} | {q['parsed_rating']} |"
                )
            lines.append("")

    # ------------------------------------------------------------------
    # 2. Paired test multi-run summary
    # ------------------------------------------------------------------
    if paired_runs:
        test_name = paired_runs[0]["test_name"]
        lines.append("## 2. Paired A/B Test – Multi-Run Summary")
        lines.append("")
        lines.append(f"**Test name:** {test_name}")
        lines.append("")

        # 2.1 primary type per run
        lines.append("### 2.1 Primary Type (and Wings) per Run (Paired)")
        lines.append("")
        for r in paired_runs:
            p = r["profile"]
            core = p["core_type"]
            core_score = p["core_score"]
            left = p["left_wing"]
            right = p["right_wing"]
            lw = p["left_wing_score"]
            rw = p["right_wing_score"]
            primary_wing = p["primary_wing"]
            lines.append(
                f"- **Run {r['run_index']}** → Core: Type {core} "
                f"(selections {core_score}); "
                f"Wings: {left} ({lw}), {right} ({rw}); "
                f"Primary wing: {primary_wing if primary_wing else 'tie'}"
            )
        lines.append("")

        # 2.2 selections by type across runs
        agg_types = aggregate_type_scores_across_runs(paired_runs, "counts_by_type")
        lines.append("### 2.2 Selections by Enneagram Type Across Runs (Paired)")
        lines.append("")
        lines.append("| Type | " + " | ".join(f"Run {i}" for i in range(1, runs_per_test + 1)) + " | Mean | σ |")
        lines.append("|------|" + "------|" * runs_per_test + "------|------|")
        for t in sorted(agg_types.keys()):
            vals = agg_types[t]["values"]
            vals = vals + [""] * (runs_per_test - len(vals))
            m = agg_types[t]["mean"]
            s = agg_types[t]["std"]
            vals_str = " | ".join(str(v) for v in vals)
            lines.append(f"| {t} | {vals_str} | {m:.2f} | {s:.2f} |")
        lines.append("")

        # 2.3 centers across runs
        centers_agg = aggregate_center_scores_across_runs(paired_runs)
        lines.append("### 2.3 Centers of Intelligence per Run (Paired)")
        lines.append("")
        lines.append(
            "Head = Types 5, 6, 7 &nbsp;&nbsp; "
            "Heart = Types 2, 3, 4 &nbsp;&nbsp; "
            "Gut = Types 8, 9, 1"
        )
        lines.append("")
        lines.append("| Center | " + " | ".join(f"Run {i}" for i in range(1, runs_per_test + 1)) + " | Mean | σ |")
        lines.append("|--------|" + "------|" * runs_per_test + "------|------|")
        for center in ["Head", "Heart", "Gut"]:
            vals = centers_agg[center]["values"]
            vals = vals + [""] * (runs_per_test - len(vals))
            m = centers_agg[center]["mean"]
            s = centers_agg[center]["std"]
            vals_str = " | ".join(str(v) for v in vals)
            lines.append(f"| {center} | {vals_str} | {m:.2f} | {s:.2f} |")
        lines.append("")

        # 2.4 derived profile per run
        lines.append("### 2.4 Derived Enneagram Profile per Run (Paired)")
        lines.append("")
        for r in paired_runs:
            p = r["profile"]
            lines.append(f"#### Paired – Run {r['run_index']} Profile")
            lines.append("")
            lines.append(f"- **Core type:** {p['core_type']} (selections {p['core_score']})")
            t1, t2, t3 = p["top3"]
            lines.append(
                f"- **Top 3 types:** "
                f"{t1[0]} ({t1[1]}), {t2[0]} ({t2[1]}), {t3[0]} ({t3[1]})"
            )
            lines.append(
                f"- **Wings:** {p['left_wing']} ({p['left_wing_score']}), "
                f"{p['right_wing']} ({p['right_wing_score']}); "
                f"primary wing: {p['primary_wing'] if p['primary_wing'] else 'tie'}"
            )
            lines.append(
                f"- **Tritype (Gut / Heart / Head):** "
                f"{p['gut_type']} / {p['heart_type']} / {p['head_type']}"
            )
            cs = p["center_scores"]
            lines.append(
                f"- **Center scores:** Head={cs['Head']}, Heart={cs['Heart']}, Gut={cs['Gut']} "
                f"(dominant center: {p['dominant_center']})"
            )
            lines.append("")

        # 2.5 transcripts
        lines.append("### 2.5 Full Question Transcripts (Paired)")
        lines.append("")
        for r in paired_runs:
            lines.append(f"#### Paired – Run {r['run_index']} Transcript")
            lines.append("")
            lines.append(
                "| # | Choice | Column | Enneagram | Raw Answer | Statement chosen | "
                "A text (column) | B text (column) |"
            )
            lines.append(
                "|---|--------|--------|-----------|-----------|------------------|"
                "-----------------|-----------------|"
            )
            for q in r["transcript"]:
                chosen = q["chosen_text"].replace("|", "\\|")
                a_text = q["a_text"].replace("|", "\\|")
                b_text = q["b_text"].replace("|", "\\|")
                raw = q["raw_response"].replace("|", "\\|")
                lines.append(
                    f"| {q['id']} | {q['choice']} | {q['column']} | {q['enneagram_type']} "
                    f"| {raw} | {chosen} | {a_text} ({q['a_column']}) | "
                    f"{b_text} ({q['b_column']}) |"
                )
            lines.append("")

    # 3. Cheat sheet / interpretation reminder
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
        "You can now compare this file across models, or rerun the same model "
        "with different prompts or temperatures and see how the Enneagram "
        "profile shifts."
    )
    lines.append("")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Have an Ollama LLM take Enneagram tests (Likert + paired) "
                    "multiple times and write a single markdown report."
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

    # Run tests N times
    likert_runs: List[Dict[str, Any]] = []
    paired_runs: List[Dict[str, Any]] = []

    for i in range(1, args.runs_per_test + 1):
        likert_runs.append(run_likert_once(args.model, likert_path, i))
    for i in range(1, args.runs_per_test + 1):
        paired_runs.append(run_paired_once(args.model, paired_path, i))

    # Build markdown report
    now = dt.datetime.now()
    slug_model = slugify(args.model)
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    out_path = outdir / f"enneagram-multi_{slug_model}_{timestamp_str}.md"

    md = build_markdown_report(
        model=args.model,
        timestamp=now,
        runs_per_test=args.runs_per_test,
        likert_runs=likert_runs,
        paired_runs=paired_runs,
    )
    out_path.write_text(md, encoding="utf-8")

    print(f"\nMulti-run report written to: {out_path}")


if __name__ == "__main__":
    main()
