# Enneagram LLM Multi-Run Report

- **Model:** `mistral`
- **Date:** 2025-12-06
- **Runs per test:** 3

This file aggregates multiple runs of two Enneagram tests for the same LLM model to analyze consistency, variability, and centers.

## 1. Likert Test – Multi-Run Summary

**Test name:** Enneagram Assessment (Likert Scale)

### 1.1 Primary Type Tally per Run (Likert)

- Run 1: top types → Type 7 (72), Type 5 (69), Type 9 (66)
- Run 2: top types → Type 7 (71), Type 5 (70), Type 8 (69)
- Run 3: top types → Type 7 (71), Type 3 (66), Type 8 (65)

### 1.2 Scores by Enneagram Type Across Runs (Likert)

| Type | Run 1 | Run 2 | Run 3 | Mean | σ |
|------|------|------|------|------|------|
| 1 | 61 | 60 | 57 | 59.33 | 1.70 |
| 2 | 65 | 63 | 64 | 64.00 | 0.82 |
| 3 | 64 | 68 | 66 | 66.00 | 1.63 |
| 4 | 61 | 59 | 60 | 60.00 | 0.82 |
| 5 | 69 | 70 | 64 | 67.67 | 2.62 |
| 6 | 65 | 65 | 62 | 64.00 | 1.41 |
| 7 | 72 | 71 | 71 | 71.33 | 0.47 |
| 8 | 65 | 69 | 65 | 66.33 | 1.89 |
| 9 | 66 | 63 | 63 | 64.00 | 1.41 |

### 1.3 Centers of Intelligence per Run (Likert)

Head = Types 5, 6, 7 &nbsp;&nbsp; Heart = Types 2, 3, 4 &nbsp;&nbsp; Gut = Types 8, 9, 1

| Center | Run 1 | Run 2 | Run 3 | Mean | σ |
|------|------|------|------|------|------|
| Head | 206 | 206 | 197 | 203.00 | 4.24 |
| Heart | 190 | 190 | 190 | 190.00 | 0.00 |
| Gut | 192 | 192 | 185 | 189.67 | 3.30 |

## 2. Paired A/B Test – Multi-Run Summary

**Test name:** Enneagram Test Assessment

### 2.1 Primary Type Tally per Run (Paired)

- Run 1: top types → Type 5 (6), Type 6 (6), Type 4 (5)
- Run 2: top types → Type 5 (6), Type 6 (6), Type 4 (5)
- Run 3: top types → Type 5 (6), Type 6 (6), Type 4 (5)

### 2.2 Selections by Enneagram Type Across Runs (Paired)

| Center | Run 1 | Run 2 | Run 3 | Mean | σ |
|------|------|------|------|------|------|
| 2 | 4 | 4 | 4 | 4.00 | 0.00 |
| 3 | 4 | 4 | 4 | 4.00 | 0.00 |
| 4 | 5 | 5 | 5 | 5.00 | 0.00 |
| 5 | 6 | 6 | 6 | 6.00 | 0.00 |
| 6 | 6 | 6 | 6 | 6.00 | 0.00 |
| 7 | 3 | 3 | 3 | 3.00 | 0.00 |
| 8 | 4 | 4 | 4 | 4.00 | 0.00 |
| 9 | 4 | 4 | 4 | 4.00 | 0.00 |

### 2.3 Centers of Intelligence per Run (Paired)

Head = Types 5, 6, 7 &nbsp;&nbsp; Heart = Types 2, 3, 4 &nbsp;&nbsp; Gut = Types 8, 9, 1

| Center | Run 1 | Run 2 | Run 3 | Mean | σ |
|------|------|------|------|------|------|
| Head | 15 | 15 | 15 | 15.00 | 0.00 |
| Heart | 13 | 13 | 13 | 13.00 | 0.00 |
| Gut | 8 | 8 | 8 | 8.00 | 0.00 |

## 3. How to Use These Stats (Cheat Sheet)

- **High consistency (low σ) for a type** → stable trait in the model.
- **High variability (high σ) for a type** → volatile or prompt-sensitive trait.
- **Dominant center across runs** → primary processing mode:
  - Head (5/6/7) → thinking, anticipating, planning
  - Heart (2/3/4) → relating, identity, image
  - Gut (8/9/1) → instinct, control, anger

You can now compare this file across models, or rerun the same model with different prompts or temperatures and see how the Enneagram profile shifts.