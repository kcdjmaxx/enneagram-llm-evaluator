# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository is an **Enneagram LLM Testing Framework** that administers personality assessments to local LLMs via Ollama. It runs two types of Enneagram tests (Likert-scale and paired forced-choice) multiple times and generates detailed Markdown reports with statistical analysis.

## Commands

### Running Tests

**Basic single run (both tests):**
```bash
python3 enneagram_runner.py --model mistral
```

**Run only one test type:**
```bash
python3 enneagram_runner.py --model mistral --run likert
python3 enneagram_runner.py --model mistral --run paired
```

**Multi-run version (v3, recommended for statistical analysis):**
```bash
python3 enneagram_runner_v3_3run.py --model mistral --runs-per-test 3
```

### Common Options
- `--model <name>`: Ollama model name (e.g., mistral, llama3, qwen2:7b)
- `--runs-per-test <N>`: Number of times to run each test (default: 3)
- `--outdir <dir>`: Output directory for results (default: results/)
- `--tests-dir <dir>`: Directory containing test JSON files (default: tests/)

### Dependencies
```bash
pip install requests numpy
```

### Prerequisites
- Python 3.9+
- Ollama installed and running locally
- At least one model pulled: `ollama pull mistral`

## Architecture

### Core Components

**1. Runner Versions**
- `enneagram_runner.py`: Original single-run version, outputs separate reports per test
- `enneagram_runner_v3_3run.py`: **Current production version** - multi-run with statistical analysis, unlabeled prompts (no type hints to LLM)
- `enneagram_runner_v2*.py`: Intermediate versions (legacy)

**2. Test Definitions (JSON)**
- `tests/enneagram_likert.json`: Likert-scale test (1-5 rating per statement)
  - Structure: `types` → `A-I` → each maps to an Enneagram type (1-9)
  - Each type has ~20 statements
- `tests/enneagram_test.json`: Paired forced-choice test (A/B questions)
  - Structure: `columns` → map to Enneagram types, `items` → question pairs
  - ~36 pairs total

**3. Ollama Integration**
- Uses HTTP API at `http://localhost:11434/api/generate`
- Non-streaming mode with 600s timeout
- Two prompt types:
  - `ask_likert_1_to_5()`: Extracts 1-5 rating from LLM response
  - `ask_forced_choice_ab()`: Extracts A or B choice

**4. Scoring System**
- **Enneagram Types**: 1-9 (primary personality types)
- **Centers of Intelligence**:
  - Head (5, 6, 7)
  - Heart (2, 3, 4)
  - Gut (8, 9, 1)
- **Derived Metrics**:
  - Core type (highest scoring type)
  - Primary wing (adjacent type with higher score)
  - Tritype (best from each center: Gut/Heart/Head)

**5. Statistical Analysis (v3)**
- Runs each test N times (default 3)
- Computes:
  - Mean and standard deviation per type
  - Consistency across runs
  - Center stability
- Single consolidated Markdown report per run

### Output Files

All outputs go to `results/` directory with timestamp-based naming:

**Single run version:**
- `enneagram-assessment-likert-scale_<model>_<YYYY-MM-DD>.md`
- `enneagram-test-assessment_<model>_<YYYY-MM-DD>.md`

**Multi-run version (v3):**
- `enneagram-multi_<model>_<YYYY-MM-DD_HH-MM-SS>_v3_unlabeled.md`

### Key Design Patterns

**Prompt Engineering:**
- v3 uses **unlabeled prompts** - no type or Enneagram labels shown to LLM during test
- Prevents priming bias
- Question text is pure statement without metadata

**Response Parsing:**
- Defensive extraction with regex and fallbacks
- For Likert: searches for digits 1-5, defaults to 3
- For A/B: searches for letters A or B, defaults to A

**Type Mapping:**
- Likert: Personality types A-I each map to Enneagram types 1-9
- Paired: Columns A-I map to types, chosen column determines type scoring

## Important Implementation Notes

- Always ensure Ollama is running before executing tests (will timeout otherwise)
- The v3 runner is preferred for research/analysis due to statistical rigor
- Test JSON files define the entire assessment - modifying them changes test content
- Transcripts include both raw LLM responses and parsed values for debugging
- Model responses are logged per-question with timestamps for temporal analysis
