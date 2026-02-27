# Enneagram LLM Testing Framework

A Python testing harness for administering Enneagram personality assessments to LLMs. Supports local models via **Ollama** and cloud APIs via **Anthropic**, **OpenAI**, and **OpenRouter**.

Two test types:

1. **Likert-Scale Enneagram Test** (1-5 rating per statement)
2. **Paired Forced-Choice Enneagram Test** (A/B questions)

The script runs both tests multiple times (default: 3) per model and outputs a unified Markdown report with full transcripts, scores, center distributions, statistical analysis, wings, tritype estimations, and variability across runs.

---

## Project Structure

```
/
├── providers.py                              # LLM provider abstraction
├── enneagram_runner_v3-2_3run.py             # Main runner (recommended)
├── enneagram_runner_v3-2_3run_NoContext.py   # NoContext variant (clears Ollama context)
├── run_all_models.py                         # Batch runner for multiple models
├── tests/
│   ├── enneagram_likert.json
│   └── enneagram_test.json
├── results/                                  # Generated reports
└── README.md
```

---

## Requirements

- Python 3.9+
- `pip install requests`
- For Ollama: Ollama installed and running, at least one model pulled
- For cloud providers: API key via env var or `--api-key` flag

---

## Quick Start

### Local (Ollama)

```bash
python3 enneagram_runner_v3-2_3run.py --model mistral
python3 enneagram_runner_v3-2_3run.py --model llama3 --runs-per-test 5
```

### Anthropic (Claude)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python3 enneagram_runner_v3-2_3run.py --provider anthropic --model claude-sonnet-4-20250514
```

### OpenAI (GPT)

```bash
export OPENAI_API_KEY="sk-..."
python3 enneagram_runner_v3-2_3run.py --provider openai --model gpt-4o
```

### OpenRouter (any model)

```bash
export OPENROUTER_API_KEY="sk-or-..."
python3 enneagram_runner_v3-2_3run.py --provider openrouter --model anthropic/claude-sonnet-4
```

### Batch Run (all local Ollama models)

```bash
python3 run_all_models.py
python3 run_all_models.py --exclude mistral:latest
```

### Batch Run (cloud provider)

```bash
python3 run_all_models.py --provider anthropic --models claude-sonnet-4-20250514 claude-haiku-4-5-20251001
```

---

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model name (required) | -- |
| `--provider` | ollama, anthropic, openai, openrouter | ollama |
| `--api-key` | API key for cloud providers | env var |
| `--runs-per-test` | Number of test repetitions | 3 |
| `--temperature` | Sampling temperature | model default |
| `--outdir` | Output directory | results/ |
| `--tests-dir` | Test JSON directory | tests/ |

---

## How It Works

1. Loads both JSON test files
2. Sends formatted prompts to the LLM via the configured provider
3. Captures all responses, including transcripts
4. Scores the answers
5. Computes full Enneagram dynamics: dominant type, wings, centers of intelligence, tritype inference
6. Repeats N times
7. Computes mean score per type, standard deviation, consistency scoring
8. Outputs a Markdown report

### NoContext Variant

`enneagram_runner_v3-2_3run_NoContext.py` explicitly clears Ollama's context between requests to prevent personality drift from context accumulation. For cloud APIs, this has no effect (each request is already stateless).

---

## Understanding the Output

### Per Run
- Full transcript of each test
- Answers selected
- Numeric scoring per type
- Center weighting (Head/Heart/Gut)
- Wing analysis
- Tritype hypothesis

### Across All Runs
- Score matrix
- Mean and standard deviation per type
- Consistency index
- Center stability
