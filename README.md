# Enneagram LLM Testing Framework

This repository contains a Python-based testing harness for administering two Enneagram personality assessments to any local LLM running via **Ollama**:

1. **Likert-Scale Enneagram Test**
2. **Paired Forced-Choice Enneagram Test**

The script runs both tests multiple times (default: 3) per model and outputs a unified Markdown report that includes:

- Full test transcripts  
- All answers chosen by the LLM  
- Scores per type  
- Center distributions  
- Variability across runs  
- Statistical analysis (mean, standard deviation)  
- Determined wings, tritype estimations, and behavioral patterns  
- Time of day for each run  
- Combined summary and interpretation  

---

## 📁 Project Structure

```
/
├── enneagram_runner.py
├── tests/
│   ├── enneagram_likert.json
│   ├── enneagram_test.json
├── results/
│   └── (generated reports go here)
└── README.md
```

---

## 🚀 Requirements

- Python 3.9+
- Ollama installed and running
- At least one model pulled (example: `mistral`)

Install Python dependencies:

```bash
pip install requests numpy
```

---

## 🧠 How It Works

The script:

1. Loads both JSON test files  
2. Sends formatted prompts to the LLM via the Ollama API  
3. Captures all responses, including transcripts  
4. Scores the answers  
5. Computes full Enneagram dynamics:
   - Dominant type
   - Wings
   - Centers of intelligence
   - Tri-type inference
   - Stress/security movement analysis  
6. Repeats N times  
7. Computes:
   - Mean score per type  
   - Standard deviation  
   - Consistency scoring  
8. Outputs a full Markdown report named:

```
enneagram_multi_<model>_<YYYY-MM-DD_HH-MM-SS>.md
```

---

## ▶️ Running the Script

To run with defaults (3 runs, both tests):

```bash
python3 enneagram_runner.py --model mistral
```

Run only one test:

```bash
python3 enneagram_runner.py --model mistral --run likert
python3 enneagram_runner.py --model mistral --run paired
```

Change number of runs:

```bash
python3 enneagram_runner.py --model mistral --runs 5
```

Change output directory:

```bash
python3 enneagram_runner.py --model mistral --outdir myreports/
```

---

## 📊 Understanding the Output

Each generated file contains:

### For EACH RUN:
- Timestamp  
- Full transcript of the test  
- Answers selected  
- Numeric scoring  
- Center weighting  
- Wing analysis  
- Tri-type hypothesis  

### ACROSS ALL RUNS:
- Score matrix  
- Mean & standard deviation per type  
- Consistency index  
- Center stability  
- Model “psychological signature”  

---

## 🌐 Git Commands (SSH)

Initialize repo:

```bash
git init
git add .
git commit -m "Initial commit"
```

Add remote via SSH:

```bash
git remote add origin git@github.com:<yourname>/<yourrepo>.git
```

Push:

```bash
git push -u origin main
```

If your branch is `master`:

```bash
git push -u origin master
```

---

## ✨ Notes

This framework is designed for:

- LLM psychological profiling  
- Behavioral reproducibility studies  
- Alignment experiments  
- Model-to-model comparison  
- “Personality drift” testing  

If you'd like, I can also generate:

- Example plots  
- A dashboard UI  
- A web app version  
- A HuggingFace Space for comparing results
