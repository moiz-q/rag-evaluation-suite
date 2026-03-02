# Quick Start - RAG Evaluation Suite

Get evaluating in 5 minutes!

## Prerequisites

1. **Install Ollama**
   - Download: https://ollama.com/download
   - Start: `ollama serve`

2. **Pull models**
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 5-Minute Demo

### Step 1: Run Initial Evaluation

```bash
python run_evaluation.py
```

Output:
```
============================================================
RAG EVALUATION SUITE
============================================================

Evaluating 1/10: What is Python?...
  Faithfulness: 0.92
  Answer Relevance: 0.88
  Context Precision: 1.00
  Context Recall: 0.85

...

============================================================
EVALUATION RESULTS
============================================================

Faithfulness:
  Mean:  0.857
  Min:   0.720
  Max:   0.950

Answer Relevance:
  Mean:  0.823
  Min:   0.680
  Max:   0.920

Context Precision:
  Mean:  0.900
  Min:   0.750
  Max:   1.000

Context Recall:
  Mean:  0.795
  Min:   0.650
  Max:   0.900
```

### Step 2: Save as Baseline

```bash
# Rename results to baseline
mv evaluation_results.json baseline_results.json
```

### Step 3: Make Improvements

Edit `simple_rag.py` to improve your RAG system:

```python
# Example: Increase number of retrieved documents
def query(self, question: str, k: int = 3):  # Changed from k=2
    ...
```

### Step 4: Compare Improvements

```bash
python compare_improvements.py
```

Output:
```
============================================================
RAG IMPROVEMENT COMPARISON
============================================================

Baseline loaded. Running current evaluation...

============================================================
COMPARISON WITH BASELINE
============================================================

Faithfulness:
  Baseline: 0.857
  Current:  0.892
  ✓ Improvement: +0.035 (+4.1%)

Answer Relevance:
  Baseline: 0.823
  Current:  0.851
  ✓ Improvement: +0.028 (+3.4%)

Context Precision:
  Baseline: 0.900
  Current:  0.875
  ✗ Regression: -0.025 (-2.8%)

Context Recall:
  Baseline: 0.795
  Current:  0.842
  ✓ Improvement: +0.047 (+5.9%)
```

**You just proved your improvement numerically!** 🎉

## What Just Happened?

1. **Baseline Evaluation** - Measured initial RAG performance
2. **Made Changes** - Increased retrieved documents from 2 to 3
3. **Re-evaluated** - Measured new performance
4. **Compared** - Saw +4.1% faithfulness, +5.9% recall

## Key Metrics Explained

### Faithfulness (0-1)
- **What:** Does answer stick to retrieved context?
- **Good:** 0.8+ (80%+ of claims supported)
- **Bad:** <0.6 (hallucinations)

### Answer Relevance (0-1)
- **What:** Is answer relevant to question?
- **Good:** 0.8+ (highly relevant)
- **Bad:** <0.5 (off-topic)

### Context Precision (0-1)
- **What:** Are retrieved docs relevant?
- **Good:** 0.9+ (90%+ relevant)
- **Bad:** <0.7 (retrieving junk)

### Context Recall (0-1)
- **What:** Did we retrieve all needed info?
- **Good:** 0.8+ (got everything)
- **Bad:** <0.6 (missing info)

## Common Commands

```bash
# Run evaluation
python run_evaluation.py

# Compare with baseline
python compare_improvements.py

# Test RAG system manually
python simple_rag.py
```

## Try These Improvements

### 1. Increase Retrieved Documents
```python
# In simple_rag.py
def query(self, question: str, k: int = 3):  # Was k=2
```

**Expected:** Higher recall, possibly lower precision

### 2. Add Query Rewriting
```python
def query(self, question: str):
    # Expand query
    expanded = self.expand_query(question)
    retrieved = self.retrieve(expanded, k=2)
    ...
```

**Expected:** Higher recall and relevance

### 3. Add Reranking
```python
def retrieve(self, query: str, k: int = 2):
    # Get more candidates
    candidates = self._get_candidates(query, k=k*2)
    # Rerank
    reranked = self._rerank(query, candidates)
    return reranked[:k]
```

**Expected:** Higher precision

## Troubleshooting

**"Connection refused"**
- Start Ollama: `ollama serve`

**"Model not found"**
- Pull models: `ollama pull llama3.2`

**Evaluation is slow**
- Normal! Each question requires multiple LLM calls
- 10 questions ≈ 2-3 minutes

**Scores seem low**
- This is normal for simple RAG systems
- Production systems: 0.8-0.9 range
- Simple systems: 0.6-0.8 range

## Next Steps

1. Read `README.md` for detailed explanation
2. Check `PROJECT_SUMMARY.md` for architecture
3. Modify `simple_rag.py` to test improvements
4. Create your own test dataset in `test_dataset.py`
5. Evaluate your production RAG system

## Key Takeaway

**Stop guessing, start measuring!** Every change should be validated with numbers. 📊
