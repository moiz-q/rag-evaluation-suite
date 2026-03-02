# 🚀 START HERE - RAG Evaluation Suite

Welcome! This is Repo 10 - learn to measure and prove RAG quality numerically.

## What is This?

An automated evaluation framework for RAG systems. Stop guessing if your RAG is good - measure it with numbers!

## Why Evaluate?

**Without evaluation:**
```
You: "I added reranking"
Boss: "Did it help?"
You: "Umm... it looks better?" 🤷
```

**With evaluation:**
```
You: "I added reranking"
Boss: "Did it help?"
You: "Yes! Faithfulness +18%, Precision +20%" 📊
Boss: "Ship it!" ✅
```

## 4 Key Metrics

1. **Faithfulness** - No hallucinations (answer sticks to context)
2. **Answer Relevance** - Answer addresses the question
3. **Context Precision** - Retrieved docs are relevant
4. **Context Recall** - All needed info was retrieved

## Quick Start (5 minutes)

### 1. Setup

```bash
# Install Ollama (if not already)
# Download from: https://ollama.com/download

# Pull models
ollama pull llama3.2
ollama pull nomic-embed-text

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Evaluation

```bash
python run_evaluation.py
```

Output:
```
Faithfulness:     0.857 (85.7% - Good!)
Answer Relevance: 0.823 (82.3% - Good!)
Context Precision: 0.900 (90.0% - Excellent!)
Context Recall:   0.795 (79.5% - Fair)
```

### 3. Make Improvements

```bash
# Save baseline
mv evaluation_results.json baseline_results.json

# Edit simple_rag.py
# Change: k=2 → k=3 (retrieve more docs)

# Compare
python compare_improvements.py
```

Output:
```
Faithfulness:  +4.1% ✓
Relevance:     +3.4% ✓
Precision:     -2.8% ✗
Recall:        +5.9% ✓

Overall: Net positive!
```

**You just proved improvement with data!** 🎉

## File Structure

```
rag-evaluation-suite/
├── run_evaluation.py        # Main script (START HERE)
├── compare_improvements.py  # Compare versions
├── test_dataset.py          # Test questions
├── metrics.py               # Evaluation metrics
├── evaluator.py             # Evaluation engine
├── simple_rag.py            # Sample RAG system
└── README.md               # Full documentation
```

## What to Read

**Just want to run it?**
→ You're done! (this file)

**Want to understand metrics?**
→ Read QUICKSTART.md (5 min)

**Want deep dive?**
→ Read README.md (30 min)

**Want technical details?**
→ Read PROJECT_SUMMARY.md (15 min)

**Want examples?**
→ Read EXAMPLES.md

## Common Commands

```bash
# Run evaluation
python run_evaluation.py

# Compare with baseline
python compare_improvements.py

# Test RAG manually
python simple_rag.py
```

## Metrics Cheat Sheet

### Faithfulness (0-1)
- **0.9+:** Excellent (no hallucinations)
- **0.8-0.9:** Good
- **0.7-0.8:** Fair
- **<0.7:** Poor (hallucinating)

### Answer Relevance (0-1)
- **0.9+:** Excellent (directly answers)
- **0.8-0.9:** Good
- **0.7-0.8:** Fair
- **<0.7:** Poor (off-topic)

### Context Precision (0-1)
- **0.9+:** Excellent (all docs relevant)
- **0.8-0.9:** Good
- **0.7-0.8:** Fair
- **<0.7:** Poor (retrieving junk)

### Context Recall (0-1)
- **0.9+:** Excellent (got everything)
- **0.8-0.9:** Good
- **0.7-0.8:** Fair
- **<0.7:** Poor (missing info)

## How It Works

```
Test Question
    ↓
RAG System (answer + contexts)
    ↓
Evaluate 4 Metrics
    ↓
Aggregate Scores
    ↓
Compare with Baseline
    ↓
Report Improvements
```

## Try These Improvements

### 1. Retrieve More Documents
```python
# In simple_rag.py, line 120
def query(self, question: str, k: int = 3):  # Was k=2
```
**Expected:** Higher recall

### 2. Add Better Prompts
```python
# In simple_rag.py, line 130
prompt = f"""Answer based ONLY on the context. Be concise.

Context: {context_text}
Question: {query}
Answer:"""
```
**Expected:** Higher faithfulness

### 3. Filter Low-Confidence Results
```python
# In simple_rag.py, line 110
if sim > 0.5:  # Only keep high-confidence
    results.append((doc_name, self.knowledge_base[doc_name]))
```
**Expected:** Higher precision

## Troubleshooting

**"Connection refused"**
→ Start Ollama: `ollama serve`

**"Model not found"**
→ Pull models: `ollama pull llama3.2`

**Evaluation is slow**
→ Normal! 10 questions ≈ 2-3 minutes

**Scores seem low**
→ Simple RAG: 0.6-0.8 (normal)
→ Production RAG: 0.8-0.9 (good)

## Prerequisites

Before this repo, you should understand:
- RAG basics (Repo 3)
- Production RAG (Repo 5)

**Haven't done those?** Check LEARNING_PATH.md in parent directory.

## Learning Goals

✅ Done when:
- You can evaluate RAG systems automatically
- You understand all 4 metrics
- You can prove improvements numerically
- You can compare different approaches
- You can set quality thresholds

## Time Required

- Quick demo: 5 minutes
- Full understanding: 1 hour
- Integration: 2-3 hours

## What's Next?

After mastering this:
1. Evaluate your production RAG
2. Create domain-specific test datasets
3. Integrate with CI/CD
4. Monitor quality over time
5. Build evaluation dashboard

## Key Insight

**You can't improve what you don't measure!**

Every production RAG system needs automated evaluation. This repo shows you how.

## Ready?

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run evaluation
python run_evaluation.py

# 3. Read QUICKSTART.md for more
```

Stop guessing, start measuring! 📊
