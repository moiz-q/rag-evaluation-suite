# RAG Evaluation Suite

Goal: Stop guessing RAG quality - measure and prove improvements numerically.

Learn how to evaluate RAG systems using automated metrics, ground-truth datasets, and systematic testing. Understand what makes a RAG system "good" and how to prove it with data.

**Uses Ollama** - completely free, runs locally!

## What is RAG Evaluation?

RAG evaluation is the process of measuring how well your RAG system performs using quantitative metrics. Instead of manually checking if answers "seem good", you measure:

1. **Faithfulness** - Does the answer stick to the retrieved context?
2. **Relevance** - Is the answer relevant to the question?
3. **Retrieval Precision** - Did we retrieve the right documents?
4. **Retrieval Recall** - Did we retrieve all relevant documents?

**Key insight:** You can't improve what you don't measure!

## Why Evaluate?

### Without Evaluation (Guessing)

```python
# Make a change
rag_system.add_reranking()

# Test manually
answer = rag_system.query("What is Python?")
print(answer)  # "Looks better... I think?"

# Ship it! 🤞
```

### With Evaluation (Knowing)

```python
# Baseline
baseline_scores = evaluate(rag_system, test_set)
# Faithfulness: 0.72, Relevance: 0.68, Precision: 0.65

# Make a change
rag_system.add_reranking()

# Measure improvement
new_scores = evaluate(rag_system, test_set)
# Faithfulness: 0.85 (+18%), Relevance: 0.82 (+21%), Precision: 0.78 (+20%)

# Ship it with confidence! ✅
```

## Evaluation Metrics

### 1. Faithfulness (Answer Quality)

**Question:** Does the answer stick to the retrieved context?

**Measures:** Hallucination rate

```python
Context: "Python was created by Guido van Rossum in 1991"
Question: "Who created Python?"

Good answer: "Python was created by Guido van Rossum" ✓
Bad answer: "Python was created by Google in 2000" ✗ (hallucination)

Faithfulness score: 1.0 (good) vs 0.0 (bad)
```

**How it works:**
- Extract claims from the answer
- Check if each claim is supported by context
- Score = (supported claims) / (total claims)

### 2. Answer Relevance

**Question:** Is the answer relevant to the question?

**Measures:** How well the answer addresses the question

```python
Question: "What are the benefits of exercise?"

Good answer: "Exercise improves cardiovascular health, strengthens muscles..." ✓
Bad answer: "Exercise can be done indoors or outdoors" ✗ (not relevant)

Relevance score: 0.9 (good) vs 0.3 (bad)
```

**How it works:**
- Generate embedding for question
- Generate embedding for answer
- Score = cosine similarity

### 3. Context Precision (Retrieval Quality)

**Question:** Are the retrieved documents relevant?

**Measures:** Precision of retrieval

```python
Question: "How to train a neural network?"

Retrieved docs:
1. "Neural network training guide" ✓ (relevant)
2. "Deep learning basics" ✓ (relevant)
3. "History of computers" ✗ (not relevant)

Precision = 2/3 = 0.67
```

**How it works:**
- For each retrieved document
- Ask: "Is this relevant to the question?"
- Score = relevant_docs / total_docs

### 4. Context Recall (Information Completeness)

**Question:** Did we retrieve all necessary information?

**Measures:** Completeness of retrieval

```python
Question: "What are Python data types?"
Ground truth: "Python has int, float, str, bool, list, dict, tuple, set"

Retrieved context mentions: int, float, str, bool, list
Missing: dict, tuple, set

Recall = 5/8 = 0.625
```

**How it works:**
- Extract key facts from ground truth
- Check if each fact is in retrieved contexts
- Score = facts_found / total_facts

## Setup

1. **Install Ollama** (if you haven't already)
   - Download from: https://ollama.com/download

2. **Pull required models**
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Run Initial Evaluation

```bash
python run_evaluation.py
```

This will:
- Evaluate a sample RAG system on 10 test questions
- Compute all 4 metrics
- Display results
- Save to `evaluation_results.json`

### 2. Save as Baseline

```bash
mv evaluation_results.json baseline_results.json
```

### 3. Make Improvements

Edit `simple_rag.py` to improve your RAG system. For example:

```python
# Increase retrieved documents
def query(self, question: str, k: int = 3):  # Was k=2
    ...
```

### 4. Compare Improvements

```bash
python compare_improvements.py
```

This will:
- Load baseline results
- Evaluate current system
- Show improvements/regressions
- Display percentage changes

## Project Structure

```
rag-evaluation-suite/
├── test_dataset.py          # Test questions + ground truth
├── metrics.py               # Evaluation metrics implementation
├── evaluator.py             # Evaluation orchestrator
├── simple_rag.py            # Sample RAG system to evaluate
├── embeddings.py            # Embedding utilities
├── llm.py                   # LLM interface
├── run_evaluation.py        # Main evaluation script
├── compare_improvements.py  # Comparison script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Evaluation Metrics Explained

### Faithfulness (0-1)

**What it measures:** Does the answer stick to the retrieved context?

**How it works:**
1. Extract claims from the answer
2. Check if each claim is supported by context
3. Score = supported_claims / total_claims

**Example:**
```python
Context: "Python was created by Guido van Rossum in 1991"
Question: "Who created Python?"

Good answer: "Python was created by Guido van Rossum"
→ Faithfulness: 1.0 (all claims supported)

Bad answer: "Python was created by Google in 2000"
→ Faithfulness: 0.0 (hallucinated facts)
```

**Interpretation:**
- 0.9-1.0: Excellent (no hallucinations)
- 0.8-0.9: Good (minor unsupported claims)
- 0.7-0.8: Fair (some hallucinations)
- <0.7: Poor (significant hallucinations)

### Answer Relevance (0-1)

**What it measures:** Is the answer relevant to the question?

**How it works:**
1. Generate embedding for question
2. Generate embedding for answer
3. Score = cosine_similarity(question_emb, answer_emb)

**Example:**
```python
Question: "What are the benefits of exercise?"

Good answer: "Exercise improves cardiovascular health, strengthens muscles..."
→ Relevance: 0.92 (highly relevant)

Bad answer: "Exercise can be done indoors or outdoors"
→ Relevance: 0.35 (not answering the question)
```

**Interpretation:**
- 0.9-1.0: Excellent (directly answers question)
- 0.8-0.9: Good (relevant with minor tangents)
- 0.7-0.8: Fair (somewhat relevant)
- <0.7: Poor (off-topic)

### Context Precision (0-1)

**What it measures:** Are the retrieved documents relevant?

**How it works:**
1. For each retrieved document
2. Ask LLM: "Is this relevant to the question?"
3. Score = relevant_docs / total_docs

**Example:**
```python
Question: "How to train a neural network?"

Retrieved:
1. "Neural network training guide" → Relevant ✓
2. "Deep learning basics" → Relevant ✓
3. "History of computers" → Not relevant ✗

Precision = 2/3 = 0.67
```

**Interpretation:**
- 0.9-1.0: Excellent (all docs relevant)
- 0.8-0.9: Good (mostly relevant)
- 0.7-0.8: Fair (some irrelevant docs)
- <0.7: Poor (retrieving junk)

### Context Recall (0-1)

**What it measures:** Did we retrieve all necessary information?

**How it works:**
1. Extract key facts from ground truth answer
2. Check if each fact is in retrieved contexts
3. Score = facts_found / total_facts

**Example:**
```python
Question: "What are Python data types?"
Ground truth: "Python has int, float, str, bool, list, dict, tuple, set"

Retrieved context mentions: int, float, str, bool, list
Missing: dict, tuple, set

Recall = 5/8 = 0.625
```

**Interpretation:**
- 0.9-1.0: Excellent (all info retrieved)
- 0.8-0.9: Good (most info retrieved)
- 0.7-0.8: Fair (missing some info)
- <0.7: Poor (missing critical info)

## Example Evaluation Output

```
============================================================
RAG EVALUATION SUITE
============================================================

Evaluating 1/10: What is Python?...
  Faithfulness: 0.92
  Answer Relevance: 0.88
  Context Precision: 1.00
  Context Recall: 0.85

Evaluating 2/10: What are the benefits of regular exercise?...
  Faithfulness: 0.87
  Answer Relevance: 0.91
  Context Precision: 1.00
  Context Recall: 0.82

...

============================================================
EVALUATION RESULTS
============================================================

Faithfulness:
  Mean:  0.857
  Min:   0.720
  Max:   0.950
  Count: 10

Answer Relevance:
  Mean:  0.823
  Min:   0.680
  Max:   0.920
  Count: 10

Context Precision:
  Mean:  0.900
  Min:   0.750
  Max:   1.000
  Count: 10

Context Recall:
  Mean:  0.795
  Min:   0.650
  Max:   0.900
  Count: 10
```

## Comparison Example

```
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

## Creating Custom Test Datasets

Edit `test_dataset.py` to add your own test cases:

```python
TEST_DATASET = [
    {
        "question": "Your question here",
        "ground_truth": "Expected answer",
        "relevant_docs": ["doc1.txt", "doc2.txt"]
    },
    # Add more test cases...
]
```

**Best practices:**
- Cover diverse question types
- Include edge cases
- Use domain-specific questions
- Ensure ground truth is accurate
- Aim for 20-50 test cases minimum

## Evaluating Your Own RAG System

Replace the `simple_rag.py` with your RAG system:

```python
from evaluator import RAGEvaluator

def my_rag_function(question: str):
    """
    Your RAG system.
    
    Returns:
        (answer: str, contexts: List[str])
    """
    # Your RAG logic here
    answer = your_rag.query(question)
    contexts = your_rag.get_contexts()
    
    return answer, contexts

# Evaluate
evaluator = RAGEvaluator(my_rag_function)
results = evaluator.evaluate()
```

## Improvement Strategies

### To Improve Faithfulness
- Use more specific prompts ("Answer based only on context")
- Add citation requirements
- Implement fact-checking
- Use smaller context windows

### To Improve Answer Relevance
- Improve query understanding
- Add query rewriting/expansion
- Use better generation prompts
- Fine-tune generation model

### To Improve Context Precision
- Improve retrieval algorithm
- Add reranking step
- Use hybrid search (BM25 + vector)
- Tune similarity thresholds
- Filter low-confidence results

### To Improve Context Recall
- Retrieve more documents (increase k)
- Use query expansion
- Improve chunking strategy
- Use multiple retrieval methods
- Implement recursive retrieval

## Common Patterns

### Pattern 1: A/B Testing

```python
# Test two RAG variants
evaluator_a = RAGEvaluator(rag_variant_a)
results_a = evaluator_a.evaluate()

evaluator_b = RAGEvaluator(rag_variant_b)
results_b = evaluator_b.evaluate()

# Compare
print(f"Variant A faithfulness: {results_a['faithfulness']['mean']}")
print(f"Variant B faithfulness: {results_b['faithfulness']['mean']}")
```

### Pattern 2: Hyperparameter Tuning

```python
best_score = 0
best_k = 0

for k in [1, 2, 3, 4, 5]:
    rag = create_rag_with_k(k)
    evaluator = RAGEvaluator(rag)
    results = evaluator.evaluate(verbose=False)
    
    score = results['faithfulness']['mean']
    if score > best_score:
        best_score = score
        best_k = k

print(f"Best k: {best_k} with score: {best_score}")
```

### Pattern 3: CI/CD Integration

```python
def test_rag_quality():
    """Run in CI pipeline."""
    evaluator = RAGEvaluator(production_rag)
    results = evaluator.evaluate()
    
    # Fail if quality drops below threshold
    assert results["faithfulness"]["mean"] >= 0.8, "Faithfulness too low!"
    assert results["answer_relevance"]["mean"] >= 0.75, "Relevance too low!"
```

## Limitations

1. **Ground truth required** - Need labeled test data
2. **LLM-as-judge bias** - Evaluation LLM may have biases
3. **Slow evaluation** - Multiple LLM calls per question
4. **Metric limitations** - No single metric captures everything
5. **Context dependency** - Metrics depend on retrieval quality

## Advanced: RAGAS Integration

This repo implements metrics similar to RAGAS (RAG Assessment). To use the official RAGAS library:

```bash
pip install ragas
```

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# Your evaluation code here
```

## Troubleshooting

**"Connection refused"**
- Start Ollama: `ollama serve`

**"Model not found"**
- Pull models: `ollama pull llama3.2` and `ollama pull nomic-embed-text`

**Evaluation is slow**
- Normal! Each question requires multiple LLM calls
- 10 questions ≈ 2-3 minutes
- Consider caching or using faster models

**Scores seem low**
- Simple RAG systems: 0.6-0.8 range (normal)
- Production systems: 0.8-0.9 range (good)
- Perfect scores (1.0) are rare

**Metrics don't match manual assessment**
- Automated metrics aren't perfect
- Use as guidance, not absolute truth
- Validate with human evaluation

## Done When

- ✅ You can evaluate RAG systems automatically
- ✅ You have baseline metrics
- ✅ You can prove improvements numerically
- ✅ You understand metric trade-offs
- ✅ You can iterate systematically

## Why This Matters

**Without evaluation:**
- Guessing if changes help
- No way to prove improvements
- Can't compare approaches
- Manual testing doesn't scale

**With evaluation:**
- Know exactly what improved
- Prove changes with data
- Compare approaches objectively
- Automated testing at scale

**Key insight:** You can't improve what you don't measure!

## Next Steps

After mastering this:
- Create domain-specific test datasets
- Integrate with CI/CD pipeline
- Add custom metrics
- Build evaluation dashboard
- Implement A/B testing framework
- Add human evaluation loop

## Real-World Applications

**Development:**
- Measure impact of changes
- Compare different approaches
- Tune hyperparameters
- Debug quality issues

**Production:**
- Monitor quality over time
- Detect regressions
- A/B test improvements
- Track user satisfaction

**Research:**
- Compare RAG architectures
- Evaluate new techniques
- Publish results
- Reproduce experiments

Stop guessing, start measuring! 📊

## Key Takeaways

1. **Measurement is essential** - Can't improve without metrics
2. **Multiple metrics needed** - No single metric tells full story
3. **Baselines matter** - Need reference point for comparison
4. **Iterate systematically** - Measure → Change → Measure
5. **Automation scales** - Manual testing doesn't work long-term
6. **Trade-offs exist** - Improving one metric may hurt another
7. **Validation required** - Automated metrics + human judgment

Evaluation transforms RAG development from art to science! 🔬
