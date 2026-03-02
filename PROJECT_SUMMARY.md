# Project Summary - RAG Evaluation Suite

## Overview

A comprehensive evaluation framework for RAG (Retrieval-Augmented Generation) systems. Provides automated metrics to measure and prove improvements numerically instead of guessing quality.

## Architecture

### Core Components

1. **Test Dataset** (`test_dataset.py`)
   - 10 sample questions with ground truth answers
   - Relevant document mappings
   - Extensible for custom datasets

2. **Metrics** (`metrics.py`)
   - Faithfulness (hallucination detection)
   - Answer Relevance (semantic similarity)
   - Context Precision (retrieval quality)
   - Context Recall (information completeness)

3. **Evaluator** (`evaluator.py`)
   - Runs evaluation on test dataset
   - Aggregates scores
   - Compares with baseline
   - Saves results to JSON

4. **Simple RAG** (`simple_rag.py`)
   - Basic RAG implementation for testing
   - Vector-based retrieval
   - LLM-based generation
   - Easily modifiable for experiments

## Evaluation Flow

```
Test Dataset
    ↓
For each question:
    ↓
RAG System (question → answer + contexts)
    ↓
Evaluate Metrics:
  • Faithfulness (answer vs contexts)
  • Answer Relevance (answer vs question)
  • Context Precision (contexts vs question)
  • Context Recall (contexts vs ground truth)
    ↓
Aggregate Scores
    ↓
Compare with Baseline
    ↓
Report Improvements
```

## Metrics Implementation

### 1. Faithfulness

**Method:** LLM-as-judge
```python
1. Extract claims from answer
2. Check if each claim is supported by context
3. Score = supported_claims / total_claims
```

**Fallback:** Keyword overlap between answer and context

### 2. Answer Relevance

**Method:** Embedding similarity
```python
1. Generate embedding for question
2. Generate embedding for answer
3. Score = cosine_similarity(question_emb, answer_emb)
```

**Fallback:** Keyword overlap

### 3. Context Precision

**Method:** LLM-as-judge
```python
For each retrieved context:
    Ask LLM: "Is this context relevant to the question?"
Score = relevant_contexts / total_contexts
```

**Fallback:** Keyword overlap

### 4. Context Recall

**Method:** LLM-as-judge
```python
1. Extract key facts from ground truth
2. Check if each fact is in retrieved contexts
3. Score = facts_found / total_facts
```

**Fallback:** Keyword overlap

## Usage Patterns

### Pattern 1: Initial Evaluation

```python
from evaluator import RAGEvaluator
from simple_rag import create_rag_function

rag_function = create_rag_function()
evaluator = RAGEvaluator(rag_function)
results = evaluator.evaluate()
evaluator.save_results("baseline_results.json")
```

### Pattern 2: Compare Improvements

```python
import json

# Load baseline
with open("baseline_results.json") as f:
    baseline = json.load(f)["aggregated"]

# Evaluate improved system
evaluator = RAGEvaluator(improved_rag_function)
results = evaluator.evaluate()

# Compare
evaluator.print_comparison(baseline)
```

### Pattern 3: Custom Test Dataset

```python
custom_dataset = [
    {
        "question": "Your question",
        "ground_truth": "Expected answer",
        "relevant_docs": ["doc1.txt"]
    },
    # ... more test cases
]

evaluator.evaluate(test_dataset=custom_dataset)
```

## File Structure

```
rag-evaluation-suite/
├── test_dataset.py          # Test questions + ground truth
├── metrics.py               # Evaluation metrics
├── evaluator.py             # Evaluation orchestrator
├── simple_rag.py            # Sample RAG system
├── embeddings.py            # Embedding utilities
├── llm.py                   # LLM interface
├── run_evaluation.py        # Main evaluation script
├── compare_improvements.py  # Comparison script
├── requirements.txt         # Dependencies
└── README.md               # Full documentation
```

## Metrics Interpretation

### Faithfulness
- **0.9-1.0:** Excellent (no hallucinations)
- **0.8-0.9:** Good (minor unsupported claims)
- **0.7-0.8:** Fair (some hallucinations)
- **<0.7:** Poor (significant hallucinations)

### Answer Relevance
- **0.9-1.0:** Excellent (directly answers question)
- **0.8-0.9:** Good (relevant with minor tangents)
- **0.7-0.8:** Fair (somewhat relevant)
- **<0.7:** Poor (off-topic)

### Context Precision
- **0.9-1.0:** Excellent (all docs relevant)
- **0.8-0.9:** Good (mostly relevant)
- **0.7-0.8:** Fair (some irrelevant docs)
- **<0.7:** Poor (retrieving junk)

### Context Recall
- **0.9-1.0:** Excellent (all info retrieved)
- **0.8-0.9:** Good (most info retrieved)
- **0.7-0.8:** Fair (missing some info)
- **<0.7:** Poor (missing critical info)

## Improvement Strategies

### To Improve Faithfulness
- Use more specific prompts
- Add "stick to context" instructions
- Implement citation requirements
- Use smaller context windows

### To Improve Answer Relevance
- Improve query understanding
- Add query rewriting
- Use better prompts
- Fine-tune generation

### To Improve Context Precision
- Improve retrieval algorithm
- Add reranking
- Use hybrid search (BM25 + vector)
- Tune similarity thresholds

### To Improve Context Recall
- Retrieve more documents
- Use query expansion
- Improve chunking strategy
- Use multiple retrieval methods

## Performance Considerations

### Evaluation Speed
- **Per question:** ~10-20 seconds
- **10 questions:** ~2-3 minutes
- **100 questions:** ~20-30 minutes

**Bottlenecks:**
- LLM calls for faithfulness/precision/recall
- Embedding generation for relevance

**Optimization:**
- Cache embeddings
- Batch LLM calls
- Use faster models for evaluation
- Parallelize evaluation

### Accuracy vs Speed Trade-offs

**High Accuracy (Slow):**
- Use LLM-as-judge for all metrics
- Multiple evaluation passes
- Detailed claim extraction

**Balanced (Medium):**
- LLM for faithfulness/precision
- Embeddings for relevance
- Keyword fallback for recall

**Fast (Lower Accuracy):**
- Keyword overlap for all metrics
- Single pass evaluation
- No LLM calls

## Limitations

1. **LLM-as-judge bias** - Evaluation LLM may have biases
2. **Ground truth required** - Need labeled test data
3. **Slow evaluation** - Multiple LLM calls per question
4. **Metric limitations** - No single metric captures everything
5. **Context dependency** - Metrics depend on retrieval quality

## Best Practices

1. **Create good test datasets**
   - Cover diverse question types
   - Include edge cases
   - Use domain-specific questions
   - Have clear ground truth

2. **Establish baselines**
   - Always measure before changes
   - Save baseline results
   - Track improvements over time

3. **Iterate systematically**
   - Change one thing at a time
   - Measure impact
   - Keep what works
   - Discard what doesn't

4. **Use multiple metrics**
   - Don't optimize for single metric
   - Balance trade-offs
   - Consider user experience

5. **Validate with humans**
   - Automated metrics aren't perfect
   - Spot-check results
   - Get user feedback

## Integration with Production

### CI/CD Integration

```python
# In your CI pipeline
def test_rag_quality():
    evaluator = RAGEvaluator(production_rag)
    results = evaluator.evaluate()
    
    # Fail if quality drops
    assert results["faithfulness"]["mean"] >= 0.8
    assert results["answer_relevance"]["mean"] >= 0.75
```

### Monitoring

```python
# Track metrics over time
def monitor_rag_quality():
    results = evaluate_rag_system(rag_function)
    
    # Log to monitoring system
    log_metric("rag.faithfulness", results["faithfulness"]["mean"])
    log_metric("rag.relevance", results["answer_relevance"]["mean"])
    
    # Alert if quality drops
    if results["faithfulness"]["mean"] < 0.7:
        send_alert("RAG quality degraded!")
```

## Dependencies

- **requests:** HTTP calls to Ollama
- **numpy:** Vector operations
- **Ollama:** LLM and embedding models

## Key Learnings

1. **Measurement is essential** - Can't improve what you don't measure
2. **Baselines matter** - Need reference point for comparison
3. **Multiple metrics needed** - No single metric tells full story
4. **Trade-offs exist** - Improving one metric may hurt another
5. **Automation saves time** - Manual evaluation doesn't scale
6. **Ground truth is hard** - Creating good test data takes effort
7. **Iterate systematically** - Measure, change, measure again

## Comparison to Manual Testing

### Manual Testing
- ❌ Subjective ("looks good")
- ❌ Slow (minutes per question)
- ❌ Not reproducible
- ❌ Doesn't scale
- ❌ Hard to compare versions

### Automated Evaluation
- ✅ Objective (numerical scores)
- ✅ Fast (seconds per question)
- ✅ Reproducible
- ✅ Scales to 100s of questions
- ✅ Easy to compare versions

## Conclusion

This evaluation suite provides the foundation for systematic RAG improvement. By measuring faithfulness, relevance, precision, and recall, you can prove improvements numerically instead of guessing.

**Key insight:** Every production RAG system needs automated evaluation. This repo shows you how to build it.
