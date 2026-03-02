# RAG Evaluation Examples

Real-world examples showing how to use the evaluation suite.

## Example 1: Initial Evaluation

```bash
$ python run_evaluation.py

============================================================
RAG EVALUATION SUITE
============================================================

Initializing RAG system...
Building embeddings index...
Indexed 5 documents

Starting evaluation...

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

Results saved to evaluation_results.json
```

## Example 2: Comparing Improvements

```bash
# Save baseline
$ mv evaluation_results.json baseline_results.json

# Make improvements (edit simple_rag.py)
# Change: k=2 → k=3 (retrieve more documents)

# Compare
$ python compare_improvements.py

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

**Analysis:**
- Retrieving more documents improved recall (+5.9%)
- Faithfulness also improved (+4.1%)
- But precision dropped slightly (-2.8%) - some irrelevant docs
- Overall: Net positive improvement

## Example 3: Custom Evaluation

```python
from evaluator import RAGEvaluator

def my_rag_system(question: str):
    """Your custom RAG system."""
    # Your logic here
    answer = "Python is a programming language"
    contexts = ["Python is a high-level language..."]
    return answer, contexts

# Evaluate
evaluator = RAGEvaluator(my_rag_system)
results = evaluator.evaluate()

print(f"Faithfulness: {results['faithfulness']['mean']:.3f}")
print(f"Relevance: {results['answer_relevance']['mean']:.3f}")
```

## Example 4: Custom Test Dataset

```python
from evaluator import RAGEvaluator
from simple_rag import create_rag_function

# Custom test cases
custom_tests = [
    {
        "question": "What is machine learning?",
        "ground_truth": "Machine learning is a method where computers learn from data...",
        "relevant_docs": ["machine_learning.txt"]
    },
    {
        "question": "How does exercise help?",
        "ground_truth": "Exercise improves cardiovascular health...",
        "relevant_docs": ["exercise_guide.txt"]
    }
]

# Evaluate on custom dataset
rag_function = create_rag_function()
evaluator = RAGEvaluator(rag_function)
results = evaluator.evaluate(test_dataset=custom_tests)
```

## Example 5: Hyperparameter Tuning

```python
from evaluator import RAGEvaluator
from simple_rag import SimpleRAG

# Test different k values
results_by_k = {}

for k in [1, 2, 3, 4, 5]:
    print(f"\nTesting k={k}...")
    
    rag = SimpleRAG()
    
    def rag_function(question):
        return rag.query(question, k=k)
    
    evaluator = RAGEvaluator(rag_function)
    results = evaluator.evaluate(verbose=False)
    
    results_by_k[k] = results

# Find best k
best_k = max(results_by_k.keys(), 
             key=lambda k: results_by_k[k]['faithfulness']['mean'])

print(f"\nBest k: {best_k}")
print(f"Faithfulness: {results_by_k[best_k]['faithfulness']['mean']:.3f}")
```

Output:
```
Testing k=1...
Testing k=2...
Testing k=3...
Testing k=4...
Testing k=5...

Best k: 3
Faithfulness: 0.892
```

## Example 6: A/B Testing

```python
from evaluator import RAGEvaluator

# Variant A: Basic RAG
def rag_variant_a(question):
    # Basic retrieval
    return basic_rag.query(question)

# Variant B: RAG with reranking
def rag_variant_b(question):
    # Retrieval + reranking
    return advanced_rag.query_with_reranking(question)

# Evaluate both
evaluator_a = RAGEvaluator(rag_variant_a)
results_a = evaluator_a.evaluate()

evaluator_b = RAGEvaluator(rag_variant_b)
results_b = evaluator_b.evaluate()

# Compare
print("\nVariant A (Basic):")
print(f"  Faithfulness: {results_a['faithfulness']['mean']:.3f}")
print(f"  Precision: {results_a['context_precision']['mean']:.3f}")

print("\nVariant B (Reranking):")
print(f"  Faithfulness: {results_b['faithfulness']['mean']:.3f}")
print(f"  Precision: {results_b['context_precision']['mean']:.3f}")

# Winner
if results_b['faithfulness']['mean'] > results_a['faithfulness']['mean']:
    print("\n✓ Variant B wins!")
else:
    print("\n✓ Variant A wins!")
```

## Example 7: Metric Analysis

```python
from evaluator import RAGEvaluator
from simple_rag import create_rag_function
import json

# Run evaluation
rag_function = create_rag_function()
evaluator = RAGEvaluator(rag_function)
results = evaluator.evaluate()

# Analyze individual results
for result in evaluator.results:
    question = result['question']
    scores = result['scores']
    
    # Find low-scoring questions
    if scores['faithfulness'] < 0.7:
        print(f"\nLow faithfulness: {question}")
        print(f"  Score: {scores['faithfulness']:.2f}")
        print(f"  Answer: {result['answer'][:100]}...")
        print(f"  Issue: Possible hallucination")
    
    if scores['context_recall'] < 0.7:
        print(f"\nLow recall: {question}")
        print(f"  Score: {scores['context_recall']:.2f}")
        print(f"  Issue: Missing information in retrieval")
```

## Example 8: Saving and Loading Results

```python
import json
from evaluator import RAGEvaluator
from simple_rag import create_rag_function

# Run evaluation
rag_function = create_rag_function()
evaluator = RAGEvaluator(rag_function)
results = evaluator.evaluate()

# Save detailed results
evaluator.save_results("my_evaluation.json")

# Load and analyze later
with open("my_evaluation.json", 'r') as f:
    data = json.load(f)
    
    print("Aggregated scores:")
    for metric, stats in data['aggregated'].items():
        print(f"  {metric}: {stats['mean']:.3f}")
    
    print(f"\nTotal questions evaluated: {len(data['results'])}")
```

## Example 9: CI/CD Integration

```python
# test_rag_quality.py
from evaluator import RAGEvaluator
from production_rag import get_rag_function

def test_rag_meets_quality_standards():
    """Test that RAG system meets quality standards."""
    rag_function = get_rag_function()
    evaluator = RAGEvaluator(rag_function)
    results = evaluator.evaluate(verbose=False)
    
    # Assert minimum quality thresholds
    assert results["faithfulness"]["mean"] >= 0.8, \
        f"Faithfulness too low: {results['faithfulness']['mean']:.3f}"
    
    assert results["answer_relevance"]["mean"] >= 0.75, \
        f"Relevance too low: {results['answer_relevance']['mean']:.3f}"
    
    assert results["context_precision"]["mean"] >= 0.85, \
        f"Precision too low: {results['context_precision']['mean']:.3f}"
    
    print("✓ All quality checks passed!")

if __name__ == "__main__":
    test_rag_quality()
```

Run in CI:
```bash
pytest test_rag_quality.py
```

## Example 10: Monitoring Over Time

```python
import json
from datetime import datetime
from evaluator import RAGEvaluator
from production_rag import get_rag_function

def monitor_rag_quality():
    """Monitor RAG quality over time."""
    rag_function = get_rag_function()
    evaluator = RAGEvaluator(rag_function)
    results = evaluator.evaluate(verbose=False)
    
    # Add timestamp
    results['timestamp'] = datetime.now().isoformat()
    
    # Append to history
    try:
        with open("quality_history.json", 'r') as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []
    
    history.append(results)
    
    with open("quality_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Check for degradation
    if len(history) >= 2:
        prev = history[-2]
        curr = history[-1]
        
        for metric in ['faithfulness', 'answer_relevance']:
            prev_score = prev[metric]['mean']
            curr_score = curr[metric]['mean']
            
            if curr_score < prev_score - 0.05:  # 5% drop
                print(f"⚠️  Warning: {metric} dropped from {prev_score:.3f} to {curr_score:.3f}")

# Run daily
monitor_rag_quality()
```

## Tips for Best Results

1. **Start with baseline** - Always measure before making changes
2. **Change one thing** - Isolate variables for clear attribution
3. **Use enough test cases** - 20-50 minimum for reliable results
4. **Validate manually** - Spot-check automated scores
5. **Track over time** - Monitor quality trends
6. **Set thresholds** - Define minimum acceptable scores
7. **Iterate systematically** - Measure → Change → Measure

## Common Patterns

### Pattern: Iterative Improvement
```
1. Baseline evaluation
2. Identify lowest metric
3. Make targeted improvement
4. Re-evaluate
5. Keep if improved, discard if not
6. Repeat
```

### Pattern: Trade-off Analysis
```
Improvement A: +10% recall, -5% precision
Improvement B: +5% recall, +2% precision

Choose B: Better overall balance
```

### Pattern: Regression Detection
```
Before deployment:
  Faithfulness: 0.85

After deployment:
  Faithfulness: 0.72

Action: Rollback and investigate
```

## Next Steps

- Create your own test dataset
- Evaluate your production RAG
- Set up CI/CD integration
- Monitor quality over time
- Build evaluation dashboard

Happy evaluating! 📊
