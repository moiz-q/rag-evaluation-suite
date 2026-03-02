"""
RAG System Evaluator.

Evaluates a RAG system against a test dataset and computes metrics.
"""

import json
from typing import List, Dict, Callable
from metrics import evaluate_all
from test_dataset import get_test_dataset


class RAGEvaluator:
    """Evaluates RAG systems using test datasets."""
    
    def __init__(self, rag_function: Callable):
        """
        Initialize evaluator.
        
        Args:
            rag_function: Function that takes a question and returns (answer, contexts)
        """
        self.rag_function = rag_function
        self.results = []
    
    def evaluate(self, test_dataset: List[Dict] = None, verbose: bool = True) -> Dict:
        """
        Evaluate RAG system on test dataset.
        
        Args:
            test_dataset: List of test cases with questions and ground truth
            verbose: Print progress
        
        Returns:
            Dictionary with aggregated metrics
        """
        if test_dataset is None:
            test_dataset = get_test_dataset()
        
        self.results = []
        
        for i, test_case in enumerate(test_dataset):
            if verbose:
                print(f"\nEvaluating {i+1}/{len(test_dataset)}: {test_case['question'][:50]}...")
            
            question = test_case["question"]
            ground_truth = test_case["ground_truth"]
            
            # Get RAG system response
            try:
                answer, contexts = self.rag_function(question)
            except Exception as e:
                print(f"Error getting RAG response: {e}")
                continue
            
            # Evaluate metrics
            scores = evaluate_all(question, answer, contexts, ground_truth)
            
            # Store result
            result = {
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "contexts": contexts,
                "scores": scores
            }
            self.results.append(result)
            
            if verbose:
                print(f"  Faithfulness: {scores['faithfulness']:.2f}")
                print(f"  Answer Relevance: {scores['answer_relevance']:.2f}")
                print(f"  Context Precision: {scores['context_precision']:.2f}")
                if "context_recall" in scores:
                    print(f"  Context Recall: {scores['context_recall']:.2f}")
        
        # Aggregate scores
        aggregated = self._aggregate_scores()
        
        if verbose:
            print("\n" + "="*60)
            print("EVALUATION RESULTS")
            print("="*60)
            self._print_summary(aggregated)
        
        return aggregated
    
    def _aggregate_scores(self) -> Dict:
        """Aggregate scores across all test cases."""
        if not self.results:
            return {}
        
        metrics = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]
        aggregated = {}
        
        for metric in metrics:
            scores = [r["scores"][metric] for r in self.results if metric in r["scores"]]
            if scores:
                aggregated[metric] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores)
                }
        
        return aggregated
    
    def _print_summary(self, aggregated: Dict):
        """Print evaluation summary."""
        for metric, stats in aggregated.items():
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  Mean:  {stats['mean']:.3f}")
            print(f"  Min:   {stats['min']:.3f}")
            print(f"  Max:   {stats['max']:.3f}")
            print(f"  Count: {stats['count']}")
    
    def save_results(self, filepath: str = "evaluation_results.json"):
        """Save detailed results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump({
                "results": self.results,
                "aggregated": self._aggregate_scores()
            }, f, indent=2)
        print(f"\nResults saved to {filepath}")
    
    def compare_with_baseline(self, baseline_results: Dict) -> Dict:
        """
        Compare current results with baseline.
        
        Args:
            baseline_results: Previous evaluation results
        
        Returns:
            Dictionary with improvements/regressions
        """
        current = self._aggregate_scores()
        comparison = {}
        
        for metric in current.keys():
            if metric in baseline_results:
                current_score = current[metric]["mean"]
                baseline_score = baseline_results[metric]["mean"]
                improvement = current_score - baseline_score
                improvement_pct = (improvement / baseline_score * 100) if baseline_score > 0 else 0
                
                comparison[metric] = {
                    "baseline": baseline_score,
                    "current": current_score,
                    "improvement": improvement,
                    "improvement_pct": improvement_pct
                }
        
        return comparison
    
    def print_comparison(self, baseline_results: Dict):
        """Print comparison with baseline."""
        comparison = self.compare_with_baseline(baseline_results)
        
        print("\n" + "="*60)
        print("COMPARISON WITH BASELINE")
        print("="*60)
        
        for metric, stats in comparison.items():
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  Baseline: {stats['baseline']:.3f}")
            print(f"  Current:  {stats['current']:.3f}")
            
            improvement = stats['improvement']
            improvement_pct = stats['improvement_pct']
            
            if improvement > 0:
                print(f"  ✓ Improvement: +{improvement:.3f} (+{improvement_pct:.1f}%)")
            elif improvement < 0:
                print(f"  ✗ Regression: {improvement:.3f} ({improvement_pct:.1f}%)")
            else:
                print(f"  = No change")


def evaluate_rag_system(rag_function: Callable, verbose: bool = True) -> Dict:
    """
    Convenience function to evaluate a RAG system.
    
    Args:
        rag_function: Function that takes a question and returns (answer, contexts)
        verbose: Print progress
    
    Returns:
        Aggregated evaluation metrics
    """
    evaluator = RAGEvaluator(rag_function)
    return evaluator.evaluate(verbose=verbose)
