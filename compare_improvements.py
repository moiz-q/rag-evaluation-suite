"""
Compare RAG system improvements.

This script compares baseline results with improved results.
"""

import json
from evaluator import RAGEvaluator
from simple_rag import create_rag_function


def load_baseline(filepath: str = "baseline_results.json"):
    """Load baseline evaluation results."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            return data.get("aggregated", {})
    except FileNotFoundError:
        print(f"Baseline file not found: {filepath}")
        print("Run evaluation first and save as baseline_results.json")
        return None


def main():
    print("="*60)
    print("RAG IMPROVEMENT COMPARISON")
    print("="*60)
    
    # Load baseline
    baseline = load_baseline()
    if baseline is None:
        print("\nNo baseline found. Running initial evaluation...")
        rag_function = create_rag_function()
        evaluator = RAGEvaluator(rag_function)
        results = evaluator.evaluate(verbose=True)
        evaluator.save_results("baseline_results.json")
        print("\nBaseline saved! Make improvements and run this script again.")
        return
    
    print("\nBaseline loaded. Running current evaluation...")
    
    # Run current evaluation
    rag_function = create_rag_function()
    evaluator = RAGEvaluator(rag_function)
    current_results = evaluator.evaluate(verbose=False)
    
    # Compare
    evaluator.print_comparison(baseline)
    
    # Save current results
    evaluator.save_results("current_results.json")
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print("\nCurrent results saved to current_results.json")
    print("\nInterpretation:")
    print("  ✓ Positive values = Improvement")
    print("  ✗ Negative values = Regression")
    print("  = Zero = No change")


if __name__ == "__main__":
    main()
