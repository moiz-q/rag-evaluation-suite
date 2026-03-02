"""
Main evaluation script.

Run this to evaluate your RAG system.
"""

import json
from evaluator import RAGEvaluator
from simple_rag import create_rag_function
from test_dataset import get_test_dataset


def main():
    print("="*60)
    print("RAG EVALUATION SUITE")
    print("="*60)
    print("\nThis will evaluate your RAG system on a test dataset.")
    print("Metrics: Faithfulness, Answer Relevance, Context Precision, Context Recall")
    print()
    
    # Create RAG function
    print("Initializing RAG system...")
    rag_function = create_rag_function()
    
    # Create evaluator
    evaluator = RAGEvaluator(rag_function)
    
    # Run evaluation
    print("\nStarting evaluation...")
    results = evaluator.evaluate(verbose=True)
    
    # Save results
    evaluator.save_results("evaluation_results.json")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print("\nResults saved to evaluation_results.json")
    print("\nNext steps:")
    print("1. Review the scores above")
    print("2. Make improvements to your RAG system")
    print("3. Run evaluation again to measure improvement")
    print("4. Use compare_improvements.py to see the difference")


if __name__ == "__main__":
    main()
