"""
RAG evaluation metrics.

Implements:
- Faithfulness (answer grounded in context)
- Answer Relevance (answer relevant to question)
- Context Precision (retrieved docs are relevant)
- Context Recall (all relevant info retrieved)
"""

import json
from typing import List, Dict
from llm import call_llm
from embeddings import get_embedding, cosine_similarity


def evaluate_faithfulness(
    question: str,
    answer: str,
    context: List[str]
) -> float:
    """
    Evaluate if answer is faithful to the context (no hallucinations).
    
    Returns:
        Score between 0 and 1 (1 = fully faithful, 0 = hallucinated)
    """
    context_text = "\n\n".join(context)
    
    prompt = f"""Given a question, answer, and context, determine if the answer is faithful to the context.
An answer is faithful if all claims in the answer are supported by the context.

Question: {question}

Context:
{context_text}

Answer: {answer}

Task: Extract claims from the answer and check if each is supported by the context.

Respond in JSON format:
{{
    "claims": ["claim1", "claim2", ...],
    "supported": [true, false, ...],
    "faithfulness_score": 0.0-1.0
}}"""
    
    try:
        response = call_llm(prompt, temperature=0.0, json_mode=True)
        result = json.loads(response)
        return float(result.get("faithfulness_score", 0.0))
    except:
        # Fallback: simple keyword matching
        answer_lower = answer.lower()
        context_lower = context_text.lower()
        
        # Check if answer words are in context
        answer_words = set(answer_lower.split())
        context_words = set(context_lower.split())
        
        overlap = len(answer_words & context_words)
        score = overlap / len(answer_words) if answer_words else 0.0
        
        return min(score, 1.0)


def evaluate_answer_relevance(
    question: str,
    answer: str
) -> float:
    """
    Evaluate if answer is relevant to the question.
    
    Returns:
        Score between 0 and 1 (1 = highly relevant, 0 = not relevant)
    """
    try:
        # Use embeddings for semantic similarity
        question_emb = get_embedding(question)
        answer_emb = get_embedding(answer)
        
        similarity = cosine_similarity(question_emb, answer_emb)
        
        # Normalize to 0-1 range (cosine similarity is already -1 to 1, but we use normalized vectors)
        score = (similarity + 1) / 2
        
        return float(score)
    except:
        # Fallback: keyword overlap
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        overlap = len(question_words & answer_words)
        score = overlap / len(question_words) if question_words else 0.0
        
        return min(score, 1.0)


def evaluate_context_precision(
    question: str,
    contexts: List[str],
    ground_truth: str = None
) -> float:
    """
    Evaluate if retrieved contexts are relevant to the question.
    
    Returns:
        Score between 0 and 1 (1 = all contexts relevant, 0 = none relevant)
    """
    if not contexts:
        return 0.0
    
    relevant_count = 0
    
    for context in contexts:
        prompt = f"""Is the following context relevant to answering the question?

Question: {question}

Context: {context}

Answer with just "yes" or "no"."""
        
        try:
            response = call_llm(prompt, temperature=0.0).lower()
            if "yes" in response:
                relevant_count += 1
        except:
            # Fallback: keyword matching
            question_words = set(question.lower().split())
            context_words = set(context.lower().split())
            overlap = len(question_words & context_words)
            if overlap >= 2:  # At least 2 words in common
                relevant_count += 1
    
    return relevant_count / len(contexts)


def evaluate_context_recall(
    question: str,
    contexts: List[str],
    ground_truth: str
) -> float:
    """
    Evaluate if contexts contain all information needed to answer the question.
    
    Returns:
        Score between 0 and 1 (1 = all info present, 0 = missing info)
    """
    context_text = "\n\n".join(contexts)
    
    prompt = f"""Given a question, ground truth answer, and retrieved contexts, determine if the contexts contain all information needed to produce the ground truth answer.

Question: {question}

Ground Truth Answer: {ground_truth}

Retrieved Contexts:
{context_text}

Task: Check if all key facts from the ground truth are present in the contexts.

Respond in JSON format:
{{
    "key_facts": ["fact1", "fact2", ...],
    "facts_found": [true, false, ...],
    "recall_score": 0.0-1.0
}}"""
    
    try:
        response = call_llm(prompt, temperature=0.0, json_mode=True)
        result = json.loads(response)
        return float(result.get("recall_score", 0.0))
    except:
        # Fallback: check if ground truth words are in context
        ground_truth_words = set(ground_truth.lower().split())
        context_words = set(context_text.lower().split())
        
        overlap = len(ground_truth_words & context_words)
        score = overlap / len(ground_truth_words) if ground_truth_words else 0.0
        
        return min(score, 1.0)


def evaluate_all(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str = None
) -> Dict[str, float]:
    """
    Evaluate all metrics for a single QA pair.
    
    Returns:
        Dictionary with all metric scores
    """
    scores = {
        "faithfulness": evaluate_faithfulness(question, answer, contexts),
        "answer_relevance": evaluate_answer_relevance(question, answer),
        "context_precision": evaluate_context_precision(question, contexts, ground_truth)
    }
    
    if ground_truth:
        scores["context_recall"] = evaluate_context_recall(question, contexts, ground_truth)
    
    return scores
