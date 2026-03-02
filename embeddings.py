"""
Embedding utilities for evaluation.
Reused from previous repos.
"""

import requests
import numpy as np
from typing import List


def get_embedding(
    text: str,
    model: str = "nomic-embed-text",
    normalize: bool = True
) -> np.ndarray:
    """
    Generate embedding vector for text using Ollama.
    
    Args:
        text: Input text to embed
        model: Ollama embedding model name
        normalize: Whether to normalize vector to unit length
    
    Returns:
        numpy array of floats (embedding vector)
    """
    url = "http://localhost:11434/api/embed"
    
    payload = {
        "model": model,
        "input": text
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        embedding = np.array(response.json()["embeddings"][0], dtype=np.float32)
        
        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding
    
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        print("Make sure Ollama is running: ollama serve")
        print(f"And model is pulled: ollama pull {model}")
        raise


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(vec1, vec2))
