"""
Simple RAG system for evaluation.
This is a basic RAG implementation to demonstrate evaluation.
"""

import os
import sys
from typing import List, Tuple
from llm import call_llm
from embeddings import get_embedding, cosine_similarity
import numpy as np


# Sample knowledge base (in production, load from files)
KNOWLEDGE_BASE = {
    "python_intro.txt": """Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991. Python emphasizes code readability with its notable use of significant whitespace. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.""",
    
    "python_basics.md": """# Python Basics

## Variables
A variable is a named storage location in memory that holds a value. In Python, you don't need to declare variable types explicitly.

Example:
```python
name = "Alice"
age = 25
```

## Data Types
Python has several built-in data types:
- int: integers (whole numbers)
- float: floating-point numbers (decimals)
- str: strings (text)
- bool: booleans (True/False)
- list: ordered, mutable collections
- tuple: ordered, immutable collections
- dict: key-value pairs
- set: unordered collections of unique items

## Functions
A function in Python is a reusable block of code that performs a specific task. Functions are defined using the 'def' keyword.

Example:
```python
def greet(name):
    return f"Hello, {name}!"
```""",
    
    "machine_learning.txt": """Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.

There are three main types of machine learning:

1. Supervised Learning: The algorithm learns from labeled training data. The model is trained on input-output pairs and learns to predict outputs for new inputs. Examples include classification and regression.

2. Unsupervised Learning: The algorithm learns patterns from unlabeled data. It tries to find hidden structure in the data. Examples include clustering and dimensionality reduction.

3. Reinforcement Learning: The algorithm learns through trial and error, receiving rewards or penalties for actions. It's commonly used in robotics and game playing.""",
    
    "exercise_guide.txt": """# Exercise Guide

## Benefits of Regular Exercise
Regular physical activity is one of the most important things you can do for your health. Exercise provides numerous benefits:

- Improves cardiovascular health and strengthens the heart
- Helps maintain healthy weight
- Strengthens muscles and bones
- Reduces stress and anxiety
- Improves mental health and mood
- Boosts energy levels
- Improves sleep quality
- Reduces risk of chronic diseases

## Types of Exercise

### Cardiovascular Exercise
Cardiovascular exercise, or cardio, is physical activity that raises your heart rate and improves the efficiency of your cardiovascular system. Examples include:
- Running and jogging
- Swimming
- Cycling
- Dancing
- Brisk walking

### Strength Training
Strength training involves using resistance to build muscle strength and endurance. This can include:
- Weight lifting
- Bodyweight exercises (push-ups, squats)
- Resistance bands
- Gym machines

## Recommendations
Adults should aim for at least 150 minutes of moderate-intensity aerobic activity or 75 minutes of vigorous-intensity activity per week, plus muscle-strengthening activities on 2 or more days per week.""",
    
    "healthy_eating.txt": """# Healthy Eating Guide

## Balanced Diet
A balanced diet includes a variety of foods from all food groups:

- Fruits and vegetables: 5+ servings per day
- Whole grains: brown rice, whole wheat bread, oats
- Lean proteins: chicken, fish, beans, nuts
- Healthy fats: avocados, olive oil, nuts
- Dairy or alternatives: milk, yogurt, cheese

## Heart-Healthy Foods
Foods that are particularly good for heart health include:
- Fatty fish rich in omega-3 (salmon, mackerel, sardines)
- Nuts and seeds (almonds, walnuts, flaxseeds)
- Berries (blueberries, strawberries, raspberries)
- Leafy green vegetables (spinach, kale)
- Whole grains
- Avocados
- Olive oil

Limit saturated fats, trans fats, sodium, and added sugars.

## Hydration
Water is essential for health. Most health authorities recommend drinking about 8 glasses (64 ounces or 2 liters) of water per day. Individual needs vary based on:
- Activity level
- Climate and temperature
- Overall health
- Pregnancy or breastfeeding

Signs of proper hydration include clear or light-colored urine and not feeling thirsty."""
}


class SimpleRAG:
    """Simple RAG system for evaluation."""
    
    def __init__(self):
        self.knowledge_base = KNOWLEDGE_BASE
        self.embeddings_cache = {}
        self._build_index()
    
    def _build_index(self):
        """Build embeddings index for knowledge base."""
        print("Building embeddings index...")
        for doc_name, content in self.knowledge_base.items():
            self.embeddings_cache[doc_name] = get_embedding(content)
        print(f"Indexed {len(self.embeddings_cache)} documents")
    
    def retrieve(self, query: str, k: int = 2) -> List[Tuple[str, str]]:
        """
        Retrieve top-k relevant documents.
        
        Returns:
            List of (doc_name, content) tuples
        """
        query_emb = get_embedding(query)
        
        # Calculate similarities
        similarities = []
        for doc_name, doc_emb in self.embeddings_cache.items():
            sim = cosine_similarity(query_emb, doc_emb)
            similarities.append((doc_name, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        results = []
        for doc_name, sim in similarities[:k]:
            results.append((doc_name, self.knowledge_base[doc_name]))
        
        return results
    
    def generate(self, query: str, contexts: List[str]) -> str:
        """Generate answer using LLM."""
        context_text = "\n\n".join(contexts)
        
        prompt = f"""Answer the question based on the context below. Be concise and accurate.

Context:
{context_text}

Question: {query}

Answer:"""
        
        return call_llm(prompt, temperature=0.0)
    
    def query(self, question: str, k: int = 2) -> Tuple[str, List[str]]:
        """
        Query the RAG system.
        
        Returns:
            (answer, list of context strings)
        """
        # Retrieve relevant documents
        retrieved = self.retrieve(question, k=k)
        contexts = [content for _, content in retrieved]
        
        # Generate answer
        answer = self.generate(question, contexts)
        
        return answer, contexts


def create_rag_function():
    """Create a RAG function for evaluation."""
    rag = SimpleRAG()
    
    def rag_function(question: str) -> Tuple[str, List[str]]:
        return rag.query(question)
    
    return rag_function


if __name__ == "__main__":
    # Test the RAG system
    rag = SimpleRAG()
    
    test_questions = [
        "What is Python?",
        "What are the benefits of exercise?",
        "What is a variable?"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        answer, contexts = rag.query(question)
        print(f"Answer: {answer}")
        print(f"Used {len(contexts)} contexts")
