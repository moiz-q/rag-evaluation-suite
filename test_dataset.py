"""
Test dataset with ground truth answers.

This is a sample evaluation dataset for RAG systems.
In production, you'd have a larger, domain-specific dataset.
"""

from typing import List, Dict


# Sample test dataset
TEST_DATASET = [
    {
        "question": "What is Python?",
        "ground_truth": "Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
        "relevant_docs": ["python_intro.txt"]
    },
    {
        "question": "What are the benefits of regular exercise?",
        "ground_truth": "Regular exercise improves cardiovascular health, strengthens muscles and bones, helps maintain healthy weight, reduces stress, and improves mental health.",
        "relevant_docs": ["exercise_guide.txt"]
    },
    {
        "question": "What is a variable in programming?",
        "ground_truth": "A variable is a named storage location in memory that holds a value. Variables can store different types of data like numbers, text, or boolean values.",
        "relevant_docs": ["python_basics.md"]
    },
    {
        "question": "How does machine learning work?",
        "ground_truth": "Machine learning is a method where computers learn patterns from data without being explicitly programmed. It uses algorithms to analyze data, learn from it, and make predictions or decisions.",
        "relevant_docs": ["machine_learning.txt"]
    },
    {
        "question": "What foods are good for heart health?",
        "ground_truth": "Foods good for heart health include fruits, vegetables, whole grains, lean proteins, nuts, and foods rich in omega-3 fatty acids like fish. Limiting saturated fats and sodium is also important.",
        "relevant_docs": ["healthy_eating.txt"]
    },
    {
        "question": "What is a function in Python?",
        "ground_truth": "A function in Python is a reusable block of code that performs a specific task. Functions are defined using the 'def' keyword and can accept parameters and return values.",
        "relevant_docs": ["python_basics.md"]
    },
    {
        "question": "What is cardiovascular exercise?",
        "ground_truth": "Cardiovascular exercise, or cardio, is physical activity that raises your heart rate and improves the efficiency of your cardiovascular system. Examples include running, swimming, and cycling.",
        "relevant_docs": ["exercise_guide.txt"]
    },
    {
        "question": "What is supervised learning?",
        "ground_truth": "Supervised learning is a type of machine learning where the algorithm learns from labeled training data. The model is trained on input-output pairs and learns to predict outputs for new inputs.",
        "relevant_docs": ["machine_learning.txt"]
    },
    {
        "question": "What are data types in Python?",
        "ground_truth": "Python has several built-in data types including integers (int), floating-point numbers (float), strings (str), booleans (bool), lists, tuples, dictionaries, and sets.",
        "relevant_docs": ["python_basics.md", "python_intro.txt"]
    },
    {
        "question": "How much water should you drink daily?",
        "ground_truth": "Most health authorities recommend drinking about 8 glasses (64 ounces or 2 liters) of water per day, though individual needs vary based on activity level, climate, and health conditions.",
        "relevant_docs": ["healthy_eating.txt"]
    }
]


def get_test_dataset() -> List[Dict]:
    """Get the test dataset."""
    return TEST_DATASET


def get_questions() -> List[str]:
    """Get just the questions from the test dataset."""
    return [item["question"] for item in TEST_DATASET]


def get_ground_truth(question: str) -> str:
    """Get ground truth answer for a question."""
    for item in TEST_DATASET:
        if item["question"] == question:
            return item["ground_truth"]
    return None


def get_relevant_docs(question: str) -> List[str]:
    """Get list of relevant document names for a question."""
    for item in TEST_DATASET:
        if item["question"] == question:
            return item["relevant_docs"]
    return []
