"""
Naive Bayes model for Distracted Driver Detection.
"""

import time
from typing import Dict, Any
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

def train_naive_bayes(X_train: np.ndarray, y_train: np.ndarray) -> GaussianNB:
    """
    Train a Gaussian Naive Bayes classifier.
    
    Args:
        X_train: Training features (flattened image vectors).
        y_train: Training labels.
        
    Returns:
        Trained GaussianNB model.
    """
    print("Training Gaussian Naive Bayes...")
    start_time = time.time()
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    return model

def evaluate_naive_bayes(model: GaussianNB, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate the Gaussian Naive Bayes model.
    
    Args:
        model: Trained GaussianNB model.
        X_val: Validation features.
        y_val: Validation labels.
        
    Returns:
        Dictionary containing accuracy, predictions, and classification report.
    """
    print("Evaluating Gaussian Naive Bayes...")
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    report = classification_report(y_val, predictions, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'report': report
    }
