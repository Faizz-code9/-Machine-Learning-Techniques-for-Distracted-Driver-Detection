"""
Support Vector Machine (SVM) model for Distracted Driver Detection.
"""

import time
from typing import Dict, Any
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def train_svm(X_train: np.ndarray, y_train: np.ndarray, C: float = 1.0, max_iter: int = 2000) -> Pipeline:
    """
    Train a Linear Support Vector Classification model.
    Uses a pipeline with StandardScaler for better performance on large datasets.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        C: Regularization parameter.
        max_iter: Maximum number of iterations for the solver.
        
    Returns:
        Trained Pipeline model containing StandardScaler and LinearSVC.
    """
    print(f"Training Linear SVC (C={C}, max_iter={max_iter})...")
    start_time = time.time()
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', LinearSVC(C=C, max_iter=max_iter, random_state=42, dual=False))
    ])
    
    model.fit(X_train, y_train)
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    return model

def evaluate_svm(model: Pipeline, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate the Linear SVC model.
    
    Args:
        model: Trained SVM pipeline.
        X_val: Validation features.
        y_val: Validation labels.
        
    Returns:
        Dictionary containing accuracy, predictions, and classification report.
    """
    print("Evaluating Linear SVC...")
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    report = classification_report(y_val, predictions, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'report': report
    }
