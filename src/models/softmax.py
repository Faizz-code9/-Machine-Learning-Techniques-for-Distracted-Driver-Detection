"""
Softmax (Multinomial Logistic Regression) model for Distracted Driver Detection.
"""

import time
from typing import Dict, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def train_softmax(X_train: np.ndarray, y_train: np.ndarray, C: float = 1.0, max_iter: int = 2000) -> Pipeline:
    """
    Train a Multinomial Logistic Regression (Softmax) model.
    Uses a pipeline with StandardScaler for better performance.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        C: Inverse of regularization strength.
        max_iter: Maximum number of iterations for the solver.
        
    Returns:
        Trained Pipeline model containing StandardScaler and LogisticRegression.
    """
    print(f"Training Softmax/Logistic Regression (C={C}, max_iter={max_iter})...")
    start_time = time.time()
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('softmax', LogisticRegression(
            multi_class='multinomial', 
            solver='lbfgs', 
            C=C, 
            max_iter=max_iter, 
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    model.fit(X_train, y_train)
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    return model

def evaluate_softmax(model: Pipeline, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate the Softmax model.
    
    Args:
        model: Trained Softmax pipeline.
        X_val: Validation features.
        y_val: Validation labels.
        
    Returns:
        Dictionary containing accuracy, predictions, and classification report.
    """
    print("Evaluating Softmax/Logistic Regression...")
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    report = classification_report(y_val, predictions, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'report': report
    }
