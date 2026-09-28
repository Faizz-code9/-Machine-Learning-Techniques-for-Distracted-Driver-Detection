"""
Random Forest model for Distracted Driver Detection.
"""

import time
from typing import Dict, Any, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 100, max_depth: Optional[int] = None) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of the trees.
        
    Returns:
        Trained RandomForestClassifier model.
    """
    print(f"Training Random Forest (n_estimators={n_estimators}, max_depth={max_depth})...")
    start_time = time.time()
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    return model

def evaluate_random_forest(model: RandomForestClassifier, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate the Random Forest model.
    
    Args:
        model: Trained RandomForestClassifier model.
        X_val: Validation features.
        y_val: Validation labels.
        
    Returns:
        Dictionary containing accuracy, predictions, and classification report.
    """
    print("Evaluating Random Forest...")
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    report = classification_report(y_val, predictions, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'report': report
    }
