"""
Decision Tree model for Distracted Driver Detection.
"""

import time
from typing import Dict, Any, Optional
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

def train_decision_tree(X_train: np.ndarray, y_train: np.ndarray, max_depth: Optional[int] = None) -> DecisionTreeClassifier:
    """
    Train a Decision Tree classifier with optional cross-validation.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        max_depth: Maximum depth of the tree.
        
    Returns:
        Trained DecisionTreeClassifier model.
    """
    print(f"Training Decision Tree (max_depth={max_depth})...")
    start_time = time.time()
    
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    
    # Optional: cross-validation to check performance during training
    # cv_scores = cross_val_score(model, X_train, y_train, cv=3, n_jobs=-1)
    # print(f"3-fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    model.fit(X_train, y_train)
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    return model

def evaluate_decision_tree(model: DecisionTreeClassifier, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate the Decision Tree model.
    
    Args:
        model: Trained DecisionTreeClassifier model.
        X_val: Validation features.
        y_val: Validation labels.
        
    Returns:
        Dictionary containing accuracy, predictions, and classification report.
    """
    print("Evaluating Decision Tree...")
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    report = classification_report(y_val, predictions, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'report': report
    }
