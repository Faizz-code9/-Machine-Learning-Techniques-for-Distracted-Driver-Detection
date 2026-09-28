import json
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def compute_metrics(y_true, y_pred, class_names=None):
    """
    Computes accuracy, per-class accuracy, classification report, and confusion matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    return {
        'accuracy': acc,
        'per_class_accuracy': per_class_acc.tolist(),
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }

def print_results_table(results_dict):
    """
    Prints a formatted comparison table of all models based on accuracy.
    """
    print("-" * 40)
    print(f"{'Model Name':<20} | {'Accuracy':<15}")
    print("-" * 40)
    for model_name, metrics in results_dict.items():
        acc = metrics.get('accuracy', 0.0)
        print(f"{model_name:<20} | {acc:.4f}")
    print("-" * 40)

def save_results(results_dict, save_path='results/results.json'):
    """
    Saves the results dictionary to a JSON file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"Results saved to {save_path}")
