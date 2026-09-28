"""
Visualization utilities for the Distracted Driver Detection project.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix

def _ensure_dir(path: str):
    """Ensure the directory for the given path exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

def plot_sample_images(data_dir: str, class_names: list, n_per_class: int = 2):
    """
    Shows a grid of sample images from each class.

    Args:
        data_dir (str): Root directory containing class subdirectories.
        class_names (list): List of class names.
        n_per_class (int): Number of images to show per class.
    """
    fig, axes = plt.subplots(len(class_names), n_per_class, figsize=(n_per_class * 4, len(class_names) * 3))
    if len(class_names) == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for i, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, f'c{i}')
        if not os.path.isdir(class_path):
            continue
        
        img_paths = glob.glob(os.path.join(class_path, '*.jpg'))[:n_per_class]
        
        for j, img_path in enumerate(img_paths):
            img = Image.open(img_path)
            ax = axes[i, j] if n_per_class > 1 else axes[i]
            ax.imshow(img)
            ax.axis('off')
            if j == 0:
                ax.set_title(class_name, loc='left', fontsize=12)

    plt.tight_layout()
    save_path = 'results/sample_images.png'
    _ensure_dir(save_path)
    plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plots a confusion matrix heatmap using seaborn.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class names.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    save_path = 'results/confusion_matrix.png'
    _ensure_dir(save_path)
    plt.savefig(save_path)
    plt.show()

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """
    Plots loss and accuracy curves.

    Args:
        train_losses: List of training losses.
        val_losses: List of validation losses.
        train_accs: List of training accuracies.
        val_accs: List of validation accuracies.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()

    plt.tight_layout()
    save_path = 'results/training_history.png'
    _ensure_dir(save_path)
    plt.savefig(save_path)
    plt.show()

def plot_per_class_accuracy(y_true, y_pred, class_names):
    """
    Horizontal bar chart of per-class accuracy.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class names.
    """
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=per_class_acc, y=class_names, palette='viridis')
    plt.xlabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.xlim(0, 1)
    plt.tight_layout()
    save_path = 'results/per_class_accuracy.png'
    _ensure_dir(save_path)
    plt.savefig(save_path)
    plt.show()

def plot_model_comparison(results_dict):
    """
    Bar chart comparing validation accuracy of all models.

    Args:
        results_dict: Dictionary mapping model names to their validation accuracy.
    """
    models = list(results_dict.keys())
    accuracies = list(results_dict.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=accuracies, y=models, palette='magma')
    plt.xlabel('Validation Accuracy')
    plt.title('Model Comparison')
    plt.xlim(0, 1)
    
    for i, acc in enumerate(accuracies):
        plt.text(acc + 0.01, i, f'{acc:.4f}', va='center')
        
    plt.tight_layout()
    save_path = 'results/model_comparison.png'
    _ensure_dir(save_path)
    plt.savefig(save_path)
    plt.show()
