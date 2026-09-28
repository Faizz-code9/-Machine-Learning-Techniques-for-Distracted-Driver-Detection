"""
Main training pipeline for Distracted Driver Detection.
Ties together all models, data loading, evaluation, and visualization.
"""

import os
import sys
import argparse
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CLASS_NAMES, DEVICE, RANDOM_SEED, NUM_EPOCHS_NN, NUM_EPOCHS_CNN, LEARNING_RATE_NN, LEARNING_RATE_CNN
from data_loader import load_images_flat, get_data_loaders
from evaluate import compute_metrics, print_results_table, save_results
from visualize import (
    plot_confusion_matrix, plot_training_history,
    plot_per_class_accuracy, plot_model_comparison
)

# Traditional ML models
from models.naive_bayes import train_naive_bayes, evaluate_naive_bayes
from models.decision_tree import train_decision_tree, evaluate_decision_tree
from models.random_forest import train_random_forest, evaluate_random_forest
from models.svm import train_svm, evaluate_svm
from models.softmax import train_softmax, evaluate_softmax

# Neural Network models
from models.neural_net import TwoLayerNet, train_neural_net, evaluate_neural_net
from models.cnn_model import create_cnn_model, train_cnn, evaluate_cnn


def run_traditional_ml(data_dir: str) -> dict:
    """
    Trains and evaluates all Traditional ML models on flattened image vectors.

    Args:
        data_dir: Path to dataset directory containing c0-c9 subdirectories.

    Returns:
        Dictionary mapping model names to their evaluation results.
    """
    print("=" * 60)
    print("PHASE 1: Traditional ML Models")
    print("=" * 60)

    X_train, X_val, y_train, y_val = load_images_flat(data_dir)
    print(f"\nData loaded: {X_train.shape[0]} train, {X_val.shape[0]} val samples")
    print(f"Feature dimension: {X_train.shape[1]}\n")

    results = {}

    # 1. Naive Bayes
    print("-" * 40)
    nb_model = train_naive_bayes(X_train, y_train)
    nb_results = evaluate_naive_bayes(nb_model, X_val, y_val)
    results['Naive Bayes'] = nb_results
    print(f"Naive Bayes Accuracy: {nb_results['accuracy']:.4f}\n")

    # 2. Decision Tree
    print("-" * 40)
    dt_model = train_decision_tree(X_train, y_train)
    dt_results = evaluate_decision_tree(dt_model, X_val, y_val)
    results['Decision Tree'] = dt_results
    print(f"Decision Tree Accuracy: {dt_results['accuracy']:.4f}\n")

    # 3. Random Forest
    print("-" * 40)
    rf_model = train_random_forest(X_train, y_train, n_estimators=100)
    rf_results = evaluate_random_forest(rf_model, X_val, y_val)
    results['Random Forest'] = rf_results
    print(f"Random Forest Accuracy: {rf_results['accuracy']:.4f}\n")

    # 4. Linear SVM
    print("-" * 40)
    svm_model = train_svm(X_train, y_train, C=1.0, max_iter=2000)
    svm_results = evaluate_svm(svm_model, X_val, y_val)
    results['Linear SVM'] = svm_results
    print(f"Linear SVM Accuracy: {svm_results['accuracy']:.4f}\n")

    # 5. Softmax (Logistic Regression)
    print("-" * 40)
    sm_model = train_softmax(X_train, y_train, C=1.0, max_iter=2000)
    sm_results = evaluate_softmax(sm_model, X_val, y_val)
    results['Softmax'] = sm_results
    print(f"Softmax Accuracy: {sm_results['accuracy']:.4f}\n")

    return results, y_val


def run_neural_net(data_dir: str) -> tuple:
    """
    Trains and evaluates the 2-Layer Neural Network on flattened image vectors.

    Returns:
        Tuple of (model, history, results_dict, y_val, y_pred)
    """
    print("=" * 60)
    print("PHASE 2: 2-Layer Neural Network")
    print("=" * 60)

    X_train, X_val, y_train, y_val = load_images_flat(data_dir)

    model = TwoLayerNet(input_size=X_train.shape[1], hidden_size=512, num_classes=10)
    model, history = train_neural_net(
        model, X_train, y_train, X_val, y_val,
        epochs=NUM_EPOCHS_NN, lr=LEARNING_RATE_NN
    )

    accuracy, predictions = evaluate_neural_net(model, X_val, y_val)
    print(f"\n2-Layer Neural Net Accuracy: {accuracy:.4f}")

    results = {'accuracy': accuracy, 'predictions': predictions}
    return model, history, results, y_val, predictions


def run_cnn(data_dir: str, model_name: str = 'resnet50') -> tuple:
    """
    Trains and evaluates the CNN model with transfer learning.

    Returns:
        Tuple of (model, history, results_dict, y_true, y_pred)
    """
    print("=" * 60)
    print(f"PHASE 3: CNN Transfer Learning ({model_name})")
    print("=" * 60)

    train_loader, val_loader = get_data_loaders(data_dir, img_size=224, batch_size=32)

    model = create_cnn_model(model_name=model_name, num_classes=10, pretrained=True)
    model, history = train_cnn(
        model, train_loader, val_loader,
        epochs=NUM_EPOCHS_CNN, lr=LEARNING_RATE_CNN, device=str(DEVICE)
    )

    accuracy, predictions, targets = evaluate_cnn(model, val_loader, device=str(DEVICE))
    print(f"\n{model_name} Accuracy: {accuracy:.4f}")

    results = {'accuracy': accuracy, 'predictions': predictions}
    return model, history, results, targets, predictions


def run_all(data_dir: str):
    """
    Runs all training pipelines, generates visualizations, and saves results.
    """
    all_results = {}

    # Phase 1: Traditional ML
    trad_results, y_val_trad = run_traditional_ml(data_dir)
    for name, res in trad_results.items():
        all_results[name] = {'accuracy': res['accuracy']}

    # Phase 2: Neural Network
    nn_model, nn_history, nn_results, y_val_nn, y_pred_nn = run_neural_net(data_dir)
    all_results['2-Layer Neural Net'] = {'accuracy': nn_results['accuracy']}

    # Phase 3: CNN
    cnn_model, cnn_history, cnn_results, y_true_cnn, y_pred_cnn = run_cnn(data_dir)
    all_results['ResNet50 (Transfer)'] = {'accuracy': cnn_results['accuracy']}

    # Print comparison table
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print_results_table(all_results)

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Model comparison chart
    acc_dict = {name: res['accuracy'] for name, res in all_results.items()}
    plot_model_comparison(acc_dict)

    # Neural Net training curves
    plot_training_history(
        nn_history['train_loss'], nn_history['val_loss'],
        nn_history['train_acc'], nn_history['val_acc']
    )

    # CNN confusion matrix
    plot_confusion_matrix(y_true_cnn, y_pred_cnn, CLASS_NAMES)
    plot_per_class_accuracy(y_true_cnn, y_pred_cnn, CLASS_NAMES)

    # Save results
    save_results(all_results)
    print("\nAll done! Check the results/ directory for plots and metrics.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train ML models for Distracted Driver Detection"
    )
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Path to the dataset train directory (containing c0-c9 folders)'
    )
    parser.add_argument(
        '--mode', type=str, default='all',
        choices=['all', 'traditional', 'neural_net', 'cnn'],
        help='Which models to run'
    )
    args = parser.parse_args()

    np.random.seed(RANDOM_SEED)

    if args.mode == 'all':
        run_all(args.data_dir)
    elif args.mode == 'traditional':
        run_traditional_ml(args.data_dir)
    elif args.mode == 'neural_net':
        run_neural_net(args.data_dir)
    elif args.mode == 'cnn':
        run_cnn(args.data_dir)
