# 🚗 Machine Learning Techniques for Distracted Driver Detection

A comprehensive machine learning project that classifies driver behavior from in-car images into 10 categories (safe driving + 9 distraction types) using both traditional ML and deep learning approaches.

## 📋 Problem Statement

Distracted driving is a leading cause of road accidents. This project builds and compares multiple ML classifiers to detect distracted driving activities from dashboard camera images, using the [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection) dataset.

## 🏷️ Classes

| Label | Activity | Label | Activity |
|-------|----------|-------|----------|
| c0 | Safe Driving | c5 | Operating Radio |
| c1 | Texting - Right | c6 | Drinking |
| c2 | Talking on Phone - Right | c7 | Reaching Behind |
| c3 | Texting - Left | c8 | Hair and Makeup |
| c4 | Talking on Phone - Left | c9 | Talking to Passenger |

## 🔬 Models Implemented

### Traditional ML (scikit-learn)
- **Gaussian Naive Bayes** — Probabilistic baseline
- **Decision Tree** — Non-parametric classifier
- **Random Forest** — Ensemble of decision trees
- **Linear SVM** — Support Vector Machine with linear kernel
- **Softmax (Logistic Regression)** — Multinomial logistic regression

### Deep Learning (PyTorch)
- **2-Layer Neural Network** — Fully connected network with ReLU
- **CNN with Transfer Learning** — ResNet50 / VGG16 pretrained on ImageNet

## 📁 Project Structure

```
Project-ML/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── distracted_driver_detection.ipynb   # Main Kaggle notebook
├── src/
│   ├── __init__.py
│   ├── config.py              # Hyperparameters & constants
│   ├── data_loader.py         # Dataset loading & preprocessing
│   ├── train.py               # Training pipeline
│   ├── evaluate.py            # Evaluation & metrics
│   ├── visualize.py           # Plots & visualizations
│   └── models/
│       ├── __init__.py
│       ├── naive_bayes.py
│       ├── decision_tree.py
│       ├── random_forest.py
│       ├── svm.py
│       ├── softmax.py
│       ├── neural_net.py
│       └── cnn_model.py
├── results/                   # Saved plots & metrics
└── docs/
    ├── writeup.pdf
    └── presentation.pptx
```

## 🚀 Setup & Usage

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/distracted-driver-detection.git
cd distracted-driver-detection

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

1. Install Kaggle API: `pip install kaggle`
2. Place your `kaggle.json` API key in `~/.kaggle/`
3. Download the dataset:
```bash
kaggle competitions download -c state-farm-distracted-driver-detection
unzip state-farm-distracted-driver-detection.zip -d data/
```

### Running on Kaggle (Recommended)

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Create a new notebook
3. Add the "State Farm Distracted Driver Detection" dataset
4. Upload `notebooks/distracted_driver_detection.ipynb`
5. Enable GPU accelerator
6. Run all cells

### Running Locally

```bash
# Run all models
python -m src.train --data_dir data/train

# Or run specific model types
python -m src.train --data_dir data/train --mode traditional
python -m src.train --data_dir data/train --mode neural_net
python -m src.train --data_dir data/train --mode cnn
```

## 📊 Empirical Results & Comparison

| Classifier | Input Format | Stanford CS229 Baseline | Our Validation Accuracy |
| :--- | :---: | :---: | :---: |
| **Gaussian Naive Bayes** | 64×64 Flat (12,288) | 54.99% | **59.81%** |
| **Decision Tree** | 64×64 Flat (12,288) | 84.73% | **90.78%** |
| **Random Forest (100 Trees)** | 64×64 Flat (12,288) | *N/A* | **99.26%** |
| **Linear SVM (SGD)** | 64×64 Flat (12,288) | 71.39% | **97.88%** |
| **Softmax (Logistic Regression)**| 64×64 Flat (12,288) | 82.31% | **97.57%** |
| **2-Layer Neural Network** | 64×64 Flat (12,288) | 92.24% | **99.06%** |
| **ResNet-50 (Transfer + Fine-Tuning)** | 224×224×3 Tensors | *N/A* | **98.63%** |


## 📚 References

1. [Stanford CS229 Report — ML Techniques for Distracted Driver Detection](https://cs229.stanford.edu/proj2019spr/report/24.pdf)
2. [State Farm Distracted Driver Detection — Kaggle](https://www.kaggle.com/c/state-farm-distracted-driver-detection)

## 👥 Team

- Member 1 — Data preprocessing, Traditional ML models
- Member 2 — Neural networks, CNN, Evaluation & analysis

---

*UE24CS352A — Machine Learning Mini-Project, Sep–Oct 2026*
