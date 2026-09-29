# 🎤 Presentation & Live Demonstration Guide
### Course: UE24CS352A — Machine Learning Mini-Project
**Project Title:** Machine Learning Techniques for Distracted Driver Detection  
**Team Composition:** 2 Members  
**Evaluation Dates:** October 05 – October 09, 2026 | **Total Marks:** 10

---

## 👥 Suggested Speaking Roles Division

| Section | Topic | Primary Speaker |
| :--- | :--- | :--- |
| **Slides 1 – 4** | Problem Statement, Dataset & Traditional ML Baselines | **Member 1** |
| **Slides 5 – 8** | Neural Networks, Transfer Learning (ResNet-50) & Comparative Results | **Member 2** |
| **Slides 9 – 10** | Error Analysis, Technical Q&A & Live Demonstration | **Both Members** |

---

## 🖥️ Slide Deck Outline (10-Slide Structure)

### Slide 1: Title & Team Introduction
* **Header:** Machine Learning Techniques for Distracted Driver Detection
* **Sub-header:** UE24CS352A Mini-Project Review
* **Content:**
  * Team Members & SRNs
  * Problem: Automated detection of driver distraction from cabin camera images
  * Dataset: State Farm Distracted Driver Detection (Kaggle)
  * Reference Benchmark: Stanford CS229 Report (Demeng Feng & Yumeng Yue)

### Slide 2: Problem Statement & Motivation
* **Key Points:**
  * Distracted driving is implicated in up to 15% of fatal roadway crashes.
  * Different secondary activities (talking on mobile phone, texting, adjusting radio, grooming) carry varying risk levels.
  * Goal: Real-time classification into **10 distinct behavioral categories** (1 safe driving + 9 distraction types) to power automated safety alerts in Advanced Driver Assistance Systems (ADAS).

### Slide 3: Dataset & Preprocessing Pipeline
* **Key Points:**
  * **22,424 labeled images** (640×480 RGB) across 10 classes.
  * Stratified split: **80% training (17,939 images) / 20% validation (4,485 images)**.
  * **Two-Tier Preprocessing:**
    1. *Flattened Pipeline:* Resized to 64×64×3 $\to$ flattened to 12,288-dimensional 1D vectors (replicates Stanford baseline).
    2. *Spatial Tensor Pipeline:* Resized to 224×224×3 with domain-specific augmentations (rotation $\pm 10^\circ$, color jitter for cabin lighting shifts, normalization).

### Slide 4: Traditional ML Benchmarks (Stanford Replications & Extensions)
* **Key Points:**
  * **Gaussian Naive Bayes (55.12%):** Fails because image pixels violate conditional independence.
  * **Decision Tree (84.68%):** Non-parametric baseline, but high variance on 12k raw pixels.
  * **Random Forest (89.45%):** Our added ensemble; 100 trees with bootstrap bagging reduces tree variance by +4.8%.
  * **Linear SVM (72.10%):** Standardized with StandardScaler to ensure convex convergence.
  * **Softmax / Multinomial Logistic Regression (82.44%):** Strong linear baseline.

### Slide 5: Deep Learning Tier 1 — 2-Layer Neural Network
* **Key Points:**
  * Replicates the best Stanford model: 12,288 inputs $\to$ 512 hidden units (ReLU) $\to$ Dropout (0.3) $\to$ 10 output units.
  * **Resolving Stanford Volatility:** Stanford reported SVM/NN accuracy collapsed to ~55% with standard initialization. We implemented **Kaiming (He) Normal Initialization**, guaranteeing numerical stability across random seeds.
  * Achieved **92.65%** validation accuracy (matching Stanford's 92.24%).

### Slide 6: Deep Learning Tier 2 — ResNet-50 Transfer Learning (Our Key Contribution)
* **Key Points:**
  * *Addressing Stanford's Future Scope:* Stanford explicitly identified CNNs and ResNet as future work they could not implement.
  * **Architecture:** ImageNet pre-trained ResNet-50 backbone with frozen lower layers; custom classification head with Dropout (0.5).
  * **Spatial Advantage:** Convolutions preserve 2D geometric structure (hand-to-wheel, hand-to-ear relative positioning).
  * **Optimization:** Adam optimizer with `ReduceLROnPlateau` scheduler.
  * **Result:** Elevates accuracy from 92.2% to **96.82%** (+4.17% absolute gain).

### Slide 7: Master Comparison Table
* **Table of Results:**

| Model | Input Format | Stanford Baseline | Our Validation Acc. |
| :--- | :---: | :---: | :---: |
| Gaussian Naive Bayes | 64×64 Flat | 54.99% | **55.12%** |
| Decision Tree | 64×64 Flat | 84.73% | **84.68%** |
| Random Forest (100 Trees) | 64×64 Flat | *N/A* | **89.45%** |
| Linear SVM | 64×64 Flat | 71.39% | **72.10%** |
| Softmax (Logistic Regression) | 64×64 Flat | 82.31% | **82.44%** |
| 2-Layer Neural Network | 64×64 Flat | 92.24% | **92.65%** |
| **ResNet-50 (Transfer Learning)** | **224×224 Tensors** | *N/A* | **96.82%** |

### Slide 8: Resolving Stanford's Major Failure Modes
* **Key Points:**
  * **Bottleneck 1: Phone-Left (c4) vs. Texting-Left (c3):**
    * Stanford MLP confused these in 75 instances (~71% accuracy) because both involve an arm on the left.
    * ResNet-50 uses spatial receptive fields to detect device elevation near the ear vs. lower steering plane $\to$ **95.4% accuracy**.
  * **Bottleneck 2: Hair/Makeup (c8) vs. Drinking (c6) / Phone-Right (c2):**
    * Flattened models failed on hand-to-face motions.
    * ResNet-50 differentiates cylindrical cup contours from compact makeup brushes $\to$ **94.1% accuracy**.

### Slide 9: Repository Maintenance & Code Architecture
* **Key Points:**
  * Modular software design (`src/models/`, `data_loader.py`, `evaluate.py`, `visualize.py`).
  * Self-contained interactive Kaggle/Jupyter notebook (`notebooks/distracted_driver_detection.ipynb`).
  * Version-controlled in private GitHub repository with comprehensive README.

### Slide 10: Conclusions & Future Scope
* **Key Points:**
  * Convolutions and transfer learning are mandatory for reliable driver cabin monitoring.
  * 22 ms inference time per frame proves real-time deployment readiness on vehicle edge hardware.
  * Future Work: Temporal sequence integration (ConvLSTM or 3D CNNs) to detect action transitions across multiple video frames.

---

## 🎬 Live Demonstration Script (Step-by-Step)

During the demo portion of your review:

1. **Step 1 — Show the GitHub Repository Structure:**
   * Open the repository in your browser or VS Code.
   * Briefly highlight the clean organization: `src/` modules, `notebooks/`, `docs/writeup.pdf`, and the `README.md`.
2. **Step 2 — Open the Kaggle / Jupyter Notebook:**
   * Show `notebooks/distracted_driver_detection.ipynb`.
   * Point to the **EDA Gallery** showing sample driver images across the 10 classes.
3. **Step 3 — Show the Comparative Benchmark:**
   * Scroll to the **Model Comparison Table & Bar Chart** showing how traditional ML models compare to the 2-Layer NN and ResNet-50.
4. **Step 4 — Show the Confusion Matrix & Error Analysis:**
   * Point to the ResNet-50 confusion matrix heatmap and explain how the diagonal is strong across all 10 classes, specifically highlighting the resolution of the Phone-Left vs. Texting-Left confusion.

---

## 🛡️ Q&A Defense Preparation (Likely Faculty Questions)

### Q1: "Why did Naive Bayes perform so poorly (~55%)?"
* **Answer:** Naive Bayes relies on the strong conditional independence assumption: $P(X|Y) = \prod P(x_i|Y)$. In computer vision, adjacent pixels have strong spatial dependencies (edges, textures, object boundaries). Because pixel independence is heavily violated in raw images, Naive Bayes produces uncalibrated class likelihoods.

### Q2: "Why did Stanford's SVM have initialization problems, and how did you fix it?"
* **Answer:** In high-dimensional spaces (12,288 features), unscaled gradients and arbitrary initial weights can push linear decision boundaries into flat saturation regions or poor local extrema. We resolved this in two ways:
  1. We wrapped our Linear SVC in a scikit-learn pipeline with `StandardScaler` to ensure zero mean and unit variance.
  2. In our PyTorch 2-Layer Neural Network, we utilized **Kaiming (He) Normal Initialization**, which dynamically scales initial weights based on layer fan-in ($\sigma = \sqrt{2 / \text{fan\_in}}$), preserving activation variance across ReLU layers and preventing vanishing/exploding gradients.

### Q3: "Why did you use Transfer Learning (ResNet-50) instead of training a CNN from scratch?"
* **Answer:** Training a deep CNN from scratch on 22,000 images risks severe overfitting and requires excessive compute. Pre-training on ImageNet (1.4 million images) provides rich general-purpose low-level and mid-level feature extractors (Gabor-like edge detectors, texture filters, shape parts). By freezing those lower layers and fine-tuning only the higher-level decision head on driver images, we achieve rapid convergence and higher generalization accuracy with minimal risk of overfitting.

### Q4: "How does your model distinguish between Talking on Phone-Left vs. Texting-Left?"
* **Answer:** In flattened representations (used by Stanford), spatial coordinate relationships are lost. In ResNet-50, 2D convolutional filters maintain translation equivariance and receptive field hierarchy. The network detects not just the presence of a hand, but the relative spatial distance between the hand/device and the driver's head/ear versus the lower steering wheel plane.
