# Part 1: Neural Network Fundamentals and Training Behavior Analysis

## Overview

Binary classification neural network to predict **customer churn** using the `customer_churn_nn.csv` dataset.

| Property | Value |
|---|---|
| Dataset | customer_churn_nn.csv |
| Task | Binary Classification (Churn: 0 / 1) |
| Rows | 2,000 |
| Features | 16 (11 numerical, 4 categorical, 1 ID) |
| Framework | TensorFlow / Keras |
| Key Challenge | Heavy class imbalance (~1.55% churn) |

---

## Repository Structure

```
part-1-neural-network-analysis/
│
├── README.md
├── notebook.ipynb              ← Main notebook (all 6 tasks)
├── requirements.txt
└── results/
    ├── target_distribution.png
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── experiment_comparison.png
    ├── model_comparison_table.png
    └── evaluation_outputs.png
```

---

## Tasks Covered

| Task | Description |
|---|---|
| Task 1 | Dataset exploration — shape, types, missing values, distribution |
| Task 2 | Preprocessing — drop ID, OHE encoding, StandardScaler, train/test split, class weights |
| Task 3 | Model architecture — Input → Dense(ReLU) → BatchNorm → Dropout → Dense(Sigmoid) |
| Task 4 | Training & evaluation — accuracy, AUC, confusion matrix, classification report |
| Task 5 | 5 hyperparameter experiments (layers, neurons, LR, batch size, activation) |
| Task 6 | Written reflection — weights/biases, activation functions, LR effects, overfitting |

---

## Model Architecture (Baseline)

```
Input (27 features)
   ↓
Dense(64, ReLU) + BatchNormalization + Dropout(0.3)
   ↓
Dense(32, ReLU) + Dropout(0.2)
   ↓
Dense(1, Sigmoid)  ← churn probability output

Loss      : Binary Crossentropy
Optimizer : Adam (lr=0.001)
Regularizer: L2(0.001)
Class Weights: Applied (~63x for minority churn class)
```

---

## Hyperparameter Experiments

| Exp | Change | Key Finding |
|---|---|---|
| Exp 1 | Baseline (2 layers, 64 neurons, lr=0.001, ReLU) | Good AUC baseline |
| Exp 2 | Deeper (4 layers, 128 neurons) | May overfit on small churn set |
| Exp 3 | High LR (0.01) | Faster convergence, risk of instability |
| Exp 4 | Low LR (0.0001) | Slow convergence, may underfit with early stopping |
| Exp 5 | Tanh activation + batch=128 | Smoother gradients, larger batches need more epochs |

---

## Setup & Run

```bash
# Clone repo and install dependencies
pip install -r requirements.txt

# Place customer_churn_nn.csv in the same folder as notebook.ipynb
# Then launch Jupyter
jupyter notebook notebook.ipynb
```

---

## Key Insight

Raw accuracy (~98%) is misleading due to class imbalance. Focus on **AUC-ROC** and **Recall (Churn class)** as primary metrics. Class weighting is critical for the model to learn meaningful patterns from the minority churn class.
