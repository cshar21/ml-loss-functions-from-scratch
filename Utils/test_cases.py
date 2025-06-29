"""test_cases.py
A central place to define example inputs and expected outputs for loss functions.

Helps you test each function under the same conditions for fair comparison.

Good for automated testing."""

# utils/test_cases.py

import numpy as np

# === Regression Losses ===
from Regression.mse import mse_loss
from Regression.mae import mae_loss
from Regression.huber import huber_loss

# === Classification Losses ===
from Classification.binary_crossentropy import binary_crossentropy
from Classification.categorical_crossentropy import categorical_crossentropy
from Classification.hinge import hinge_loss

print("✅ Running standard test cases...\n")

# Sample regression data
y_true_reg = np.array([3.0, -0.5, 2.0, 7.0])
y_pred_reg = np.array([2.5, 0.0, 2.0, 8.0])

# Sample binary classification data
y_true_bin = np.array([0, 1, 1, 0])
y_pred_bin = np.array([0.1, 0.9, 0.8, 0.2])

# Sample categorical classification data (one-hot)
y_true_cat = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
y_pred_cat = np.array([
    [0.9, 0.05, 0.05],
    [0.1, 0.8, 0.1],
    [0.2, 0.2, 0.6]
])

# Sample hinge classification data
y_true_hinge = np.array([1, -1, 1, -1])
y_pred_hinge = np.array([1.5, -0.8, 0.5, 1.0])


#  Run all test cases

print(" MSE Loss:", mse_loss(y_true_reg, y_pred_reg))
print(" MAE Loss:", mae_loss(y_true_reg, y_pred_reg))
print(" Huber Loss:", huber_loss(y_true_reg, y_pred_reg))

print("\n Binary Crossentropy:", binary_crossentropy(y_true_bin, y_pred_bin))
print(" Categorical Crossentropy:", categorical_crossentropy(y_true_cat, y_pred_cat))
print(" Hinge Loss:", hinge_loss(y_true_hinge, y_pred_hinge))
