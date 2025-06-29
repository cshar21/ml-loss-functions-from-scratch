# regression/mae.py

import numpy as np
import matplotlib.pyplot as plt

# =========================================================
#  Mean Absolute Error (MAE) — Regression Loss Function
# =========================================================
# Definition:
#   Measures the average of the absolute differences between predicted
#   and actual values.
#
# Formula:
#   MAE = (1/n) * Σ |y_pred_i - y_true_i|
#
# Properties:
#   - More robust to outliers than MSE (does not square the error)
#   - penalizes all errors linearly
#   - Not differentiable at zero, but subgradient can be used
#   - L1 loss function

def mae(y_true, y_pred):
    """
    Compute the Mean Absolute Error (MAE) between predicted and true values.

    Parameters:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: The MAE value.
    """
    # abs_errors = np.abs(y_true - y_pred)
    # return np.mean(abs_errors)

    return np.mean(np.abs(y_pred - y_true))

def mae_derivative(y_true, y_pred):
    """
    Compute the subgradient of the MAE loss with respect to the predictions.

    Note: MAE is not differentiable at 0, but we can define the subgradient.
    The subgradient is the derivative of the MAE loss when the predicted value 
    is greater than the true value, and the negative derivative when the predicted
    value is less than the true value.

    Case wise derivates:
    - If y_true >= y_pred,loss increases as y_pred increases, then d/da MAE = -1
    - If y_true <= y_pred,loss decreases as y_pred increases, then d/da MAE = 1
    - If y_true == y_pred, then d/da MAE = 0

    Parameters:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        np.ndarray: Subgradient array (sign of error).
        # sign() returns -1 if negative, +1 if positive, 0 if zero
    """
    sign = np.sign(y_pred - y_true)
    # Divide by number of elements
    return sign / len(y_true)


# Run test and visualize
if __name__ == "__main__":
    # Example data
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 2.0])

    # Compute and print MAE
    print("✅ MAE:", mae(y_true, y_pred))

    # Compute and print gradient
    print("📐 MAE Derivative (Sign):", mae_derivative(y_true, y_pred))

    # Visualize the loss curve
    errors = np.linspace(-3, 3, 100)
    loss_values = np.abs(errors)

    plt.figure(figsize=(6, 4))
    #This creates a V-shaped graph
    #It’s piecewise linear, unlike MSE which was a smooth parabola.
    #Constant gradient = ±1
    #Sharp point (non-differentiable) at y_true = y_pred

    plt.plot(errors, loss_values, label='MAE Loss Curve', color='orange')
    plt.title('📈 Mean Absolute Error Loss vs. Prediction Error')
    plt.xlabel('Prediction Error (y_pred - y_true)')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Save plot to file 
    plt.savefig(r"C:\Users\DELL\OneDrive\Documents\CLUB WORK\ML loss functions\graphs\mae_graph.png")
    plt.show()
